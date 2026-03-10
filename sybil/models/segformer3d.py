import math
from typing import Optional, Union, List
import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean, scatter_max

eps = 1e-6


def compute_instance_avg(risk_map, instance_mask, hidden_dim, aggregation="mean"):
    # Flatten
    risk_flat = risk_map.view(hidden_dim, -1)  # [H*W]
    inst_flat = instance_mask.view(-1)  # [H*W]
    # Remove background if needed
    valid = inst_flat > 0  # or > 0 if 0 is background
    risk_flat = risk_flat[:, valid]
    inst_flat = inst_flat[valid].long()
    # Compute mean risk per instance
    if aggregation == "mean":
        avg_risks = scatter_mean(risk_flat, inst_flat, -1)  # [num_instances]
    elif aggregation == "sum":
        avg_risks = scatter_add(risk_flat, inst_flat, -1)  # [num_instances]
    elif aggregation == "max":
        avg_risks = scatter_max(risk_flat, inst_flat, -1)[0]  # [num_instances]
    return avg_risks  # indexed by instance ID

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channel: int = 4,
        embed_dim: int = 768,
        kernel_size: Union[int, List[int]] = [7, 7, 7],
        stride: Union[int, List[int]] = [4, 4, 4],
        padding: Union[int, List[int]] = [3, 3, 3],
    ):
        super().__init__()
        # Convert single integers to lists if necessary
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(stride, int):
            stride = [stride] * 3
        if isinstance(padding, int):
            padding = [padding] * 3

        self.patch_embeddings = nn.Conv3d(
            in_channel,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Store for shape calculations
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        patches = self.patch_embeddings(x)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        return patches

    def get_output_shape(self, input_shape):
        """Calculate output spatial dimensions after convolution"""
        d, h, w = input_shape
        od = ((d + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]) + 1
        oh = ((h + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]) + 1
        ow = ((w + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2]) + 1
        return (od, oh, ow)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, spatial_shape):
        B, N, C = x.shape
        d, h, w = spatial_shape
        x = x.transpose(1, 2).view(B, C, d, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_dim = embed_dim // num_heads
        self.scale = self.attention_head_dim**-0.5

        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, spatial_shape, output_attentions: bool = False):
        D, H, W = spatial_shape
        B, N, C = x.shape
        q = (
            self.query(x)
            .reshape(B, N, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = (
                self.key_value(x_)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.key_value(x)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        if output_attentions:
            return x, attn
        return (x,)


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_features, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, hidden_features)
        self.conv = DWConv(hidden_features)
        self.linear_2 = nn.Linear(hidden_features, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spatial_shape):
        x = self.linear_1(x)
        x = self.conv(x, spatial_shape)
        x = self.linear_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)

        self.mlp = MLP(embed_dim, hidden_features, proj_dropout)

    def forward(self, x, spatial_shape, output_attentions: bool = False):
        attention_outputs = self.attention(
            self.norm1(x), spatial_shape, output_attentions
        )
        x = x + attention_outputs[0]
        x = x + self.mlp(self.norm2(x), spatial_shape)

        outputs = (x,) + attention_outputs[1:] if output_attentions else (x,)
        return outputs, spatial_shape


class SegFormer3DModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: list = [4, 2, 1, 1],
        embed_dims: list = [32, 64, 160, 256],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [4, 4, 4, 4],
        num_heads: list = [1, 2, 5, 8],
        depths: list = [2, 2, 2, 2],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        decoder_dropout: float = 0.0,
    ):
        super().__init__()

        # Encoder components
        self.encoders = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()

        # Build hierarchical encoder stages
        for i in range(4):
            # Patch embedding for this stage
            encoder = PatchEmbedding(
                in_channel=in_channels if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                kernel_size=patch_kernel_size[i],
                stride=patch_stride[i],
                padding=patch_padding[i],
            )
            self.encoders.append(encoder)

            # Transformer blocks for this stage
            stage_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        embed_dim=embed_dims[i],
                        num_heads=num_heads[i],
                        sr_ratio=sr_ratios[i],
                        mlp_ratio=mlp_ratios[i],
                    )
                    for _ in range(depths[i])
                ]
            )
            self.transformer_blocks.append(stage_blocks)

            # Layer norm for this stage
            self.encoder_norms.append(nn.LayerNorm(embed_dims[i]))

        # Decoder components
        self.decoder = SegFormer3DDecoderHead(
            input_feature_dims=embed_dims,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes,
            dropout=decoder_dropout,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            )
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv3d):
            fan_out = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.kernel_size[2]
                * module.out_channels
            )
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def get_input_embeddings(self):
        return self.encoders[0].patch_embeddings

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        encoder_hidden_states = []
        all_attentions = [] if output_attentions else None

        x = pixel_values

        # Process through encoder stages
        for stage_idx in range(4):
            # Track input spatial dimensions
            if stage_idx == 0:
                spatial_shape = pixel_values.shape[2:]  # (D, H, W)

            # Patch embedding
            x = self.encoders[stage_idx](x)
            spatial_shape = self.encoders[stage_idx].get_output_shape(spatial_shape)
            B, N, C = x.shape

            # Transformer blocks
            for block in self.transformer_blocks[stage_idx]:
                block_outputs, spatial_shape = block(
                    x, spatial_shape, output_attentions
                )
                x = block_outputs[0]
                if output_attentions:
                    all_attentions.append(block_outputs[1])

            # Layer norm
            x = self.encoder_norms[stage_idx](x)

            # Reshape and store hidden state using calculated dimensions
            d, h, w = spatial_shape
            x_reshaped = x.reshape(B, d, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
            encoder_hidden_states.append(x_reshaped)

            # Prepare input for next stage if not last stage
            if stage_idx < 3:
                x = x_reshaped

        # Decode features
        x = self.decoder(encoder_hidden_states)
        return x


class SegFormer3DDecoderHead(nn.Module):
    def __init__(
        self,
        input_feature_dims: list = [512, 320, 128, 64],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Linear layers for each encoder stage
        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, decoder_head_embedding_dim),
                    nn.LayerNorm(decoder_head_embedding_dim),
                )
                for dim in input_feature_dims[::-1]
            ]
        )

        # Feature fusion
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * decoder_head_embedding_dim,
                out_channels=decoder_head_embedding_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)
        self.linear_pred = nn.Conv3d(
            decoder_head_embedding_dim, num_classes, kernel_size=1
        )
        self.upsample = nn.Upsample(
            scale_factor=4, mode="trilinear", align_corners=False
        )

    def forward(self, encoder_hidden_states):
        # Process features from each encoder stage
        B = encoder_hidden_states[-1].shape[0]

        # Linear projection and upsampling of each stage's features
        decoded_features = []
        for i, features in enumerate(
            encoder_hidden_states[::-1]
        ):  # Process in reverse order
            d, h, w = features.shape[2:]
            projected = (
                self.linear_layers[i](features.flatten(2).transpose(1, 2))
                .transpose(1, 2)
                .reshape(B, -1, d, h, w)
            )

            # Upsample if not the last feature map
            if i != len(encoder_hidden_states[::-1]):
                projected = torch.nn.functional.interpolate(
                    projected,
                    size=encoder_hidden_states[0].shape[
                        2:
                    ],  # Size of first stage features
                    mode="trilinear",
                    align_corners=False,
                )
            decoded_features.append(projected)

        # Fuse all features
        fused_features = self.linear_fuse(torch.cat(decoded_features, dim=1))

        # Final prediction
        x = self.dropout(fused_features)
        x = self.linear_pred(x)
        x = self.upsample(x)

        return x

