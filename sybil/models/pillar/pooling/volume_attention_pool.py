import torch
import torch.nn as nn
from einops import rearrange
from .basic_pool_layers import GlobalMaxPool
from sybil.models.pillar.pooling.abstract import AbstractPooling


class AttentivePooling(nn.Module):
    """
    Attentive pooling layer for 3D data.
    Uses conv with same padding to get an attention map (attention weight for each location).
    The weights sum to 1 across the volume.
    """

    def __init__(self, input_dim, output_dim=1, kernel=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(
            input_dim, output_dim, kernel, stride=stride, padding=padding
        )
        nn.init.xavier_normal_(self.conv.weight)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, W, H, D)
        returns:
            - pooled_features: tensor of shape (B, C)
            - attention_map: tensor of shape (B, 1, W, H, D)
        """
        B, C, W, H, D = x.size()
        attention_map = self.conv(x)  # (B, 1, W, H, D)
        attention_map = rearrange(attention_map, "b c w h d -> b (c w h d)")
        attention_map = self.softmax(attention_map)
        attention_map = rearrange(
            attention_map, "b (c w h d) -> b c w h d", c=1, w=W, h=H, d=D
        )
        weighted_x = x * attention_map  # (B, C, W, H, D)
        pooled_features = rearrange(weighted_x, "b c w h d -> b c (w h d)").sum(
            dim=-1
        )  # (B, C)
        return pooled_features, attention_map


class VolumeAttentionPool(AbstractPooling):
    """
    Combines the convolution-based attention pooling with global max pooling.
    """

    def __init__(
        self, args, input_dim=512, hidden_dim=512, kernel=1, stride=1, padding=0
    ):
        super(VolumeAttentionPool, self).__init__(args)

        # Attentive pooling
        self.attention_pool = AttentivePooling(
            input_dim=input_dim,
            output_dim=1,
            kernel=kernel,
            stride=stride,
            padding=padding,
        )

        # Max pooling
        self.max_pool = GlobalMaxPool(args=args)

        # Fully connected layer for dimensionality refinement
        self.hidden_fc = nn.Linear(2 * input_dim, hidden_dim)

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, W, H, D)
        returns:
            - output: dict with:
                + "hidden": Combined hidden representation (B, C)
                + "attention_map": Attention map (B, 1, W, H, D)
        """
        output = {}

        # Perform attentive pooling
        attention_hidden, attention_map = self.attention_pool(x)

        # Perform max pooling
        maxpool_out = self.max_pool(x)
        maxpool_hidden = maxpool_out["hidden"]  # (B, C)

        # Combine attention and max pooling results
        combined_hidden = torch.cat([attention_hidden, maxpool_hidden], dim=1)
        combined_hidden = torch.flatten(combined_hidden, 1)
        combined_hidden = self.hidden_fc(combined_hidden)

        # Populate output dictionary
        output["hidden"] = combined_hidden
        output["volume_attention_map"] = attention_map

        return output


class VolumeAttentionPoolDetached(AbstractPooling):
    """
    Combines the convolution-based attention pooling with global max pooling.
    But: it does  not use the attention pooling for the predictions, only to inform the head training
    """

    def __init__(
        self, args, input_dim=512, hidden_dim=512, kernel=1, stride=1, padding=0
    ):
        super(VolumeAttentionPoolDetached, self).__init__(args)

        # Attentive pooling
        self.attention_pool = AttentivePooling(
            input_dim=input_dim,
            out_dim=1,
            kernel=kernel,
            stride=stride,
            padding=padding,
        )

        # Max pooling
        self.max_pool = GlobalMaxPool(args=args)

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, W, H, D)
        returns:
            - output: dict with:
                + "hidden": Combined hidden representation (B, C)
                + "attention_map": Attention map (B, 1, W, H, D)
        """
        output = {}

        # Perform attentive pooling. Do not use it for prediction
        attention_hidden, attention_map = self.attention_pool(x)
        output["volume_attention_map"] = attention_map

        # Perform max pooling
        maxpool_out = self.max_pool(x)
        output["hidden"] = maxpool_out["hidden"]

        return output
