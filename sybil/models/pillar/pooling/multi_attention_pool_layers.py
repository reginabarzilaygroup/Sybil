# This file is based on Sybil's implementations:
# https://github.com/reginabarzilaygroup/Sybil/blob/main/sybil/models/pooling_layer.py

# The implementation is designed to be as close to the original implementation as possible in order to compare across implementations.

import torch
import torch.nn as nn

from sybilx.models.pillar.pooling.basic_pool_layers import (
    GlobalMaxPool,
    SimpleAttentionPool,
)
from sybilx.models.pillar.pooling.abstract import AbstractPooling


class PerFrameMaxPool(nn.Module):
    """
    Pool to obtain the maximum value for each slice in 3D input
    """

    def __init__(self):
        super(PerFrameMaxPool, self).__init__()

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict.
                + output['multi_image_hidden'] is (B, C, T)
        """
        assert len(x.shape) == 5
        output = {}
        spatially_flat_size = (*x.size()[:3], -1)
        x = x.view(spatially_flat_size)
        output["multi_image_hidden"], _ = torch.max(x, dim=-1)
        return output


class Conv1dAttnPool(nn.Module):
    """
    Pool to learn an attention over the slices after convolution
    """

    def __init__(self, args, **kwargs):
        super(Conv1dAttnPool, self).__init__()
        self.conv1d = nn.Conv1d(
            kwargs["hidden_dim"],
            kwargs["hidden_dim"],
            kernel_size=kwargs["conv_pool_kernel_size"],
            stride=kwargs["stride"],
            padding=kwargs["conv_pool_kernel_size"] // 2,
            bias=False,
        )
        self.aggregate = SimpleAttentionPool(args, **kwargs)

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, T)
        returns:
            - output: dict
                + output['attention_scores']: tensor (B, C)
                + output['hidden']: tensor (B, C)
        """
        # X: B, C, N
        x = self.conv1d(x)  # B, C, N'
        return self.aggregate(x)


class SimpleAttentionPool_MultiImg(nn.Module):
    """
    Pool to learn an attention over the slices and the volume
    """

    def __init__(self, **kwargs):
        super(SimpleAttentionPool_MultiImg, self).__init__()

        self.attention_fc = nn.Linear(kwargs["hidden_dim"], 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict
                + output['attention_scores']: tensor (B, T, C)
                + output['multi_image_hidden']: tensor (B, T, C)
                + output['hidden']: tensor (B, T*C)
        """
        output = {}
        B, C, T, W, H = x.size()
        x = x.permute([0, 2, 1, 3, 4])
        x = x.contiguous().view(B * T, C, W * H)
        attention_scores = self.attention_fc(
            x.transpose(1, 2).contiguous()
        )  # BT, WH , 1

        output["image_attention"] = self.logsoftmax(
            attention_scores.transpose(1, 2)
        ).view(B, T, -1)
        attention_scores = self.softmax(attention_scores.transpose(1, 2))  # BT, 1, WH
        output["image_attention_softmax"] = attention_scores.view(B, T, 1, W, H)

        x = x * attention_scores  # BT, C, WH
        x = torch.sum(x, dim=-1)
        output["multi_image_hidden"] = x.view(B, T, C).permute([0, 2, 1])
        output["hidden"] = x.view(B, T * C)
        return output


class MultiAttentionPool(AbstractPooling):
    def __init__(
        self, args, hidden_dim=512, conv_pool_kernel_size=11, conv_pool_stride=1
    ):
        super(MultiAttentionPool, self).__init__(args)
        params = {
            "hidden_dim": hidden_dim,
            "conv_pool_kernel_size": conv_pool_kernel_size,
            "stride": conv_pool_stride,
        }
        self.image_pool1 = SimpleAttentionPool_MultiImg(**params)
        self.volume_pool1 = SimpleAttentionPool(
            args, output_attention_scores_name="volume_attention", **params
        )

        self.image_pool2 = PerFrameMaxPool()
        self.volume_pool2 = Conv1dAttnPool(
            args, output_attention_scores_name="volume_attention", **params
        )

        self.global_max_pool = GlobalMaxPool(args=args)

        self.hidden_fc = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, x):
        # X dim: B, C, T, W, H
        output = {}

        # contains keys: "multi_image_hidden", "image_attention"
        image_pool_out1 = self.image_pool1(x)
        # contains keys: "hidden", "volume_attention"
        volume_pool_out1 = self.volume_pool1(image_pool_out1["multi_image_hidden"])

        # contains keys: "multi_image_hidden"
        image_pool_out2 = self.image_pool2(x)
        # contains keys: "hidden", "volume_attention"
        volume_pool_out2 = self.volume_pool2(image_pool_out2["multi_image_hidden"])

        for pool_out, num in [
            (image_pool_out1, 1),
            (volume_pool_out1, 1),
            (image_pool_out2, 2),
            (volume_pool_out2, 2),
        ]:
            for key, val in pool_out.items():
                output["{}_{}".format(key, num)] = val

        maxpool_out = self.global_max_pool(x)
        output["maxpool_hidden"] = maxpool_out["hidden"]

        hidden = torch.cat(
            [
                volume_pool_out1["hidden"],
                volume_pool_out2["hidden"],
                output["maxpool_hidden"],
            ],
            dim=-1,
        )
        output["hidden"] = self.hidden_fc(hidden)

        return output
