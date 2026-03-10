import torch
import torch.nn as nn

from sybilx.models.pillar.pooling.abstract import AbstractPooling


class GlobalMaxPool(AbstractPooling):
    """
    Pool to obtain the maximum value for each channel
    """

    def __init__(self, args, **extras):
        super(GlobalMaxPool, self).__init__(args)

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict. output['hidden'] is (B, C)
        """
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        hidden, _ = torch.max(x, dim=-1)
        return {"hidden": hidden}


class GlobalAvgPool(AbstractPooling):
    """
    Pool to obtain the average value for each channel
    """

    def __init__(self, args, **extras):
        super(GlobalAvgPool, self).__init__(args)

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict. output['hidden'] is (B, C)
        """
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        hidden = torch.mean(x, dim=-1)
        return {"hidden": hidden}


class SimpleAttentionPool(AbstractPooling):
    """
    Pool to learn an attention over tokens or slices
    """

    def __init__(
        self,
        args,
        hidden_dim,
        output_attention_scores_name="attention_scores",
        **extras,
    ):
        super(SimpleAttentionPool, self).__init__(args)

        self.attention_fc = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # In Sybil, this is "volume_attention". We allow setting the name for more flexibility.
        self.output_attention_scores_name = output_attention_scores_name

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, N)
        returns:
            - output: dict
                + output[self.output_attention_scores_name]: tensor (B, C)
                + output['hidden']: tensor (B, C)
        """
        output = {}
        B = x.shape[0]
        spatially_flat_size = (*x.size()[:2], -1)  # B, C, N

        x = x.view(spatially_flat_size)
        attention_scores = self.attention_fc(x.transpose(1, 2).contiguous())  # B, N, 1

        output[self.output_attention_scores_name] = self.logsoftmax(
            attention_scores.transpose(1, 2)
        ).view(B, -1)
        attention_scores = self.softmax(attention_scores.transpose(1, 2))  # B, 1, N
        output[f"{self.output_attention_scores_name}_softmax"] = attention_scores

        x = x * attention_scores  # B, C, N
        output["hidden"] = torch.sum(x, dim=-1)
        return output


class MaxAvgPool(AbstractPooling):
    """
    Pool to obtain the maximum and average value for each channel
    """

    def __init__(self, hidden_dim=512, **extras):
        super(MaxAvgPool, self).__init__()
        # Pooling
        self.max_pool = GlobalMaxPool()
        self.avg_pool = GlobalAvgPool()

        # Fully connected layer for dimensionality refinement
        self.hidden_fc = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x):
        """
        args:
            - x: tensor of shape (B, C, W, H, D)
        returns:
            - output: dict with:
                + "hidden": Combined hidden representation (B, C)
        """
        output = {}

        # Perform max pooling
        maxpool_out = self.max_pool(x)
        maxpool_hidden = maxpool_out["hidden"]  # (B, C)

        # Perform average pooling
        avgpool_out = self.avg_pool(x)
        avgpool_hidden = avgpool_out["hidden"]  # (B, C)

        # Combine max and avg pooling results
        combined_hidden = torch.cat([maxpool_hidden, avgpool_hidden], dim=1)
        combined_hidden = torch.flatten(combined_hidden, 1)
        combined_hidden = self.hidden_fc(combined_hidden)

        # Populate output dictionary
        output["hidden"] = combined_hidden

        return output
