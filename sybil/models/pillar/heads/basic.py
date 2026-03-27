import torch.nn as nn
import torch.nn.functional as F

from sybil.models.pillar.abstract_model import AbstractModel


class Linear(AbstractModel):
    def __init__(self, args, input_dim, output_dim, dropout_rate=0.0):
        super().__init__(args)

        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, **extras):
        return self.linear(self.dropout(x))


class MLP(AbstractModel):
    def __init__(
        self,
        args,
        input_dim,
        output_dim,
        hidden_dims=[],
        activation="ReLU",
        dropout_rate=0.0,
    ):
        super().__init__(args)

        # Get activation function from nn module
        # .title() capitalizes first letter: "relu" -> "Relu"
        # This matches PyTorch's naming convention: nn.ReLU
        activation_function = getattr(nn, activation)

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_function())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, **extras):
        return self.layers(x)
