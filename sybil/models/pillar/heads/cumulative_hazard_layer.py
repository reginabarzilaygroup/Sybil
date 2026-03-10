import torch
import torch.nn as nn

from sybilx.models.pillar.abstract_model import AbstractModel


class CumulativeProbabilityLayer(AbstractModel):
    def __init__(self, args, input_dim=None, max_followup=None, **extras):
        super(CumulativeProbabilityLayer, self).__init__(args)
        self.hazard_fc = nn.Linear(input_dim, max_followup)
        self.base_hazard_fc = nn.Linear(input_dim, 1)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask).contiguous(), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        hazards = self.hazards(x)
        B, T = hazards.size()  # hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(
            B, T, T
        )  # expanded_hazards is (B,T, T)
        masked_hazards = (
            expanded_hazards * self.upper_triagular_mask
        )  # masked_hazards now (B,T, T)
        base_hazard = self.base_hazard_fc(x)
        cum_prob = torch.sum(masked_hazards, dim=1) + base_hazard
        return cum_prob
