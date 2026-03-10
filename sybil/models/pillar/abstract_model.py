import torch.nn as nn


class AbstractModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True