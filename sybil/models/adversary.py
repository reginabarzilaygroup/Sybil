import torch
import torch.nn as nn

class AlignmentMLP(nn.Module):
    '''
        Simple MLP discriminator to be used as adversary for alignment of hiddens.
    '''
    def __init__(self, args):
        super(AlignmentMLP, self).__init__()
        self.args = args

        # calculate input size based on chosn layers (default: just 'hidden')
        discrim_input_size = args.hidden_dim 

        # init discriminator
        self.model = nn.Sequential(
            nn.Linear(discrim_input_size, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, model_output, batch=None):
        # concatenate hiddens of chosen layers
        hiddens = model_output['hidden']
        # pass hiddens through mlp
        output = {
            'logit': self.model(hiddens)
        }

        return output
