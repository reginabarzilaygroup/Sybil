import argparse
import datetime
import os

from sybil import Serie, Sybil

def test_create_sybilnet():
    from sybil.models.sybil import SybilNet

    fake_args = argparse.Namespace(
        dropout=0.1,
        max_followup=5,
        )

    sybil_net = SybilNet(fake_args)

    assert sybil_net.hidden_dim == 512
    assert sybil_net.prob_of_failure_layer is not None
