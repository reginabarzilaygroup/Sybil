"""Models for selector implementation."""
import os.path

import numpy as np
import torch
import torch.nn as nn

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import sybil.selector.features as features


class SelectiveNet(torch.nn.Module):
    """Implements a feed-forward MLP."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout=0.0,
    ):
        super(SelectiveNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, 1)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)

    @property
    def dtype(self):
        return next(self.net.parameters()).dtype


class Selector:
    def __init__(self, all_models_dict=None):
        self._meta_models = None
        self._meta_scaler = None
        self._selector = None

        if all_models_dict is not None:
            self.load_models(all_models_dict)

    def load_models(self, all_models_dict):
        meta_models_dict = all_models_dict["meta_models"]
        self._meta_models = features.load_outlier_models(meta_models_dict["meta_models"])
        self._meta_scaler = features.FeatureScaler().load_state_dict(meta_models_dict["meta_scaler"])
        assert len(self._meta_models) == 6, f"Expected 6 meta models, found {len(self._meta_models)}"
        input_dim, hidden_dim, num_layers = all_models_dict["selector_model"]["model"]
        selector = SelectiveNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers)
        selector.load_state_dict(all_models_dict["selector_model"]["state_dict"])
        self._selector = selector

    @property
    def dtype(self):
        return self._selector.dtype

    @classmethod
    def load_from_file(cls, path):
        all_models_dict = torch.load(path)
        return cls(all_models_dict)

    def forward(self, base_model_results, device=None):
        keep_cols = [-1]
        with torch.no_grad():
            base_features = base_model_results["hidden"].to(device)

            base_features_np = base_features.detach().cpu().numpy()
            base_logits = base_model_results["logit"].to(device)
            base_logits = base_logits[:, keep_cols]

        output_probs = torch.sigmoid(base_logits)
        output_probs_np = output_probs.detach().cpu().numpy()
        # Need the output_probs to have at least 2 classes
        if output_probs_np.shape[-1] == 1:
            output_probs_np = np.concatenate([1 - output_probs_np, output_probs_np], axis=-1)

        base_confidence = torch.max(output_probs, dim=1).values
        base_confidence_np = base_confidence.detach().cpu().numpy()

        meta_features = self.generate_meta_features(input_features=base_features_np,
                                                    output_probs=output_probs_np)
        meta_features = torch.from_numpy(meta_features).to(device).to(self.dtype)

        selector_output = self._selector(meta_features)
        selector_output_np = selector_output.detach().cpu().numpy()
        selector_probs_np = torch.sigmoid(selector_output).detach().cpu().numpy()

        output_dict = {
            "base_confidence": base_confidence_np,
            "logit": selector_output_np,
            "prob": selector_probs_np
        }
        return output_dict

    def __call__(self, base_model_results, device=None):
        return self.forward(base_model_results, device)

    def generate_meta_features(self, input_features, output_probs, skip_class_based_features=False):

        input_scaler, knn, kde, osvm, isoforest, lof = \
            self._meta_models["input_scaler"], self._meta_models["knn"], self._meta_models["kde"], \
            self._meta_models["osvm"], self._meta_models["isoforest"], self._meta_models["lof"]

        probs = output_probs
        num_samples = input_features.shape[0]

        # Gather meta features.
        meta_features = []

        # Class-based features.
        if not skip_class_based_features:
            meta_features.append(probs)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(num_samples), np.argmax(probs, axis=1)] = 1
            meta_features.append(one_hot)

        # Confidence-based features.
        meta_features.append(np.max(probs, axis=-1, keepdims=True))
        meta_features.append(entropy(probs).reshape(-1, 1))

        # Outlier/novelty-based features.
        features = input_scaler.scale(input_features)
        meta_features.append(knn.score_samples(features).reshape(-1, 1))
        meta_features.append(kde.score_samples(features).reshape(-1, 1))
        meta_features.append(osvm.score_samples(features).reshape(-1, 1))
        meta_features.append(isoforest.score_samples(features).reshape(-1, 1))
        meta_features.append(lof.score_samples(features).reshape(-1, 1))

        # Combine and scale
        meta_features = np.concatenate(meta_features, axis=-1)
        meta_features = self._meta_scaler.scale(meta_features)

        return meta_features

    def to(self, arg):
        return self._selector.to(arg)

def entropy(probs):
    """Compute entropy of predicted distribution."""
    probs = np.clip(probs, a_min=1e-8, a_max=1)
    return -np.sum(probs * np.log(probs), axis=-1)

def example_load():
    # Test code
    import os
    from sybil.models.sybil import SybilNet
    mypath = os.path.expanduser("~/calibrated-selective-classification/data/processed/nlst/28a7cd44f5bcd3e6cc760b65c7e0d54depoch=10_calibrated_v2/28a7cd44f5bcd3e6cc760b65c7e0d54depoch=10.v2.ckpt")
    device = "cpu"

    base_net = SybilNet.load(mypath)

    # Load image data
    image_dir = os.path.expanduser("~/Projects/Sybil_general/sybil_demo_data")
    input_files = os.listdir(image_dir)
    input_files = [os.path.join(image_dir, x) for x in input_files if not x.startswith(".")]
    input_files = [x for x in input_files if os.path.isfile(x)]

    from sybil.serie import Serie
    serie = Serie(input_files)
    volume = serie.get_volume().to(device)

    # Get base model predictions
    with torch.no_grad():
        base_model_results = base_net(volume)

    base_selector_output = base_model_results["selector"]
    print(f"Base selector output: {base_selector_output}")

    selector_output = base_net._selector(base_model_results, device=device)

    print(f"Selector output:\n{selector_output}")

if __name__ == "__main__":
    example_load()
