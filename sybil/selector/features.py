"""Processing meta features for use in the selector."""

import copy
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import torch


class FeatureScaler:
    """Standard feature scaler to mean 0 and unit variance."""

    def __init__(self, binary_columns=True):
        self.binary_columns = binary_columns
        self.mu = None
        self.std = None

    def fit(self, X):
        print('\nFitting scaling...')
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if self.binary_columns:
            binary_columns = []
            for col in range(X.shape[1]):
                vals = np.unique(X[:, col])
                if len(vals) > 2:
                    continue
                binary_columns.append(col)
            print(f'Num binary columns = {len(binary_columns)}')
            self.mu[binary_columns] = 0
            self.std[binary_columns] = 1

        # Fix features with std 0 to be 1.
        self.std[self.std == 0] = 1

        return self

    def scale(self, X):
        return (X - self.mu) / self.std

    def state_dict(self):
        return {'mu': self.mu, 'std': self.std, "binary_columns": self.binary_columns}

    def load_state_dict(self, state_dict):
        self.binary_columns = state_dict['binary_columns']
        self.mu = state_dict['mu']
        self.std = state_dict['std']
        return self


class KNNDistance:
    """Compute average distance to k nearest neighbors in training set."""

    def __init__(self, k=8):
        """Initialize class with number of neighbors to average over."""
        self.k = k
        self.source_vecs = None

    def fit(self, source_vecs):
        """Store training set vectors."""
        self.source_vecs = torch.from_numpy(source_vecs).unsqueeze(0)

    def score_samples(self, query_vecs):
        """Score input examples based on average distance to source vectors."""
        query_vecs = torch.from_numpy(query_vecs).unsqueeze(0)
        dists = torch.cdist(query_vecs, self.source_vecs).squeeze(0)
        knn_dists = torch.topk(dists, self.k, dim=1, largest=False).values
        return knn_dists.mean(dim=1).view(-1).numpy()

    def state_dict(self):
        return {'k': self.k, "source_vecs": self.source_vecs}

    def load_state_dict(self, state_dict):
        self.k = state_dict['k']
        self.source_vecs = state_dict['source_vecs']
        return self


def load_outlier_models(models_dict):
    # keys = ["input_scaler", "knn", "kde", "osvm", "isoforest", "lof"]
    output = copy.deepcopy(models_dict)
    output["input_scaler"] = FeatureScaler().load_state_dict(models_dict["input_scaler"])
    output["knn"] = KNNDistance().load_state_dict(models_dict["knn"])
    return output


def entropy(probs):
    """Compute entropy of predicted distribution."""
    probs = np.clip(probs, a_min=1e-8, a_max=1)
    return -np.sum(probs * np.log(probs), axis=-1)