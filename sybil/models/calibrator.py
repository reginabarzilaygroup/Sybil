import json
import os
from typing import List

import numpy as np

"""
Calibrator for Sybil prediction models.

We calibrate probabilities using isotonic regression. 
Previously this was done with scikit-learn, here we use a custom implementation to avoid versioning issues.
"""


class SimpleClassifierGroup:
    """
    A class to represent a calibrator for prediction models.
    Behavior and coefficients are taken from the sklearn.calibration.CalibratedClassifierCV class.
    Make a custom class to avoid sklearn versioning issues.
    """

    def __init__(self, calibrators: List["SimpleIsotonicRegressor"]):
        self.calibrators = calibrators

    def predict_proba(self, X, expand=False):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_probabilities,)
            The input probabilities to recalibrate.
        expand : bool, default=False
            Whether to return the probabilities for each class separately.
            This is intended for binary classification which can be done in 1D,
            expand=True will return a 2D array with shape (n_probabilities, 2).

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered by
            lexicographic order.
        """
        proba = np.array([calibrator.transform(X) for calibrator in self.calibrators])
        pos_prob = np.mean(proba, axis=0)
        if expand and len(self.calibrators) == 1:
            return np.array([1.-pos_prob, pos_prob])
        else:
            return pos_prob

    def to_json(self):
        return [calibrator.to_json() for calibrator in self.calibrators]

    @classmethod
    def from_json(cls, json_list):
        return cls([SimpleIsotonicRegressor.from_json(json_dict) for json_dict in json_list])

    @classmethod
    def from_json_grouped(cls, json_path):
        """
        We store calibrators in a diction of {year (str): [calibrators]}.
        This is a convenience method to load that dictionary from a file path.
        """
        json_dict = json.load(open(json_path, "r"))
        output_dict = {key: cls.from_json(json_list) for key, json_list in json_dict.items()}
        return output_dict


class SimpleIsotonicRegressor:
    def __init__(self, coef, intercept, x0, y0, x_min=-np.inf, x_max=np.inf):
        self.coef = coef
        self.intercept = intercept
        self.x0 = x0
        self.y0 = y0
        self.x_min = x_min
        self.x_max = x_max

    def transform(self, X):
        T = X
        T = T @ self.coef + self.intercept
        T = np.clip(T, self.x_min, self.x_max)
        return np.interp(T, self.x0, self.y0)

    @classmethod
    def from_classifier(cls, classifer: "_CalibratedClassifier"):
        assert len(classifer.calibrators) == 1, "Only one calibrator per classifier is supported."
        calibrator = classifer.calibrators[0]
        return cls(classifer.base_estimator.coef_, classifer.base_estimator.intercept_,
                   calibrator.f_.x, calibrator.f_.y, calibrator.X_min_, calibrator.X_max_)

    def to_json(self):
        return {
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "x0": self.x0.tolist(),
            "y0": self.y0.tolist(),
            "x_min": self.x_min,
            "x_max": self.x_max
        }

    @classmethod
    def from_json(cls, json_dict):
        return cls(
            np.array(json_dict["coef"]),
            np.array(json_dict["intercept"]),
            np.array(json_dict["x0"]),
            np.array(json_dict["y0"]),
            json_dict["x_min"],
            json_dict["x_max"]
        )

    def __repr__(self):
        return f"SimpleIsotonicRegressor(x={self.x0}, y={self.y0})"


def export_calibrator(input_path, output_path):
    import pickle
    import sklearn
    sk_cal_dict = pickle.load(open(input_path, "rb"))
    simple_cal_dict = dict()
    for key, cal in sk_cal_dict.items():
        calibrators = [SimpleIsotonicRegressor.from_classifier(classifier) for classifier in cal.calibrated_classifiers_]
        simple_cal_dict[key] = SimpleClassifierGroup(calibrators).to_json()

    json.dump(simple_cal_dict, open(output_path, "w"), indent=2)


def export_by_name(base_dir, model_name, overwrite=False):
    sk_input_path = os.path.expanduser(f"{base_dir}/{model_name}.p")
    simple_output_path = os.path.expanduser(f"{base_dir}/{model_name}_simple_calibrator.json")

    version = "1.4.0"
    scores_output_path = f"{base_dir}/{model_name}_v{version}_calibrations.json"

    if overwrite or not os.path.exists(simple_output_path):
        run_test_calibrations(sk_input_path, scores_output_path)

    if overwrite or not os.path.exists(simple_output_path):
        export_calibrator(sk_input_path, simple_output_path)


def export_all_default_calibrators(base_dir="~/.sybil", overwrite=False):
    base_dir = os.path.expanduser(base_dir)
    model_names = ["sybil_1", "sybil_2", "sybil_3", "sybil_4", "sybil_5", "sybil_ensemble"]
    for model_name in model_names:
        export_by_name(base_dir, model_name, overwrite=overwrite)


def run_test_calibrations(sk_input_path, scores_output_path, overwrite=False):
    """
    For regression testing. Output calibrated probabilities for a range of input probabilities.
    """
    import pickle
    sk_cal_dict = pickle.load(open(sk_input_path, "rb"))

    test_probs = np.arange(0, 1, 0.001).reshape(-1, 1)

    output_dict = {"x": test_probs.flatten().tolist()}
    for key, model in sk_cal_dict.items():
        output_dict[key] = model.predict_proba(test_probs)[:, -1].flatten().tolist()

    if overwrite or not os.path.exists(scores_output_path):
        with open(scores_output_path, "w") as f:
            json.dump(output_dict, f, indent=2)


if __name__ == "__main__":
    export_all_default_calibrators(overwrite=False)
