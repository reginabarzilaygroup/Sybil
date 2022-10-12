from typing import NamedTuple, Union, Dict, List, Optional
import os
from argparse import Namespace
import gdown

import torch
import numpy as np
import pickle

from sybil.serie import Serie
from sybil.models.sybil import SybilNet
from sybil.utils.metrics import get_survival_metrics


NAME_TO_FILE = {
    "sybil_base": {
        "checkpoint": ["28a7cd44f5bcd3e6cc760b65c7e0d54d"],
        "google_checkpoint_id": ["1ftYbav_BbUBkyR3HFCGnsp-h4uH1yhoz"],
        "google_calibrator_id": "1F5TOtzueR-ZUvwl8Yv9Svs2NPP5El3HY",
    },
    "sybil_1": {
        "checkpoint": ["28a7cd44f5bcd3e6cc760b65c7e0d54d"],
        "google_checkpoint_id": ["1ftYbav_BbUBkyR3HFCGnsp-h4uH1yhoz"],
        "google_calibrator_id": "1F5TOtzueR-ZUvwl8Yv9Svs2NPP5El3HY",
    },
    "sybil_2": {
        "checkpoint": ["56ce1a7d241dc342982f5466c4a9d7ef"],
        "google_checkpoint_id": ["1rscGi1grSxaVGzn-tqKtuAR3ipo0DWgA"],
        "google_calibrator_id": "1zKLVYBaiuMOx7p--e2zabs1LbQ-XXxcZ",
    },
    "sybil_3": {
        "checkpoint": ["624407ef8e3a2a009f9fa51f9846fe9a"],
        "google_checkpoint_id": ["1DV0Ge7n9r8WAvBXyoNRPwyA7VL43csAr"],
        "google_calibrator_id": "1qh4nawgE2Kjf_H97XuuTpL7XUIX7JOJn",
    },
    "sybil_4": {
        "checkpoint": ["64a91b25f84141d32852e75a3aec7305"],
        "google_checkpoint_id": ["1Acz_yzdJMpkz3PRrjXy526CjAboMEIHX"],
        "google_calibrator_id": "1QIvvCYLaesPGMEiE2Up77pKL3ygDdGU2",
    },
    "sybil_5": {
        "checkpoint": ["65fd1f04cb4c5847d86a9ed8ba31ac1a"],
        "google_checkpoint_id": ["1uV58SD-Qtb6xElTzWPDWWnloH1KB_zrP"],
        "google_calibrator_id": "1yDq1_A5w-fSdxzq4K2YSBRNcQQkDnH0K",
    },
    "sybil_ensemble": {
        "checkpoint": [
            "28a7cd44f5bcd3e6cc760b65c7e0d54d",
            "56ce1a7d241dc342982f5466c4a9d7ef",
            "624407ef8e3a2a009f9fa51f9846fe9a",
            "64a91b25f84141d32852e75a3aec7305",
            "65fd1f04cb4c5847d86a9ed8ba31ac1a",
        ],
        "google_checkpoint_id": [
            "1ftYbav_BbUBkyR3HFCGnsp-h4uH1yhoz",
            "1rscGi1grSxaVGzn-tqKtuAR3ipo0DWgA",
            "1DV0Ge7n9r8WAvBXyoNRPwyA7VL43csAr",
            "1Acz_yzdJMpkz3PRrjXy526CjAboMEIHX",
            "1uV58SD-Qtb6xElTzWPDWWnloH1KB_zrP",
        ],
        "google_calibrator_id": "1FxHNo0HqXYyiUKE_k2bjatVt9e64J9Li",
    },
}


class Prediction(NamedTuple):
    scores: List[List[float]]


class Evaluation(NamedTuple):
    auc: List[float]
    c_index: float
    scores: List[List[float]]


def download_sybil(name, cache):
    """Download trained models and calibrator from Google Drive

    Parameters
    ----------
    name (str): name of model to use. A key in NAME_TO_FILE
    cache (str): path to directory where files are downloaded

    Returns
    -------
        download_model_paths (list): paths to .ckpt models
        download_calib_path (str): path to calibrator
    """
    # Create cache folder if not exists
    cache = os.path.expanduser(cache)
    os.makedirs(cache, exist_ok=True)

    # Download if neded
    model_files = NAME_TO_FILE[name]

    # Download models
    download_model_paths = []
    for model_name, google_id in zip(
        model_files["checkpoint"], model_files["google_checkpoint_id"]
    ):
        model_path = os.path.join(cache, f"{model_name}.ckpt")
        if not os.path.exists(model_path):
            print(f"Downloading model to {cache}")
            gdown.download(id=google_id, output=model_path, quiet=False)
        download_model_paths.append(model_path)

    # download calibrator
    download_calib_path = os.path.join(cache, f"{name}.p")
    if not os.path.exists(download_calib_path):
        gdown.download(
            id=model_files["google_calibrator_id"],
            output=download_calib_path,
            quiet=False,
        )

    return download_model_paths, download_calib_path


class Sybil:
    def __init__(
        self,
        name_or_path: Union[List[str], str] = "sybil_base",
        cache: str = "~/.sybil/",
        calibrator_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize a trained Sybil model for inference.

        Parameters
        ----------
        name_or_path: list or str
            Alias to a provided pretrained Sybil model or path
            to a sybil checkpoint.
        cache: str
            Directory to download model checkpoints to
        calibrator_path: str
            Path to calibrator pickle file corresponding with model
        device: str
            If provided, will run inference using this device.
            By default uses GPU, if available.

        """
        # Download if needed
        if isinstance(name_or_path, str) and (name_or_path in NAME_TO_FILE):
            name_or_path, calibrator_path = download_sybil(name_or_path, cache)

        elif not all(os.path.exists(p) for p in name_or_path):
            raise ValueError(
                "No saved model or local path: {}".format(
                    [p for p in name_or_path if not os.path.exists(p)]
                )
            )

        # Check calibrator path before continuing
        if (calibrator_path is not None) and (not os.path.exists(calibrator_path)):
            raise ValueError(f"Path not found for calibrator {calibrator_path}")

        # Set device
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.ensemble = torch.nn.ModuleList()
        for path in name_or_path:
            self.ensemble.append(self.load_model(path))

        if calibrator_path is not None:
            self.calibrator = pickle.load(open(calibrator_path, "rb"))

    def load_model(self, path):
        """Load model from path.

        Parameters
        ----------
        path : str
            Path to a sybil checkpoint.

        Returns
        -------
        model
            Pretrained Sybil model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location="cpu")
        args = checkpoint["args"]
        self._max_followup = args.max_followup
        self._censoring_dist = args.censoring_distribution
        model = SybilNet(args)

        # Remove model from param names
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)  # type: ignore
        if self.device == "cuda":
            model.to("cuda")

        # Set eval
        model.eval()
        print(f"Loaded model from {path}")
        return model

    def _calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate raw predictions

        Parameters
        ----------
        calibrator: Optional[dict]
            Dictionary of sklearn.calibration.CalibratedClassifierCV for each year, otherwise None.
        scores: np.ndarray
            risk scores as numpy array

        Returns
        -------
            np.ndarray: calibrated risk scores as numpy array
        """
        if self.calibrator is None:
            return scores

        calibrated_scores = []
        for YEAR in range(scores.shape[1]):
            probs = scores[:, YEAR].reshape(-1, 1)
            probs = self.calibrator["Year{}".format(YEAR + 1)].predict_proba(probs)[
                :, 1
            ]
            calibrated_scores.append(probs)

        return np.stack(calibrated_scores, axis=1)

    def _predict(
        self,
        model: SybilNet,
        series: Union[Serie, List[Serie]],
    ) -> np.ndarray:
        """Run predictions over the given serie(s).

        Parameters
        ----------
        model: SybilNet
            Instance of SybilNet
        series : Union[Serie, Iterable[Serie]]
            One or multiple series to run predictions for.

        Returns
        -------
        Prediction
            Output prediction as risk scores.

        """
        if isinstance(series, Serie):
            series = [series]
        elif not isinstance(series, list):
            raise ValueError("Expected either a Serie object or list of Serie objects.")

        scores: List[List[float]] = []
        for serie in series:
            if not isinstance(serie, Serie):
                raise ValueError("Expected a list of Serie objects.")

            volume = serie.get_volume()
            if self.device == "cuda":
                volume = volume.cuda()

            with torch.no_grad():
                out = model(volume)
                score = out["logit"].sigmoid().squeeze(0).cpu().numpy()
                scores.append(score)

        return np.stack(scores)

    def predict(self, series: Union[Serie, List[Serie]]) -> Prediction:
        """Run predictions over the given serie(s) and ensemble

        Parameters
        ----------
        series : Union[Serie, Iterable[Serie]]
            One or multiple series to run predictions for.

        Returns
        -------
        Prediction
            Output prediction. See details for :class:`~sybil.model.Prediction`".

        """
        scores = []
        for sybil in self.ensemble:
            pred = self._predict(sybil, series)
            scores.append(pred)
        scores = np.mean(np.array(scores), axis=0)
        calib_scores = self._calibrate(scores).tolist()
        return Prediction(scores=calib_scores)

    def evaluate(self, series: Union[Serie, List[Serie]]) -> Evaluation:
        """Run evaluation over the given serie(s).

        Parameters
        ----------
        series : Union[Serie, List[Serie]]
            One or multiple series to run evaluation for.

        Returns
        -------
        Evaluation
            Output evaluation. See details for :class:`~sybil.model.Evaluation`".

        """
        if isinstance(series, Serie):
            series = [series]
        elif not isinstance(series, list):
            raise ValueError(
                "Expected either a Serie object or an iterable over Serie objects."
            )

        # Check all have labels
        if not all(serie.has_label() for serie in series):
            raise ValueError("All series must have a label for evaluation")

        # Get scores and labels
        scores = self.predict(series).scores
        labels = [serie.get_label(self._max_followup) for serie in series]

        # Convert to format for survival metrics
        input_dict = {
            "probs": torch.tensor(scores),
            "censors": torch.tensor([label.censor_time for label in labels]),
            "golds": torch.tensor([label.y for label in labels]),
        }
        args = Namespace(
            max_followup=self._max_followup, censoring_distribution=self._censoring_dist
        )
        out = get_survival_metrics(input_dict, args)
        auc = [float(out[f"{i + 1}_year_auc"]) for i in range(self._max_followup)]
        c_index = float(out["c_index"])

        return Evaluation(auc=auc, c_index=c_index, scores=scores)
