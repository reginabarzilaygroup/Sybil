from argparse import Namespace
from io import BytesIO
import os
from typing import NamedTuple, Union, Dict, List, Optional, Tuple
from urllib.request import urlopen
from zipfile import ZipFile
# import gdown

import torch
import numpy as np
import pickle

from sybil.serie import Serie
from sybil.models.sybil import SybilNet
from sybil.utils.logging_utils import get_logger
from sybil.utils.device_utils import get_default_device
from sybil.utils.metrics import get_survival_metrics


# Leaving this here for a bit; these are IDs to download the models from Google Drive
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

CHECKPOINT_URL = "https://github.com/reginabarzilaygroup/Sybil/releases/download/v1.0.3/sybil_checkpoints.zip"


class Prediction(NamedTuple):
    scores: List[List[float]]
    attentions: List[Dict[str, np.ndarray]] = None


class Evaluation(NamedTuple):
    auc: List[float]
    c_index: float
    scores: List[List[float]]
    attentions: List[Dict[str, np.ndarray]] = None


def download_sybil_gdrive(name, cache):
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


def download_sybil(name, cache) -> Tuple[List[str], str]:
    """Download trained models and calibrator"""
    # Create cache folder if not exists
    cache = os.path.expanduser(cache)
    os.makedirs(cache, exist_ok=True)

    # Download models
    model_files = NAME_TO_FILE[name]
    checkpoints = model_files["checkpoint"]
    download_calib_path = os.path.join(cache, f"{name}.p")
    have_all_files = os.path.exists(download_calib_path)

    download_model_paths = []
    for checkpoint in checkpoints:
        cur_checkpoint_path = os.path.join(cache, f"{checkpoint}.ckpt")
        have_all_files &= os.path.exists(cur_checkpoint_path)
        download_model_paths.append(cur_checkpoint_path)

    if not have_all_files:
        print(f"Downloading models to {cache}")
        download_and_extract(CHECKPOINT_URL, cache)

    return download_model_paths, download_calib_path


def download_and_extract(remote_model_url: str, local_model_dir) -> List[str]:
    resp = urlopen(remote_model_url)
    os.makedirs(local_model_dir, exist_ok=True)
    with ZipFile(BytesIO(resp.read())) as zip_file:
        all_files_and_dirs = zip_file.namelist()
        zip_file.extractall(local_model_dir)
    return all_files_and_dirs


class Sybil:
    def __init__(
        self,
        name_or_path: Union[List[str], str] = "sybil_ensemble",
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
        self._logger = get_logger()
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
        else:
            self.calibrator = None

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
        self._logger.info(f"Loaded model from {path}")
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
        return_attentions: bool = False,
    ) -> Prediction:
        """Run predictions over the given serie(s).

        Parameters
        ----------
        model: SybilNet
            Instance of SybilNet
        series : Union[Serie, Iterable[Serie]]
            One or multiple series to run predictions for.
        return_attentions : bool
            If True, returns attention scores for each serie. See README for details.

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
        attentions: List[Dict[str, np.ndarray]] = [] if return_attentions else None
        for serie in series:
            if not isinstance(serie, Serie):
                raise ValueError("Expected a list of Serie objects.")

            volume = serie.get_volume()
            if self.device == "cuda":
                volume = volume.cuda()

            with torch.no_grad():
                out = model(volume)
                score = out["logit"].sigmoid().squeeze(0).cpu().numpy()
                scores.append(score.tolist())
                if return_attentions:
                    attentions.append(
                        {
                            "image_attention_1": out["image_attention_1"]
                            .detach()
                            .cpu(),
                            "volume_attention_1": out["volume_attention_1"]
                            .detach()
                            .cpu(),
                        }
                    )

        return Prediction(scores=scores, attentions=attentions)

    def predict(
        self, series: Union[Serie, List[Serie]], return_attentions: bool = False
    ) -> Prediction:
        """Run predictions over the given serie(s) and ensemble

        Parameters
        ----------
        series : Union[Serie, Iterable[Serie]]
            One or multiple series to run predictions for.
        return_attentions : bool
            If True, returns attention scores for each serie. See README for details.

        Returns
        -------
        Prediction
            Output prediction. See details for :class:`~sybil.model.Prediction`".

        """
        scores = []
        attentions_ = [] if return_attentions else None
        attention_keys = None
        for sybil in self.ensemble:
            pred = self._predict(sybil, series, return_attentions)
            scores.append(pred.scores)
            if return_attentions:
                attentions_.append(pred.attentions)
                if attention_keys is None:
                    attention_keys = pred.attentions[0].keys()

        scores = np.mean(np.array(scores), axis=0)
        calib_scores = self._calibrate(scores).tolist()

        attentions = None
        if return_attentions:
            attentions = []
            for i in range(len(series)):
                att = {}
                for key in attention_keys:
                    att[key] = np.stack([
                        attentions_[j][i][key] for j in range(len(self.ensemble))
                    ])
                attentions.append(att)

        return Prediction(scores=calib_scores, attentions=attentions)

    def evaluate(
        self, series: Union[Serie, List[Serie]], return_attentions: bool = False
    ) -> Evaluation:
        """Run evaluation over the given serie(s).

        Parameters
        ----------
        series : Union[Serie, List[Serie]]
            One or multiple series to run evaluation for.
        return_attentions : bool
            If True, returns attention scores for each serie. See README for details.

        Returns
        -------
        Evaluation
            Output evaluation. See details for :class:`~sybil.model.Evaluation`.

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
        predictions = self.predict(series, return_attentions)
        scores = predictions.scores
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

        return Evaluation(auc=auc, c_index=c_index, scores=scores, attentions=predictions.attentions)
