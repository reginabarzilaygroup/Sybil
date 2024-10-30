from argparse import Namespace
from io import BytesIO
import os
from typing import NamedTuple, Union, Dict, List, Optional, Tuple
from urllib.request import urlopen
from zipfile import ZipFile

import torch
import numpy as np

from sybil.serie import Serie
from sybil.models.sybil import SybilNet
from sybil.models.calibrator import SimpleClassifierGroup
from sybil.utils.logging_utils import get_logger
from sybil.utils.device_utils import get_default_device, get_most_free_gpu, get_device_mem_info


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

CHECKPOINT_URL = os.getenv("SYBIL_CHECKPOINT_URL", "https://github.com/reginabarzilaygroup/Sybil/releases/download/v1.5.0/sybil_checkpoints.zip")


class Prediction(NamedTuple):
    scores: List[List[float]]
    attentions: List[Dict[str, np.ndarray]] = None


class Evaluation(NamedTuple):
    auc: List[float]
    c_index: float
    scores: List[List[float]]
    attentions: List[Dict[str, np.ndarray]] = None


def download_sybil(name, cache) -> Tuple[List[str], str]:
    """Download trained models and calibrator"""
    # Create cache folder if not exists
    cache = os.path.expanduser(cache)
    os.makedirs(cache, exist_ok=True)

    # Download models
    model_files = NAME_TO_FILE[name]
    checkpoints = model_files["checkpoint"]
    download_calib_path = os.path.join(cache, f"{name}_simple_calibrator.json")
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


def download_and_extract(remote_url: str, local_dir: str) -> List[str]:
    os.makedirs(local_dir, exist_ok=True)
    resp = urlopen(remote_url)
    with ZipFile(BytesIO(resp.read())) as zip_file:
        all_files_and_dirs = zip_file.namelist()
        zip_file.extractall(local_dir)
    return all_files_and_dirs


def _torch_set_num_threads(threads) -> int:
    """
    Set the number of CPU threads for torch to use.
    Set to a negative number for no-op.
    Set to 0 for the number of CPUs.
    """
    if threads < 0:
        return torch.get_num_threads()
    if threads is None or threads == 0:
        # I've never seen a benefit to going higher than 8 and sometimes there is a big slowdown
        threads = min(8, os.cpu_count())

    torch.set_num_threads(threads)
    return torch.get_num_threads()


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
            By default, uses GPU with the most free memory, if available.

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

        # Set device.
        # If set manually, use it and stay there.
        # Otherwise, pick the most free GPU now and at predict time.
        self._device_flexible = True
        if device is not None:
            self.device = device
            self._device_flexible = False
        else:
            self.device = get_default_device()

        self.ensemble = torch.nn.ModuleList()
        for path in name_or_path:
            self.ensemble.append(self.load_model(path))
        self.to(self.device)

        if calibrator_path is not None:
            self.calibrator = SimpleClassifierGroup.from_json_grouped(calibrator_path)
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
        if self.device is not None:
            model.to(self.device)

        # Set eval
        model.eval()
        self._logger.info(f"Loaded model from {path}")
        return model

    def _calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate raw predictions

        Parameters
        ----------
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
            probs = self.calibrator["Year{}".format(YEAR + 1)].predict_proba(probs)[:, -1]
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
            if self.device is not None:
                volume = volume.to(self.device)

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
                            "hidden": out["hidden"]
                            .detach()
                            .cpu(),
                        }
                    )

        return Prediction(scores=scores, attentions=attentions)

    def predict(
        self, series: Union[Serie, List[Serie]], return_attentions: bool = False, threads=0,
    ) -> Prediction:
        """Run predictions over the given serie(s) and ensemble

        Parameters
        ----------
        series : Union[Serie, Iterable[Serie]]
            One or multiple series to run predictions for.
        return_attentions : bool
            If True, returns attention scores for each serie. See README for details.
        threads : int
            Number of CPU threads to use for PyTorch inference.

        Returns
        -------
        Prediction
            Output prediction. See details for :class:`~sybil.model.Prediction`".

        """

        # Set CPU threads available to torch
        num_threads = _torch_set_num_threads(threads)
        self._logger.debug(f"Using {num_threads} threads for PyTorch inference")

        if self._device_flexible:
            self.device = self._pick_device()
            self.to(self.device)
        self._logger.debug(f"Beginning prediction on device: {self.device}")

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
        from sybil.utils.metrics import get_survival_metrics
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

    def to(self, device: str):
        """Move model to device.

        Parameters
        ----------
        device : str
            Device to move model to.
        """
        self.device = device
        self.ensemble.to(device)

    def _pick_device(self):
        """
        Pick the device to run inference on.
        This is based on the device with the most free memory, with a preference for remaining
        on the current device.

        Motivation is to enable multiprocessing without the processes needed to communicate.
        """
        if not torch.cuda.is_available():
            return get_default_device()

        # Get size of the model in memory (approximate)
        model_mem = 9*sum(p.numel() * p.element_size() for p in self.ensemble.parameters())

        # Check memory available on current device.
        # If it seems like we're the only thing on this GPU, stay.
        free_mem, total_mem = get_device_mem_info(self.device)
        cur_allocated = total_mem - free_mem
        min_to_move = int(1.01 * model_mem)
        if cur_allocated < min_to_move:
            return self.device
        else:
            # Otherwise, get the most free GPU
            return get_most_free_gpu()
