from typing import NamedTuple, Union, Dict, List, Optional
import os
from argparse import Namespace
import gdown 

import torch
import numpy as np

from sybil.serie import Serie
from sybil.models.sybil import SybilNet
from sybil.utils.metrics import get_survival_metrics


NAME_TO_FILE = {
    "sybil_base": ["28a7cd44f5bcd3e6cc760b65c7e0d54depoch=10.ckpt"],
    "sybil_ensemble": [
        "28a7cd44f5bcd3e6cc760b65c7e0d54depoch=10.ckpt",
        "56ce1a7d241dc342982f5466c4a9d7efepoch=10.ckpt",
        "624407ef8e3a2a009f9fa51f9846fe9aepoch=10.ckpt",
        "64a91b25f84141d32852e75a3aec7305epoch=10.ckpt",
        "65fd1f04cb4c5847d86a9ed8ba31ac1aepoch=10.ckpt",
    ],
}


class Prediction(NamedTuple):
    scores: List[List[float]]


class Evaluation(NamedTuple):
    auc: List[float]
    c_index: float
    scores: List[List[float]]


def download_sybil(name, cache):
    # Create cache folder if not exists
    cache = os.path.expanduser(cache)
    os.makedirs(cache, exist_ok=True)

    # Download if neded
    file_ids = NAME_TO_FILE[name]
    download_paths = []
    for file_id in file_ids:
        path = os.path.join(cache, f"{file_id}.ckpt")
        if not os.path.exists(path):
            print(f"Downloading model to {cache}")
            gdown.download(id=file_id, output=path, quiet = False)
        download_paths.append(path)
    return download_paths


class Sybil:
    def __init__(
        self,
        name_or_path: Union[List[str], str] = ["sybil_base"],
        cache: str = "~/.sybil/",
        device: Optional[str] = None,
    ):
        """Initialize a trained Sybil model for inference.

        Parameters
        ----------
        name_or_path : str
            Alias to a provided pretrained Sybil model or path
            to a sybil checkpoint.
        cache: str
            Directory to download model checkpoints to
        device: str
            If provided, will run inference using this device.
            By default uses GPU, if available.

        """
        # Download if needed
        if isinstance(name_or_path, str) and name_or_path in NAME_TO_FILE:
            name_or_path = download_sybil(name_or_path, cache)

        elif not all(os.path.exists(p) for p in name_or_path):
            raise ValueError(
                "No saved model or local path: {}".format(
                    [p for p in name_or_path if not os.path.exists(p)]
                )
            )

        # Set device
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.ensemble = torch.nn.ModuleList()
        for nop in name_or_path:
            self.ensemble.append(self.load_model(nop))

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
        return model

    def _predict(
        self, model: SybilNet, series: Union[Serie, List[Serie]]
    ) -> Prediction:
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
            Output prediction. See details for :class:`~sybil.model.Prediction`".

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
                score = out["logit"].sigmoid().squeeze(0).cpu().numpy().tolist()
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
            scores.append(pred.scores)
        scores = np.mean(np.array(scores), axis=0).tolist()
        return Prediction(scores=scores)

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
