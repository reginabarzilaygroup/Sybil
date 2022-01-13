from typing import NamedTuple, Union, Dict, List, Optional
import os
from argparse import Namespace

import torch

from sybil.serie import Serie
from sybil.models.sybil import SybilNet
from sybil.utils.download import download_file_from_google_drive
from sybil.utils.metrics import get_survival_metrics


NAME_TO_FILE: Dict[str, str] = {"test": "1P7rKz9Ir8Gd99AisaKLFddtS9uOczPG0"}


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
    file_id = NAME_TO_FILE[name]
    path = os.path.join(cache, f"{file_id}.ckpt")
    if not os.path.exists(path):
        print(f"Downloading model to {cache}")
        download_file_from_google_drive(file_id, path)

    return path


class Sybil:
    def __init__(
        self,
        name_or_path: str = "sybil_base",
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
        if name_or_path in NAME_TO_FILE:
            name_or_path = download_sybil(name_or_path, cache)

        elif not os.path.exists(name_or_path):
            raise ValueError(f"No saved model or local path: {name_or_path}")

        # Set device
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load checkpoint
        checkpoint = torch.load(name_or_path, map_location="cpu")
        hparams = checkpoint['hyper_parameters']
        self._max_followup = hparams["max_followup"]
        self._censoring_dist = hparams["censoring_distribution"]
        self.model = SybilNet(Namespace(**hparams))

        # Remove model from param names
        state_dict = {k[6:]: v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)  # type: ignore
        if self.device == "cuda":
            self.model.to("cuda")

        # Set eval
        self.model.eval()

    def predict(self, series: Union[Serie, List[Serie]]) -> Prediction:
        """Run predictions over the given serie(s).

        Parameters
        ----------
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
            raise ValueError(
                "Expected either a Serie object or list of Serie objects."
            )

        scores: List[List[float]] = []
        for serie in series:
            if not isinstance(serie, Serie):
                raise ValueError("Expected a list of Serie objects.")

            volume = serie.get_volume()
            if self.device == "cuda":
                volume = volume.cuda()

            with torch.no_grad():
                out = self.model(volume)
                score = out["logit"].sigmoid().cpu().numpy().tolist()
                scores.append(score)

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
            "golds": torch.tensor([label.y for label in labels])
        }
        args = Namespace(
            max_followup=self._max_followup,
            censoring_distribution=self._censoring_dist
        )
        out = get_survival_metrics(input_dict, args)
        auc = [float(out[f"{i + 1}_year_auc"]) for i in range(self._max_followup)]
        c_index = float(out["c_index"])

        return Evaluation(auc=auc, c_index=c_index, scores=scores)
