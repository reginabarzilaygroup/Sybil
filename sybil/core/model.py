from typing import NamedTuple, Iterable, Union, Optional
from sybil.core.serie import Serie


class Prediction(NamedTuple):
    pass


class Evaluation(NamedTuple):
    pass


class Sybil:

    def __init__(self, name_or_path: Optional[str] = "sybil_base"):
        """Initialize a trained Sybil model for inference.

        Parameters
        ----------
        name_or_path : str
            Alias to a provided pretrained Sybil model or path
            to a sybil checkpoint.

        """
        pass

    def predict(self, series: Union[Serie, Iterable[Serie]]) -> Prediction:
        """Run predictions over the given serie(s).

        Parameters
        ----------
        series : Union[Serie, Iterable[Serie]]
            One or multiple series to run predictions for.

        Returns
        -------
        Prediction
            Output prediction. See details for :class:`~sybil.core.model.Prediction`".

        """
        pass

    def evaluate(self, series: Union[Serie, Iterable[Serie]]) -> Evaluation:
        """Run evaluation over the given serie(s).

        Parameters
        ----------
        series : Union[Serie, Iterable[Serie]]
            One or multiple series to run evaluation for.

        Returns
        -------
        Evaluation
            Output evaluation. See details for :class:`~sybil.core.model.Evaluation`".

        """
        pass
