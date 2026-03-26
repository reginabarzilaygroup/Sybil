"""
Dataset for Sybil v2 inference from a CSV manifest.

CSV columns (required):  patient_id, timepoint, ct_dir
CSV columns (optional):  label, censor_time

Each row is one CT scan (one timepoint) for a patient.  Multiple rows with
the same ``patient_id`` are collected into a single multi-timepoint Serie.
"""

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from sybil.serie import Serie

REQUIRED_COLS = {"patient_id", "timepoint", "ct_dir"}


class SybilV2Dataset(Dataset):
    """``torch.utils.data.Dataset`` over a CSV manifest of CT studies.

    Each ``__getitem__`` call returns one patient as a ``(Serie, metadata)``
    tuple.  Returns ``None`` on failure so that a ``None``-filtering
    ``collate_fn`` can skip bad samples without crashing the DataLoader.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.  Must contain columns ``patient_id``,
        ``timepoint``, and ``ct_dir``.  Optional columns ``label`` (0/1) and
        ``censor_time`` (int, years) enable labelled mode.
    cache_dir : str
        Directory for intermediate NIfTI files required by the v2 pipeline.
        Created automatically if it does not exist.
    file_type : str
        ``"dicom"`` (default) or ``"png"``.
    file_extension : str, optional
        Override the glob extension used to find CT files inside each
        ``ct_dir`` (e.g. ``".dcm"``, ``".IMA"``, ``".png"``).
        Defaults to ``".dcm"`` for DICOM and ``".png"`` for PNG.
    """

    def __init__(
        self,
        csv_path: str,
        cache_dir: str,
        file_type: str = "dicom",
        file_extension: Optional[str] = None,
    ):
        self.cache_dir = cache_dir
        self.file_type = file_type
        self.file_extension = (
            file_extension
            if file_extension is not None
            else (".dcm" if file_type == "dicom" else ".png")
        )
        os.makedirs(cache_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        self.has_labels = {"label", "censor_time"}.issubset(df.columns)
        self.samples = self._build_samples(df)
        logger.info(
            f"SybilV2Dataset: {len(self.samples)} patient(s) loaded from {csv_path}"
        )

    # ------------------------------------------------------------------
    def _build_samples(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for patient_id, group in df.groupby("patient_id", sort=False):
            timepoint_paths: Dict[str, List[str]] = {}

            for _, row in group.iterrows():
                ct_dir = str(row["ct_dir"])
                if not os.path.isdir(ct_dir):
                    logger.warning(f"CT directory not found, skipping: {ct_dir}")
                    continue
                paths = sorted(
                    glob.glob(os.path.join(ct_dir, f"*{self.file_extension}"))
                )
                if not paths:
                    logger.warning(
                        f"No {self.file_extension} files found in {ct_dir}, skipping"
                    )
                    continue
                timepoint_paths[str(row["timepoint"])] = paths

            if not timepoint_paths:
                logger.warning(
                    f"No valid timepoints for patient {patient_id}, skipping"
                )
                continue

            entry: Dict[str, Any] = {
                "patient_id": patient_id,
                "timepoint_paths": timepoint_paths,
            }
            if self.has_labels:
                # label / censor_time are patient-level; take from the first row
                entry["label"] = int(group.iloc[0]["label"])
                entry["censor_time"] = int(group.iloc[0]["censor_time"])

            samples.append(entry)
        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Optional[Tuple[Serie, Dict[str, Any]]]:
        """Return ``(Serie, metadata_dict)`` for the patient at ``idx``.

        ``metadata_dict`` always contains ``patient_id`` and ``timepoints``.
        When the dataset has labels it also contains ``label`` and
        ``censor_time``.

        Returns ``None`` on failure so that a filtering collate function can
        skip bad samples gracefully.
        """
        entry = self.samples[idx]
        patient_id = entry["patient_id"]
        patient_cache = os.path.join(self.cache_dir, str(patient_id))
        os.makedirs(patient_cache, exist_ok=True)

        label = entry.get("label")
        censor_time = entry.get("censor_time")

        try:
            serie = Serie(
                dicoms=entry["timepoint_paths"],
                label=label,
                censor_time=censor_time,
                file_type=self.file_type,
                version="v2",
                cache_dir=patient_cache,
            )
        except Exception as exc:
            logger.warning(
                f"Failed to construct Serie for patient {patient_id}: {exc}"
            )
            return None

        meta: Dict[str, Any] = {
            "patient_id": patient_id,
            "timepoints": sorted(entry["timepoint_paths"].keys()),
        }
        if self.has_labels:
            meta["label"] = label
            meta["censor_time"] = censor_time

        return serie, meta


def collate_series(batch):
    """Collate function for use with ``SybilV2Dataset``.

    Filters out ``None`` items (failed samples) and returns a
    ``(series_list, meta_list)`` tuple, or ``None`` if the entire batch
    is empty after filtering.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    series_list = [b[0] for b in batch]
    meta_list = [b[1] for b in batch]
    return series_list, meta_list
