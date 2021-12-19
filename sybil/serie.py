from time import thread_time
from typing import List, Optional, NamedTuple

import numpy as np
from sybil.datasets.utils import order_slices
from sybil.loaders.image_loaders import DicomLoader 
import pydicom
from ast import literal_eval

#? use imports
from sybil.augmentations import get_augmentations 
from argparse import Namespace

class Volume(NamedTuple):
    paths : list
    thickness: float
    pixel_spacing: float
    manifacturer: str
    imagetype: str
    slice_positions: list


class Label(NamedTuple):
    y: int
    y_seq: np.ndarray
    y_mask: np.ndarray
    censor_time: int


class Serie:
    def __init__(
        self,
        dicoms: List[str],
        label: Optional[int] = None,
        censor_time: Optional[int] = None
    ):
        """Initialize a Serie.

        Parameters
        ----------
        dicoms : List[str]
            [description]
        label : Optional[int], optional
            Whether the patient associated with this serie
            has or ever developped cancer.
        censor_time : Optional[int]
            Number of years until cancer diagnostic.
            If less than 1 year, should be 0.

        """
        if label is not None and censor_time is None:
            raise ValueError("censor_time should also provided with label.")
        # ! How to init loader
        self._loader = DicomLoader(None, [], Namespace(*{}) )
        self._label = label
        self._censor_time = censor_time
        self._vol = self.get_volume_metadata(dicoms)
        # sort paths
        self._dicom_paths, self._dicom_order = order_slices(dicoms, self._vol.slice_positions)

    def has_label(self) -> bool:
        """Check if there is a label associated with this serie.

        Returns
        -------
        bool
            [description]
        """
        return self._label is not None

    def get_label(self) -> int:
        """Check if there is a label associated with this serie.

        Returns
        -------
        bool
            [description]
        """
        if not self.has_label():
            raise ValueError("No label associated with this serie.")

        return self._label  # type: ignore

    def get_volume_metadata(self, dicom_paths):
        """Extract metadata from dicom files efficiently

        Parameters
        ----------
        dicom_paths : List[str]
            List of paths to dicom files

        Returns
        -------
        Tuple[list]
            slice_positions: list of indices for dicoms along z-axis
        """
        slice_positions = []
        for path in dicom_paths:
            dcm = pydicom.dcmread(path, stop_before_pixels = True)
            slice_positions.append(
                float(literal_eval(dcm.ImagePositionPatient)[-1])
            )

        thickness = float(dcm.SliceThickness) 
        pixel_spacing: map(float, eval(dcm.PixelSpacing))
        manifacturer: dcm.Manufacturer
        imagetype: dcm.TransferSyntaxUID

        volume = Volume(
            paths=dicom_paths,
            thickness=thickness, 
            pixel_spacing=pixel_spacing, 
            manifacturer=manifacturer, 
            imagetype=imagetype, 
            slice_positions = slice_positions
            )
        
        return volume


    def get_processed_volume(self) -> Volume:
        """[summary]

        Returns
        -------
        np.array
            [description]
        """
        # load volume 
        # ! how to use loader
        # x, mask = self._loader(self._dicom_paths, sample['additionals'], sample)
        raise NotImplementedError

    def get_processed_label(self, max_followup: int = 6) -> Label:
        """[summary]

        Parameters
        ----------
        max_followup : int, optional
            [description], by default 6

        Returns
        -------
        Tuple[bool, np.array, np.array, int]
            [description]

        Raises
        ------
        ValueError
            [description]
        """

        if not self.has_label():
            raise ValueError("No label associated with this serie.")

        # First convert months to years
        year_to_cancer = self._censor_time // 12  # type: ignore

        y_seq = np.zeros(max_followup, dtype=np.float64)
        y = int((year_to_cancer < max_followup) and self._label)  # type: ignore
        if y:
            y_seq[year_to_cancer:] = 1
        else:
            year_to_cancer = min(year_to_cancer, max_followup - 1)

        y_mask = np.array(
            [1] * (year_to_cancer + 1) + [0] * (max_followup - (year_to_cancer + 1)),
            dtype=np.float64,
        )
        return Label(y=y, y_seq=y_seq, y_mask=y_mask, censor_time=year_to_cancer)
