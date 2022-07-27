from typing import List, Optional, NamedTuple, Literal
from argparse import Namespace

import torch
import numpy as np
import pydicom
import torchio as tio

from sybil.datasets.utils import order_slices, VOXEL_SPACING
from sybil.utils.loading import get_sample_loader


class Meta(NamedTuple):
    paths: list
    thickness: float
    pixel_spacing: list
    manufacturer: str
    slice_positions: list
    voxel_spacing: torch.Tensor


class Label(NamedTuple):
    y: int
    y_seq: np.ndarray
    y_mask: np.ndarray
    censor_time: int


class Serie:
    def __init__(
        self,
        dicoms: List[str],
        voxel_spacing: Optional[List[float]] = None,
        label: Optional[int] = None,
        censor_time: Optional[int] = None,
        file_type: Literal["png", "dicom"] = "dicom",
        split: Literal["train", "dev", "test"] = "test",
    ):
        """Initialize a Serie.

        Parameters
        ----------
        `dicoms` : List[str]
            [description]
        `voxel_spacing`: Optional[List[float]], optional
            The voxel spacing associated with input CT
            as (row spacing, col spacing, slice thickness)
        `label` : Optional[int], optional
            Whether the patient associated with this serie
            has or ever developped cancer.
        `censor_time` : Optional[int]
            Number of years until cancer diagnostic.
            If less than 1 year, should be 0.
        `file_type`: Literal['png', 'dicom']
            File type of CT slices
        `split`: Literal['train', 'dev', 'test']
            Dataset split into which the serie falls into.
            Assumed to be test by default
        """
        if label is not None and censor_time is None:
            raise ValueError("censor_time should also provided with label.")

        self._censor_time = censor_time
        self._label = label
        args = self._load_args(file_type)
        self._loader = get_sample_loader(split, args)
        self._meta = self._load_metadata(dicoms, voxel_spacing, file_type)
        self._check_valid(args)
        self.resample_transform = tio.transforms.Resample(target=VOXEL_SPACING)
        self.padding_transform = tio.transforms.CropOrPad(
            target_shape=tuple(args.img_size + [args.num_images]), padding_mode=0
        )

    def has_label(self) -> bool:
        """Check if there is a label associated with this serie.

        Returns
        -------
        bool
            [description]
        """
        return self._label is not None

    def get_label(self, max_followup: int = 6) -> Label:
        """Get the label for this Serie.

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
            raise ValueError("No label in this serie.")

        # First convert months to years
        year_to_cancer = self._censor_time # type: ignore

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

    def get_volume(self) -> torch.Tensor:
        """
        Load loaded 3D CT volume

        Returns
        -------
        torch.Tensor
            CT volume of shape (1, C, N, H, W)
        """

        sample = {"seed": np.random.randint(0, 2**32 - 1)}

        input_dicts = [self._loader.get_image(path, sample) for path in self._meta.paths]
        
        x = torch.cat( [i["input"].unsqueeze(0) for i in input_dicts], dim = 0)
        
        # Convert from (T, C, H, W) to (C, T, H, W)
        x = x.permute(1, 0, 2, 3)

        x = tio.ScalarImage(
            affine=torch.diag(self._meta.voxel_spacing),
            tensor=x.permute(0, 2, 3, 1),
        )
        x = self.resample_transform(x)
        x = self.padding_transform(x)
        x = x.data.permute(0, 3, 1, 2)
        x.unsqueeze_(0)
        return x

    def _load_metadata(self, paths, voxel_spacing, file_type):
        """Extract metadata from dicom files efficiently

        Parameters
        ----------
        `dicom_paths` : List[str]
            List of paths to dicom files
        `voxel_spacing`: Optional[List[float]], optional
            The voxel spacing associated with input CT
            as (row spacing, col spacing, slice thickness)
        `file_type` : Literal['png', 'dicom']
            File type of CT slices

        Returns
        -------
        Tuple[list]
            slice_positions: list of indices for dicoms along z-axis
        """
        if file_type == "dicom":
            slice_positions = []
            processed_paths = []
            for path in paths:
                dcm = pydicom.dcmread(path, stop_before_pixels=True)
                processed_paths.append(path)
                slice_positions.append(float(dcm.ImagePositionPatient[-1]))

            processed_paths, slice_positions = order_slices(
                processed_paths, slice_positions
            )

            thickness = float(dcm.SliceThickness)
            pixel_spacing = list(map(float, dcm.PixelSpacing))
            manufacturer = dcm.Manufacturer
            voxel_spacing = torch.tensor(pixel_spacing + [thickness, 1])
        elif file_type == "png":
            processed_paths = paths
            slice_positions = list(range(len(paths)))
            thickness = voxel_spacing[-1] if voxel_spacing is not None else None
            pixel_spacing = []
            manufacturer = ""
            voxel_spacing = (
                torch.tensor(voxel_spacing + [1]) if voxel_spacing is not None else None
            )

        meta = Meta(
            paths=processed_paths,
            thickness=thickness,
            pixel_spacing=pixel_spacing,
            manufacturer=manufacturer,
            slice_positions=slice_positions,
            voxel_spacing=voxel_spacing,
        )
        return meta

    def _load_args(self, file_type):
        """
        Load default args required for a single Serie volume

        Parameters
        ----------
        file_type : Literal['png', 'dicom']
            File type of CT slices

        Returns
        -------
        Namespace
            args with preset values
        """
        args = Namespace(
            **{
                "img_size": [256, 256],
                "img_mean": [128.1722],
                "img_std": [87.1849],
                "num_images": 200,
                "img_file_type": file_type,
                "num_chan": 3,
                "cache_path": None,
                "use_annotations": False,
                "fix_seed_for_multi_image_augmentations": True,
                "slice_thickness_filter": 5,
            }
        )
        return args

    def _check_valid(self, args):
        """
        Check if serie is acceptable:

        Parameters
        ----------
        `args` : Namespace
            manually set args used to develop model

        Raises
        ------
        ValueError if:
            - serie doesn't have a label, OR
            - slice thickness is too big
        """
        if (self._meta.thickness is None) or (
            self._meta.thickness > args.slice_thickness_filter
        ):
            raise ValueError(
                "slice thickness is greater than {}.".format(
                    args.slice_thickness_filter
                )
            )
        if self._meta.voxel_spacing is None:
            raise ValueError("voxel spacing either not set or not found in DICOM")
