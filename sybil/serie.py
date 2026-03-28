import functools
import os
import shutil
from typing import Any, Dict, List, Optional, NamedTuple, Literal, Union, Tuple
from argparse import Namespace
import cc3d
import torch
import numpy as np
import pydicom
import SimpleITK as sitk
import torchio as tio
import torch.nn.functional as F
from monai.data import MetaTensor
from loguru import logger
from sybil.datasets.utils import order_slices, VOXEL_SPACING
from sybil.utils.loading import get_sample_loader
from sybil.utils.dicom_to_nifti import read_with_sitk


def _pad_axis(lo: int, hi: int, min_size: int, max_size: int):
    """Symmetrically expand [lo, hi+1) to at least min_size, clamped to [0, max_size]."""
    size = hi - lo + 1
    pad = max(0, min_size - size)
    lo = max(0, lo - pad // 2)
    hi = min(max_size, lo + max(size, min_size))
    lo = max(0, hi - max(size, min_size))
    return lo, hi


class Meta(NamedTuple):
    paths: list
    thickness: float
    pixel_spacing: list
    manufacturer: str
    slice_positions: list
    voxel_spacing: torch.Tensor
    nifti_path: Optional[str] = None
    identifier: Optional[str] = None


class Label(NamedTuple):
    y: int
    y_seq: np.ndarray
    y_mask: np.ndarray
    censor_time: int


class InputV2(NamedTuple):
    segmentation_volume: torch.Tensor
    rve_volume: Optional[torch.Tensor] = None
    lungmask_volume: Optional[torch.Tensor] = None


class Serie:
    def __init__(
        self,
        dicoms: Union[List[str], Dict[int, List[str]]],
        voxel_spacing: Optional[List[float]] = None,
        label: Optional[int] = None,
        censor_time: Optional[int] = None,
        file_type: Literal["png", "dicom"] = "dicom",
        split: Literal["train", "dev", "test"] = "test",
        version: Literal["v1", "v2"] = "v1",
        cache_dir: Optional[str] = None,
    ):
        """Initialize a Serie.

        Parameters
        ----------
        `dicoms` : Union[List[str], [str, dict]]
            List of dicom paths or dicom metadata dicts.
            If dicts are provided and `multi_scan` is False,
            they are converted into a flat list of paths.
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
        `version`: Literal['v1', 'v2']
            Version of the dataset. Affects how metadata is extracted from DICOM files.
            Assumed to be v1 by default.
        `cache_dir`: Optional[str]
            Optional directory to use for caching processed images. If None, no caching is used.
            Caching can speed up loading for large datasets, but requires additional disk space.
        """
        if label is not None and censor_time is None:
            raise ValueError("censor_time should also provided with label.")
        if file_type == "png" and voxel_spacing is None:
            raise ValueError("voxel_spacing should be provided for PNG files.")

        self._is_version1 = version == "v1"
        self._is_version2 = version == "v2"
        if self._is_version1 and isinstance(dicoms, dict) and len(dicoms) > 1:
            raise ValueError(
                "Multiple dicom dicts provided for version 1. Expected a single dict or a list of paths."
            )

        if self._is_version2:
            assert cache_dir is not None, (
                "Version 2 requires a cache directory for storing intermediate NIfTI files."
            )

        self._cache_dir = cache_dir

        if self._is_version1 and isinstance(dicoms, dict):
            dicoms = self._convert_dicom_dicts_to_paths(dicoms)

        self._censor_time = censor_time
        self._label = label
        if self._is_version1:
            args = self._load_argsv1(file_type)
            self._args = args
            self._meta = self._load_metadata(dicoms, voxel_spacing, file_type)
            self._check_valid(args)
            self.resample_transform = tio.transforms.Resample(target=VOXEL_SPACING)
            self.padding_transform = tio.transforms.CropOrPad(
                target_shape=tuple(args.img_size + [args.num_images]), padding_mode=0
            )
        elif self._is_version2:
            args = self._load_argsv2(file_type)
            self._args = args
            self._meta = {
                k: self._load_metadata(dcms, voxel_spacing, file_type)
                for k, dcms in dicoms.items()
            }

        self._loader = get_sample_loader(split, args, version=version)

    def _convert_dicom_dicts_to_paths(self, dicoms: Dict[str, List[str]]) -> List[str]:
        assert len(dicoms) == 1, (
            "Expected only one dicom dict when multi_scan is False."
        )
        key = list(dicoms.keys())[0]
        return dicoms[key]

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
        year_to_cancer = self._censor_time  # type: ignore

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

    def get_raw_images(self) -> List[np.ndarray]:
        """
        Load raw images from serie

        Returns
        -------
        List[np.ndarray]
            List of CT slices of shape (1, C, H, W)
        """

        loader = get_sample_loader("test", self._args, apply_augmentations=False)
        input_dicts = [loader.get_image(path) for path in self._meta.paths]
        images = [i["input"] for i in input_dicts]
        return images

    @functools.lru_cache
    def get_volume(self) -> Union[torch.Tensor, Dict[str, InputV2]]:
        if self._is_version1:
            return self._get_volume_v1()
        elif self._is_version2:
            return self._get_volume_v2()
        else:
            raise ValueError("Invalid version. Expected 'v1' or 'v2'.")

    def _get_volume_v2(self):
        volumes = {}
        for key, meta in self._meta.items():
            # rve sample
            nifti_volume, sitk_volume = read_with_sitk(
                meta.paths, depth_first=True
            )
            # logger.debug(f"Saved NIfTI for {key} at {meta.nifti_path}")
            rve_volume = self._get_volume_for_rve(sitk_volume, "{}_{}".format(meta.identifier, key))
            lungmask_volume, segmentation_volume = self._get_volume_for_segmentation(
                nifti_volume, meta
            )
            volumes[key] = InputV2(
                lungmask_volume=lungmask_volume,  # shared with confidence model
                segmentation_volume=segmentation_volume,  # shared with confidence model
                rve_volume=rve_volume,
            )
        return volumes

    def _get_volume_for_rve(self, sitk_volume: sitk.Image, accession: str) -> Optional[torch.Tensor]:
        rve_path = self._loader["pillar"].rve_processor(sitk_volume, output_dir=self._cache_dir, accession=accession, series_number=1)
        volume = self._loader["pillar"].load_input(rve_path)["input"]
        # delete nifti_file after loading to save space
        if self._cache_dir is not None and os.path.exists(rve_path):
            if os.path.isdir(rve_path):
                shutil.rmtree(rve_path)
                logger.debug(
                    f"Deleted cached directory at {rve_path} after loading RVE volume."
                )
            else:
                os.remove(rve_path)
                logger.debug(
                    f"Deleted cached file at {rve_path} after loading RVE volume."
                )
        return volume

    def _get_volume_for_segmentation(
        self, image: np.ndarray, meta
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._loader["nifti"].load_input(image)["input"]
        affine = torch.diag(meta.voxel_spacing)
        image = MetaTensor(
            image,
            affine=affine,
            dtype=torch.float32,
        )

        # Ensure image has the correct spatial size (H, W) = (1024, 1024)
        # image is (D, H, W) from NiftiLoader
        H, W = 1024, 1024
        img_h, img_w = image.shape[1], image.shape[2]
        if (img_h, img_w) != (H, W):
            # Resize H, W slice-by-slice: (D, H, W) -> (D, 1, H, W) -> interpolate -> (D, 1, H', W')
            resize_image = image.unsqueeze(1)
            resize_image = F.interpolate(
                resize_image,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            resize_image = resize_image.squeeze(1).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        else:
            resize_image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        return image, resize_image

    def _get_volume_v1(self) -> torch.Tensor:
        """
        Load loaded 3D CT volume

        Returns
        -------
        torch.Tensor
            CT volume of shape (1, C, N, H, W)
        """
        input_dicts = [self._loader.get_image(path) for path in self._meta.paths]

        x = torch.cat([i["input"].unsqueeze(0) for i in input_dicts], dim=0)

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
        `paths` : List[str]
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

        identifier = paths[0].split("/")[-2]  # folder name
        nifti_path = (
            os.path.join(self._cache_dir, f"{identifier}.nii.gz")
            if self._cache_dir is not None
            else None
        )

        meta = Meta(
            paths=processed_paths,
            thickness=thickness,
            pixel_spacing=pixel_spacing,
            manufacturer=manufacturer,
            slice_positions=slice_positions,
            voxel_spacing=voxel_spacing,
            nifti_path=nifti_path,
            identifier=identifier,
        )
        return meta

    def _load_argsv1(self, file_type):
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

    def _load_argsv2(self, file_type):
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
                "img_file_type": file_type,
                "cache_path": None,
                "use_annotations": False,
                "fix_seed_for_multi_image_augmentations": True,
                "slice_thickness_filter": 3,
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
        if self._meta.thickness is None:
            raise ValueError("slice thickness not found")
        if self._meta.thickness > args.slice_thickness_filter:
            raise ValueError(
                f"slice thickness {self._meta.thickness} is greater than {args.slice_thickness_filter}."
            )
        if self._meta.voxel_spacing is None:
            raise ValueError("voxel spacing either not set or not found in DICOM")

    def prepare_for_confidence_model(
        self,
        sparse_seg: torch.Tensor,
        image: torch.Tensor,
        nodule_mask: torch.Tensor,
        crop_size: Tuple[int, int, int] = (128, 128, 32),
    ) -> torch.Tensor:
        """Prepare 2-channel (CT + segmentation probability) crops for the confidence model.

        For each nodule in ``sparse_seg``, the bounding box is symmetrically padded to
        ``crop_size``, then both the CT image and the soft nodule probability are cropped
        and stacked into a 2-channel patch.

        Parameters
        ----------
        sparse_seg : torch.Tensor (sparse COO)
            Sparse tensor of shape (H, W, D) with integer nodule IDs as values.
        image : torch.Tensor
            CT volume of shape (H, W, D).
        nodule_mask : torch.Tensor
            Binary mask of shape (D, H, W) from the segmentation model.
        crop_size : tuple
            (min_height, min_width, min_depth) for each patch.

        Returns
        -------
        torch.Tensor
            Shape (N_nodules, 2, H_crop, W_crop, D_crop).
        """
        H_CROP, W_CROP, D_CROP = crop_size
        img_h, img_w, img_d = image.shape

        sparse_seg = sparse_seg.coalesce()
        nodule_ids = sparse_seg.values().unique()
        nodule_ids = nodule_ids[nodule_ids > 0]

        patches = []
        for nid in nodule_ids:
            mask = sparse_seg.values() == nid
            ys, xs, zs = sparse_seg.indices()[:, mask]

            ymin, ymax = ys.min().item(), ys.max().item()
            xmin, xmax = xs.min().item(), xs.max().item()
            zmin, zmax = zs.min().item(), zs.max().item()

            # symmetrically pad bounding box to reach minimum crop dimensions
            y1, y2 = _pad_axis(ymin, ymax, H_CROP, img_h)
            x1, x2 = _pad_axis(xmin, xmax, W_CROP, img_w)
            z1, z2 = _pad_axis(zmin, zmax, D_CROP, img_d)

            patchx = image[y1:y2, x1:x2, z1:z2]  # (H_crop, W_crop, D_crop)

            # place nodule probability into a zero canvas, then crop
            patch_seg = torch.zeros(img_h, img_w, img_d, dtype=nodule_mask.dtype)
            patch_seg[ymin : ymax + 1, xmin : xmax + 1, zmin : zmax + 1] = nodule_mask[
                zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1
            ].permute(1, 2, 0)
            patchl = patch_seg[y1:y2, x1:x2, z1:z2]

            patches.append(torch.stack([patchx, patchl]))  # (2, H_crop, W_crop, D_crop)

        if patches:
            return torch.stack(patches)  # (N, 2, H_crop, W_crop, D_crop)
        return torch.zeros(0, 2, H_CROP, W_CROP, D_CROP)

    def prepare_for_malignancy_model(
        self,
        sparse_seg: torch.Tensor,
        image: torch.Tensor,
        crop_size: Tuple[int, int, int] = (128, 128, 32),
    ) -> torch.Tensor:
        """Prepare CT crops centred on each nodule for the malignancy model.

        Parameters
        ----------
        sparse_seg : torch.Tensor (sparse COO)
            Sparse tensor of shape (H, W, D) with integer nodule IDs as values.
        image : torch.Tensor
            CT volume of shape (H, W, D).
        crop_size : tuple
            (height, width, depth) of the output patch.

        Returns
        -------
        torch.Tensor
            Shape (N_nodules, H_crop, W_crop, D_crop).
        """
        H_CROP, W_CROP, D_CROP = crop_size
        img_h, img_w, img_d = image.shape

        sparse_seg = sparse_seg.coalesce()
        nodule_ids = sparse_seg.values().unique()
        nodule_ids = nodule_ids[nodule_ids > 0]

        patches = []
        for nid in nodule_ids:
            mask = sparse_seg.values() == nid
            ys, xs, zs = sparse_seg.indices()[:, mask]

            ycenter = (ys.min().item() + ys.max().item()) // 2
            xcenter = (xs.min().item() + xs.max().item()) // 2
            zcenter = (zs.min().item() + zs.max().item()) // 2

            ymin = max(0, ycenter - H_CROP // 2)
            ymax = min(img_h, ymin + H_CROP)
            if ymax - ymin < H_CROP:
                ymin = max(0, ymax - H_CROP)

            xmin = max(0, xcenter - W_CROP // 2)
            xmax = min(img_w, xmin + W_CROP)
            if xmax - xmin < W_CROP:
                xmin = max(0, xmax - W_CROP)

            zmin = max(0, zcenter - D_CROP // 2)
            zmax = min(img_d, zmin + D_CROP)

            patches.append(
                image[ymin:ymax, xmin:xmax, zmin:zmax]
            )  # (H_crop, W_crop, D_crop)

        if patches:
            return torch.stack(patches)  # (N, H_crop, W_crop, D_crop)
        return torch.zeros(0, H_CROP, W_CROP, D_CROP)
