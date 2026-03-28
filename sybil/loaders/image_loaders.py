from sybil.loaders.abstract_loader import abstract_loader
import cv2
import rve
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from sybil.loaders.rve import  NiftiToRVE
LOADING_ERROR = "LOADING ERROR! {}"

def apply_windowing(image, center, width, bit_size=16, use_tensor=False):
    """Windowing function to transform image pixels for presentation.
    Must be run after a DICOM modality LUT is applied to the image.
    Windowing algorithm defined in DICOM standard:
    http://dicom.nema.org/medical/dicom/2020b/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2
    Reference implementation:
    https://github.com/pydicom/pydicom/blob/da556e33b/pydicom/pixel_data_handlers/util.py#L460
    Args:
        image (ndarray or Tensor): Numpy image array or PyTorch tensor
        center (float): Window center (or level)
        width (float): Window width
        bit_size (int): Max bit size of pixel
        use_tensor (bool): Whether the input image is a tensor
    Returns:
        ndarray or Tensor: Transformed image in the same format as input
    """
    if use_tensor or isinstance(image, torch.Tensor):
        image = image.clone()
        y_min = 0
        y_max = 2**bit_size - 1
        y_range = y_max - y_min

        c = center - 0.5
        w = width - 1

        below = image <= (c - w / 2)
        above = image > (c + w / 2)
        between = ~below & ~above

        image[below] = y_min
        image[above] = y_max
        if between.any():
            image[between] = ((image[between] - c) / w + 0.5) * y_range + y_min
    else:
        image = image.copy()
        y_min = 0
        y_max = 2**bit_size - 1
        y_range = y_max - y_min

        c = center - 0.5
        w = width - 1

        below = image <= (c - w / 2)
        above = image > (c + w / 2)
        between = np.logical_and(~below, ~above)

        image[below] = y_min
        image[above] = y_max
        if between.any():
            image[between] = ((image[between] - c) / w + 0.5) * y_range + y_min

    return image


def apply_pillar_windowing(
    volume: torch.Tensor, center: float, width: float
) -> torch.Tensor:
    """
    Apply traditional center/width windowing to medical images.

    Args:
        volume: Input volume
        center: Window center (level)
        width: Window width

    Returns:
        Windowed volume with values in [0, 1]
    """

    # Calculate window bounds
    min_val = center - width / 2
    max_val = center + width / 2

    # Apply windowing
    windowed = (volume - min_val) / (max_val - min_val + 1e-8)
    windowed = torch.clamp(windowed, 0, 1)

    return windowed

class OpenCVLoader(abstract_loader):

    def load_input(self, path):
        """
        loads as grayscale image
        """
        return {"input": cv2.imread(path, 0)}

    @property
    def cached_extension(self):
        return ".png"


class DicomLoader(abstract_loader):
    def __init__(self, cache_path, augmentations, args, apply_augmentations=True):
        super(DicomLoader, self).__init__(cache_path, augmentations, args, apply_augmentations)
        self.window_center = -600
        self.window_width = 1500

    def load_input(self, path):
        try:
            dcm = pydicom.dcmread(path)
            dcm = apply_modality_lut(dcm.pixel_array, dcm)
            arr = apply_windowing(dcm, self.window_center, self.window_width)
            arr = arr//256  # parity with images loaded as 8 bit
        except Exception:
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD DICOM."))
        return {"input": arr}

    @property
    def cached_extension(self):
        return ""

class SegmentationLoader(abstract_loader):
    def __init__(self, cache_path, augmentations, args, apply_augmentations=True):
        super(SegmentationLoader, self).__init__(cache_path, augmentations, args, apply_augmentations)
        self.window_center = -600
        self.window_width = 1600

    def load_input(self, path):
        try:
            dcm = pydicom.dcmread(path)
            dcm = dcm.pixel_array.astype(np.float32) * dcm.RescaleSlope + dcm.RescaleIntercept
            arr = apply_windowing(dcm, self.window_center, self.window_width)
            arr = arr//256  # parity with images loaded as 8 bit
        except Exception:
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD DICOM."))
        return {"input": arr}

    @property
    def cached_extension(self):
        return ""


class NiftiLoader(abstract_loader):
    def __init__(self, cache_path, augmentations, args, apply_augmentations=True):
        super(NiftiLoader, self).__init__(cache_path, augmentations, args, apply_augmentations)
        self.window_center = -600
        self.window_width = 1600

    def load_input(self, path):
        try:
            if isinstance(path, str):
                arr = np.transpose(nib.load(path).get_fdata(), (2, 1, 0))  # (x, y, z) -> (z, y, x)
            else:
                arr = path
            arr = apply_windowing(arr, self.window_center, self.window_width)
            arr = arr//256  # parity with images loaded as 8 bit
        except Exception:
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD NIFTI."))
        return {"input": arr}

    @property
    def cached_extension(self):
        return ""
    
class PillarLoader(abstract_loader):
    def __init__(self, cache_path, augmentations, args, apply_augmentations=True):
        super(PillarLoader, self).__init__(cache_path, augmentations, args, apply_augmentations)
        self.rve_processor = NiftiToRVE()
        self.anatomical_windows = {
            "CT": {
                "lung": {"center": -600, "width": 1500},
                "mediastinum": {"center": 50, "width": 400},
                "abdomen": {"center": 40, "width": 400},
                "liver": {"center": 80, "width": 150},
                "bone": {"center": 400, "width": 1800},
                "brain": {"center": 40, "width": 80},
                "subdural": {"center": 75, "width": 215},
                "stroke": {"center": 40, "width": 40},
                "temporal_bone": {"center": 600, "width": 2800},
                "soft_tissue": {"center": 50, "width": 350},
            }
        }

    def load_input(self, path):
        try:
            image = rve.load_sample(path, use_hardware_acceleration=False)[None]
            image = self._pad_along_depth(image, 256)
            image = self.process_image_for_pillar(image)
        except Exception:
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD PILLAR IMAGE."))
        return {"input": image}

    @property
    def cached_extension(self):
        return ""

    def _pad_along_depth(self, tensor: torch.Tensor, target_depth: int) -> torch.Tensor:
        """
        Pad a (C, D, H, W) or (D, H, W) tensor symmetrically along D to target_depth.
        """
        if tensor.dim() == 3:
            depth = tensor.shape[0]
            if depth < target_depth:
                pad_total = target_depth - depth
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                return F.pad(tensor, (0, 0, 0, 0, pad_left, pad_right))
            elif depth > target_depth:
                crop_total = depth - target_depth
                crop_left = crop_total // 2
                crop_right = crop_total - crop_left
                if crop_right == 0:
                    return tensor[crop_left:]
                return tensor[crop_left:-crop_right]
            return tensor
        elif tensor.dim() == 4:
            depth = tensor.shape[1]
            if depth < target_depth:
                pad_total = target_depth - depth
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                return F.pad(tensor, (0, 0, 0, 0, pad_left, pad_right))
            elif depth > target_depth:
                crop_total = depth - target_depth
                crop_left = crop_total // 2
                crop_right = crop_total - crop_left
                if crop_right == 0:
                    return tensor[:, crop_left:]
                return tensor[:, crop_left:-crop_right]
            return tensor
        else:
            raise ValueError(
                f"Unexpected tensor shape for D padding: {tuple(tensor.shape)}"
            )
    
    def process_image_for_pillar(self, image):
        # min-max normalize
        minmax = (image - image.min()) / (image.max() - image.min() + 1e-8)
        minmax = torch.clamp(minmax, 0, 1)
        # apply anatomical windows
        image = [
            apply_pillar_windowing(image, window["center"], window["width"])
            for bodypart, window in self.anatomical_windows["CT"].items()
        ] + [minmax]
        image = torch.concat(image, 0)  #  channels, y, x, z
        return image