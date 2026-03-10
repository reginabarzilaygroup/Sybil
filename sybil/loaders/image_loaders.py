from sybil.loaders.abstract_loader import abstract_loader
import cv2
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np

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

