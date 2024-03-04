from sybil.loaders.abstract_loader import abstract_loader
import cv2
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np

LOADING_ERROR = "LOADING ERROR! {}"


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


def apply_windowing(image, center, width, bit_size=16):
    """Windowing function to transform image pixels for presentation.
    Must be run after a DICOM modality LUT is applied to the image.
    Windowing algorithm defined in DICOM standard:
    http://dicom.nema.org/medical/dicom/2020b/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2
    Reference implementation:
    https://github.com/pydicom/pydicom/blob/da556e33b/pydicom/pixel_data_handlers/util.py#L460
    Args:
        image (ndarray): Numpy image array
        center (float): Window center (or level)
        width (float): Window width
        bit_size (int): Max bit size of pixel
    Returns:
        ndarray: Numpy array of transformed images
    """
    y_min = 0
    y_max = 2 ** bit_size - 1
    y_range = y_max - y_min

    c = center - 0.5
    w = width - 1

    below = image <= (c - w / 2)  # pixels to be set as black
    above = image > (c + w / 2)  # pixels to be set as white
    between = np.logical_and(~below, ~above)

    image[below] = y_min
    image[above] = y_max
    if between.any():
        image[between] = ((image[between] - c) / w + 0.5) * y_range + y_min

    return image
