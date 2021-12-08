from sybil.loaders.abstract_loader import abstract_loader
import cv2
import torch
import pydicom

LOADING_ERROR = "LOADING ERROR! {}"


class OpenCVLoader(abstract_loader):
    def configure_path(self, path, additional, sample):
        return path

    def load_input(self, path, additional):
        """
        loads as grayscale image
        """
        return cv2.imread(path, 0)

    @property
    def cached_extension(self):
        return ".png"


class DicomLoader(abstract_loader):
    def configure_path(self, path, additional, sample):
        return path

    def load_input(self, path, additional):
        try:
            dcm = pydicom.dcmread(path)
            arr = dcm.pixel_array
        except Exception:
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD DICOM."))
        return arr

    @property
    def cached_extension(self):
        return ""

    @property
    def apply_augmentations(self):
        return False

    def reshape_images(self, images):
        images = torch.cat(images, dim=0)
        rp = (self.args.num_chan, 1, 1, 1)
        images = images.unsqueeze(0).repeat(rp)
        return images
