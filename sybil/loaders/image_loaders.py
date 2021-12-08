from sybil.loaders.abstract_loader import abstract_loader
import cv2
import torch

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


class TensorLoader(abstract_loader):
    def configure_path(self, path, additional, sample):
        return path

    def load_input(self, path, additional):
        try:
            arr = torch.load(path)
        except Exception:
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD TENSOR."))
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
