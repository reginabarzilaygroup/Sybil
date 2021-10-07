from sandstone.datasets.loader.factory import RegisterInputLoader
from sandstone.datasets.loader.abstract_loader import abstract_loader
import cv2
import numpy as np
import pydicom
import os
import math
import torch 

LOADING_ERROR = 'LOADING ERROR! {}'

@RegisterInputLoader('default_image_loader')
class OpenCVLoader(abstract_loader):
    def configure_path(self, path, additional, sample):
        return path

    def load_input(self, path, additional):
        '''
        loads as grayscale image
        '''
        return cv2.imread(path, 0)

    @property
    def cached_extension(self):
        return '.png'

@RegisterInputLoader('tensor_loader')
class TensorLoader(abstract_loader):
    def configure_path(self, path, additional, sample):
        return path

    def load_input(self, path, additional):
        try:
            arr = torch.load(path)
        except:
            raise Exception(LOADING_ERROR.format('COULD NOT LOAD TENSOR.'))
        return arr

    @property
    def cached_extension(self):
        return ''

    @property
    def apply_augmentations(self):
        return False

    def reshape_images(self, images):
        images = torch.cat(images, dim=0)
        rp = (self.args.num_chan, 1, 1, 1)
        images = images.unsqueeze(0).repeat(rp)
        return images
