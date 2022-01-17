import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Literal
from abc import ABCMeta, abstractmethod
import numpy as np
import random

ROTATE_DEGREES = {'min': -20, 'max': 20}

def get_augmentations(split: Literal['train', 'dev', 'test'], args):
    if split == 'train':
        augmentations = [
            Scale_2d(args, {}), 
            Rotate_Range(args, ROTATE_DEGREES), 
            ToTensor(),
            Force_Num_Chan_Tensor_2d(args, {}), 
            Normalize_Tensor_2d(args, {})
            ]
    else:
        augmentations = [
            Scale_2d(args, {}), 
            ToTensor(),
            Force_Num_Chan_Tensor_2d(args, {}), 
            Normalize_Tensor_2d(args, {})]
    
    return augmentations 
  
class Abstract_augmentation(object):
    """
    Abstract-transformer.
    Default - non cachable
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self._is_cachable = False
        self._trans_sep = '@' 
        self._attr_sep = '#'
        self.name = self.__str__().split('sybil.augmentations.')[-1].split(' ')[0].lower()

    @abstractmethod
    def __call__(self, img, mask=None, additional=None):
        pass

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def cachable(self):
        return self._is_cachable

    def set_cachable(self, *keys):
        '''
        Sets the transformer as cachable
        and sets the _caching_keys according to the input variables.
        '''
        self._is_cachable = True
        name_str = '{}{}'.format(self._trans_sep, self.name)
        keys_str = ''.join(self._attr_sep + str(k) for k in keys)
        self._caching_keys = '{}{}'.format(name_str, keys_str)
        return

    def caching_keys(self):
        return self._caching_keys

class ComposeAug(Abstract_augmentation):
    """
    composes multiple augmentations
    """

    def __init__(self, augmentations):
        super(ComposeAug, self).__init__()
        self.augmentations = augmentations

    def __call__(self, img, mask=None, additional=None):
        for transformer in self.augmentations:
            img, mask = transformer(img, mask, additional)

        return img, mask

class ToTensor(Abstract_augmentation):
    """
    torchvision.transforms.ToTensor wrapper.
    """

    def __init__(self):
        super(ToTensor, self).__init__()
        self.transform = ToTensorV2()
        self.name = "totensor"

    def __call__(self, img, mask=None, additional=None):
        if mask is None:
            return torch.from_numpy(img).float(), mask
        return torch.from_numpy(img).float(), torch.from_numpy(mask)

class Scale_2d(Abstract_augmentation):
    """
        Given PIL image, enforce its some set size
        (can use for down sampling / keep full res)
    """

    def __init__(self, args, kwargs):
        super(Scale_2d, self).__init__()
        assert len(kwargs.keys()) == 0
        width, height = args.img_size
        self.set_cachable(width, height)
        self.transform = A.Resize(height, width)

    def __call__(self, img, mask=None, additional=None):
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]

class Rotate_Range(Abstract_augmentation):
    """
    Rotate image counter clockwise by random
    kwargs['min'] - kwargs['max'] degrees.

    Example: 'rotate/min=-20/max=20' will rotate by up to +/-20 deg
    """

    def __init__(self, args, kwargs):
        super(Rotate_Range, self).__init__()
        assert len(kwargs.keys()) == 2
        self.max_angle = int(kwargs["max"])
        self.min_angle = int(kwargs["min"])
        self.transform = A.Rotate(limit=self.max_angle, p=0.5)

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and "seed" in additional:
            self.set_seed(additional["seed"])
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]
    
class Normalize_Tensor_2d(Abstract_augmentation):
    """
    torchvision.transforms.Normalize wrapper.
    """

    def __init__(self, args, kwargs):
        super(Normalize_Tensor_2d, self).__init__()
        assert len(kwargs) == 0
        channel_means = [args.img_mean] if len(args.img_mean) == 1 else args.img_mean
        channel_stds = [args.img_std] if len(args.img_std) == 1 else args.img_std

        self.transform = torchvision.transforms.Normalize(
            torch.Tensor(channel_means), torch.Tensor(channel_stds)
        )


    def __call__(self, img, mask=None, additional=None):
        if len(img.size()) == 2:
            img = img.unsqueeze(0)
        return self.transform(img), mask
    
class Force_Num_Chan_Tensor_2d(Abstract_augmentation):
    """
    Convert gray scale images to image with args.num_chan num channels.
    """

    def __init__(self, args, kwargs):
        super(Force_Num_Chan_Tensor_2d, self).__init__()
        assert len(kwargs) == 0
        self.args = args

    def __call__(self, img, mask=None, additional=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        num_dims = len(img.shape)
        if num_dims == 2:
            img = img.unsqueeze(0)
        existing_chan = img.size()[0]
        if not existing_chan == self.args.num_chan:
            return img.expand(self.args.num_chan, *img.size()[1:]), mask
        return img, mask
