import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Literal
from abc import ABCMeta, abstractmethod
import numpy as np
import random


def get_augmentations(split: Literal["train", "dev", "test"], args):
    if split == "train":
        augmentations = [
            Scale_2d(args, {}),
            Rotate_Range(args, {"deg": 20}),
            ToTensor(),
            Force_Num_Chan_Tensor_2d(args, {}),
            Normalize_Tensor_2d(args, {}),
        ]
    else:
        augmentations = [
            Scale_2d(args, {}),
            ToTensor(),
            Force_Num_Chan_Tensor_2d(args, {}),
            Normalize_Tensor_2d(args, {}),
        ]

    return augmentations


class Abstract_augmentation(object):
    """
    Abstract-transformer.
    Default - non cachable
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self._is_cachable = False
        self._trans_sep = "@"
        self._attr_sep = "#"
        self.name = (
            self.__str__().split("sybil.augmentations.")[-1].split(" ")[0].lower()
        )

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
        """
        Sets the transformer as cachable
        and sets the _caching_keys according to the input variables.
        """
        self._is_cachable = True
        name_str = "{}{}".format(self._trans_sep, self.name)
        keys_str = "".join(self._attr_sep + str(k) for k in keys)
        self._caching_keys = "{}{}".format(name_str, keys_str)
        return

    def caching_keys(self):
        return self._caching_keys


class ComposeAug(Abstract_augmentation):
    """
    Composes multiple augmentations
    """

    def __init__(self, augmentations):
        super(ComposeAug, self).__init__()
        self.augmentations = augmentations

    def __call__(self, input_dict, sample=None):
        for transformer in self.augmentations:
            input_dict = transformer(input_dict, sample)

        return input_dict


class ToTensor(Abstract_augmentation):
    """
    torchvision.transforms.ToTensor wrapper.
    """

    def __init__(self):
        super(ToTensor, self).__init__()
        self.transform = ToTensorV2()
        self.name = "totensor"

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = torch.from_numpy(input_dict["input"]).float()
        if input_dict.get("mask", None) is not None:
            input_dict["mask"] = torch.from_numpy(input_dict["mask"]).float()
        return input_dict


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

    def __call__(self, input_dict, sample=None):
        out = self.transform(
            image=input_dict["input"], mask=input_dict.get("mask", None)
        )
        input_dict["input"] = out["image"]
        input_dict["mask"] = out["mask"]
        return input_dict


class Rotate_Range(Abstract_augmentation):
    """
    Rotate image counter clockwise by random degree https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.Rotate

        kwargs
            deg: max degrees to rotate
    """

    def __init__(self, args, kwargs):
        super(Rotate_Range, self).__init__()
        assert len(kwargs.keys()) == 1
        self.max_angle = int(kwargs["deg"])
        self.transform = A.Rotate(limit=self.max_angle, p=0.5)

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        out = self.transform(
            image=input_dict["input"], mask=input_dict.get("mask", None)
        )
        input_dict["input"] = out["image"]
        input_dict["mask"] = out["mask"]
        return input_dict


class Normalize_Tensor_2d(Abstract_augmentation):
    """
    Normalizes input by channel
    wrapper for torchvision.transforms.Normalize wrapper.
    """

    def __init__(self, args, kwargs):
        super(Normalize_Tensor_2d, self).__init__()
        assert len(kwargs) == 0
        channel_means = [args.img_mean] if len(args.img_mean) == 1 else args.img_mean
        channel_stds = [args.img_std] if len(args.img_std) == 1 else args.img_std

        self.transform = torchvision.transforms.Normalize(
            torch.Tensor(channel_means), torch.Tensor(channel_stds)
        )

        self.permute = args.img_file_type in [
            "png",
        ]

    def __call__(self, input_dict, sample=None):
        img = input_dict["input"]
        if len(img.size()) == 2:
            img = img.unsqueeze(0)

        if self.permute:
            img = img.permute(2, 0, 1)
            input_dict["input"] = self.transform(img).permute(1, 2, 0)
        else:
            input_dict["input"] = self.transform(img)

        return input_dict


class Force_Num_Chan_Tensor_2d(Abstract_augmentation):
    """
    Convert gray scale images to image with args.num_chan num channels.
    """

    def __init__(self, args, kwargs):
        super(Force_Num_Chan_Tensor_2d, self).__init__()
        assert len(kwargs) == 0
        self.args = args

    def __call__(self, input_dict, sample=None):
        img = input_dict["input"]
        mask = input_dict.get("mask", None)
        if mask is not None:
            input_dict["mask"] = mask.unsqueeze(0)

        num_dims = len(img.shape)
        if num_dims == 2:
            img = img.unsqueeze(0)
        existing_chan = img.size()[0]
        if not existing_chan == self.args.num_chan:
            input_dict["input"] = img.expand(self.args.num_chan, *img.size()[1:])

        return input_dict
