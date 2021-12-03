import numpy as np
from PIL import Image
import albumentations as A
from sybil.augmentations.abstract import Abstract_augmentation


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


class Pad(Abstract_augmentation):
    """
        Given PIL image, enforce its some set size
        (can use for down sampling / keep full res)
    """

    def __init__(self, args, kwargs):
        super(Pad, self).__init__()
        width, height = (int(kwargs["w"]), int(kwargs["h"]))
        self.set_cachable(width, height)
        self.transform = A.PadIfNeeded(min_height=height, min_width=width)

    def __call__(self, img, additional=None):
        return self.transform(image=img)["image"]


class Random_Horizontal_Flip(Abstract_augmentation):
    def __init__(self, args, kwargs):
        super(Random_Horizontal_Flip, self).__init__()
        self.args = args
        assert len(kwargs.keys()) == 0
        self.transform = A.HorizontalFlip()

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and "seed" in additional:
            self.set_seed(additional["seed"])
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]


class Random_Vertical_Flip(Abstract_augmentation):
    """
    random vertical flip.
    """

    def __init__(self, args, kwargs):
        super(Random_Vertical_Flip, self).__init__()
        self.args = args
        assert len(kwargs.keys()) == 0
        self.transform = A.VerticalFlip()

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and "seed" in additional:
            self.set_seed(additional["seed"])
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]


class RandomResizedCrop(Abstract_augmentation):
    """
        torchvision.transforms.RandomResizedCrop wrapper
        size of cropping will be decided by the 'h' and 'w' kwargs.
        'padding' kwarg is also available.
    """

    def __init__(self, args, kwargs):
        super(RandomResizedCrop, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert (kwargs_len >= 2) and (kwargs_len <= 6)
        h, w = (int(kwargs["h"]), int(kwargs["w"]))
        min_scale = float(kwargs["min_scale"]) if "min_scale" in kwargs else 0.08
        max_scale = float(kwargs["max_scale"]) if "max_scale" in kwargs else 1.0
        min_ratio = float(kwargs["min_ratio"]) if "min_ratio" in kwargs else 0.75
        max_ratio = float(kwargs["max_ratio"]) if "max_ratio" in kwargs else 1.33
        self.transform = A.RandomResizedCrop(
            height=h,
            width=w,
            scale=(min_scale, max_scale),
            ratio=(min_ratio, max_ratio),
        )

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and "seed" in additional:
            self.set_seed(additional["seed"])
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]


class Random_Crop(Abstract_augmentation):
    """
        torchvision.transforms.RandomCrop wrapper
        size of cropping will be decided by the 'h' and 'w' kwargs.
        'padding' kwarg is also available.
    """

    def __init__(self, args, kwargs):
        super(Random_Crop, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len in [2, 3]
        size = (int(kwargs["h"]), int(kwargs["w"]))
        self.transform = A.RandomCrop(*size)

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and "seed" in additional:
            self.set_seed(additional["seed"])
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]


class Blur(Abstract_augmentation):
    """
    Randomly blurs image with kernel size limit
    """

    def __init__(self, args, kwargs):
        super(Blur, self).__init__()
        limit = float(kwargs["limit"]) if "limit" in kwargs else 3
        self.transform = A.Blur(blur_limit=limit)

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and "seed" in additional:
            self.set_seed(additional["seed"])
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]


class Center_Crop(Abstract_augmentation):
    """
        torchvision.transforms.CenterCrop wrapper
        size of cropping will be decided by the 'h' and 'w' kwargs.
        'padding' kwarg is also available.
    """

    def __init__(self, args, kwargs):
        super(Center_Crop, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len in [2, 3]
        size = (int(kwargs["h"]), int(kwargs["w"]))
        self.set_cachable(*size)
        self.transform = A.CenterCrop(*size)

    def __call__(self, img, mask=None, additional=None):
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]


class Color_Jitter(Abstract_augmentation):
    def __init__(self, args, kwargs):
        super(Color_Jitter, self).__init__()
        self.args = args
        self.transform = A.HueSaturationValue()

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and "seed" in additional:
            self.set_seed(additional["seed"])
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]


class Hue_Saturation_Value(Abstract_augmentation):
    """
        HueSaturationValue wrapper
    """

    def __init__(self, args, kwargs):
        super(Hue_Saturation_Value, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len <= 3
        val, satur, hue = (
            int(kwargs["val"]) if "val" in kwargs else 0,
            int(kwargs["satur"]) if "satur" in kwargs else 0,
            int(kwargs["hue"]) if "hue" in kwargs else 0,
        )
        self.transform = A.HueSaturationValue(
            hue_shift_limit=hue, sat_shift_limit=satur, val_shift_limit=val, p=1
        )

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and "seed" in additional:
            self.set_seed(additional["seed"])
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]


class Random_Brightness_Contrast(Abstract_augmentation):
    """
        RandomBrightnessContrast wrapper
    """

    def __init__(self, args, kwargs):
        super(Random_Brightness_Contrast, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len <= 2
        contrast = float(kwargs["contrast"]) if "contrast" in kwargs else 0
        brightness = float(kwargs["brightness"]) if "brightness" in kwargs else 0

        self.transform = A.RandomBrightnessContrast(
            brightness_limit=brightness, contrast_limit=contrast, p=1
        )

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and "seed" in additional:
            self.set_seed(additional["seed"])
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]


class Gaussian_Blur(Abstract_augmentation):
    """
        albumentations.augmentations.transforms.GuassianBlur wrapper
        blur must odd and in range [3, inf). Default: (3, 7).
    """

    def __init__(self, args, kwargs):
        super(Gaussian_Blur, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len >= 1
        min_blur = int(kwargs["min_blur"]) if "min_blur" in kwargs else 3
        max_blur = int(kwargs["max_blur"])
        assert (max_blur % 2 == 1) and (min_blur % 2 == 1)
        self.transform = A.GaussianBlur(blur_limit=(min_blur, max_blur), p=1)

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and "seed" in additional:
            self.set_seed(additional["seed"])
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


class RandomRotate90(Abstract_augmentation):
    """
    Rotate image by 90s, one or more times
    """

    def __init__(self, args, kwargs):
        super(RandomRotate90, self).__init__()
        self.transform = A.RandomRotate90()

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and "seed" in additional:
            self.set_seed(additional["seed"])
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]


class Align_To_Left(Abstract_augmentation):
    """
    Aligns all images s.t. the breast will face left.
    Note: this should be applied after the scaling since the mask
    is the size of args.img_size.
    """

    def __init__(self, args, kwargs):
        super(Align_To_Left, self).__init__()
        assert len(kwargs.keys()) == 0

        self.set_cachable(args.img_size)

        # Create black image
        mask_r = Image.new("1", args.img_size)
        # Paint right side in white
        mask_r.paste(1, ((mask_r.size[0] * 3 // 4), 0, mask_r.size[0], mask_r.size[1]))
        mask_l = mask_r.transpose(Image.FLIP_LEFT_RIGHT)

        self.mask_r = mask_r
        self.mask_l = mask_l
        self.black = Image.new("I", args.img_size)

    def __call__(self, img, mask=None, additional=None):
        raise NotImplementedError(
            "This function must be reimplemented to"
            " support albumentations instead of torchvision"
        )
        left = img.copy()
        left.paste(self.black, mask=self.mask_l)
        left_sum = np.array(left.getdata()).sum()
        right = img.copy()
        right.paste(self.black, mask=self.mask_r)
        right_sum = np.array(right.getdata()).sum()
        if right_sum > left_sum:
            if mask is not None:
                return (
                    img.transpose(Image.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.FLIP_LEFT_RIGHT),
                )
            else:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask

        else:
            return img, mask


class Grayscale(Abstract_augmentation):
    """
    Given PIL image, converts it to grayscale
    with args.num_chan channels.
    """

    def __init__(self, args, kwargs):
        super(Grayscale, self).__init__()
        assert len(kwargs.keys()) == 0
        self.set_cachable(args.num_chan)

        self.transform = A.ToGray()

    def __call__(self, img, mask=None, additional=None):
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"]
