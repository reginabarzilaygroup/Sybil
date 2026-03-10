import numpy as np
import torch
from monai.transforms import (
    ScaleIntensityd,
    Compose,
    LoadImaged,
    RandGaussianNoised,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGibbsNoised,
    RandFlipd,
    RandAffined,
    EnsureTyped,
    EnsureChannelFirstd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    Spacingd,
    ResizeWithPadOrCropd,
    Identityd,
    RandShiftIntensityd,
    Resized,
    Transposed,
    EnsureChannelFirst,
    EnsureType,
    Spacing,
    Identity,
    Transpose,
    Resize,
    ResizeWithPadOrCrop,
    ScaleIntensity,
    Transform,
    MapTransform,
)
import random


class CLIPTransformd(MapTransform):
    """Dictionary-based wrapper of CLIPTransform."""

    def __init__(self, keys, min_value, max_value, allow_missing_keys=False):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            min_value: minimum value to clip to.
            max_value: maximum value to clip to.
            allow_missing_keys: do not raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = torch.clip(d[key], self.min_value, self.max_value)
        return d


def min_max_normalize_batch(tensor):
    # tensor shape: (B, 1, D, H, W)
    # Compute min and max over (D, H, W) for each batch
    min_vals = tensor.amin(dim=(2, 3, 4), keepdim=True)
    max_vals = tensor.amax(dim=(2, 3, 4), keepdim=True)
    normalized = (tensor - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized


def pad_3d_box_centered(box, image, target_height=30, target_width=30, target_depth=3):
    """
    Expand a 3D bounding box to the specified target_height, target_width, and target_depth,
    centering the original box (nodule) within the new box as much as possible.

    Returns a new box dict and the corresponding slices.
    """
    # Extract original box
    z1, z2 = box["z_start"], box["z_stop"]
    y1, y2 = box["y_start"], box["y_stop"]
    x1, x2 = box["x_start"], box["x_stop"]

    d = z2 - z1
    h = y2 - y1
    w = x2 - x1

    # Compute center of the original box
    center_z = (z1 + z2) // 2
    center_y = (y1 + y2) // 2
    center_x = (x1 + x2) // 2

    # Compute new box coordinates centered on the nodule
    new_d1 = max(center_z - target_depth // 2, 0)
    new_d2 = new_d1 + target_depth

    new_y1 = max(center_y - target_height // 2, 0)
    new_y2 = new_y1 + target_height

    new_x1 = max(center_x - target_width // 2, 0)
    new_x2 = new_x1 + target_width

    # Ensure box does not exceed image boundaries
    img_h, img_w, img_d = image.shape
    if new_y2 > img_h:
        new_y2 = img_h
        new_y1 = max(new_y2 - target_height, 0)
    if new_x2 > img_w:
        new_x2 = img_w
        new_x1 = max(new_x2 - target_width, 0)
    if new_d2 > img_d:
        new_d2 = img_d
        new_d1 = max(new_d2 - target_depth, 0)

    new_box = {
        "z_start": new_d1,
        "z_stop": new_d2,
        "y_start": new_y1,
        "y_stop": new_y2,
        "x_start": new_x1,
        "x_stop": new_x2,
    }

    cbbox = (
        slice(new_y1, new_y2),
        slice(new_x1, new_x2),
        slice(new_d1, new_d2),
    )
    return cbbox


def random_pad_3d_box(
    box, image, min_height=30, min_width=30, min_depth=3, random_hw=True, random_d=True
):
    """
    Expand a 3D bounding box randomly to at least min_height and min_width
    while preserving the original box inside it and returning new coordinates
    in the original coordinate space.

    Returns a new box dict.
    """
    # Extract original box
    z1, z2 = box["z_start"], box["z_stop"]
    y1, y2 = box["y_start"], box["y_stop"]
    x1, x2 = box["x_start"], box["x_stop"]

    d = z2 - z1
    h = y2 - y1
    w = x2 - x1

    # Randomly determine target size (at least min + up to 20 more)
    if random_hw:
        target_h = random.randint(max(h, min_height), max(h, min_height) + 20)
        target_w = random.randint(max(w, min_width), max(w, min_width) + 20)
    else:
        # If not random, use fixed sizes
        target_h = max(min_height, h)
        target_w = max(min_width, w)
    if random_d:
        target_z = random.randint(max(d, min_depth), max(d, min_depth) + 10)
    else:
        target_z = max(min_depth, d)

    # Compute padding needed
    pad_h = target_h - h
    pad_w = target_w - w
    pad_d = target_z - d

    # Random offset of original box inside new box
    if random_hw:
        offset_y = random.randint(0, pad_h)
        offset_x = random.randint(0, pad_w)
    else:
        offset_y = pad_h // 2
        offset_x = pad_w // 2
    if random_d:
        offset_d = random.randint(0, pad_d)
    else:
        offset_d = pad_d // 2

    # Expand box in y and x directions
    new_y1 = max(y1 - offset_y, 0)
    new_y2 = new_y1 + target_h

    new_x1 = max(x1 - offset_x, 0)
    new_x2 = new_x1 + target_w

    new_d1 = max(z1 - offset_d, 0)
    new_d2 = new_d1 + target_z

    # z dimension is unchanged
    new_box = {
        "z_start": new_d1,
        "z_stop": new_d2,
        "y_start": new_y1,
        "y_stop": new_y2,
        "x_start": new_x1,
        "x_stop": new_x2,
    }

    img_h, img_w, img_d = image.shape
    cbbox = (
        slice(max(new_box["y_start"], 0), min(new_box["y_stop"], img_h)),
        slice(max(new_box["x_start"], 0), min(new_box["x_stop"], img_w)),
        slice(max(new_box["z_start"], 0), min(new_box["z_stop"], img_d)),
    )
    return cbbox


class AnatomixAugmentations:
    def __init__(self, args, split="train"):
        self.split = split
        self.train_transforms = self.get_train_transforms(args)
        self.val_transforms = self.get_val_transforms(args)
        self.test_transforms = self.get_test_transforms(args)

    def get_train_transforms(self, args):
        crop_size = args.anatomix_crop_size
        pixdim = tuple(args.ct_pixel_spacing)
        return Compose(
            [
                # LoadImaged(keys=["image"]),
                EnsureChannelFirstd(
                    keys=["image", "label"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image", "label"], allow_missing_keys=True),
                ScaleIntensityd(keys="image"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                ),  # Paper didn't use this, we're just doing this for Colab
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=crop_size,
                    pos=0.8,
                    neg=0.2,
                    allow_missing_keys=True,
                ),  # This is for MSD-Heart, use plain randcrop for other datasets
                RandGaussianNoised(keys=["image"], prob=0.33),
                RandBiasFieldd(keys=["image"], prob=0.33, coeff_range=(0.0, 0.05)),
                RandGibbsNoised(keys=["image"], prob=0.33, alpha=(0.0, 0.33)),
                RandAdjustContrastd(keys=["image"], prob=0.33),
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=0.33,
                    sigma_x=(0.0, 0.1),
                    sigma_y=(0.0, 0.1),
                    sigma_z=(0.0, 0.1),
                ),
                RandGaussianSharpend(keys=["image"], prob=0.33),
                RandAffined(
                    keys=["image", "label"],
                    prob=0.98,
                    mode=("bilinear", "nearest"),
                    rotate_range=(np.pi / 4, np.pi / 4, np.pi / 4),
                    scale_range=(0.2, 0.2, 0.2),
                    shear_range=(0.2, 0.2, 0.2),
                    spatial_size=crop_size,
                    padding_mode="zeros",
                    allow_missing_keys=True,
                ),
                ScaleIntensityd(keys="image"),
            ]
        )

    def get_val_transforms(self, args):
        pixdim = tuple(args.ct_pixel_spacing)
        return Compose(
            [
                EnsureChannelFirstd(
                    keys=["image", "label"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image", "label"], allow_missing_keys=True),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                ),  # Paper didn't use this, we're just doing this for Colab
                ScaleIntensityd(keys="image"),
            ]
        )

    def get_test_transforms(self, args):
        pixdim = tuple(args.ct_pixel_spacing)
        return Compose(
            [
                EnsureChannelFirstd(
                    keys=["image", "label"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                ),  # Paper didn't use this, we're just doing this for Colab
                ScaleIntensityd(keys="image"),
            ]
        )

    def __call__(self, sample):
        if self.split == "train":
            return self.train_transforms(sample)
        elif self.split == "dev":
            return self.val_transforms(sample)
        elif self.split == "test":
            return self.test_transforms(sample)
        else:
            raise ValueError(f"Unknown split: {self.split}")


class PatchAugmentations:
    def __init__(self, args, split="train"):
        self.split = "predict" if args.predict else split
        self.resample_pixel_spacing = args.resample_pixel_spacing
        self.augment_before_cropping = args.augment_before_cropping
        self.train_transforms = self.get_train_transforms(args)
        self.val_transforms = self.get_val_transforms(args)
        self.test_transforms = self.get_test_transforms(args)
        self.predict_transforms = self.get_predict_transforms(args)
        crop_size = args.anatomix_crop_size
        self._transform = ResizeWithPadOrCropd(
            keys=["image", "label"], spatial_size=crop_size, allow_missing_keys=True
        )

    def get_train_transforms(self, args):
        crop_size = args.anatomix_crop_size
        pixdim = tuple(args.ct_pixel_spacing)
        return Compose(
            [
                # LoadImaged(keys=["image"]),
                EnsureChannelFirstd(
                    keys=["image", "label"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image", "label"], allow_missing_keys=True),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                )
                if self.resample_pixel_spacing
                else Identityd(keys=["image", "label"], allow_missing_keys=True),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=2,
                    prob=0.5,
                    allow_missing_keys=True,
                ),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
                RandAdjustContrastd(keys=["image"], prob=0.33, gamma=(0.7, 1.3)),
                RandGaussianNoised(keys=["image"], prob=0.2),
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=0.33,
                    sigma_x=(0.0, 0.1),
                    sigma_y=(0.0, 0.1),
                    sigma_z=(0.0, 0.1),
                ),
                RandGaussianSharpend(keys=["image"], prob=0.33),
                Identityd(keys=["image", "label"], allow_missing_keys=True)
                if self.augment_before_cropping
                else ResizeWithPadOrCropd(
                    keys=["image", "label"],
                    spatial_size=crop_size,
                    allow_missing_keys=True,
                ),
                ScaleIntensityd(keys="image"),
            ]
        )

    def get_val_transforms(self, args):
        crop_size = args.anatomix_crop_size
        pixdim = tuple(args.ct_pixel_spacing)
        return Compose(
            [
                EnsureChannelFirstd(
                    keys=["image", "label"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image", "label"], allow_missing_keys=True),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                )
                if self.resample_pixel_spacing
                else Identityd(
                    keys=["image", "label"]
                ),  # Paper didn't use this, we're just doing this for Colab
                Identityd(keys=["image", "label"])
                if self.augment_before_cropping
                else ResizeWithPadOrCropd(
                    keys=["image", "label"], spatial_size=crop_size
                ),
                ScaleIntensityd(keys="image"),
            ]
        )

    def get_test_transforms(self, args):
        crop_size = args.anatomix_crop_size
        pixdim = tuple(args.ct_pixel_spacing)
        return Compose(
            [
                EnsureChannelFirstd(
                    keys=["image", "label"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                )
                if self.resample_pixel_spacing
                else Identityd(
                    keys=["image", "label"], allow_missing_keys=True
                ),  # Paper didn't use this, we're just doing this for Colab
                Identityd(keys=["image", "label"], allow_missing_keys=True)
                if self.augment_before_cropping
                else ResizeWithPadOrCropd(
                    keys=["image", "label"],
                    spatial_size=crop_size,
                    allow_missing_keys=True,
                ),
                ScaleIntensityd(keys="image"),
            ]
        )

    def get_predict_transforms(self, args):
        crop_size = args.anatomix_crop_size
        pixdim = tuple(args.ct_pixel_spacing)
        return Compose(
            [
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                EnsureTyped(keys=["image"]),
                Spacingd(
                    keys=["image"],
                    pixdim=pixdim,
                    mode=("bilinear"),
                )
                if self.resample_pixel_spacing
                else Identityd(
                    keys=["image"]
                ),  # Paper didn't use this, we're just doing this for Colab
                ScaleIntensityd(keys="image"),
            ]
        )

    def __call__(self, sample):
        if self.split == "train":
            return self.train_transforms(sample)
        elif self.split == "dev":
            return self.val_transforms(sample)
        elif self.split == "test":
            return self.test_transforms(sample)
        elif self.split == "predict":
            return self.predict_transforms(sample)
        else:
            raise ValueError(f"Unknown split: {self.split}")


class FullAugmentations(PatchAugmentations):
    def __init__(self, args, split="train"):
        super().__init__(args, split)
        self.segmentation_transform = self.get_segmentation_transforms(args)

    def get_segmentation_transforms(self, args):
        crop_size = args.anatomix_crop_size
        img_size = (-1,) + tuple(args.img_size)
        return Compose(
            [
                EnsureChannelFirst(channel_dim="no_channel"),
                EnsureType(),
                Transpose(indices=(0, 3, 1, 2)),
                Resize(
                    spatial_size=img_size,
                    mode="linear",
                ),
                Transpose(indices=(0, 2, 3, 1)),
                ResizeWithPadOrCrop(spatial_size=crop_size),
            ]
        )

    def get_train_transforms(self, args):
        crop_size = args.anatomix_crop_size
        img_size = (-1,) + tuple(args.img_size)
        pixdim = tuple(args.ct_pixel_spacing)
        return Compose(
            [
                EnsureChannelFirstd(
                    keys=["image", "lung"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image", "lung"], allow_missing_keys=True),
                Spacingd(
                    keys=["image", "lung"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                )
                if self.resample_pixel_spacing
                else Identityd(keys=["image", "lung"], allow_missing_keys=True),
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 3, 1, 2),
                    allow_missing_keys=True,
                ),
                Resized(
                    keys=["image", "lung"],
                    spatial_size=img_size,
                    mode="linear",
                    allow_missing_keys=True,
                ),
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 2, 3, 1),
                    allow_missing_keys=True,
                ),
                # RandFlipd(keys=["image",  "lung"], spatial_axis=2, prob=0.5),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
                RandAdjustContrastd(keys=["image"], prob=0.33, gamma=(0.7, 1.3)),
                RandGaussianNoised(keys=["image"], prob=0.2),
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=0.33,
                    sigma_x=(0.0, 0.1),
                    sigma_y=(0.0, 0.1),
                    sigma_z=(0.0, 0.1),
                ),
                RandGaussianSharpend(keys=["image"], prob=0.33),
                Identityd(keys="image"),
                ResizeWithPadOrCropd(
                    keys=["image", "lung"],
                    spatial_size=crop_size,
                    allow_missing_keys=True,
                ),
            ]
        )

    def get_val_transforms(self, args):
        crop_size = args.anatomix_crop_size
        pixdim = tuple(args.ct_pixel_spacing)
        img_size = (-1,) + tuple(args.img_size)
        return Compose(
            [
                EnsureChannelFirstd(
                    keys=["image", "lung"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image", "lung"], allow_missing_keys=True),
                Spacingd(
                    keys=["image", "lung"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                )
                if self.resample_pixel_spacing
                else Identityd(keys=["image", "lung"], allow_missing_keys=True),
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 3, 1, 2),
                    allow_missing_keys=True,
                ),
                Resized(
                    keys=["image", "lung"],
                    spatial_size=img_size,
                    mode="linear",
                    allow_missing_keys=True,
                ),
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 2, 3, 1),
                    allow_missing_keys=True,
                ),
                Identityd(keys="image"),
                ResizeWithPadOrCropd(
                    keys=["image", "lung"],
                    spatial_size=crop_size,
                    allow_missing_keys=True,
                ),
            ]
        )

    def get_test_transforms(self, args):
        return self.get_val_transforms(args)

    def get_predict_transforms(self, args):
        crop_size = args.anatomix_crop_size
        img_size = (-1,) + tuple(args.img_size)
        pixdim = tuple(args.ct_pixel_spacing)
        return Compose(
            [
                EnsureChannelFirstd(
                    keys=["image", "lung"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image", "lung"], allow_missing_keys=True),
                Spacingd(
                    keys=["image", "lung"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                )
                if self.resample_pixel_spacing
                else Identityd(keys=["image", "lung"], allow_missing_keys=True),
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 3, 1, 2),
                    allow_missing_keys=True,
                ),
                Resized(
                    keys=["image", "lung"],
                    spatial_size=img_size,
                    mode="linear",
                    allow_missing_keys=True,
                ),
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 2, 3, 1),
                    allow_missing_keys=True,
                ),
                Identityd(keys="image"),
                ResizeWithPadOrCropd(
                    keys=["image", "lung"],
                    spatial_size=crop_size,
                    allow_missing_keys=True,
                ),
            ]
        )

    def __call__(self, sample):
        if self.split == "train":
            return self.train_transforms(sample)
        elif self.split == "dev":
            return self.val_transforms(sample)
        elif self.split == "test":
            return self.test_transforms(sample)
        elif self.split == "predict":
            return self.predict_transforms(sample)
        else:
            raise ValueError(f"Unknown split: {self.split}")


class PillarAugmentations:
    def __init__(self, args, split="train"):
        self.split = "predict" if args.predict else split
        self.resample_pixel_spacing = args.resample_pixel_spacing
        self.augment_before_cropping = args.augment_before_cropping
        self.train_transforms = self.get_train_transforms(args)
        self.val_transforms = self.get_val_transforms(args)
        self.test_transforms = self.get_test_transforms(args)
        self.predict_transforms = self.get_predict_transforms(args)
        crop_size = args.anatomix_crop_size
        self._transform = ResizeWithPadOrCropd(
            keys=["image", "label"], spatial_size=crop_size, allow_missing_keys=True
        )
        self.segmentation_transform = self.get_segmentation_transforms(args)

    def get_segmentation_transforms(self, args):
        crop_size = args.anatomix_crop_size
        img_size = (-1,) + tuple(args.img_size)
        return Compose(
            [
                EnsureChannelFirst(channel_dim="no_channel"),
                EnsureType(),
                # Transpose(indices=(0, 3, 1, 2)),
                # Resize(
                #     spatial_size=img_size,
                #     mode="linear",
                # ),
                # Transpose(indices=(0, 2, 3, 1)),
                ResizeWithPadOrCrop(spatial_size=crop_size),
            ]
        )

    def get_train_transforms(self, args):
        crop_size = args.anatomix_crop_size
        img_size = (-1,) + tuple(args.img_size)
        pixdim = tuple(args.ct_pixel_spacing)
        return Compose(
            [
                EnsureChannelFirstd(
                    keys=["image", "lung"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image", "lung"], allow_missing_keys=True),
                # resample
                Spacingd(
                    keys=["image", "lung"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                )
                if self.resample_pixel_spacing
                else Identityd(keys=["image", "lung"], allow_missing_keys=True),
                # pad/crop in 2d
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 3, 1, 2),
                    allow_missing_keys=True,
                ),
                ResizeWithPadOrCropd(
                    keys=["image", "lung"],
                    spatial_size=img_size,
                    allow_missing_keys=True,
                ),
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 2, 3, 1),
                    allow_missing_keys=True,
                ),
                # clip
                CLIPTransformd(
                    keys=["image"],
                    min_value=-1024,
                    max_value=3071,
                    allow_missing_keys=True,
                ),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
                RandAdjustContrastd(keys=["image"], prob=0.33, gamma=(0.7, 1.3)),
                RandGaussianNoised(keys=["image"], prob=0.2),
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=0.33,
                    sigma_x=(0.0, 0.1),
                    sigma_y=(0.0, 0.1),
                    sigma_z=(0.0, 0.1),
                ),
                RandGaussianSharpend(keys=["image"], prob=0.33),
                # pad in z if needed
                ResizeWithPadOrCropd(
                    keys=["image", "lung"],
                    spatial_size=crop_size,
                    allow_missing_keys=True,
                ),
            ]
        )

    def get_val_transforms(self, args):
        crop_size = args.anatomix_crop_size
        pixdim = tuple(args.ct_pixel_spacing)
        img_size = (-1,) + tuple(args.img_size)
        return Compose(
            [
                EnsureChannelFirstd(
                    keys=["image", "lung"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image", "lung"], allow_missing_keys=True),
                Spacingd(
                    keys=["image", "lung"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                )
                if self.resample_pixel_spacing
                else Identityd(keys=["image", "lung"], allow_missing_keys=True),
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 3, 1, 2),
                    allow_missing_keys=True,
                ),
                ResizeWithPadOrCropd(
                    keys=["image", "lung"],
                    spatial_size=img_size,
                    allow_missing_keys=True,
                ),
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 2, 3, 1),
                    allow_missing_keys=True,
                ),
                CLIPTransformd(
                    keys=["image"],
                    min_value=-1024,
                    max_value=3071,
                    allow_missing_keys=True,
                ),
                ResizeWithPadOrCropd(
                    keys=["image", "lung"],
                    spatial_size=crop_size,
                    allow_missing_keys=True,
                ),
            ]
        )

    def get_test_transforms(self, args):
        return self.get_val_transforms(args)

    def get_predict_transforms(self, args):
        crop_size = args.anatomix_crop_size
        img_size = (-1,) + tuple(args.img_size)
        pixdim = tuple(args.ct_pixel_spacing)
        return Compose(
            [
                EnsureChannelFirstd(
                    keys=["image", "lung"],
                    channel_dim="no_channel",
                    allow_missing_keys=True,
                ),
                EnsureTyped(keys=["image", "lung"], allow_missing_keys=True),
                Spacingd(
                    keys=["image", "lung"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True,
                )
                if self.resample_pixel_spacing
                else Identityd(keys=["image", "lung"], allow_missing_keys=True),
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 3, 1, 2),
                    allow_missing_keys=True,
                ),
                ResizeWithPadOrCropd(
                    keys=["image", "lung"],
                    spatial_size=img_size,
                    allow_missing_keys=True,
                ),
                Transposed(
                    keys=["image", "lung"],
                    indices=(0, 2, 3, 1),
                    allow_missing_keys=True,
                ),
                CLIPTransformd(
                    keys=["image"],
                    min_value=-1024,
                    max_value=3071,
                    allow_missing_keys=True,
                ),
                ResizeWithPadOrCropd(
                    keys=["image", "lung"],
                    spatial_size=crop_size,
                    allow_missing_keys=True,
                ),
            ]
        )

    def __call__(self, sample):
        if self.split == "train":
            return self.train_transforms(sample)
        elif self.split == "dev":
            return self.val_transforms(sample)
        elif self.split == "test":
            return self.test_transforms(sample)
        elif self.split == "predict":
            return self.predict_transforms(sample)
        else:
            raise ValueError(f"Unknown split: {self.split}")
