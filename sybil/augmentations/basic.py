import torch
import torchvision
from sybil.augmentations.abstract import Abstract_augmentation
from albumentations.pytorch import ToTensorV2


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


class Permute3d(Abstract_augmentation):
    """Permute tensor (T, C, H, W) ==> (C, T, H, W)"""

    def __init__(self):
        super(Permute3d, self).__init__()

        def permute_3d(tensor):
            return tensor.permute(1, 0, 2, 3)

        self.transform = torchvision.transforms.Lambda(permute_3d)

    def __call__(self, tensor, mask=None, additional=None):
        if mask is None:
            return self.transform(tensor), mask
        return self.transform(tensor), self.transform(mask)


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
