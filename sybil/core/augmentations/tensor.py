import torchvision
import torch
from sandstone.augmentations.factory import RegisterTensorAugmentation
import numpy as np
from sandstone.augmentations.abstract import Abstract_augmentation
import pdb

@RegisterTensorAugmentation("normalize_2d")
class Normalize_Tensor_2d(Abstract_augmentation):
    '''
    torchvision.transforms.Normalize wrapper.
    '''
    def __init__(self, args, kwargs):
        super(Normalize_Tensor_2d, self).__init__()
        assert len(kwargs) == 0
        channel_means = [args.img_mean] if len(args.img_mean) == 1 else args.img_mean
        channel_stds = [args.img_std] if len(args.img_std) == 1 else args.img_std

        self.transform = torchvision.transforms.Normalize(torch.Tensor(channel_means),
                                                          torch.Tensor(channel_stds))

        self.permute = args.input_loader_name in ['numpy_image_loader', 'color_image_loader']

    def __call__(self, img, mask=None, additional=None):
        if len(img.size()) == 2:
            img = img.unsqueeze(0)
        if self.permute:
            img = img.permute(2,0,1)
            if mask is None:
                return self.transform(img).permute(1,2,0), mask
            return self.transform(img).permute(1,2,0),self.transform(mask).permute(1,2,0)
        return self.transform(img), mask

@RegisterTensorAugmentation("inv_normalize_2d")
class Inv_Normalize_Tensor_2d(Abstract_augmentation):
    '''
    torchvision.transforms.Normalize wrapper.
    '''
    def __init__(self, args, kwargs):
        super(Inv_Normalize_Tensor_2d, self).__init__()
        assert len(kwargs) == 0
        channel_means = [args.img_mean] if len(args.img_mean) == 1 else args.img_mean
        channel_stds = [args.img_std] if len(args.img_std) == 1 else args.img_std

        self.transform = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0. for _ in channel_means],
                                                     std = [1/s for s in channel_stds]),
                                torchvision.transforms.Normalize(mean = [-m for m in channel_means],
                                                     std = [1. for _ in channel_stds]),
                               ])

        self.permute = args.input_loader_name in ['numpy_image_loader', 'color_image_loader']

    def __call__(self, img, additional=None):
        if len(img.size()) == 2:
            img = img.unsqueeze(0)
        if self.permute:
            img = img.permute(2,0,1)
            return self.transform(img).permute(1,2,0)
        return self.transform(img)


@RegisterTensorAugmentation("normalize_2d_img_per_chan")
class Normalize_Tensor_2d_Img_Per_Chan(Abstract_augmentation):
    '''
    torchvision.transforms.Normalize wrapper.
    '''

    def __init__(self, args, kwargs):
        super(Normalize_Tensor_2d_Img_Per_Chan, self).__init__()
        assert len(kwargs) == 0
        self.channel_means = [args.img_mean] if len(args.img_mean) == 1 else args.img_mean
        self.channel_stds = [args.img_std] if len(args.img_std) == 1 else args.img_std

    def __call__(self, img, mask=None, additional=None):
        channel = additional['chan'] - 1
        return img.sub_(self.channel_means[channel]).div_(self.channel_stds[channel]), mask



@RegisterTensorAugmentation("cutout")
class CutOut(Abstract_augmentation):
    '''
        Randomly sets a patch to black.
        size of patch will be decided by the 'h' and 'w' kwargs. Done with probablity p
        From: https://arxiv.org/pdf/1708.04552.pdf
    '''

    def __init__(self, args, kwargs):
        super(CutOut, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len == 3
        mask_w, mask_h, p = (int(kwargs['w']), int(kwargs['h']), float(kwargs['p']))
        img_w, img_h = self.args.img_size
        mask = 0

        def cutout(image):
            if np.random.random() > p:
                return image
            center_x, center_y = np.random.randint(0, img_w), np.random.randint(0, img_h)

            x_min, x_max = center_x - (mask_w // 2), center_x + (mask_w // 2)
            y_min, y_max = center_y - (mask_h // 2), center_y + (mask_h // 2)
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(img_w, y_min), min(img_h, y_max)
            image[y_min:y_max, x_min:x_max] *= mask

            return image
        self.transform = torchvision.transforms.Lambda(cutout)

    def __call__(self, img, mask=None, additional=None):
        if additional is not None and 'seed' in additional:
            self.set_seed(additional['seed'])
        if mask is not None:
            mask = self.transform(mask)
        return self.transform(img), mask

@RegisterTensorAugmentation("add_guassian_noise")
class GaussianNoise(Abstract_augmentation):
    '''
        Add gaussian noise to img.
    '''

    def __init__(self, args, kwargs):
        super(GaussianNoise, self).__init__()
        kwargs_len = len(kwargs.keys())
        assert kwargs_len == 2
        self.mu = float(kwargs['mean'])
        self.sigma = float(kwargs['std'])

    def __call__(self, image, mask = None, additional=None):
        if additional is not None and 'seed' in additional:
            self.set_seed(additional['seed'])
        return image + torch.randn(image.shape[-2:])*self.sigma + self.mu, mask

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mu, self.sigma)

@RegisterTensorAugmentation("channel_shift")
class Channel_Shift_Tensor(Abstract_augmentation):
    '''
    Randomly shifts values in a channel by a random number uniformly sampled
    from -shift:shift.
    '''

    def __init__(self, args, kwargs):
        super(Channel_Shift_Tensor, self).__init__()
        assert len(kwargs) == 1
        shift = float(kwargs['shift'])

        def apply_shift(img):
            shift_val = float(np.random.uniform(low=-shift, high=shift, size=1))
            return img + shift_val

        self.transform = torchvision.transforms.Lambda(apply_shift)

    def __call__(self, img, mask = None, additional=None):
        if additional is not None and 'seed' in additional:
            self.set_seed(additional['seed'])
        return self.transform(img), mask


@RegisterTensorAugmentation("force_num_chan_2d")
class Force_Num_Chan_Tensor_2d(Abstract_augmentation):
    '''
    Convert gray scale images to image with args.num_chan num channels.
    '''

    def __init__(self, args, kwargs):
        super(Force_Num_Chan_Tensor_2d, self).__init__()
        assert len(kwargs) == 0
        self.args = args

    def __call__(self, img, mask = None, additional=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        num_dims = len(img.shape)
        if num_dims == 2:
            img = img.unsqueeze(0)
        existing_chan = img.size()[0]
        if not existing_chan == self.args.num_chan:
            return img.expand(self.args.num_chan, *img.size()[1:]), mask
        return img, mask
