from __future__ import division
import torch
import math
import sys
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

from . import functional as F
from torchvision import transforms

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class RandomCrop(transforms.RandomCrop):
    def __init__(self, *args, **kwargs):
        super(RandomCrop, self).__init__(*args, **kwargs)
        self.playing = False

    def get_params(self, img, output_size):
        while True:
            w, h = img.size
            th, tw = output_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            self.playing = True
            while self.playing:
                yield i, j, th, tw

    def __call__(self, imgs):
        self.playing = False
        return list(map(super(RandomCrop, self).__call__, imgs))


class CenterCrop(transforms.CenterCrop):
    def __init__(self, *args, **kwargs):
        super(CenterCrop, self).__init__(*args, **kwargs)

    def __call__(self, imgs):
        return list(map(super(CenterCrop, self).__call__, imgs))


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given PIL Images randomly with a given probability.

    Args:
        p (float): probability of the images being flipped. Default value is 0.5
        is_flow (bool): whether the input data is optical flow or not. Default
            value is False.
    """
    def __init__(self, p=0.5, is_flow=False):
        super(RandomHorizontalFlip, self).__init__(p)
        self.is_flow = is_flow

    def __call__(self, imgs):
        """
        Args:
            imgs (List of PIL Images): Images to be flipped.

        Returns:
            List of PIL Images: Randomly flipped images.
        """
        if random.random() >= self.p:
            return imgs
        imgs = list(map(F.hflip, imgs))
        if self.is_flow:
            for i in range(0, len(imgs), 2):
                # invert flow pixel values when flipping
                imgs[i] = ImageOps.invert(imgs[i])
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, is_flow={1})'.format(self.p, self.is_flow)


class Normalize2d(transforms.Normalize):
    def __init__(self, *args, **kwargs):
        super(Normalize2d, self).__init__(*args, **kwargs)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor images of size (T * C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor images.
        """
        mean = self.mean * (tensor.size()[0] // len(self.mean))
        std = self.std * (tensor.size()[0] // len(self.std))
        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
        tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        return tensor


class Normalize3d(transforms.Normalize):
    def __init__(self, *args, **kwargs):
        super(Normalize3d, self).__init__(*args, **kwargs)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor images of size (C, T, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor images.
        """
        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return tensor


class Resize(transforms.Resize):
    def __init__(self, *args, **kwargs):
        super(Resize, self).__init__(*args, **kwargs)

    def __call__(self, imgs):
        return list(map(super(Resize, self).__call__, imgs))


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR
        self.offset_tmpl = self._get_offset_tmpl()
        self.length_tmpl = self._get_length_tmpl()

    def _get_offset_tmpl(self):
        ret = list()
        ret.append(lambda w, h: (0, 0))          # upper left
        ret.append(lambda w, h: (4 * w, 0))      # upper right
        ret.append(lambda w, h: (0, 4 * h))      # lower left
        ret.append(lambda w, h: (4 * w, 4 * h))  # lower right
        ret.append(lambda w, h: (2 * w, 2 * h))  # center

        if self.more_fix_crop:
            ret.append(lambda w, h: (0, 2 * h))      # center left
            ret.append(lambda w, h: (4 * w, 2 * h))  # center right
            ret.append(lambda w, h: (2 * w, 4 * h))  # lower center
            ret.append(lambda w, h: (2 * w, 0 * h))  # upper center

            ret.append(lambda w, h: (1 * w, 1 * h))  # upper left quarter
            ret.append(lambda w, h: (3 * w, 1 * h))  # upper right quarter
            ret.append(lambda w, h: (1 * w, 3 * h))  # lower left quarter
            ret.append(lambda w, h: (3 * w, 3 * h))  # lower right quarter

        return ret

    def _get_length_tmpl(self):
        ret = list()
        for i, h_scale in enumerate(self.scales):
            for j, w_scale in enumerate(self.scales):
                if abs(i - j) <= self.max_distort:
                    ret.append((w_scale, h_scale))
        return ret

    def _get_random_length(self, base):
        w_scale, h_scale = random.choice(self.length_tmpl)
        return int(base * w_scale), int(base * h_scale)

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size[0], im_size[1])
        ret = [img.crop(
            (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)).resize(
            (self.input_size[0], self.input_size[1]), self.interpolation) for img in img_group]
        return ret

    def _sample_crop_size(self, image_w, image_h):
        crop_w, crop_h = self._sample_length(image_w, image_h)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_w)
            h_offset = random.randint(0, image_h - crop_h)
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h,
                                                         crop_w, crop_h)

        return crop_w, crop_h, w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offset = random.choice(self.offset_tmpl)
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4
        return offset(w_step, h_step)

    def _sample_length(self, image_w, image_h):
        base_size = min(image_w, image_h)
        crop_w, crop_h = self._get_random_length(base_size)
        crop_w = self.input_size[0] if abs(crop_w - self.input_size[0]) < 3 else crop_w
        crop_h = self.input_size[1] if abs(crop_h - self.input_size[1]) < 3 else crop_h
        return crop_w, crop_h


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, explicit=False):
        # whether explicit expand temporal dim or not
        self.explicit = explicit

    def __call__(self, img_group):
        # H, W, C
        if self.explicit:
            if img_group[0].mode == 'L':
                return np.stack(img_group, axis=0).reshape(
                    (-1, 2) + img_group[0].size).transpose((3, 0, 1, 2))
            elif img_group[0].mode == 'RGB':
                return np.stack(img_group, axis=0).transpose((3, 0, 1, 2))
        else:
            if img_group[0].mode == 'L':
                return np.stack(img_group, axis=0)
            elif img_group[0].mode == 'RGB':
                return np.concatenate(img_group, axis=2)


class ToTensor(object):
    """ Converts a numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True, modality='RGB'):
        self.div = div
        self.modality = modality

    def __call__(self, pic):
        if len(pic.shape) == 4 or self.modality == 'Flow':
            img = torch.from_numpy(pic).contiguous()
        else:
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        return img.float().div(255) if self.div else img


class IdentityTransform(object):

    def __call__(self, data):
        return data
