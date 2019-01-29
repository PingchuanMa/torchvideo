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

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Resize", "Scale", "CenterCrop", "Pad",
           "Lambda", "RandomApply", "RandomChoice", "RandomOrder", "RandomCrop", "RandomHorizontalFlip",
           "RandomVerticalFlip", "RandomResizedCrop", "RandomSizedCrop", "FiveCrop", "TenCrop", "LinearTransformation",
           "ColorJitter", "RandomRotation", "RandomAffine", "Grayscale", "RandomGrayscale"]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, vid):
        for t in self.transforms:
            vid = t(vid)
        return vid

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, vid):
        """
        Args:
            vid (list of PIL Image or list of numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(vid)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToPILImage(object):
    """Convert a tensor or an ndarray to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
             - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
             - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
             - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
             - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
              ``short``).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or list of numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return F.to_pil_image(pic, self.mode)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, vid):
        """
        Args:
            vid (list of PIL Image): Images to be scaled.

        Returns:
            list of PIL Image: Rescaled images.
        """
        return [F.resize(img, self.size, self.interpolation) for img in vid]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class Scale(Resize):
    """
    Note: This transform is deprecated in favor of Resize.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                      "please use transforms.Resize instead.")
        super(Scale, self).__init__(*args, **kwargs)


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, vid):
        """
        Args:
            vid (list of PIL Image): Images to be cropped.

        Returns:
            list of PIL Image: Cropped images.
        """
        return [F.center_crop(img, self.size) for img in vid]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Pad(object):
    """Pad the given PIL Image on all sides with the given "pad" value.

    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image

            - reflect: pads with reflection of image without repeating the last value on the edge

                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge

                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, vid):
        """
        Args:
            vid (list of PIL Image): Images to be padded.

        Returns:
            list of PIL Image: Padded images.
        """
        return [F.pad(img, self.padding, self.fill, self.padding_mode) for img in vid]

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, vid):
        return [self.lambd(img) for img in vid]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTransforms(object):
    """Base class for a list of transformations with randomness

    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    @staticmethod
    def _apply(transforms, img):
        for t in transforms:
            img = t(img)
        return img

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, vid):
        if self.p < random.random():
            return vid
        return [self._apply(self.transforms, img) for img in vid]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order
    """
    def __call__(self, vid):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        return [self._apply(self._random_iter(order), img) for img in vid]

    def _random_iter(self, order):
        for i in order:
            yield self.transforms[i]


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """
    def __call__(self, vid):
        t = random.choice(self.transforms)
        return [t(img) for img in vid]


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        """
        Args:
            vid (list): Images to be cropped.

        Returns:
            list: Cropped images.
        """
        if self.padding is not None:
            vid = [F.pad(img, self.padding, self.fill, self.padding_mode) for img in vid]

        # pad the width if needed
        if self.pad_if_needed and vid[0].size[0] < self.size[1]:
            vid = [F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode) for img in vid]
        # pad the height if needed
        if self.pad_if_needed and vid[0].size[1] < self.size[0]:
            vid = [F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode) for img in vid]

        i, j, h, w = self.get_params(vid[0], self.size)

        return [F.crop(img, i, j, h, w) for img in vid]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Images randomly with a given probability.

    Args:
        p (float): probability of the images being flipped. Default value is 0.5
        flow (bool): whether the input data is optical flow or not. It should be
            explicitly assigned since a grayscale video also consists of `L`-mode
            images. Default value is False.
    """
    def __init__(self, p=0.5, flow=False):
        self.p = p
        self.flow = flow

    def __call__(self, vid):
        """
        Args:
            vid (list of PIL Image): Images to be flipped.

        Returns:
            list of PIL Image: Randomly flipped images.
        """
        if self.p < random.random():
            return vid
        if self.flow:
            return [F.invert(F.hflip(img)) if i % 2 == 0 else F.hflip(img) for i, img in enumerate(vid)]
        return list(map(F.hflip, vid))

    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, is_flow={1})'.format(self.p, self.is_flow)


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        flow (bool): whether the input data is optical flow or not. It should be
            explicitly assigned since a grayscale video also consists of `L`-mode
            images. Default value is False.
    """

    def __init__(self, p=0.5, flow=False):
        self.p = p
        self.flow = flow

    def __call__(self, vid):
        """
        Args:
            vid (list of PIL Image): Images to be flipped.

        Returns:
            list of PIL Image: Randomly flipped images.
        """
        if self.p < random.random():
            return vid
        if self.flow:
            return [F.invert(F.vflip(img)) if i % 2 == 1 else F.hflip(img) for i, img in enumerate(vid)]
        return list(map(F.hflip, vid))

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


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
            (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
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
