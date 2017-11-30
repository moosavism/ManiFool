import math

import torch
import torch.cuda

try:
    from .gpu_projective import proj_warp_gpu
except Exception as e:
    print('GPU not available. Ignore if not using GPU')

# necessary now, but should eventually not be
from skimage.transform import warp, AffineTransform
import numpy as np


def transform_matrix_offset_center(matrix, x, y):
    """Apply offset to a transform matrix so that the image is
    transformed about the center of the image.

    Arguments
    ---------
    matrix : 3x3 matrix/array

    x : integer
        height dimension of image to be transformed

    y : integer
        width dimension of image to be transformed
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = torch.DoubleTensor([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = torch.DoubleTensor([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = torch.mm(torch.mm(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, out, transform, fill_mode='nearest', fill_value=0., interp_order = 0):
    """Applies an affine transform to a 2D array, or to each channel of a 3D array.

    Arguments
    ---------
    x : np.ndarray
        array to transform. NOTE: array should be ordered CHW

    transform : 3x3 affine transform matrix
        matrix to apply
    """
    transform = transform_matrix_offset_center(transform, x.size()[2], x.size()[1])

    if x.is_cuda:
        if out is None:
            out = torch.cuda.FloatTensor(x.size())
        proj_warp_gpu(transform.float().cuda(), x, out)

    else:
        x = x.numpy()
        atfm = AffineTransform(matrix=transform.numpy())
        channel_images = [warp(x_channel,
                               inverse_map=atfm,
                               order=interp_order,
                               mode=fill_mode,
                               cval=fill_value,
                               preserve_range=True)
                               for x_channel in x]
        out = torch.from_numpy(np.stack(channel_images, axis=0).astype('float32'))

    return out

class Affine(object):

    def __init__(self,
                 tform_matrix=None,
                 rotation=None,
                 translation=None,
                 shear=None,
                 zoom=None,
                 fill_mode='constant',
                 fill_value=0.,
                 target_fill_mode='nearest',
                 target_fill_value=0.,
                 interp_order = 0):

        transforms = []
        if translation:
            if isinstance(translation, float):
                translation = (translation, translation)
            translation_tform = torch.DoubleTensor([[1, 0, translation[0]],
                                              [0, 1, translation[1]],
                                              [0, 0, 1]])
            transforms.append(translation_tform)

        if rotation:
            rotation_tform = torch.DoubleTensor([[math.cos(rotation), -math.sin(rotation), 0],
                                           [math.sin(rotation), math.cos(rotation), 0],
                                           [0, 0, 1]])
            transforms.append(rotation_tform)

        if shear:
            shear_tform = torch.DoubleTensor([[1, shear, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]])
            transforms.append(shear_tform)

        if zoom:
            if isinstance(zoom, float):
                zoom = (zoom, zoom)
            zoom_tform = torch.DoubleTensor([[zoom[0], 0, 0],
                                       [0, zoom[1], 0],
                                       [0, 0, 1]])
            transforms.append(zoom_tform)

        # collect all of the lazily returned tform matrices
        if tform_matrix is None:
            self.tform_matrix = torch.eye(3).double()
        else:
            self.tform_matrix = tform_matrix

        for tform in transforms[0:]:
            self.tform_matrix = torch.mm(self.tform_matrix, tform)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.interp_order = interp_order

    def __call__(self, x, out=None):

        x_transformed = apply_transform(x, out, self.tform_matrix,
                                        fill_mode=self.fill_mode,
                                        fill_value=self.fill_value,
                                        interp_order=self.interp_order)


        if out is None:
            return x_transformed

    def compose_(self, x):

        if type(x) is Affine:
            self.tform_matrix = x.tform_matrix.mm(self.tform_matrix)
        elif x.size() == (3,3):
            self.tform_matrix = x.mm(self.tform_matrix)
        else:
            raise ValueError(x.size())
    def compose(self, x):

        if type(x) is Affine:
            tform_matrix = x.tform_matrix.mm(self.tform_matrix)
        elif x.size() == (3,3):
            tform_matrix = x.mm(self.tform_matrix)
        else:
            raise ValueError(x.size())

        return Affine(tform_matrix=tform_matrix,
                      fill_mode=self.fill_mode,
                      fill_value=self.fill_value,
                      target_fill_mode=self.target_fill_mode,
                      target_fill_value=self.target_fill_value,
                      interp_order=self.interp_order)
