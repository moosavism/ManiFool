import math

import torch
import torch.cuda

try:
    from .gpu_projective import proj_warp_gpu
except Exception as e:
    print('GPU not available. Ignore if not using GPU')

from skimage.transform import warp, AffineTransform
import numpy as np


def transform_matrix_offset_center(matrix, x, y):
    """Apply offset to a transform matrix so that the image is transformed about the center of the image.

    Inputs:
    :Tensor matrix: 3x3 transformation matrix
    :int x: height of the image
    :int y: width of the image

    Outputs:
    :Tensor transform_matrix: the shifted 3x3 transform matrix
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = torch.DoubleTensor([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = torch.DoubleTensor([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = torch.mm(torch.mm(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform, fill_mode='constant', fill_value=0., interp_order = 0):
    """
    Applies an affine transform to an image. Uses GPU for transformation if the input image is a CUDA Tensor. GPU
    transformation only uses bilinear interpolation in its current form.

    Inputs:
    :Tensor x: image to be transformed
    :Tensor transform: 3x3 transformation metric
    :string fill_mode: filling method for points outside the image
                       (only used if CPU is used, more info in skimage.transform.warp)
    :double fill_value: filling value for points outside the image
                       (only used if CPU is used and fill mode is constant, more info in skimage.transform.warp)
    :int interp_order: interpolation method
                       (only used if CPU is used, more info in skimage.transform.warp)

    Outputs:
    :Tensor out: transformed image

    """
    transform = transform_matrix_offset_center(transform, x.size()[2], x.size()[1])

    if x.is_cuda:
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
    """
    A class to perform an affine transformation. The transformation can be constructed by using various sub
    transformations(rotation, translation, shear, scaling) or simply using a 3x3 transformation matrix. The class
    only holds the transformation matrix and transformation parameters, and the transformation itself is done using only
    one interpolation.

    Attributes:
    :Tensor tform_matrix: 3x3 transform matrix
    :string fill_mode: filling method for points outside the image
                       (only used if CPU is used, more info in skimage.transform.warp)
    :double fill_value: filling value for points outside the image
                       (only used if CPU is used and fill mode is constant, more info in skimage.transform.warp)
    :int interp_order: interpolation method
                       (only used if CPU is used, more info in skimage.transform.warp)
    """

    def __init__(self,
                 tform_matrix=None,
                 rotation=None,
                 translation=None,
                 shear=None,
                 zoom=None,
                 fill_mode='constant',
                 fill_value=0.,
                 interp_order = 0):
        """
        Initializes the Affine class

        Inputs:
        :Tensor tform_matrix: 3x3 transform matrix
        :float rotation: rotation angle in radians
        :(float,float) translation: translation values in x and y directions. If translation is a single float, the
                                    image is translated in both direction by the same amount
        :float shear: shear value
        :(float,float) zoom: scaling value in x and y directions. If translation is a single float, the
                             image is scaled in both direction by the same amount
        :string fill_mode: filling method for points outside the image
                           (only used if CPU is used, more info in skimage.transform.warp)
        :double fill_value: filling value for points outside the image
                           (only used if CPU is used and fill mode is constant, more info in skimage.transform.warp)
        :int interp_order: interpolation method
                           (only used if CPU is used, more info in skimage.transform.warp)
        """

        #Generate the matrices for each transformation
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

        if tform_matrix is None:
            self.tform_matrix = torch.eye(3).double()
        else:
            self.tform_matrix = tform_matrix

        # Collect all generated transform matrices
        for tform in transforms[0:]:
            self.tform_matrix = torch.mm(self.tform_matrix, tform)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interp_order = interp_order

    def __call__(self, x):
        """
        Transforms the image by calling apply transform.

        Input:
        :Tensor x: image to be transformed

        Output:
        :Tensor x_transformed: transformed image
        """
        x_transformed = apply_transform(x, self.tform_matrix,
                                        fill_mode=self.fill_mode,
                                        fill_value=self.fill_value,
                                        interp_order=self.interp_order)



        return x_transformed

    def compose_(self, x):
        """
        In-place composition of two affine transformations

        Input:
        :Affine or Tensor x: transformation to be composed. If x is a tensor, than it has to be 3x3 affine
                             transformation matrix
        """

        if type(x) is Affine:
            self.tform_matrix = x.tform_matrix.mm(self.tform_matrix)
        elif x.size() == (3,3):
            self.tform_matrix = x.mm(self.tform_matrix)
        else:
            raise ValueError(x.size())
    def compose(self, x):
        """
        Returns the composition of two affine transformations

        Input:
        :Affine or Tensor x: transformation to be composed with self. If x is a tensor, than it has to be 3x3 affine
                             transformation matrix

        Output:
        :Affine out: composition of x and self
        """

        if type(x) is Affine:
            tform_matrix = x.tform_matrix.mm(self.tform_matrix)
        elif x.size() == (3,3):
            tform_matrix = x.mm(self.tform_matrix)
        else:
            raise ValueError(x.size())

        return Affine(tform_matrix=tform_matrix,
                      fill_mode=self.fill_mode,
                      fill_value=self.fill_value,
                      interp_order=self.interp_order)
