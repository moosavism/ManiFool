# import scipy.ndimage as ndi
import numpy as np
import torch
from skimage.transform import warp, ProjectiveTransform
from .affine_transformsv2 import Affine


def transform_matrix_offset_center(matrix, x, y):
    """
    Apply offset to a transform matrix so that the image is transformed about the center of the image.

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

def projective_transform_2d(im,
                            tform_matrix,
                            fill_mode='constant',
                            fill_value=0.,
                            interp_order = 1):
    """
    Applies a projective transform to an image. Uses GPU for transformation if the input image is a CUDA Tensor. GPU
    transformation only uses bilinear interpolation in its current form.

    Inputs:
    :Tensor im: image to be transformed
    :Tensor transform: 3x3 transformation metric
    :string fill_mode: filling method for points outside the image
                       (only used if CPU is used, more info in skimage.transform.warp)
    :double fill_value: filling value for points outside the image
                       (only used if CPU is used and fill mode is constant, more info in skimage.transform.warp)
    :int interp_order: interpolation method
                       (only used if CPU is used, more info in skimage.transform.warp)

    Outputs:
    :Tensor x: transformed image

    """
    transform = transform_matrix_offset_center(tform_matrix,im.size()[2],im.size()[1])

    if im.is_cuda:
        from .gpu_projective import proj_warp_gpu
        x = torch.cuda.FloatTensor(im.size())
        proj_warp_gpu(transform.float().cuda(), im, x)

    else:
        ptfm = ProjectiveTransform(transform.numpy())
        channel_images = [warp(x_channel,
                               inverse_map=ptfm,
                               order=interp_order,
                               mode=fill_mode,
                               cval=fill_value,
                               preserve_range=True)
                               for x_channel in im.numpy()]
        x = torch.from_numpy(np.stack(channel_images,axis=0).astype('float32'))

    return x

class Projective(object):
    """
    A class to perform a projective transformation. The transformation is represented by using a 3x3 transformation
    matrix. The class only holds the transformation matrix and transformation parameters, and the transformation itself
    is done using only one interpolation.

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
                 tform_matrix,
                 fill_mode='constant',
                 fill_value=0.,
                 interp_order = 1):
        """
        Initializes the Projective object.

        Inputs:
        :Tensor tform_matrix: 3x3 transform matrix
        :string fill_mode: filling method for points outside the image
                           (only used if CPU is used, more info in skimage.transform.warp)
        :double fill_value: filling value for points outside the image
                           (only used if CPU is used and fill mode is constant, more info in skimage.transform.warp)
        :int interp_order: interpolation method
                           (only used if CPU is used, more info in skimage.transform.warp)
        """
        self.tform_matrix = tform_matrix
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interp_order = interp_order

    def __call__(self, x):
        """
        Transforms the image.

        Input:
        :Tensor x: image to be transformed

        Output:
        :Tensor x_transformed: transformed image
        """
        x_transformed = projective_transform_2d(x,
                                                self.tform_matrix,
                                                fill_mode=self.fill_mode,
                                                fill_value=self.fill_value,
                                                interp_order=self.interp_order)

        return x_transformed

    def compose_(self, x):
        """
        In-place composition of two projective transformations

        Input:
        :Projective, Affine or Tensor x: transformation to be composed
        """
        if type(x) is Projective or type(x) is Affine:
            self.tform_matrix = x.tform_matrix.mm(self.tform_matrix)
        elif x.size() == (3,3):
            self.tform_matrix = x.mm(self.tform_matrix)
        #else:
            #TODO throw an error

    def compose(self, x):
        """
        Returns the composition of two projective transformations

        Input:
        :Projective, Affine or Tensor x: transformation to be composed with self

        Output:
        :Projective out: composition of x and self
        """
        if type(x) is Projective or type(x) is Affine:
            tform_matrix = x.tform_matrix.mm(self.tform_matrix)
        elif x.size() == (3,3):
            tform_matrix = x.mm(self.tform_matrix)
        #else:
            #TODO throw an error

        return Projective(tform_matrix=tform_matrix,
                          fill_mode=self.fill_mode,
                          fill_value=self.fill_value,
                          interp_order=self.interp_order)
