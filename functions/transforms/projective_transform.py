# import scipy.ndimage as ndi
import numpy as np
import torch
from skimage.transform import warp, ProjectiveTransform


def transform_matrix_offset_center(matrix, x, y):
    """Apply offset to a transform matrix so that the image is
    transformed about the center of the image.

    NOTE: This is a fairly simple operaion, so can easily be
    moved to full torch.

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

def projective_map(output_coord, tform_matrix):

    output_coord = np.append(np.array(output_coord),1)
    input_coord = tform_matrix.dot(output_coord)

    return (input_coord[0]/input_coord[2],input_coord[1]/input_coord[2])


def projective_transform_2d(im,
                            tform_matrix,
                            fill_mode='constant',
                            fill_value=0.,
                            target_fill_mode='nearest',
                            target_fill_value=0.,
                            interp_order = 1):

    transform = transform_matrix_offset_center(tform_matrix,im.size()[2],im.size()[1])
    # transform = tform_matrix

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

    def __init__(self,
                 tform_matrix,
                 fill_mode='constant',
                 fill_value=0.,
                 target_fill_mode='nearest',
                 target_fill_value=0.,
                 interp_order = 1):

        self.tform_matrix = tform_matrix
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.interp_order = interp_order

    def __call__(self, x):

        x_transformed = projective_transform_2d(x,
                                                self.tform_matrix,
                                                fill_mode=self.fill_mode,
                                                fill_value=self.fill_value,
                                                interp_order=self.interp_order)

        return x_transformed
    def compose_(self, x):

        if type(x) is Projective or type(x) is Affine:
            self.tform_matrix = x.tform_matrix.mm(self.tform_matrix)
        elif x.size() == (3,3):
            self.tform_matrix = x.mm(self.tform_matrix)
        #else:
            #TODO throw an error

    def compose(self, x):

        if type(x) is Projective or type(x) is Affine:
            tform_matrix = x.tform_matrix.mm(self.tform_matrix)
        elif x.size() == (3,3):
            tform_matrix = x.mm(self.tform_matrix)
        #else:
            #TODO throw an error

        return Projective(tform_matrix=tform_matrix,
                          fill_mode=self.fill_mode,
                          fill_value=self.fill_value,
                          target_fill_mode=self.target_fill_mode,
                          target_fill_value=self.target_fill_value,
                          interp_order=self.interp_order)
