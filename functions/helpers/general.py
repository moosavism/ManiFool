import numbers

import numpy as np
import scipy.linalg as slin
import torch
from torch.autograd import Variable

from ..transforms.affine_transformsv2 import Affine
from ..transforms.projective_transform import Projective


def eval_network (im, net):
    """
    Feeds the image to the network and returns the output label

    Inputs:
    :Tensor im: input image
    :Module net: network

    Output:
    :int output_index: output label of the image
    """
    x = Variable(im.unsqueeze(0),requires_grad=True)
    output = net(x)
    _, output_index = torch.max(output.data,1)

    return output_index[0]

def make_algebra_matrix(tau, mode):
    """
    Generates the matrix on the Lie algebra of the transformation group for vector tau

    Input:
    :Tensor tau: parameter vector
    :String mode: transformation set

    Output:
    :array B: 3x3 matrix on the Lie algebra of mode
    """
    tau.squeeze_()

    if mode == 'rotation':
        B = np.array([[0,-tau[0],0],[tau[0],0,0],[0,0,0]])

    elif mode == 'translation':
        B = np.array([[0,0,tau[0]],[0,0,tau[1]],[0,0,0]])

    elif mode == 'rotation+scaling':
        B = np.array([[tau[1],-tau[0],0],[tau[0],tau[1],0],[0,0,0]])

    elif mode == 'rotation+translation':
        B = np.array([[0,-tau[0],tau[1]],[tau[0],0,tau[2]],[0,0,0]])

    elif mode == 'scaling+translation':
        B = np.array([[tau[0],0,tau[1]],[0,tau[0],tau[2]],[0,0,0]])

    elif mode == 'similarity':
        B = np.array([[tau[3],-tau[0],tau[1]],[tau[0],tau[3],tau[2]],[0,0,0]])

    elif mode == 'affine':
        B = np.array([[tau[3]+tau[4],-tau[0]+tau[5],tau[1]],
                     [tau[0]+tau[5],tau[3]-tau[4],tau[2]],[0,0,0]])

    elif mode == 'projective':
        B = np.array([[tau[0],tau[1],tau[2]],[tau[3],tau[4],tau[5]],
                      [tau[6],tau[7],-tau[0]-tau[4]]])

    return B

def para2tfm (tau, mode, interp):
    """
    Maps a vector from the parameter space to the transformation set

    Inputs:
    :Tensor tau: parameter vector
    :String mode: transformation set
    :int interp: interpolation method(0: nearest neighbour, 1: bilinear interpolation, 2:quadratic interpolation)

    Output:
    :Transform tfm: mapped transform
    """
    B = make_algebra_matrix(tau, mode)

    tform_matrix = torch.from_numpy(slin.expm(B))

    if mode == 'projective':
        tfm = Projective(tform_matrix = tform_matrix, interp_order=interp)
    else:
        tfm = Affine(tform_matrix = tform_matrix ,interp_order=interp)

    return tfm

def check_if_in_range(tau, mode):
    """
    Checks if a vector is valid, i.e. if its lie algebra counterpart is in the range of matrix logarithm

    Input:
    :Tensor tau: parameter vector to be checked
    :string mode: transformation set

    Output:
    :bool check: True if the vector tau is valid
    """
    B = make_algebra_matrix(tau, mode)

    B_logexp = slin.logm(slin.expm(B))

    return np.allclose(B,B_logexp)

def init_param(mode):
    """
    Generates the identity vector for the given transformation set

    Input:
    :string mode: transformation set

    Output:
    :Tensor tau: identity vector for mode
    """
    if mode == 'rotation':
        tau = torch.zeros(1)
    elif mode == 'translation' or mode == 'rotation+scaling':
        tau = torch.zeros(2)
    elif mode == 'rotation+translation' or mode == 'scaling+translation':
        tau = torch.zeros(3)
    elif mode == 'similarity':
        tau = torch.zeros(4)
    elif mode == 'affine':
        tau = torch.zeros(6)
    elif mode == 'projective':
        tau = torch.zeros(8)
    else:
        raise NameError('Wrong mode name entered')


    return tau

def jacobian (I_org, I, tfm, mode,interp_order):
    """
    Numerically calculates the Jacobian matrix of the image appearance manifold at image I.

    Inputs:
    :Tensor I_org: original image
    :Tensor I: current image - the original image(I_org) transformed by tfm
    :Transform tfm: transformation that generated I
    :string mode: transformation set
    :int interp_order: interpolation method(0: nearest neighbour, 1: bilinear interpolation, 2:quadratic interpolation)

    Outputs:
    :Tensor J: Jacobian matrix
    """

    #

    eps = 1e-4
    tau = init_param(mode)
    d = tau.size()[0]

    if I_org.is_cuda:
        J = torch.cuda.FloatTensor(torch.Size([d])+I.size())
    else:
        J = torch.Tensor(torch.Size([d])+I.size())

    for k in range(d):
        tau_ = tau.clone()
        tau_[k]+=eps
        tfm_J = tfm.compose(para2tfm(tau_,mode,interp=interp_order))
        I_t = tfm_J(I_org)
        J[k,:,:,:] = (I_t-I)/eps

    return J

def jacobian_from_gradient(I,mode):
    """
    Analytically calculates the Jacobian matrix using the image gradient

    Inputs:
    :Tensor I: original image
    :Transform mode: transformation set

    Outputs:
    :Tensor J: Jacobian matrix
    """

    c, h, w = I.size()
    x = np.arange(w) - float(w) / 2 + 0.5
    y = - np.arange(h) + float(h) / 2 - 0.5

    X_,Y_,_= np.meshgrid(x,-y,range(c),indexing='ij')

    # Compute the image gradient horizontally and vertically
    if I.is_cuda:
        from .gpu_gradient import gradient as gpu_grad

        X = torch.from_numpy(X_.T.astype('float32',copy=False)).cuda()
        Y = torch.from_numpy(Y_.T.astype('float32',copy=False)).cuda()

        grad_x, grad_y = gpu_grad(I, grad_type=2)
    else:
        X = torch.from_numpy(X_.T.astype('float32',copy=False))
        Y = torch.from_numpy(Y_.T.astype('float32',copy=False))

        grad_y, grad_x = np.gradient(I.numpy(),axis=(1,2))
        grad_y = torch.from_numpy(grad_y.astype('float32',copy=False))
        grad_x = torch.from_numpy(grad_x.astype('float32',copy=False))


    # Compute the Jacobian using the image gradient
    if mode == 'rotation':
        grad_rot = -grad_x*Y + grad_y*X
        J = grad_rot.unsqueeze_(0)
    elif mode == 'translation':
        J = torch.stack([grad_x,grad_y])
    elif mode == 'rotation+scaling':
        grad_rot = -grad_x*Y + grad_y*X
        grad_sc  = grad_x*X + grad_y*Y
        J = torch.stack([grad_rot,grad_sc])
    elif mode == 'rotation+translation':
        grad_rot = grad_x*Y - grad_y*X
        J = torch.stack([grad_rot,grad_x,grad_y])
    elif mode == 'scaling+translation':
        grad_sc  = grad_x*X + grad_y*Y
        J = torch.stack([grad_sc,grad_x,grad_y])
    elif mode == 'similarity':
        grad_rot = -grad_x*Y + grad_y*X
        grad_sc = grad_x*X + grad_y*Y
        J = torch.stack([grad_rot,grad_x,grad_y,grad_sc])
    elif mode == 'affine':
        grad_rot = -grad_x*Y + grad_y*X
        grad_sc  = grad_x*X + grad_y*Y
        grad_sh1 = grad_x*X - grad_y*Y
        grad_sh2 = grad_x*Y + grad_y*X
        J = torch.stack([grad_rot,grad_x,grad_y,grad_sc,grad_sh1,grad_sh2])
    elif mode == 'projective':
        grad_1 = 2*grad_x*X + grad_y*Y
        grad_2 = grad_x*Y
        grad_3 = grad_x
        grad_4 = grad_y*X
        grad_5 = 2*grad_y*Y + grad_x*X
        grad_6 = grad_y
        grad_7 = -grad_x*X**2 - grad_y*X*Y
        grad_8 = -grad_x*X*Y - grad_y*Y**2
        J = torch.stack([grad_1,grad_2,grad_3,grad_4,grad_5,grad_6,grad_7,grad_8])
    else:
        raise NameError('Wrong mode name entered')

    return J

def center_crop_tensor(I,size):
    """
    Crop the center of the image for given size

    Inputs:
    :Tensor I: the image to be cropped
    :(int,int) size: the size of cropping. If size is a single integer, than (size,size) is used as the crop size.

    Outputs:
    :Tensor II: the cropped image.
    """
    if isinstance(size,numbers.Number):
        size = (size,size)

    if I.dim() == 3: # Crop for a single image
        _, h, w = I.size()

        x1 = int(round((w-size[1])/2.))
        y1 = int(round((h-size[0])/2.))


        if x1 > 0 and y1 > 0:
            II = I[:,y1:y1+size[0],x1:x1+size[1]]
        elif x1 > 0:
            II = I[:,:,x1:x1+size[1]]
        elif y1 > 0:
            II = I[:,y1:y1+size[0],:]
        else:
            II = I
    elif I.dim() == 4: # Crop for multiple images in 1 Tensor
        _, _, h, w = I.size()

        x1 = int(round((w-size[1])/2.))
        y1 = int(round((h-size[0])/2.))


        if x1 > 0 and y1 > 0:
            II = I[:,:,y1:y1+size[0],x1:x1+size[1]]
        elif x1 > 0:
            II = I[:,:,:,x1:x1+size[1]]
        elif y1 > 0:
            II = I[:,:,y1:y1+size[0],:]
        else:
            II = I

    return II.clone()


