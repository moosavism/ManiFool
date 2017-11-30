from math import floor

import numpy as np
import scipy.linalg as sli
import torch

from functions.helpers.general import check_if_in_range
from .general import para2tfm


def geodesic_distance (image, step_size, tfm_mtx, mode, interp=1):
    """
    Computes the geodesic distance using a direct path. See [1] for details.

    Inputs:
    :Tensor image: image for which the image appearance manifold is generated
    :double step_size: step size for discretizing the path
    :Tensor tfm_mtx: 3x3 transformation matrix for projective transforms & subsets
    :string mode: transformation set
    :int interp: interpolation method(0: nearest neighbour, 1: bilinear interpolation, 2:quadratic interpolation)

    Outputs:
    :double dist: geodesic distance of transform from the identity.
    :double tau_norm: norm of the parameter vector of the transformation

    [1] C. Kanbak, S.-M. Moosavi-Dezfooli, P. Frossard,"Geometric robustness of deep networks:analysis and improvement",
    arxiv:1711.09115[cs],Nov. 2017. url: https://arxiv.org/abs/1711.09115
    """

    lie_basis = sli.logm(tfm_mtx.numpy())
    tau_org = tform_2_tau(np.real(lie_basis),mode)
    tau_norm = tau_org.norm()
    if tau_norm == 0.0:
        return 0.0, 0.0
    num_of_steps = floor(tau_norm/step_size)
    normalized_tau = tau_org/tau_norm
    tau = step_size*normalized_tau

    dist = 0
    x = image
    for i in range(1,num_of_steps+1):
        tfm_step = para2tfm(i*tau, mode, interp)
        x_next = tfm_step(image)
        dist += (x_next-x).norm()
        x = x_next

    final_tfm = para2tfm(tau_org, mode, interp)
    x_next = final_tfm(image)
    dist += (x_next-x).norm()

    return dist, tau_org.norm()


def tform_2_tau(log_tfm, mode):
    """
    Parametrizes a matrix in the Lie algebra of mode.

    Input:
    :Tensor log_tfm: 3x3 matrix from the Lie algebra
    :string mode: transformation set

    Output:
    :Tensor tau: parameter vector
    """
    if mode == 'rotation':
        tau = torch.FloatTensor([log_tfm[1, 0]])
    elif mode == 'translation':
        tau = torch.FloatTensor([log_tfm[0, 2], log_tfm[1, 2]])
    elif mode == 'rotation+scaling':
        tau = torch.FloatTensor([log_tfm[1, 0], log_tfm[0, 0]])
    elif mode == 'rotation+translation':
        tau = torch.FloatTensor([log_tfm[1, 0], log_tfm[0, 2], log_tfm[1, 2]])
    elif mode == 'scaling+translation':
        tau = torch.FloatTensor([log_tfm[0, 0], log_tfm[0, 2], log_tfm[1, 2]])
    elif mode == 'similarity':
        tau = torch.FloatTensor([log_tfm[1, 0], log_tfm[0, 2], log_tfm[1, 2], log_tfm[0, 0]])
    elif mode == 'affine':
        tau = torch.FloatTensor([(log_tfm[1, 0] - log_tfm[0, 1]) / 2,
                                 log_tfm[0, 2], log_tfm[1, 2],
                                 (log_tfm[0, 0] + log_tfm[1, 1]) / 2,
                                 (log_tfm[0, 0] - log_tfm[1, 1]) / 2,
                                 (log_tfm[1, 0] + log_tfm[0, 1]) / 2])
    elif mode == 'projective':
        tau = torch.from_numpy(log_tfm.astype('float32').flatten()[0:8])
    else:
        raise NameError('Wrong mode name entered')

    return tau


def rand_trans_normalize (r,vec,im,mode,
                          step = 0.01,
                          tol = None,
                          interp = 1,
                          maxIter = 1000):
    """
    Given a parameter vector, scale it to get a transformation with geodesic score r for the image im.

    Input:
    :double r: the requested geodesic score
    :Tensor vec: input parameter vector with norm(vec) = 1
    :Tensor im: image for which the geodesic score is measured with
    :string mode: transformation set
    :double step: step size
    :double tol: tolerance value for the output. The output transform will have a geodesic score r+-tol.
    :int interp: interpolation method(0: nearest neighbour, 1: bilinear interpolation, 2:quadratic interpolation)
    :param maxIter: maximum number of allowed iterations

    Outputs:
    :Tensor vec_p: scaled parameter vector
    :bool found: shows if the algorithm found a suitable parameter vector
    """
    if tol is None:
        tol = step/2

    tau = step*vec

    total_dist = 0.0
    x = im
    diff = r
    vec_p = 0*tau
    found = True
    im_norm = im.norm()

    i = 0
    rho = 1
    delta = 1
    binary_search = False
    # Main Loop
    while diff > tol and i < maxIter:
        # increase vec_p using the step size.
        vec_p = rho*tau
        tfm_step = para2tfm(vec_p, mode, interp)
        x_next = tfm_step(im)
        dist = (x-x_next).norm()
        total_dist += dist

        dist_sc = total_dist/im_norm
        diff = np.abs(r-dist_sc)

        if dist == 0:
            found = False
            break

        if dist_sc < r:
            if binary_search:
                delta = delta/2
            rho += delta

            if delta >= 1:
                x = x_next
            else:
                total_dist -= dist
        else: # if r is passed, start binary search
            binary_search = True
            delta = delta/2
            rho -= delta
            total_dist -= dist


        i += 1

    if i == maxIter or not check_if_in_range(vec_p, mode):
        found = False

    return vec_p, found


def simple_geo_dist (image, step_size, tau_org, mode, interp=1):
    """
    Computes the geodesic score given a parameter vector - (does not have the logarithmic map)

     Inputs:
    :Tensor image: image for which the image appearance manifold is generated
    :double step_size: step size for discretizing the path
    :Tensor tau_org: input parameter vector
    :string mode: transformation set
    :int interp: interpolation method(0: nearest neighbour, 1: bilinear interpolation, 2:quadratic interpolation)

    Outputs:
    :double dist: geodesic distance of transform from the identity.
    """
    tau_norm = tau_org.norm()
    if tau_norm == 0.0:
        return 0.0
    num_of_steps = floor(tau_norm/step_size)
    normalized_tau = tau_org/tau_norm
    tau = step_size*normalized_tau

    dist = 0
    x = image
    for i in range(1,num_of_steps+1):
        tfm_step = para2tfm(i*tau, mode, interp)
        x_next = tfm_step(image)
        dist += (x_next-x).norm()
        x = x_next

    final_tfm = para2tfm(tau_org, mode, interp)
    x_next = final_tfm(image)
    dist += (x_next-x).norm()

    return dist