import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import compare_ssim

def prepare_image(I, mean, std):

    I = I.clone()

    if I.max() > 10:
        I = I/255
    else:
        tf = Denormalize(mean=mean, std=std)
        I = tf(I)

    return I

def compare_images (I, I_t, label1, label2, mean=[0], std=[1], block=False):


    I = prepare_image(I,mean,std)
    I_t = prepare_image(I_t,mean,std)

    plt.figure()
    if I.size()[0] == 1:
        ax1 = plt.subplot(121)
        plt.imshow(I.squeeze_().numpy(),cmap='gray')
        ax2 = plt.subplot(122)
        plt.imshow(I_t.squeeze_().numpy(),cmap='gray')
    else:
        ax1 = plt.subplot(121)
        plt.imshow(np.transpose(I.numpy(),(1,2,0)))
        ax2 = plt.subplot(122)
        plt.imshow(np.transpose(I_t.numpy(),(1,2,0)))

    ax1.set_xlabel(label1)
    ax2.set_xlabel(label2)

    plt.show(block=block)
    # pad =round(I.size()[1]/10.)
    #
    # imglist = make_grid([I, I_t],padding = pad)
    # plt.imshow(np.transpose(imglist.numpy(),(1,2,0)))
    # plt.show()

def compare_ssim_images (I_org, I_p, I_t, mean, std):

    I_p = prepare_image(I_p, mean, std).permute(1,2,0).numpy()
    I_org = prepare_image(I_org, mean, std).permute(1,2,0).numpy()
    I_t = prepare_image(I_t, mean, std).permute(1,2,0).numpy()

    ssim_p = compare_ssim(I_org, I_p, multichannel=True, gaussian_weigths=True, sigma=1.5, use_sample_covariance=False)
    ssim_t = compare_ssim(I_org, I_t, multichannel=True, gaussian_weigths=True, sigma=1.5, use_sample_covariance=False)

    # I_p[I_p>1.0] = 1.0
    print(I_p.shape)

    plt.figure()
    ax1 = plt.subplot(121)
    plt.imshow(I_p)
    ax2 = plt.subplot(122)
    plt.imshow(I_t)

    ax1.set_title('Perturbed Image')
    ax1.set_xlabel('SSIM: {}'.format(ssim_p))
    ax2.set_title('Transformed Image')
    ax2.set_xlabel('SSIM: {}'.format(ssim_t))

    plt.show(block=False)

    return ssim_p,ssim_t

class Denormalize(object):
    """ Given mean: (R, G, B) and std: (R, G, B),
    will denormalize each channel of the torch.*Tensor, i.e.
    channel = (channel*std + mean)
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        tens = tensor.clone()
        for t, m, s in zip(tens, self.mean, self.std):
            t.mul_(s).add_(m)
        return tens
