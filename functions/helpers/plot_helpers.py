import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import compare_ssim

def prepare_image(I, mean, std):
    """
    Prepares an image for imshow, i.e. denormalizes or rescales the image.

    Inputs:
    :Tensor I: the image to be prepared
    :(double,double,double) mean: original mean values of the pixels for denormalization(the values are ordered as (R,G,B))
    :(double,double,double) std: original standard deviations of the pixels for denormalization(the values are ordered as (R,G,B))

    Outputs:
    :Tensor I: the prepared image
    """

    I = I.clone()

    if I.max() > 10:
        I = I/255
    else:
        tf = Denormalize(mean=mean, std=std)
        I = tf(I)

    return I

def compare_images (I, I_t, label1, label2, mean=[0], std=[1], block=False):
    """
    Draws two images side by side on a figure for comparing.

    Inputs:
    :Tensor I: first image, will be printed on left
    :Tensor I_t: second image, will be printed on right
    :String label1: label of the first image
    :String label2: label of the second image
    :(double,double,double) mean: original mean values of the pixels for denormalization(the values are ordered as (R,G,B))
    :(double,double,double) std: original standard deviations of the pixels for denormalization(the values are ordered as (R,G,B))
    :bool block: if True, the printed figure pauses the program.

    """

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

class Denormalize(object):
    """
    Given mean: (R, G, B) and std: (R, G, B), will denormalize each channel of the torch.*Tensor, i.e.
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
