import numpy as np
import torch
import torch.cuda
from pycuda.compiler import SourceModule

from .pytorchholder import Holder


_kernel_ = """
#include <stdio.h>
__global__ void forward_diff_image_gradient(
float* img,
float* gradient_x,
float* gradient_y,
int nPts,
int nx,
int ny,
int nChannels
)
{
/*  Calculates the gradient of an image in a forward difference sense (f(x+1)-f(x))
*  
*   Inputs:
*   float* img: The image in question
*   float* gradient_x: A container array to output the gradient in x direction
*   float* gradient_y: A container array to output the gradient in y direction
*   int nPts: number of pixels in the image
*   int nx: width of the image
*   int ny: height of the image
*   int nChannels: number of channels
*/

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx>=nPts)
        return;


    int y = idx/nx;
    int x = idx - nx*y;

    int idx01, idx10;
    float f00,f01,f10;
    float grad_x, grad_y;

    idx01 = idx + 1;
    idx10 = idx + nx;

    // loop on different color channels
    for (int i=0;i < nChannels; i++){
       f00 = img[idx + i*nPts];//f[x,y]

       if (x >= nx-1)
           f01 = 0.0;
       else
           f01 = img[idx01 + i*nPts];//f[x+1,y]

       if (y >= ny-1)
           f10 = 0.0;
       else
           f10 = img[idx10 + i*nPts];//f[x,y+1]

       grad_x = f01-f00;
       grad_y = f10-f00;

       gradient_x[idx + i*nPts] = grad_x;
       gradient_y[idx + i*nPts] = grad_y;
    }

    return;
}

__global__ void central_diff_image_gradient(
float* img,
float* gradient_x,
float* gradient_y,
int nPts,
int nx,
int ny,
int nChannels
)
{
/*  Calculates the gradient of an image using central difference (f(x+1)-f(x-1))
*  
*   Inputs:
*   float* img: The image in question
*   float* gradient_x: A container array to output the gradient in x direction
*   float* gradient_y: A container array to output the gradient in y direction
*   int nPts: number of pixels in the image
*   int nx: width of the image
*   int ny: height of the image
*   int nChannels: number of channels
*/

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx>=nPts)
        return;


    int y = idx/nx;
    int x = idx - nx*y;

    int idx01, idx10, idx0_1, idx_10;
    float f01,f10, f0_1, f_10;
    float grad_x, grad_y;

    idx01 = idx + 1;
    idx10 = idx + nx;
    idx0_1 = idx - 1;
    idx_10 = idx - nx;

    // loop on different color channels 
    for (int i=0;i < nChannels; i++){

       if (x == nx-1)
           f01 = 0.0;
       else
           f01 = img[idx01 + i*nPts];//f[x+1,y]

       if (y == ny-1)
           f10 = 0.0;
       else
           f10 = img[idx10 + i*nPts];//f[x,y+1]

       if (x == 0)
           f0_1 = 0.0;
       else
           f0_1 = img[idx0_1 + i*nPts];//f[x-1,y]

       if (y == 0)
           f_10 = 0.0;
       else
           f_10 = img[idx_10 + i*nPts];//f[x,y-1]


       grad_x = (f01-f0_1)/2;
       grad_y = (f10-f_10)/2;

       gradient_x[idx + i*nPts] = grad_x;
       gradient_y[idx + i*nPts] = grad_y;
    }

    return;
}

"""

x = torch.cuda.FloatTensor(8)

mod = SourceModule(_kernel_)
diff_gradient = mod.get_function("forward_diff_image_gradient")
cent_gradient = mod.get_function("central_diff_image_gradient")

def gradient(im_gpu, grad_type = 1, threadsPerBlock = 1024):
    """
    Calculates the image gradient using GPU. Requires PyCUDA to work.

    Inputs:
    :CUDA Tensor im_gpu: the image whose gradient is calculated
    :int grad_type: gradient method (1:forward difference, 2:central difference)
    :int threadsPerBlock: number of CUDA threads per block (must be multiple of 32, 1024 is max)

    Outputs:
    :CUDA Tensor grad_x: image gradient in x direction
    :CUDA Tensor grad_y: image gradient in y direction
    """
    if not isinstance(im_gpu,torch.cuda.FloatTensor):
        raise TypeError(type(im_gpu))

    ny,nx = im_gpu.size()[1:]
    nPts = ny*nx
    nChannels=im_gpu.size()[0]
    nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock)))

    grad_x = torch.cuda.FloatTensor(im_gpu.size())
    grad_y = torch.cuda.FloatTensor(im_gpu.size())


    if grad_type == 1:
        grad_func = diff_gradient
    elif grad_type == 2:
        grad_func = cent_gradient
    else:
        raise ValueError(grad_type)

    grad_func(Holder(im_gpu),
              Holder(grad_x),
              Holder(grad_y),
              np.int32(nPts),
              np.int32(nx),
              np.int32(ny),
              np.int32(nChannels),
              grid = (nBlocks,1,1),
              block = (threadsPerBlock,1,1)
              )

    return grad_x, grad_y
