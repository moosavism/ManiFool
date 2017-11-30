import numpy as np
import torch
import torch.cuda

from ..helpers.pytorchholder import Holder

x = torch.cuda.FloatTensor(8)
from pycuda.compiler import SourceModule

_kernel_ = """
#include <stdio.h>

__device__ void calc_target(
int x,
int y,
float* H,
float* target_coord
)
{
    float divisor;

    divisor = x*H[6] + y*H[7] + H[8];

    target_coord[0] = (x*H[0] + y*H[1] + H[2])/divisor;
    target_coord[1] = (x*H[3] + y*H[4] + H[5])/divisor;
}


__global__ void projective_warp(
float* img,
float* img_warped,
float* H,
int nPts,
int nx,
int ny,
int nChannels
)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx>=nPts)
        return;

    int y = idx/nx;
    int x = idx - nx*y;

    float target_coord [2];
    float x_frac, y_frac;
    float x_t, y_t;

    calc_target(x,y,H, target_coord);

    x_frac = modf(target_coord[0],&x_t);
    y_frac = modf(target_coord[1],&y_t);

    int idx00, idx01, idx10, idx11;
    float f00, f01, f10, f11, new_val;

    if (x_t>=nx || y_t>=ny || x_t < -1 || y_t < -1){
        for(int i=0;i < nChannels; i++)
            img_warped[idx + i*nPts] = 0.0;
        return;
    }

    idx00 = int(x_t + y_t*nx);
    idx01 = idx00 + nx;
    idx10 = idx00 + 1;
    idx11 = idx00 + nx + 1;


    for (int i=0;i < nChannels; i++){

        if (x_t<0 || y_t<0)
           f00 = 0.0;
        else
           f00 = img[idx00 + i*nPts];

        if (x_t<0 || y_t>=ny-1)
           f01 = 0.0;
        else
           f01 = img[idx01 + i*nPts];

        if (x_t>=nx-1 || y_t<0)
           f10 = 0.0;
        else
           f10 = img[idx10 + i*nPts];

        if (x_t>=nx-1 || y_t>=ny-1)
           f11 = 0.0;
        else
           f11 = img[idx11 + i*nPts];

        new_val = f00*(1-x_frac)*(1-y_frac) + f10*x_frac*(1-y_frac)
                  +f01*(1-x_frac)*y_frac + f11*x_frac*y_frac;

        img_warped[idx + i*nPts] = new_val;

    }

    return;
}
"""

# try:
#     drv.Context.get_device()
# except Exception as e:
#     print('E')
#     import pycuda.autoinit


mod = SourceModule(_kernel_)
projective_warp = mod.get_function("projective_warp")

def proj_warp_gpu(H_gpu,
                  img_gpu,
                  img_wrapped_gpu,
                  threadsPerBlock = 1024):

    if not isinstance(img_gpu,torch.cuda.FloatTensor):
        raise TypeError(type(img_gpu))
    if not isinstance(img_wrapped_gpu,torch.cuda.FloatTensor):
        raise TypeError(type(img_wrapped_gpu))
    if not isinstance(H_gpu,torch.cuda.FloatTensor):
        raise TypeError(type(H_gpu))

    nChannels = img_gpu.size()[0]
    ny,nx = img_gpu.size()[1:]
    nPts = ny*nx

    # if nPts < threadsPerBlock:
    #     threadsPerBlock = nPts
    nBlocks = int(np.ceil(nPts/threadsPerBlock))

    projective_warp(Holder(img_gpu),
                    Holder(img_wrapped_gpu),
                    Holder(H_gpu),
                    np.int32(nPts),
                    np.int32(nx),
                    np.int32(ny),
                    np.int32(nChannels),
                    grid=(nBlocks,1,1),
                    block=(int(threadsPerBlock),1,1))
