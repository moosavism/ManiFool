import time

import torchvision
from functions.helpers.general import *
from functions.helpers.geodesic_distance import rand_trans_normalize

num_of_trials = 60
mode = 'affine'
cuda_on = True
r = 1.04

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = mean,
                                                     std = std)
                                ])


dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
            transform = transform)

tau0 = init_param(mode)
num_of_fail = 0
i = 0
t = time.time()
out_images = []
for im,label in dataset:
    if i%100 == 0:
        print('Iteration {}, time elapsed since beginning {}'.format(i,time.time()-t))

    if cuda_on:
        im = im.cuda()

    for trial in range(num_of_trials):
        u_hat = torch.randn(tau0.numel())
        u_hat = u_hat/u_hat.norm()

        u, success = rand_trans_normalize(r,u_hat,im,mode, step = 0.05, tol = 0.01)

        if success:
            tfm = para2tfm(u,mode,1)

            im_tfm = tfm(im)
            out_images.append((im_tfm,label))
            break

    if not success:
        num_of_fail +=1

    i +=1


print('Number of Fail: {}'.format(num_of_fail))


fname = './data/cifar_trans/cifar10_random_transformed.dts'
torch.save(out_images, fname)
