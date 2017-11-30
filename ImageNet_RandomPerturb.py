import functools
import operator
import random
import time
from math import inf

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from functions.helpers.general import eval_network, init_param, para2tfm, jacobian, center_crop_tensor
from functions.helpers.geodesic_distance import rand_trans_normalize, simple_geo_dist
from torch.autograd import Variable


def gen_rand_unit_vec (dim, num_of_trials, num_of_images):
    vecs = torch.randn((dim, num_of_trials, num_of_images))
    mags = vecs.norm(2,0)

    return vecs/mags

def check_misclass(k_org, l_org, transformed_images, net):

    x = Variable(transformed_images,requires_grad=True)
    misclas = np.zeros(len(net))
    for j,nn in enumerate(net):
        output = nn(x)
        _, output_index = torch.max(output.data,1)
        is_misclas = output_index!=k_org[j]
        misclas[j] = torch.sum(is_misclas)

    return misclas

# Parameters
num_of_images = 5000
num_of_trials = 40
required_successful = 10
modes = ['affine','similarity']
network = ['resnet18','resnet34','resnet50','alexnet']
norm_max = 5
step_size = 0.5
rng_seed = 13
cuda_on = True
torch.set_num_threads(4)

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
transform = transforms.Compose([transforms.Scale(256),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean,
                                                     std = std)
                                ])

# Getting the networks
net = []
if 'resnet18' in network:
    net.append(torchvision.models.resnet18(pretrained = True))
    net[-1].eval()
    net[-1].train(mode=False)
if 'resnet34' in network:
    net.append(torchvision.models.resnet34(pretrained = True))
    net[-1].eval()
    net[-1].train(mode=False)
if 'resnet50' in network:
    net.append(torchvision.models.resnet50(pretrained = True))
    net[-1].eval()
    net[-1].train(mode=False)
if 'alexnet' in network:
    net.append(torchvision.models.alexnet(pretrained=True))
    net[-1].eval()
    net[-1].train(mode=False)
if 'vgg11' in network:
    net.append(torchvision.models.vgg11(pretrained=True))
    net[-1].eval()
    net[-1].train(mode=False)
if 'vgg13' in network:
    net.append(torchvision.models.vgg13(pretrained=True))
    net[-1].eval()
    net[-1].train(mode=False)
if 'vgg16' in network:
    net.append(torchvision.models.vgg16(pretrained=True))
    net[-1].eval()
    net[-1].train(mode=False)


if cuda_on:
    for nn in net:
        nn.cuda()

# Getting the dataset
dset_path = './ILSVRC_val'
dataset = torchvision.datasets.ImageFolder(dset_path,transform)
random.seed(rng_seed)
im_ind = random.sample(range(len(dataset)),num_of_images)

# Initialization
n = functools.reduce(operator.mul, dataset.__getitem__(0)[0].size(), 1)

r_list = np.arange(start=0.0,stop=norm_max+step_size,step=step_size)
mc_vec = np.zeros((r_list.size,len(net)))

elapsed_time = np.empty(r_list.size)

t0 = time.time()

if cuda_on:
    transformed_images = torch.cuda.FloatTensor(torch.Size((required_successful,3,224,224)))
else:
    transformed_images = torch.cuda.FloatTensor(torch.Size((required_successful,3,224,224)))

k_org = torch.zeros(len(network)).int()

if cuda_on:
    k_org.cuda()

# Main Loop
for mode in modes: # loop for different transformations
    tau0 = init_param(mode)
    tfm0 = para2tfm(tau0,mode,1)

    for ii,r in enumerate(r_list): # loop for geodesic scores
        # Randomly generate transformations from unit sphere for the whole possible set
        transform_variables = gen_rand_unit_vec(tau0.numel(),num_of_trials,num_of_images)

        if cuda_on:
            transform_variables = transform_variables.cuda()

        misclassified = np.zeros(len(net))

        print('Computing the rate for {:.2f}, loop {} of {}'.format(r,ii+1,r_list.size))
        t = time.time()
        num_of_fail = 0
        for i,k in enumerate(im_ind): # loop for different images

            if i%100 == 0:
                print('Iteration {}, time elapsed since beginning {}'.format(i,time.time()-t))
            im, l_org = dataset.__getitem__(k)

            if cuda_on:
                im = im.cuda()

            im_c = center_crop_tensor(im,224)

            # Evaluate the image on all networks
            for i,nn in enumerate(net):
                k_org[i] = eval_network(im_c,nn)

            successful = 0
            for trial in range(num_of_trials): # loop for random transformations
                u_hat = transform_variables[:,trial,i]

                u, success = rand_trans_normalize(r,u_hat,im,mode, step = 0.05, tol = 0.01)

                if success: # if a transformation is found successfully with requested score, save it
                    tfm = para2tfm(u,mode,1)
                    im_tfm = tfm(im)
                    im_tfm_c = center_crop_tensor(im_tfm,224)
                    transformed_images[successful] = im_tfm_c

                    successful += 1

                if successful == required_successful:# if enough transformations are found, stop the loop
                    break

            num_of_fail += required_successful-successful
            if successful != 0:
                # Check if transformed images are misclassified
                mc = check_misclass(k_org, l_org, transformed_images[0:successful], net)
                misclassified += mc

        elapsed = time.time()-t
        print('Time elapsed: {:.2f} seconds'.format(elapsed))
        print('{} vectors were not normalized'.format(num_of_fail))

        mc_vec[ii] = misclassified/(num_of_images*required_successful-num_of_fail)

        elapsed_time[ii] = elapsed

        # Save the current calculated misclassification rate for each network
        for j,nn in enumerate(network):
            file_name = 'ImageNet_mc_vs_r_'+mode + '_' + nn
            np.savez(file_name, r_list=r_list, mc_vec=mc_vec[:,j])

print('Total time elapsed: {:.2f} seconds'.format(time.time()-t0))
