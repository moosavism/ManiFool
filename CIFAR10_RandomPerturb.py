import functools
import operator
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from functions.helpers.general import eval_network, init_param, para2tfm
from functions.helpers.geodesic_distance import rand_trans_normalize
from torch.autograd import Variable


def gen_rand_unit_vec (dim, num_of_trials, num_of_images):
    vecs = torch.randn((dim, num_of_trials, num_of_images))
    mags = vecs.norm(2,0)

    return vecs/mags

def check_misclass(k_org, transformed_images, net):

    x = Variable(transformed_images,requires_grad=True)
    misclas = np.zeros(len(net))
    for j,nn in enumerate(net):
        output = nn(x)
        _, output_index = torch.max(output.data,1)
        is_misclas = output_index!=k_org[j]
        misclas[j] = torch.sum(is_misclas)

    return misclas

#Parameters
num_of_images = 5000
num_of_trials = 60
required_successful = 20
modes = ['affine']
network = ['org','epoch0','epoch1','epoch2','epoch3','epoch4']
norm_max = 5
step_size = 0.5
rng_seed = 13
cuda_on = True
begin = 0
end = 1000
torch.set_num_threads(4)

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = mean,
                                                     std = std)
                                ])

#Getting the network and the dataset
net = []
if 'org' in network:
    pretrained = torch.load('./models/cifar_net.t7')
    net.append(pretrained['net'])
    net[-1].train(mode=False)
if 'epoch0' in network:
    pretrained = torch.load('./checkpoint/fine_tune_epoch0.t7')
    net.append(pretrained['net'])
    net[-1].train(mode=False)
if 'epoch1' in network:
    pretrained = torch.load('./checkpoint/fine_tune_epoch1.t7')
    net.append(pretrained['net'])
    net[-1].train(mode=False)
if 'epoch2' in network:
    pretrained = torch.load('./checkpoint/fine_tune_epoch2.t7')
    net.append(pretrained['net'])
    net[-1].train(mode=False)
if 'epoch3' in network:
    pretrained = torch.load('./checkpoint/fine_tune_epoch3.t7')
    net.append(pretrained['net'])
    net[-1].train(mode=False)
if 'epoch4' in network:
    pretrained = torch.load('./checkpoint/fine_tune_epoch4.t7')
    net.append(pretrained['net'])
    net[-1].train(mode=False)


if cuda_on:
    for nn in net:
        nn.cuda()

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
            transform = transform)
random.seed(rng_seed)
im_ind = random.sample(range(len(dataset)),num_of_images)
im_ind = im_ind[begin:end]

n = functools.reduce(operator.mul, dataset.__getitem__(0)[0].size(), 1)

r_list = np.arange(start=0.0,stop=norm_max+step_size,step=step_size)
mc_vec = np.empty((r_list.size,len(net)))

elapsed_time = np.empty(r_list.size)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig_list = []

t0 = time.time()

if cuda_on:
    transformed_images = torch.cuda.FloatTensor(torch.Size((required_successful,3,32,32)))
else:
    transformed_images = torch.cuda.FloatTensor(torch.Size((required_successful,3,32,32)))

k_org = torch.zeros(len(network)).int()

if cuda_on:
    k_org.cuda()

for mode in modes:
    tau0 = init_param(mode)
    tfm0 = para2tfm(tau0,mode,1)

    for ii,r in enumerate(r_list):
        transform_variables = gen_rand_unit_vec(tau0.numel(),num_of_trials,num_of_images)

        if cuda_on:
            transform_variables = transform_variables.cuda()

        misclassified = np.zeros(len(net))
        print('Computing the rate for {:.2f}, loop {} of {}'.format(r,ii+1,r_list.size))
        t = time.time()
        num_of_fail = 0
        for i,k in enumerate(im_ind):

            if i%100 == 0:
                print('Iteration {}, time elapsed since beginning {}'.format(i,time.time()-t))
            im = dataset.__getitem__(k)[0]

            if cuda_on:
                im = im.cuda()

            for i,nn in enumerate(net):
                k_org[i] = eval_network(im,nn)

            successful = 0
            for trial in range(num_of_trials):

                u_hat = transform_variables[:,trial,i]

                u, success = rand_trans_normalize(r,u_hat,im,mode, step = 0.05, tol = 0.01)

                # print(u)
                if success:
                    tfm = para2tfm(u,mode,1)
                    im_tfm = tfm(im)
                    transformed_images[successful] = im_tfm

                    successful += 1

                if successful == required_successful:
                    break

            num_of_fail += required_successful-successful
            if successful != 0:
                misclassified += check_misclass(k_org, transformed_images[0:successful], net)

        elapsed = time.time()-t
        print('Time elapsed: {:.2f} seconds'.format(elapsed))
        print('{} vectors were not normalized'.format(num_of_fail))

        mc_vec[ii] = misclassified/((end-begin)*required_successful-num_of_fail)

        elapsed_time[ii] = elapsed
        for j,nn in enumerate(network):
            file_name = 'CIFAR10_mc_vs_r_'+mode + '_' + nn + '_' + str(begin)+str(end)
            np.savez(file_name, r_list, mc_vec[:,j])

    ax1.plot(r_list,mc_vec*100,'-*',label=mode)

print('Total time elapsed: {:.2f} seconds'.format(time.time()-t0))
