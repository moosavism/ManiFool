import random
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transform

from functions.algorithms.manifool import manifool
from models.manitest_cnn import ManitestMNIST_net as Net
from torch.autograd import Variable
from torch.nn.functional import softmax

# Parameter
num_trials = 500
maxIt = 100
mode = 'similarity'
step_sizes = np.logspace(-2,np.log10(0.5),10)
gamma = 0.1
rng_seed = 13
torch.set_num_threads(8)


# Getting the network and the dataset
net = Net()
dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
            transform = transform.ToTensor())
random.seed(rng_seed)
im_ind = random.sample(range(len(dataset)),num_trials)


# Main Loop
num_of_fail = 0
t = time.time()
total_mf = 0
total_norm = 0
conf_tot = 0
geo_scores = []
for i,k in enumerate(im_ind):

    print('Iteration {}, working on image {}'.format(i,k))

    im = dataset.__getitem__(k)[0]*255

    out = manifool(im, net, mode,
                   maxIter=maxIt,
                   cuda_on=True,
                   step_sizes = step_sizes,
                   gamma = gamma,
                   numerical=True,
                   verbose = False)

    geo_sc = out['geo_dist']
    tar = out['target']
    k_org = out['org_label']
    im_out = out['output_image']

    if (k_org != tar):
        geo_scores.append(geo_sc)
        total_mf += geo_sc
        conf = softmax(net(Variable(im_out.unsqueeze(0))))
        conf_max = torch.max(conf.data)
        conf_tot += conf_max
        print('Confidence: {}, {}'.format(conf_max,conf_tot/(i+1)))

    num_of_fail += (k_org == tar)

elapsed = time.time()-t
fail_percent = num_of_fail/num_trials*100
avg_mf = total_mf/(num_trials - num_of_fail)
avg_conf = conf_tot/(num_trials - num_of_fail)

# Saving the output geodesic scores
file_name = 'MNIST_geo_scores_var_'+mode
np.save(file_name, geo_scores)

# Printing the results
print("ManiFool demo")
print('Mode: ' + mode)
print('Gamma: {}'.format(gamma))
print('*'*20)
print("Net not fooled for {}% of images".format(fail_percent))
print("Time elapsed for the whole loop: {:.4f} seconds".format(elapsed))
print("Average duration for one image: {:.4f} seconds".format(elapsed/num_trials))
print("Average Score(Calculated Directly): {}".format(avg_mf))
print("Average Confidence: {}".format(avg_conf))
