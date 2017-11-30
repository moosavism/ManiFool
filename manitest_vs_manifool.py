import datetime
import random
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transform

from functions.algorithms.manifool import manifool
from functions.algorithms.manitest import manitest
from functions.helpers.geodesic_distance import geodesic_distance
from models.manitest_cnn import ManitestMNIST_net as Net

# Parameter
num_trials = 1000
maxIt_manifool = 100
mode = 'similarity'
gamma = 0.1
step_sizes = np.logspace(-2,np.log10(0.5),10)
rng_seed = 13
torch.set_num_threads(16)


# Getting the network and the dataset
net = Net()
dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
            transform = transform.ToTensor())


random.seed(rng_seed)
im_ind = random.sample(range(len(dataset)),num_trials)

# Initialization
total_dur_manitest = 0
total_dur_manifool = 0
num_of_fail = 0
total_mf = 0
total_mt = 0
total_mt_score = 0
successful_dur_manifool = 0
total_it_mf = 0
total_it_mt = 0

mf_scores = []
mt_scores = []
images = []
# Main Loop
for i,k in enumerate(im_ind):

    im = dataset.__getitem__(k)[0]*255

    t0 = time.time()
    # Run ManiFool and unpack the output
    out = manifool(im, net, mode,
                   maxIter=maxIt_manifool,
                   cuda_on=True,
                   step_sizes = step_sizes,
                   gamma = gamma,
                   numerical=True,
                   verbose = False)

    geo_sc = out['geo_dist']
    tar = out['target']
    k_org = out['org_label']

    t1 = time.time()

    # Run Manitest
    manitest_score, _, tfm_mt, _, _, _, _, _, it_mt = manitest(im, net, mode, verbose=False)

    total_dur_manitest += time.time() - t1
    total_dur_manifool += t1 - t0


    if (k_org != tar): # if ManiFool is successful
        total_mf += geo_sc
        dist_mt, _ = geodesic_distance(im, 0.05, tfm_mt.tform_matrix, mode)
        mt_sc = dist_mt/im.norm()
        total_mt += mt_sc
        total_mt_score += manitest_score

        mf_scores.append(geo_sc)
        mt_scores.append(mt_sc)
        images.append(k)
        successful_dur_manifool += t1 - t0

        total_it_mt += it_mt
    else: # if ManiFool failed to find an example
        dist_mt = geodesic_distance(im, 0.05, tfm_mt.tform_matrix, mode)
        mt_sc = dist_mt/im.norm()
        geo_sc = np.inf
        num_of_fail += 1


    print('Iteration {}, working on image {}: Diff = {}'.format(i,k, total_dur_manitest-total_dur_manifool))

# Process the results
fail_percent = num_of_fail/num_trials*100
avg_mf = total_mf/(num_trials - num_of_fail)
avg_mt = total_mt/(num_trials - num_of_fail)
avg_mt_score = total_mt_score/(num_trials - num_of_fail)

# Display the results
print("-"*10,"Manifool","-"*10)
print("Net not fooled for {}% of images".format(fail_percent))
print("Average duration for one image: {:.4f} seconds".format(total_dur_manifool/num_trials))
print("Average duration for successul computations: {:.4f} seconds".format(successful_dur_manifool/(num_trials - num_of_fail)))
print("Average Score(Calculated Directly): {}".format(avg_mf))
print("-"*10,"Manitest","-"*10)
print("Average duration for one image: {:.4f} seconds".format(total_dur_manitest/num_trials))
print("Average Score(Calculated Directly): {}".format(avg_mt))
print("Average Score(Manitest): {}".format(avg_mt_score))

# Save the geodesic scores of both ManiTest and ManiFool
now = datetime.datetime.now()
file_name = 'MTvsMF_{}-{}-{}-{}{}'.format(now.year,now.month,now.day,now.hour,now.second)
np.savez(file_name,**{'dists_mt':np.asarray(mt_scores),'dists_mf':np.asarray(mf_scores),'im_ind':np.asarray(images)})
