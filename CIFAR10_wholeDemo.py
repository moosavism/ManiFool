import random
import time
from datetime import datetime

import numpy as np
import torchvision
import torchvision.transforms as transforms
from functions.algorithms.manifool import manifool
from models import *

#Parameter
num_trials = 5
maxIt = 100
mode = 'affine'
step_sizes = np.logspace(-2,np.log10(1),20)
rng_seed = 13
gamma = 0
torch.set_num_threads(8)

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = mean,
                                                     std = std)
                                ])

#Getting the network and the dataset
pretrained = torch.load('./models/cifar_net.t7')
net = pretrained['net']
net.train(mode=False)

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
            transform = transform)
random.seed(rng_seed)
im_ind = random.sample(range(len(dataset)),num_trials)

num_of_fail = 0
total_mf = 0
successful_dur_manifool = 0
t = time.time()
geo_scores = []
for i,k in enumerate(im_ind):

    im = dataset.__getitem__(k)[0]

    t0 = time.time()
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

    t1 = time.time()

    if (k_org != tar):
        geo_scores.append(geo_sc)
        total_mf += geo_sc
        successful_dur_manifool += t1 - t0

    num_of_fail += (k_org == tar)

    print('Iteration {}, Image {}.'
    ' Current Avg. score: {}'.format(i,k,total_mf/(i-num_of_fail+1)))


elapsed = time.time()-t

fail_percent = num_of_fail/num_trials*100
avg_mf = total_mf/(num_trials - num_of_fail)
avg_succ_dur = successful_dur_manifool/(num_trials-num_of_fail)

file_name = 'CIFAR10_geo_scores_var_'+mode
np.save(file_name, geo_scores)

info_str = ("\n"+"-"*13+"Manifool CIFAR10 Demo"+"-"*13+"\nTransformation: {}\n"
            "Number of images: {}\nGamma: {}\nMaximum number of iterations: {}".format(
            mode, num_trials, gamma, maxIt))

log_str = ("\n"+"-"*20+"Results"+"-"*21+"\nNet not fooled for {}%  of images\nTime elapsed for whole "
          "loop: {:.2f} seconds\nAverage duration for an image: {:.4f} seconds\n"
          "Average duration for successul computations: {:.4f} seconds\n"
          "Average Score(Calculated Directly): {}\n\n".format(
          fail_percent, elapsed, elapsed/num_trials, avg_succ_dur, avg_mf))

print(info_str)
print(log_str)

text_file = open('CIFAR10_wholeDemo_var_log.txt','a')
text_file.write(str(datetime.now()))
text_file.write(info_str)
text_file.write(log_str)
