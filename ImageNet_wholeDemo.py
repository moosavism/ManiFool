import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from functions.algorithms.manifool import manifool
from functions.helpers.general import center_crop_tensor
from functions.helpers.plot_helpers import Denormalize

#Parameter
num_trials = 100
maxIt = 100
mode = 'similarity'
network = 'resnet50'
step_sizes = np.logspace(-2,np.log10(1),20)
rng_seed = 2323
gamma = 0.2
torch.set_num_threads(8)

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
transform = transforms.Compose([transforms.Scale(256),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean,
                                                     std = std)
                                ])

dn = Denormalize(mean,std)

# get classes
file_name = 'synset_words.txt'
if not os.access(file_name, os.W_OK):
    synset_URL = 'https://github.com/szagoruyko/functional-zoo/raw/master/synset_words.txt'
    os.system('wget ' + synset_URL)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ', 1)[1].split(', ', 1)[0])
classes = tuple(classes)

#Loading the network
if network == 'resnet18':
    net = torchvision.models.resnet18(pretrained = True)
elif network == 'resnet34':
    net = torchvision.models.resnet34(pretrained = True)
elif network == 'resnet50':
    net = torchvision.models.resnet50(pretrained = True)
elif network == 'alexnet':
    net = torchvision.models.alexnet(pretrained=True)
elif network == 'vgg11':
    net = torchvision.models.vgg11(pretrained=True)
elif network == 'vgg13':
    net = torchvision.models.vgg13(pretrained=True)
elif network == 'vgg16':
    net = torchvision.models.vgg16(pretrained=True)

net.eval()
net.train(mode=False)

#Loading the dataset
dset_path = './ILSVRC_val'#Should be changed with the path to the dataset
dataset = torchvision.datasets.ImageFolder(dset_path,transform)
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
                   crop = 224,
                   batch_size=2,
                   verbose = False,
                   geo_step = 0.01)

    geo_sc = out['geo_dist']
    tar = out['target']
    k_org = out['org_label']
    im_out = out['output_image']
    new_label = out['fooling_label']

    t1 = time.time()

    if (k_org != tar):
        geo_scores.append(geo_sc)
        total_mf += geo_sc
        successful_dur_manifool += t1 - t0

        im_c = center_crop_tensor(im,224)
        fn = './images/ImageNet/projective/original/image_' + str(k) + '_' + classes[k_org] + '.png'
        torchvision.utils.save_image(dn(im_c),fn)
        fn = './images/ImageNet/projective/transformed/image_'+ mode + '_' + str(k) + '_' + classes[new_label] + '.png'
        torchvision.utils.save_image(dn(im_out),fn)

    num_of_fail += (k_org == tar)

    print('Iteration {}, Image {}.'
    ' Current Avg. score: {}'.format(i,k,total_mf/(i-num_of_fail+1)))
    print('Duration: {}'.format(t1-t0))


elapsed = time.time()-t

fail_percent = num_of_fail/num_trials*100
avg_mf = total_mf/(num_trials - num_of_fail)
avg_succ_dur = successful_dur_manifool/(num_trials-num_of_fail)

file_name = 'ImageNet_geo_scores_var_'+mode + '_' + network
np.save(file_name, geo_scores)

info_str = ("\n"+"-"*13+"Manifool ImageNet Demo"+"-"*13+"\nTransformation: {}\n"
            "Number of images: {}\nGamma: {}\nMaximum number of iterations: {}".format(
            mode, num_trials, gamma, maxIt))

log_str = ("\n"+"-"*20+"Results"+"-"*21+"\nNet not fooled for {}%  of images\nTime elapsed for whole "
          "loop: {:.2f} seconds\nAverage duration for an image: {:.4f} seconds\n"
          "Average duration for successul computations: {:.4f} seconds\n"
          "Average Score(Calculated Directly): {}\n\n".format(
          fail_percent, elapsed, elapsed/num_trials, avg_succ_dur, avg_mf))

print(info_str)
print(log_str)

text_file = open('ImageNet_wholeDemo_var_log.txt','a')
text_file.write(str(datetime.now()))
text_file.write(info_str)
text_file.write(log_str)
