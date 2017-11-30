import os
import time

import numpy as np
import torchvision
import torchvision.transforms as transforms
from functions.algorithms.manifool import manifool
from functions.helpers.general import center_crop_tensor
from functions.helpers.plot_helpers import compare_images, Denormalize

# Load the network
net = torchvision.models.resnet18(pretrained = True)
net.eval()
net.train(mode=False)


#get images
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
transform = transforms.Compose([transforms.Scale(256),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean,
                                                     std = std)
                                ])

dset_path = './ILSVRC_val'
dset = torchvision.datasets.ImageFolder(dset_path,transform)
i = np.random.randint(0,len(dset))

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

k = 47556
im = dset.__getitem__(k)[0]


maxIt = 100

mode = 'projective'
t = time.time()
step_sizes = np.logspace(-2,np.log10(1),20)*1e-2
out = manifool(im, net, mode,
               maxIter=maxIt,
               step_sizes = step_sizes,
               gamma = 0,
               geo_step = 0.005,
               batch_size=3,
               cuda_on=True,
               crop=224,
               numerical = True)

elapsed = time.time()-t

k_org = out['org_label']
k_f = out['fooling_label']
tar = out['target']
geo_dist = out['geo_dist']
II = out['output_image']

print('\n\n\nFinal Result -- Multi Target:')
if k_org != tar:
    print("Original Output: {}({})\nTarget {}({})\nNew Output: {}({})".format(
           k_org, classes[k_org], tar, classes[tar], k_f, classes[k_f]))
    print("Geodesic Score: {}".format(geo_dist))
    print('Time elapsed: {} seconds'.format(time.time() - t))
else:
    print("Max iteration reached before fooling")

im_c = center_crop_tensor(im,224)
compare_images(im_c, II.cpu(), classes[k_org], classes[k_f], mean, std)

input('Press Enter to end:')
