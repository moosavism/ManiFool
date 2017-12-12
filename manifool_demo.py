import os
import time
from PIL import Image

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


#load the image
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
transform = transforms.Compose([transforms.Scale(256),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean,
                                                     std = std)
                                ])
dn = Denormalize(mean,std)

img_pil = Image.open('./test_image.png')
im = transform(img_pil)

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

# Algorithm Parameters
maxIt = 100 # maximum number of iterations for binary manifool
mode = 'similarity' # transformation set
gamma = 0 # momentum parameter
geo_step = 0.05 # step size for geodesic distance calculation
batch_size = 3 # number of transformations to batch for line search
cuda_on = True # use GPU or not
numerical = True # use numerical gradients or analytical
step_sizes = np.logspace(-2,np.log10(1),20) # step sizes for line search

# Main Algorithm
t = time.time()
out = manifool(im, net, mode,
               maxIter=maxIt,
               step_sizes = step_sizes,
               gamma = gamma,
               geo_step = geo_step,
               batch_size = batch_size,
               cuda_on = cuda_on,
               crop=224,
               numerical = numerical)

elapsed = time.time()-t

# Depack the outputs
k_org = out['org_label']
k_f = out['fooling_label']
tar = out['target']
geo_dist = out['geo_dist']
II = out['output_image']

# PRint the results
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
