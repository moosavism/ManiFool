import time

import numpy as np
import torchvision
import torchvision.transforms as transform

from functions.algorithms.manifool import manifool
from functions.helpers.plot_helpers import compare_images
from models.manitest_cnn import ManitestMNIST_net as Net

# Load MNIST dataset
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                     transform=transform.ToTensor())

# Algorithm parameters
maxIt = 100
step_sizes = np.logspace(-2, np.log10(0.5), 10)
gamma = 0.8

# Algorithm inputs
mode = 'translation'  # Transformation set - see general.py for different available sets

# Choose a random image
i = np.random.randint(0, len(dataset))
I = dataset.__getitem__(i)[0] * 255

# Load the network - CNN with 2 conv layers + 1 fc before output.
net = Net()

t = time.time()

# Main Algorithm
out = manifool(I, net, mode,
               maxIter=maxIt,
               gamma=gamma,
               cuda_on=True,
               step_sizes=step_sizes,
               numerical=True,
               verbose=2)

# Unpack the output
k_org = out['org_label']
k_f = out['fooling_label']
tar = out['target']
geo_dist = out['geo_dist']
II = out['output_image']

# Print Results
print('\n\n\nFinal Result -- Multi Target:')
print("Original Output: {}\nTarget {}\nNew Output: {}".format(k_org, tar, k_f))
print("Geodesic Score: {}".format(geo_dist))
print('Time elapsed: {} seconds'.format(time.time() - t))

# Show original and transformed image
compare_images(I.cpu(), II.cpu(), k_org, k_f)

input('Press Enter to end.')
