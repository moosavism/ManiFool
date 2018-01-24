import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F


class ManitestMNIST_net(nn.Module):

    def __init__(self):
        super(ManitestMNIST_net,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding = 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64,10,5)
        self.pool = nn.MaxPool2d(2,2)

        #Loading the weights and biases
        mat_contents = sio.loadmat('models/manitestcnn.mat')
        filters1 = mat_contents['filters1']
        filters2 = mat_contents['filters2']
        filtersfc = mat_contents['filtersfc']
        biases1 = mat_contents['biases1'].flatten()
        biases2 = mat_contents['biases2'].flatten()
        biasesfc = mat_contents['biasesfc'].flatten()

        self.conv1.weight = nn.Parameter(torch.from_numpy(filters1)+0)
        self.conv1.bias = nn.Parameter(torch.from_numpy(biases1)+0)
        self.conv2.weight = nn.Parameter(torch.from_numpy(filters2)+0)
        self.conv2.bias = nn.Parameter(torch.from_numpy(biases2)+0)
        self.conv3.weight = nn.Parameter(torch.from_numpy(filtersfc)+0)
        self.conv3.bias = nn.Parameter(torch.from_numpy(biasesfc)+0)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)
        x = x.view(-1, 10)
        return x

#%%
