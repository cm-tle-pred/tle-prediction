import torch
import torch.nn as nn
from collections import OrderedDict

class NNModel(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize, activate=None):
        super().__init__()
        self.activate = nn.Sigmoid() if activate == "Sigmoid" else nn.Tanh() if activate == "Tanh" else nn.ReLU()
        self.layer1 = nn.Linear(inputSize, hiddenSize)
        self.layer2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, X):
        hidden = self.activate(self.layer1(X))
        return self.layer2(hidden)


class NNModelEx(nn.Module):
    def __init__(self, inputSize, outputSize, **kwargs):
        super().__init__()

        network = []
        p = inputSize
        for k,v in kwargs.items():
            if k.startswith('l'):
                network.append(nn.Linear(in_features=p, out_features=v))
                p=v
            elif k.startswith('d'):
                network.append(nn.Dropout(v))
            elif k.startswith('t'):
                network.append(nn.Tanh())
            elif k.startswith('s'):
                network.append(nn.Sigmoid())
            elif k.startswith('r'):
                network.append(nn.ReLU())

        network.append(nn.Linear(in_features=p, out_features=outputSize))
        #network.append(nn.ReLU())

        self.net = nn.Sequential(*network)

    def forward(self, X):
        return self.net(X)

# Based on the following
# Chen, D.; Hu, F.; Nian, G.; Yang, T. Deep Residual Learning for Nonlinear Regression. Entropy 2020, 22, 193.
# https://github.com/DowellChan/ResNetRegression/blob/master/ResNetOptimalModel.py
# DeepLearnPhysics: PyTorch 5-particle Classifier Example
# https://github.com/DeepLearnPhysics/pytorch-resnet-example/blob/master/resnet_example.py
# Residual Networks: Zuppichini
# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
# https://github.com/FrancescoSaverioZuppichini/ResNet
class ResnetDenseBlock(nn.Module):
    def __init__(self, input_size, width):
        super(ResnetDenseBlock, self).__init__()
        self.dense1 = nn.Linear(input_size, width)
        self.bn1 = nn.BatchNorm1d(num_features=width)
        self.relu = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(num_features=width)
        self.dense3 = nn.Linear(width, width)
        self.bn3 = nn.BatchNorm1d(num_features=width)
        self.shortcut = nn.Sequential(OrderedDict([
            ('dense_sc', nn.Linear(input_size, width)),
            ('bn_sc', nn.BatchNorm1d(num_features=width))
        ]))

    def forward(self, x):
        # Layer 1
        out = self.dense1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Layer 2
        out = self.dense2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Layer 3
        out = self.dense3(out)
        out = self.bn3(out)

        # Shortcut
        residual=self.shortcut(x)

        out += residual
        out = self.relu(out)

        return out


class ResnetIdentityBlock(nn.Module):
    def __init__(self, width):
        super(ResnetIdentityBlock, self).__init__()
        self.dense1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(num_features=width)
        self.relu = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(num_features=width)
        self.dense3 = nn.Linear(width, width)
        self.bn3 = nn.BatchNorm1d(num_features=width)
        self.shortcut = nn.Identity()

    def forward(self, x):
        # Layer 1
        out = self.dense1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Layer 2
        out = self.dense2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Layer 3
        out = self.dense3(out)
        out = self.bn3(out)

        # Shortcut
        residual=self.shortcut(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetStack(nn.Module):
    def __init__(self, id, input_size, width):
        super(ResNetStack, self).__init__()
        self.stack = nn.Sequential(OrderedDict([
            (f'dense_{str(id)}', ResnetDenseBlock(input_size, width)),
            (f'identity_{str(id)}a', ResnetIdentityBlock(width)),
            (f'identity_{str(id)}b', ResnetIdentityBlock(width)),
        ]))

    def forward(self, x):
        return self.stack(x)

class ResNet28(nn.Module):
    def __init__(self, input_size, output_size, width):
        super(ResNet28, self).__init__()
        self.stack1 = ResNetStack(1, input_size, width)
        self.stack2 = ResNetStack(2, width, width)
        self.stack3 = ResNetStack(3, width, width)
        self.final_bn = nn.BatchNorm1d(num_features=width)
        self.final = nn.Linear(width, output_size)

    def forward(self, x):
        out = self.stack1(x)
        out = self.stack2(out)
        out = self.stack3(out)
        out = self.final_bn(out)
        out = self.final(out)

        return out
