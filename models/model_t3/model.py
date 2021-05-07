import torch
import torch.nn as nn

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
