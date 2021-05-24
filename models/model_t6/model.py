import torch
import torch.nn as nn

def tims_mse_loss(input, target):
    worst_k = input.data.nelement() // 4
    se = (input - target) ** 2
    worsts_in_batch = torch.topk(se, worst_k, dim=0)
    return ((torch.sum(se)-torch.sum(worsts_in_batch.values)) / (input.data.nelement()-worst_k))

def tims_mae_loss(input, target):
    worst_k = input.data.nelement() // 4
    ae = torch.absolute(input - target)
    worsts_in_batch = torch.topk(ae, worst_k, dim=0)
    return ((torch.sum(ae)-torch.sum(worsts_in_batch.values)) / (input.data.nelement()-worst_k))

def tim95_mse_loss(input, target):
    worst_k = input.data.nelement() // 20
    se = (input - target) ** 2
    worsts_in_batch = torch.topk(se, worst_k, dim=0)
    return ((torch.sum(se)-torch.sum(worsts_in_batch.values)) / (input.data.nelement()-worst_k))

class NNModelXYZ(nn.Module):
    def __init__(self, inputSize, outputSize, *args):
        super().__init__()
        network = []
        p = inputSize
        for k,v in args:
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
        network.append(nn.Linear(in_features=p, out_features=outputSize-2))
        self.net = nn.Sequential(*network)

    def forward(self, X):
        out = self.net(X)
        cd1 = torch.sqrt((out[:,0:3]**2).sum(axis=1))
        cd2 = torch.sqrt((out[:,3:6]**2).sum(axis=1))
        return torch.cat((out,torch.unsqueeze(cd1, 1),torch.unsqueeze(cd2, 1)),1)