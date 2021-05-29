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

class NNModelEx(nn.Module):
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
        network.append(nn.Linear(in_features=p, out_features=outputSize))
        self.net = nn.Sequential(*network)

    def forward(self, X):
        out = self.net(X)
        return out


class NNSingleFeatureModel(nn.Module):
    def __init__(self, inputSize, feature_index, model_config):
        super().__init__()
        self.feature_index = feature_index
        self.head = NNModelEx(inputSize, 1, *model_config)

    def forward(self, X):
        out = X[:,[self.feature_index]]
        out[:,0] += self.head(X)[:,0]
        return out
