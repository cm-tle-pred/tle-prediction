import torch
import torch.nn as nn

class NNModelEx(nn.Module):
    def __init__(self, inputSize, outputSize, **kwargs):
        super().__init__()

        network = []
        p = inputSize
        for k,v in kwargs.items():
            if k.startswith('l'):
                network.append(nn.Linear(in_features=p, out_features=v))
                p=v
            elif k.startswith('b'):
                network.append(nn.Bilinear(in1_features=p, in2_features=1, out_features=v))
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

class NNBranchModel(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        # Energy map net
        
        self.linear1 = nn.Linear(in_features=inputSize-1, out_features=1000)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=1000, out_features=1000)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.lineara2 = nn.Linear(in_features=1000, out_features=1000)
        self.relu2a = nn.ReLU()
        self.dropout2a = nn.Dropout(0.5)
        self.linear2b = nn.Linear(in_features=1000, out_features=1000)
        self.relu2b = nn.ReLU()
        self.dropout2b = nn.Dropout(0.5)
        self.linear2c = nn.Linear(in_features=500, out_features=500)
        self.relu2c = nn.ReLU()
        self.dropout2c = nn.Dropout(0.5)
        self.linear3 = nn.Linear(in_features=500, out_features=250)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.linear4 = nn.Linear(in_features=250, out_features=150)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)
        self.linear5 = nn.Linear(in_features=150, out_features=100)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.5)
        self.linear6 = nn.Linear(in_features=100, out_features=100)

        self.linear_a_1 = nn.Linear(in_features=1, out_features=1)
        self.relu_a_1 = nn.ReLU()

        
        self.bilinear1 = nn.Bilinear(in1_features=1, in2_features=100, out_features=outputSize)
        

    def forward(self, X):
        out = self.linear1(X[:,1:])
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.linear4(out)
        out = self.relu4(out)
        out = self.dropout4(out)
        out = self.linear5(out)
        out = self.relu5(out)
        out = self.dropout5(out)
        out = self.linear6(out)
        
        front = self.linear_a_1(X[:,0:1])
        front = self.relu_a_1(front)

        final = self.bilinear1(front, out)
        
        return final