import torch

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y):
        'Initialization'
        self.X = torch.from_numpy(X.to_numpy()).float()
        self.y = torch.from_numpy(y.to_numpy()).float()
        self.num_X = X.shape[1]
        self.num_y = y.shape[1]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.X[index]
        y = self.y[index]

        return X, y
