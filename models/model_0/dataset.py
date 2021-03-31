import torch

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, idx_pairs, device='cpu'):
        'Initialization'
        self.data = to_device(torch.from_numpy(data.to_numpy()).float(), device)
        self.idx_pairs = to_device(torch.from_numpy(idx_pairs).long(), device)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.idx_pairs)

    def __getitem__(self, index):
        'Generates one sample of data'
        p = self.idx_pairs[index]

        # This will use the idx_pairs (x,y) to build the inputs(X) and labels (y)
        # output.  It adds the last 2 columns of y to X and removes them from y.
        X = torch.cat((self.data[p[0]], self.data[p[1]][-2:]), 0)
        y = self.data[p[1]][:-2]

        return X, y
