import pandas as pd
import os
import clean_data
import torch
import torch.nn as nn
from dataset import Dataset, to_device
from model import NNModel
from time import time
# import gc

def load_raw_data():
    print('>>> Loading raw data')
    # File created by load_data.load_data()/write_data()
    df = pd.read_pickle(os.environ['GP_HIST_PATH'] + '/raw_compiled/train.pkl' )
    return df

def clean_raw_data(df):
    print('>>> Cleaning data')
    df = clean_data.add_epoch_data(df)
    return df

def load_cleaned_data():
    print('>>> Loading cleaned data')
    # File created by load_data.load_data()/write_data()
    df = pd.read_pickle(os.environ['GP_HIST_PATH'] + '/cleaned/train_clean.pkl' )
    return df

def load_index_map():
    print('>>> Loading index map')
    # File created by clean_data.create_index_map()
    idx_pairs = clean_data.load_index_map()
    return idx_pairs

def train_model(df, idx_pairs, model_cols=None, hiddenSize=300, batchSize=2000,
                learningRate=0.01, numEpochs=1, target='cpu', num_workers=0,
                print_itr=1000):
    torch.manual_seed(0)

    device = torch.device(target)

    if model_cols is None:
        model_cols = ['MEAN_MOTION_DOT', 'MEAN_MOTION_DDOT', 'BSTAR', 'INCLINATION', 'RA_OF_ASC_NODE',
                      'ECCENTRICITY', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'MEAN_MOTION', 'epoch_jd', 'epoch_fr']

    print('>>> Loading model')
    net = NNModel(len(model_cols) + 2, len(model_cols) - 2, hiddenSize)
    to_device(net, device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)

    print('>>> Loading dataset')
    trainDataset = Dataset(df[model_cols], idx_pairs, device)
    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset,
                                              batch_size=batchSize,
                                              shuffle=True,
                                              num_workers=num_workers,
                                             )

    # # Test deleting the variable from memory...
    # del df
    # gc.collect()

    print('>>> Beginning training!')
    ts = time()
    for epoch in range(numEpochs):
        for i, (inputs, labels) in enumerate(trainLoader):
            optimizer.zero_grad()
            # Forward propagation
            outputs = net(inputs)
            # Backpropagation
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient descent
            optimizer.step()
            # Logging
            if (i+1) % print_itr == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {}, Time: {}s'.format(epoch+1,
                      numEpochs, i+1,
                      len(trainDataset)//batchSize,
                      loss,
                      round(time()-ts)))
                ts = time()

    net.eval()
    torch.save(net.state_dict(), 'model_0.pth')

    return net

if __name__ == '__main__':
    import argparse
    from time import time

    # ts = time()
    # df = load_raw_data()
    # print(f'  Took {round(time()-ts)} seconds')
    #
    # ts = time()
    # df = clean_raw_data(df)
    # print(f'  Took {round(time()-ts)} seconds')
    #
    # print('Writing cleaned data...')
    # clean_data.write_data(df)
    # print('Finished')

    ts = time()
    df = load_cleaned_data()
    print(f'  Took {round(time()-ts)} seconds')

    ts = time()
    idx_pairs = load_index_map()
    print(f'  Took {round(time()-ts)} seconds')

    ts = time()
    train_model(df, idx_pairs, num_workers=8)
    print(f'  Took {round(time()-ts)} seconds')
