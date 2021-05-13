import pandas as pd
import os
import clean_data
import torch
import torch.nn as nn
from dataset import Dataset, to_device
from model import NNModel, NNModelEx
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

def create_model(model_cols=None, **kwargs):
    if model_cols is None:
        model_cols = ['MEAN_MOTION_DOT', 'MEAN_MOTION_DDOT', 'BSTAR', 'INCLINATION', 'RA_OF_ASC_NODE',
                      'ECCENTRICITY', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'MEAN_MOTION', 'epoch_jd', 'epoch_fr']

    model = NNModelEx(inputSize=len(model_cols) + 2,
                      outputSize=len(model_cols) - 2,
                      **kwargs)
    return model

def train_model(df, idx_pairs, model_cols=None, hiddenSize=300, batchSize=2000,
                learningRate=0.01, numEpochs=1, device='cpu', num_workers=0,
                print_itr=1000, save_model=False, activate=None, loss=None,
                model=None, loss_data_points=50, **kwargs):
    torch.manual_seed(0)

    pyt_device = torch.device(device)

    if model_cols is None:
        model_cols = ['MEAN_MOTION_DOT', 'MEAN_MOTION_DDOT', 'BSTAR', 'INCLINATION', 'RA_OF_ASC_NODE',
                      'ECCENTRICITY', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'MEAN_MOTION', 'epoch_jd', 'epoch_fr']

    if not model:
        print('>>> Loading simple model')
        model = NNModel(inputSize=len(model_cols) + 2,
                        outputSize=len(model_cols) - 2,
                        hiddenSize=hiddenSize,
                        activate=activate)

    to_device(model, pyt_device)

    if not loss or loss =='MAE' or loss == 'L1':
        criterion = nn.L1Loss()
    elif loss == 'MSE' or loss == 'L2':
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    print('>>> Loading dataset')
    trainDataset = Dataset(df[model_cols], idx_pairs, pyt_device)
    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset,
                                              batch_size=batchSize,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              #pin_memory=True
                                             )

    # Determine loss output
    if loss_data_points > 0:
        loss_itr = len(idx_pairs) / batchSize * numEpochs // loss_data_points
        if loss_itr <= 0:
            loss_itr=1
        elif loss_itr > len(idx_pairs) // batchSize:
            loss_itr = len(idx_pairs) // batchSize
    loss_out = []

    print('>>> Beginning training!')
    numBatches = len(trainDataset)//batchSize
    ts = time()
    lt = time()
    for epoch in range(numEpochs):
        for i, (inputs, labels) in enumerate(trainLoader):

            optimizer.zero_grad()
            # Forward propagation
            outputs = model(inputs)
            # Backpropagation
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient descent
            optimizer.step()
            # Logging
            if print_itr > 0 and (i+1) % print_itr == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {}, Time: {}s'.format(epoch+1,
                      numEpochs, i+1,
                      numBatches,
                      loss,
                      round(time()-ts)))
                ts = time()
            if loss_data_points > 0 and (i+1) % loss_itr == 0:
                loss_out.append(dict(batch=i+1,
                                     numBatches=numBatches,
                                     epoch=epoch+1,
                                     numEpochs=numEpochs,
                                     loss=loss,
                                     time=round(time()-lt)))
                lt = time()


    print (f'Final loss: {loss}')

    model.eval()
    if save_model:
        torch.save(model.state_dict(), 'model_0.pth')
        print('Model saved as model_0.pth')

    return model, loss_out

def predict(model, X, device='cpu'):
    pyt_device = torch.device(device)

    if 'cuda' in device:
        X_tensor = to_device(torch.from_numpy(X.to_numpy()).float(), pyt_device)
        nn_results = model(X_tensor).detach().cpu().numpy()
    else:
        X_tensor = torch.from_numpy(X.to_numpy()).float()
        nn_results = model(X_tensor).detach().numpy()

    return nn_results

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
