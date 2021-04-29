import pandas as pd
import numpy as np
import os
import clean_data
import torch
import torch.nn as nn
from dataset import Dataset, to_device
from model import NNModel, NNModelEx
from time import time
from tqdm import tqdm_notebook as tqdm

def create_model(in_size, out_size, **kwargs):
    model = NNModelEx(inputSize=in_size,
                      outputSize=out_size,
                      **kwargs)
    return model

def train_model(X, y, model, device='cpu', batch_size=2000, learning_rate=0.01,
                momentum=0.9, num_epochs=1, loss=None, num_workers=0,
                loss_data_points=50, save_model=False,):
    torch.manual_seed(0)

    pyt_device = torch.device(device)

    to_device(model, pyt_device)

    if not loss or loss =='MAE' or loss == 'L1':
        criterion = nn.L1Loss()
    elif loss == 'MSE' or loss == 'L2':
        criterion = nn.MSELoss()

    print(f'batch_size={batch_size} learning_rate={learning_rate}')
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print('>>> Loading dataset')
    trainDataset = Dataset(X, y)
    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True
                                             )

    # Determine loss output
    if loss_data_points > 0:
        loss_itr = len(X) / batch_size * num_epochs // loss_data_points
        if loss_itr <= 0:
            loss_itr=1
        elif loss_itr > len(X) // batch_size:
            loss_itr = len(X) // batch_size
    loss_out = []

    print('>>> Beginning training!')
    epbar = tqdm(range(num_epochs))
    num_batches = len(trainDataset)//batch_size
    ts = time()
    lt = time()
    for epoch in epbar:
        epbar.set_description(f"Epoch {epoch+1}")
        elosses = []
        for i, (inputs, labels) in enumerate(trainLoader):
            inputs = to_device(inputs, pyt_device)
            labels = to_device(labels, pyt_device)

            optimizer.zero_grad()
            # Forward propagation
            outputs = model(inputs)
            # Backpropagation
            loss = criterion(outputs, labels)
            if 'cuda' in device:
                elosses.append(loss.data.cpu().numpy().item())
            else:
                elosses.append(loss.data.numpy().item())
            loss.backward()
            # Gradient descent
            optimizer.step()
            # Logging
            if loss_data_points > 0 and (i+1) % loss_itr == 0:
                loss_out.append(dict(batch=i+1,
                                     num_batches=num_batches,
                                     epoch=epoch+1,
                                     num_epochs=num_epochs,
                                     loss=loss,
                                     time=round(time()-lt)))
                lt = time()
        mean_elosses = np.mean(elosses)
        epbar.set_postfix({'train_loss':f"{mean_elosses:.9f}", 'time:': f"{round(time()-ts)}s"})
        ts = time()

    print (f'Final loss: {loss}')

    model.eval()
    if save_model:
        torch.save(model.state_dict(), 'model_1.pth')
        print('Model saved as model_1.pth')

    return model, loss_out

def predict(model, X, y, device='cpu'):
    pyt_device = torch.device(device)

    if 'cuda' in device:
        # Since it doesn't all fit on the GPU, we'll use a dataloader
        batch_size = 2000
        predictDataset = Dataset(X, y)
        predictLoader = torch.utils.data.DataLoader(dataset=predictDataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=5,
                                                  pin_memory=True
                                                 )
        num_elements = len(predictLoader.dataset)
        num_outputs = len(y.columns)
        num_batches = len(predictLoader)
        predictions = torch.zeros(num_elements, num_outputs)
        for i, (inputs, _) in tqdm(enumerate(predictLoader), total=num_batches):
            inputs = to_device(inputs, pyt_device)
            start = i*batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements
            pred = model(inputs)
            predictions[start:end] = pred.detach().cpu()
        nn_results = predictions.numpy()
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
