import pandas as pd
import numpy as np
import os
import clean_data
import torch
import torch.nn as nn
from dataset import Dataset, to_device
from model import ResNet28
from time import time
from tqdm import tqdm_notebook as tqdm

def legacy_load_model_with_config(config, training_set=None, force_train=False):
    # a bit hacky, but in the training phase, we never load and use the minmax scalers
    # just putting it here for when we want to load the model elsewhere THEN revert scaling
    # probably better to have the scalers saved separately....

    path = config.get('model_path', f"{os.environ['GP_HIST_PATH']}/../t_models")
    f = f"{path}/{config['model_identifier']}.pth"
    if os.path.exists(f) and not force_train:
        print("Loading existing model")
        checkpoint = torch.load(f)
        net = checkpoint['net']
        next_epoch = checkpoint['next_epoch']
        loss_func = checkpoint['loss_func']
        optimizer = checkpoint['optimizer']
        #optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'])
        mean_losses = checkpoint['mean_losses']
    else:
        if not training_set:
            raise Exception('Cannot create model without training_set')
        print("New model created")
        #net = NNModelEx(inputSize=training_set.num_X, outputSize=training_set.num_y, **config['model_definition'])
        net = ResNet28(input_size=training_set.num_X, output_size=training_set.num_y, width=config['model_width'])
        #net = ResMultiNet28(input_size=training_set.num_X, width=config['model_width'])
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
        #optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        #optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], amsgrad=config['amsgrad'])
        optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        mean_losses = []
        next_epoch = 0
        save_model_with_config(config, net=net, loss_func=loss_func,
                               mean_losses=mean_losses, next_epoch=next_epoch,
                              )
        # blank scaler when creating new model
    return net, loss_func, optimizer, mean_losses, next_epoch

def load_model_with_config(config, training_set=None, force_train=False):
    # a bit hacky, but in the training phase, we never load and use the minmax scalers
    # just putting it here for when we want to load the model elsewhere THEN revert scaling
    # probably better to have the scalers saved separately....

    path = config.get('model_path', f"{os.environ['GP_HIST_PATH']}/../t_models")
    f = f"{path}/{config['model_identifier']}.pth"
    if os.path.exists(f) and not force_train:
        print("Loading existing model")
        checkpoint = torch.load(f)
        net = checkpoint['net']
        next_epoch = checkpoint['next_epoch']
        loss_func = checkpoint['loss_func']
        optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'])
        #optimizer = checkpoint['optimizer']
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=next_epoch-1)
        #scheduler = 
        #optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], amsgrad=config['amsgrad'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        mean_losses = checkpoint['mean_losses']
    else:
        if not training_set:
            raise Exception('Cannot create model without training_set')
        print("New model created")
        #net = NNModelEx(inputSize=training_set.num_X, outputSize=training_set.num_y, **config['model_definition'])
        #net = ResNet28(input_size=training_set.num_X, output_size=training_set.num_y, width=config['model_width'])
        net = ResMultiNet28(input_size=training_set.num_X, width=config['model_width'])
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
        #optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        #optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], amsgrad=config['amsgrad'])
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1 )
        mean_losses = []
        next_epoch = 0
        save_model_with_config(config, net=net, loss_func=loss_func, optimizer=optimizer,
                               mean_losses=mean_losses, next_epoch=next_epoch,
                              )
        # blank scaler when creating new model
    return net, loss_func, optimizer, scheduler, mean_losses, next_epoch

def save_model_with_config(config, **kwargs):
    path = config.get('model_path', f"{os.environ['GP_HIST_PATH']}/../t_models")

    f = f"{path}/{config['model_identifier']}.pth"
    torch.save(kwargs, f)

def train_model(X_train, y_train, X_test, y_test, configurations, force_train=False):

    path = configurations.get('model_path', None)
    torch.manual_seed(configurations.get('random_seed',0))
    device = configurations.get('device','cpu')
    pyt_device = torch.device(device)

    training_set = Dataset(X_train, y_train)
    training_generator = torch.utils.data.DataLoader(training_set, **configurations['train_params'])
    testing_set = Dataset(X_test, y_test)
    testing_generator = torch.utils.data.DataLoader(testing_set, **configurations['test_params'])


    net, loss_func, optimizer, scheduler, mean_losses, next_epoch, = load_model_with_config(configurations, training_set, force_train)
    to_device(net, pyt_device)
    net.train()
    print(net)

    if next_epoch == configurations['max_epochs']:
        print("Model finished training. To retrain set force_train = True ")
        net.eval()
        return net, mean_losses

    epbar = tqdm(range(next_epoch, configurations['max_epochs']))
    for epoch in epbar:
        epbar.set_description(f"Epoch {epoch+1}")

        running_eloss = 0
        running_vloss = 0

        ipbar = tqdm(training_generator, leave=False)
        ipbar.set_description(f"Training")

        for i, (x, y) in enumerate(ipbar):
            x = to_device(x, pyt_device)
            y = to_device(y, pyt_device)

            optimizer.zero_grad()
            prediction = net(x)     # input x and predict based on x
            loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            running_eloss += loss.item()

        net.eval()
        mean_vlosses = 0
        if configurations['do_validate']:
            with torch.set_grad_enabled(False):
                vpbar = tqdm(testing_generator, leave=False)
                vpbar.set_description("Validating")
                for i, (x, y) in enumerate(vpbar):
                    x = to_device(x, pyt_device)
                    y = to_device(y, pyt_device)
                    prediction = net(x)
                    loss = loss_func(prediction, y)
                    running_vloss += loss.item()
            mean_vlosses = running_vloss / len(testing_generator)

        mean_elosses = running_eloss / len(training_generator)
        mean_losses.append((mean_elosses, mean_vlosses))
        scheduler.step()
        save_model_with_config(configurations, net=net, loss_func=loss_func, optimizer=optimizer,
                               mean_losses=mean_losses, next_epoch=epoch+1,)
        net.train()

        epbar.set_postfix({'train_loss':f"{mean_elosses:.9f}", 'val_loss':f"{mean_vlosses:.9f}"})
    net.eval()
    return net, mean_losses

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
