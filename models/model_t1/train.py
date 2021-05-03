import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from dataset import Dataset, to_device
from model import NNModel, NNModelEx
from time import time
from tqdm import tqdm_notebook as tqdm

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
        loss_func = checkpoint['loss_func']
        optimizer = checkpoint['optimizer']
        mean_losses = checkpoint['mean_losses']
        next_epoch = checkpoint['next_epoch']
    else:
        if not training_set:
            raise Exception('Cannot create model without training_set')
        print("New model created")
        net = NNModelEx(inputSize=training_set.num_X, outputSize=training_set.num_y, **config['model_definition'])
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
        #optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'])
        mean_losses = []
        next_epoch = 0
        save_model_with_config(config, net=net, loss_func=loss_func, optimizer=optimizer,
                               mean_losses=mean_losses, next_epoch=next_epoch,
                              )
        # blank scaler when creating new model
    return net, loss_func, optimizer, mean_losses, next_epoch

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


    net, loss_func, optimizer, mean_losses, next_epoch, = load_model_with_config(configurations, training_set, force_train)
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
        save_model_with_config(configurations, net=net, loss_func=loss_func, optimizer=optimizer,
                               mean_losses=mean_losses, next_epoch=epoch+1,
                              )
        net.train()

        epbar.set_postfix({'train_loss':f"{mean_elosses:.9f}", 'val_loss':f"{mean_vlosses:.9f}"})
    net.eval()
    return net, mean_losses


def create_model(in_size, out_size, **kwargs):
    model = NNModelEx(inputSize=in_size,
                      outputSize=out_size,
                      **kwargs)
    return model

def train_mode_old(X, y, model, device='cpu', batch_size=2000, learning_rate=0.01,
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