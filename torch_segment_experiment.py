import ast
import os
import numpy as np
import pandas as pd
import datetime

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch_mlp import MLP
from torch_segment_dataset import SegmentDataset
from data.geolife.convert_minmax_location import LocationPreprocessor

from args            import argument_parser
from torch_time_het import TimeHetNet

def metricMenhattan(y_true, y_pred):
    row = torch.abs(y_pred[:,:,:,0] - y_true[:,:,:,0])
    col = torch.abs(y_pred[:,:,:,1] - y_true[:,:,:,1])
    return torch.mean(row + col)

def metricEuclidean(y_true, y_pred):
    row = (y_pred[:,:,:,0] - y_true[:,:,:,0])**2
    col = (y_pred[:,:,:,1] - y_true[:,:,:,1])**2
    return torch.mean((row + col)** 0.5)

def make_Tensorboard_dir(dir_name, dir_format):
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime(dir_format)
    return os.path.join(root_logdir, sub_dir_name)

##### CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
##################################################

##### args
# model_type = 'mlp'
model_type= 'time-hetnet'

hidden_layer = 3
cell = 256

loss_method = 'mse'
# loss_method = 'cross'

min_length = 6
time_delta = 10
length = min_length * time_delta
y_timestep = min_length

x_attribute = 15
label_attribute = 2

sample_s = 10
sample_q = 10

batch_size = 5

args_early_stopping = True
args_epoch = 100000
args_lr = 0.001
args_patience = 5
args_factor = 0.1

train_size = 0.7
validation_size = 0.2
##################################################

##### dirs
data_dir = "data/geolife/Data/"
##################################################

##### Tensorboard
# tensorboard start
# ./tensorboard --logdir=data/geolife/runs
writer_dir_name = 'data/geolife/runs'
dir_format = '[segment_' + model_type + ']_%Y%m%d-%H%M%S'
configuration_file = 'configuration.csv'
##################################################

##### Train Phase
is_train = True
best_model_path = 'best_model.pth'
best_train_model = 'best_train_model.pth'
##################################################

##### User List
user_list_file = 'user_data_volumn.csv'#'grid_user_list.csv'
user_df = pd.read_csv('data/geolife/' + user_list_file)
user_df = user_df.loc[user_df['segment_list_10min'] >= (sample_s + sample_q), :]
locationPreprocessor = LocationPreprocessor('data/geolife/')
user_list = []
for user in user_df['user_id'].to_list():
    user_list += [locationPreprocessor.getUserId(user)]
# user_list = ["068"]#, "003", "004"]
##################################################

##### Train - Validation user list
train_len       = (int)(len(user_list) * train_size)
validation_len  = (int)(len(user_list) * validation_size)

train_list      = user_list[0:train_len]
validation_list = user_list[train_len:(train_len + validation_len)]
test_list       = user_list[(train_len + validation_len):]

# train_list = user_list[0:10]
# validation_list = user_list[10:15]
# test_list       = user_list[0:1]
##################################################

def write_configruation(conf_file):
    #--------Write Configration--------
    import pandas as pd
    conf_df = pd.DataFrame({'device':[device],
                            'model_type':[model_type],
                            'label_attribute':[label_attribute],
                            'user_list_file':[user_list_file],
                            'sample_s':[sample_s],
                            'sample_q':[sample_q],
                            'epoch':[args_epoch],
                            'patience':[args_patience],
                            'x_attribute':[x_attribute],
                            'time_delta':[time_delta],
                            'y_timestep':[y_timestep],
                            'length':[length],
                            'train_list':[train_list],
                            'val_list':[validation_list],
                            'test_list':[test_list],
                            'train_columns':[training_data.get_train_columns()]})
    conf_df.to_csv(conf_file, index=False)

print("##################################################################")
print(f"use_cuda: {use_cuda}, device: {device}")

print(f"train len: {train_len}")
print(f"validation len: {validation_len}")
print(f"test len: {len(test_list)}")

print("Building Network ...")

# Dataset
training_data           = SegmentDataset(model_type, data_dir, train_list, device, time_delta, y_timestep, length, label_attribute, sample_s, sample_q)
validation_data         = SegmentDataset(model_type, data_dir, validation_list, device, time_delta, y_timestep, length, label_attribute, sample_s, sample_q)
train_dataloader        = DataLoader(training_data, batch_size, shuffle=False)
validation_dataloader   = DataLoader(validation_data, batch_size, shuffle=False)

if is_train == True:
    print('Start Train')
    #--------Define Tensorboard--------
    writer_dir = make_Tensorboard_dir(writer_dir_name, dir_format)
    writer = SummaryWriter(writer_dir)

    best_model_path = writer_dir + "/" + best_model_path
    best_train_model = writer_dir + "/" + best_train_model

    #--------Define a Model
    if model_type == 'mlp':
        model = MLP(input_shape=[length, x_attribute], y_timestep = y_timestep, loss_fn=loss_method, label_attribute=label_attribute, cell=cell, hidden_layer=hidden_layer)

    elif model_type == 'time-hetnet':
        args = argument_parser()
        model = TimeHetNet(dims_inf = ast.literal_eval(args.dims),
                           dims_pred = ast.literal_eval(args.dims_pred),
                           activation="relu",
                           time=args.tmax_length,
                           batchnorm=False,
                           block = args.block.split(","),
                           output_shape=[y_timestep, label_attribute],
                           length = length)
        
    else:
        model = MLP(input_shape=[length, x_attribute], y_timestep = y_timestep, loss_fn=loss_method, label_attribute=label_attribute, cell=cell, hidden_layer=hidden_layer)

    #--------Define Losses annd metrics----------------
    if loss_method == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args_lr)

    #--------Define Callbacks----------------
    # lr_scheduler = None  #callback
    # if args_early_stopping:
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args_factor, patience=args_patience, verbose=True)

    #------- Train the model -----------------
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    no_improvement = 0
    
    model = model.to(torch.double)
    model = model.to(device)

    for epoch in range(args_epoch):
        print(f"epoch: {epoch}")

        model.train()
        loss_train = 0.0
        for train_idx, train_data in enumerate(train_dataloader, 0):
            task_X, task_y = train_data
            optimizer.zero_grad()
            output = model(task_X)
            # loss = criterion(output, task_y)
            loss = metricMenhattan(output, task_y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        writer.add_scalar('training loss', loss_train, epoch)

        model.eval()
        if epoch == 0:
            write_configruation(writer_dir + "/" + configuration_file)
        
        loss_val = 0.0
        with torch.no_grad():
            for val_idx, val_data in enumerate(validation_dataloader, 0):
                X_val, y_val = val_data
                val_outputs = model(X_val)     
                val_loss = metricMenhattan(y_val, val_outputs)
                # val_loss = criterion(y_val, val_outputs)
                loss_val += val_loss.item()
            writer.add_scalar('validation loss', loss_val, epoch)
            print(f"train loss: {loss_train}, validation loss: {loss_val}")
            lr_scheduler.step(loss_val)

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            no_improvement = 0
            print('save best weight and bias')
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improvement += 1
        
        if loss_train < best_train_loss:
            best_train_loss = loss_train
            print('save best train model')
            torch.save(model.state_dict(), best_train_model)

print('Finish Train')
# 예측
# with torch.no_grad():
#     model.eval()
#     predicted = model(task_X)
#     loss = metricDist(task_y, predicted)
#     print("target latitude, longitude:", task_y)
#     print("Predicted latitude, longitude:", predicted)
#     print(f'loss: {loss}')
