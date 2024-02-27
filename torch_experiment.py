import os
import gc
import numpy as np
import time
import datetime
import sys
import ast
import itertools
import random

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# Custom libraries
from data_sampling.data_gen_ts    import Task_Sampler
from data_sampling.load_gens      import *
###########
from torch.utils.tensorboard import SummaryWriter

from torch_time_model import SliceEncoderModel as SEncModel
from torch_hetnet import HetNet
from torch_time_het import TimeHetNet
# from torch_experiment_test import run_experiment
from torch_geolife_dateset import GeoLifeDataSet
from data.geolife.convert_minmax_location import LocationPreprocessor

###########
# from experiment_test import run_experiment
from args            import argument_parser
from result_save     import *
###########

from sklearn.metrics import mean_squared_error as mse

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True

def metricsSTD(y_true, y_pred):
    res = nn.MSELoss(y_true, y_pred)
    return torch.std(res, unbiased=False)

def metricMSE(y_true, y_pred):
    loss_f = nn.MSELoss()
    res = loss_f(y_true, y_pred)
    return torch.mean(res)

def metricDist(y_true, y_pred):
    row = (y_pred[:,:,:,0] - y_true[:,:,:,0])**2
    col = (y_pred[:,:,:,1] - y_true[:,:,:,1])**2
    return torch.mean(row + col) ** 0.5

def make_Tensorboard_dir(dir_name, dir_format):
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime(dir_format)
    return os.path.join(root_logdir, sub_dir_name)
# gc.enable()

args = argument_parser()
file_time = str(datetime.datetime.now()).replace(" ", "_")
ft = "file_time"
if args.name is None:
    name = file_time
else:
    name = args.name + "_" + file_time
args.name = name

print("########## argument sheet ########################################")
for arg in vars(args):
    print (f"#{arg:>15}  :  {str(getattr(args, arg))} ")
    
# torch args
data_dir = "data/geolife/Data/"
writer_dir_name = 'data/geolife/runs'
dir_format = '[grid_origin_500]_%Y%m%d-%H%M%S'

label_attribute = 2
loss_method = 'mse'
loss_val_step = 3
lr_epoch = 300

sample_s = 2
sample_q = 2

args_epoch = 5000
args_patience = 5
args_factor = 0.1

gap_min = 12 # 1 min
gap = gap_min

day = 144
week = 1008
y_timestep = (int(day/24) * 12)  #12 # must be less than length, 12 = 1 hour, 12 * 24 = 288 -> 1day
length = week

train_size = 0.7
validation_size = 0.1
batch_size = 3 # each user
batch_norm = False

## Train Phase
is_train = True
best_model_path = 'best_model.pth'
best_train_model = 'best_train_model.pth'
user_list_file = 'grid_user_list.csv'

## Test Phase
# is_train = False
# log_folder = '[grid_100]_20240217-092132'
# best_model_path = 'data/geolife/runs/' + log_folder + '/best_model.pth'

configuration_file = 'configuration.csv'
# tensorboard start
# ./tensorboard --logdir=data/geolife/runs
print("##################################################################")

model_type = 'gru'
print("Building Network ...")

###################### TimeHetNet ######################
model      = TimeHetNet(dims_inf = ast.literal_eval(args.dims),
                        dims_pred = ast.literal_eval(args.dims_pred),
                        activation="relu",
                        time=args.tmax_length,
                        batchnorm=batch_norm,
                        block = args.block.split(","),
                        output_shape=[y_timestep, label_attribute],
                        length = length)

model_type = "TimeHetNet"
print("Using Time Hetnet")
#######################################################

###################### HetNet #########################
# EncM = SEncModel(control=args.control_steps)
# # EncM = SEncModel(control=80)
# model = HetNet(EncM, "slice",
#                dims = ast.literal_eval(args.dims),
#                acti ='relu',
#                drop1 = 0.01,
#                drop2 = 0.01,
#                share_qs = False)
# print("Using Hetnet")
#######################################################

model = model.to(torch.double)
model = model.to(device)

#--------Load the data----------------
# train_gen, val_gen, _, ds_names = getGens(args, model_type)
# user_list = os.listdir(data_dir)

locationPreprocessor = LocationPreprocessor('data/geolife/')
# user_df = locationPreprocessor.get_valid_user_list()
user_list = []
# for user in user_df['valid_user_list']:
#     user_list += [locationPreprocessor.getUserId(user)]
    
# random.shuffle(user_list)

import pandas as pd
user_df = pd.read_csv('data/geolife/' + user_list_file)
for user in user_df['user_id'].to_list():
    user_list += [locationPreprocessor.getUserId(user)]

train_len       = (int)(len(user_list) * train_size)
validation_len  = (int)(len(user_list) * validation_size)

train_list      = user_list[1:train_len-10]
validation_list = user_list[train_len-10:train_len-5]
# validation_list = user_list[train_len:(train_len + validation_len)]
test_list       = user_list[(train_len + validation_len):(train_len + validation_len + 10)]

# ['068', '030', '085', '004', '067', '014', '050', '035', '013', '003', '039', '024', '036']
# train_list      = ['030', '085', '003']#['067', '004', '014']#, '050', '035', '013', '003', '039', '024', '036', '085']
# validation_list = [user_list[0]]
train_list = user_list[1:6]
validation_list = user_list[6:8]
test_list       = user_list[6:8]

# test_list       = user_list[(train_len + validation_len):]
print(f"use_cuda: {use_cuda}, device: {device}")
print(f"train_len: {train_len}, val_len: {validation_len}")
print(f"train_list:      {train_list}")
print(f"validation_list: {validation_list}")
print(f"test_list:       {test_list}")

# multi user
training_data   = GeoLifeDataSet(data_dir, train_list, sample_s, sample_q, length, y_timestep, gap, label_attribute)
validation_data = GeoLifeDataSet(data_dir, validation_list, sample_s, sample_q, length, y_timestep, gap, label_attribute)
test_data       = GeoLifeDataSet(data_dir, test_list, sample_s, sample_q, length, y_timestep, gap, label_attribute)

train_dataloader      = DataLoader(training_data, batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_data, batch_size, shuffle=False)
test_dataloader       = DataLoader(test_data, batch_size, shuffle=False)

# randomNumber = int(np.random.rand(1) * 10000000)
# print("-------ID Number is:", randomNumber)

def write_configruation(conf_file):
    #--------Write Configration--------
    import pandas as pd
    conf_df = pd.DataFrame({'label_attribute':[label_attribute],
                            'loss_method':[loss_method],
                            'loss_val_step':[loss_val_step],
                            'user_list_file':[user_list_file],
                            'sample_s':[sample_s],
                            'sample_q':[sample_q],
                            'epoch':[args_epoch],
                            'patience':[args_patience],
                            'batch_norm':[batch_norm],
                            # 'gap':[gap],
                            'day':[day],
                            'week':[week],
                            'y_timestep':[y_timestep],
                            'length':[length],
                            'train_list':[train_list],
                            'val_list':[validation_list],
                            'test_list':[test_list],
                            'train_columns':[training_data.get_train_columns()]})
    conf_df.to_csv(conf_file, index=False)
    
if is_train == True:
    print('Start Train')
    #--------Define Tensorboard--------
    writer_dir = make_Tensorboard_dir(writer_dir_name, dir_format)
    writer = SummaryWriter(writer_dir)

    best_model_path = writer_dir + "/" + best_model_path
    best_train_model = writer_dir + "/" + best_train_model

    #--------Define Losses annd metrics----------------
    if loss_method == 'cross':
        loss_object = nn.CrossEntropyLoss()
    else:
        loss_object = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #--------Define Callbacks----------------
    lr_scheduler = None  #callback
    if args.early_stopping:
        # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args_patience, verbose=True)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args_factor, patience=args_patience, verbose=True)

    #------- Train the model -----------------
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    no_improvement = 0

    # http://www.gisdeveloper.co.kr/?p=8615 - loss 값 확인 블로그
    loss_train_list = []
    loss_val_list = []
    
    loss_val_mean = 0
    val_count = 0
    for epoch in range(args_epoch):
        print(f"epoch: {epoch}")
        
        model.train()
        loss_train = 0.0
        for train_idx, train_data in enumerate(train_dataloader, 0):
            X_train, y_train = train_data
            if len(X_train) < 2:
                continue
            optimizer.zero_grad()
            outputs = model(X_train)
            # loss = loss_object(y_train, outputs)
            loss = metricDist(y_train, outputs)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        writer.add_scalar('training loss', loss_train, epoch)
        loss_train_list.append(loss_train)
        
        model.eval()
        if epoch == 0:
            write_configruation(writer_dir + "/" + configuration_file)

        loss_val = 0.0
        with torch.no_grad():
            for val_idx, val_data in enumerate(validation_dataloader, 0):
                X_val, y_val = val_data
                if len(X_val) < 2:
                    continue
                val_outputs = model(X_val)                
                val_loss = metricDist(y_val, val_outputs)
                loss_val += val_loss.item()
            writer.add_scalar('validation loss', loss_val, epoch)
            print(f"train loss: {loss_train}, validation loss: {loss_val}")
            lr_scheduler.step(loss_val)
        
        # if lr_scheduler.num_bad_epochs > lr_scheduler.patience:
        #     print(f"Early stopping after {epoch+1} epochs.")
        #     break
        
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
        # if no_improvement == args_patience:
        #     print(f"Early stopping after {epoch+1} epochs.")
        #     break

print('Finish Train')
#------- Test the model -----------------
print('Start Test')
mse = nn.MSELoss()
best_model = TimeHetNet(dims_inf = ast.literal_eval(args.dims),
                        dims_pred = ast.literal_eval(args.dims_pred), 
                        activation="relu", 
                        time=args.tmax_length,
                        batchnorm=batch_norm, 
                        block = args.block.split(","),
                        output_shape=[y_timestep, 2],
                        length=length)
print(f'best_model_path: {best_model_path}')
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()
        
class_pred_y = []
class_test_y = []
que_x_list = []

for test_idx in range(10):
    acc = 0.0
    for idx, test_data in enumerate(test_dataloader, 0):
        test_X, test_y = test_data
        if len(test_X) < 2:
            continue
        que_x, _, _ = test_X
        pred_y = best_model(test_X)
        que_x_list.append(que_x[:, :, -(y_timestep+10):, :])
        
        #######################################################
        acc = mse(pred_y, test_y)
        if is_train == True:
            writer.add_scalar('accuracy', acc/sample_q, test_idx)
            writer.close()
        print(acc)
            
print('Finish Test')