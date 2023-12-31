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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Custom libraries
from data_sampling.data_gen_ts    import Task_Sampler
from data_sampling.load_gens      import *
###########

from torch_time_model import SliceEncoderModel as SEncModel
from torch_hetnet import HetNet
from torch_time_het import TimeHetNet
from torch_experiment_test import run_experiment
from torch_geolife_dateset import GeoLifeDataSet

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
    res = nn.mse(y_true, y_pred)
    return torch.std(res, unbiased=False)

def metricMSE(y_true, y_pred):
    res = mse(y_true, y_pred)
    return torch.mean(res)

gc.enable()

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
sample_s = 5
sample_q = 5
length = 1000
y_timestep = 100
train_size = 0.2
validation_size = 0.1
batch_size = 1
print("##################################################################")

model_type = 'gru'
print("Building Network ...")

###################### TimeHetNet ######################
model      = TimeHetNet(dims_inf = ast.literal_eval(args.dims),
                        dims_pred = ast.literal_eval(args.dims_pred), 
                        activation="relu", 
                        time=args.tmax_length,
                        batchnorm=args.batchnorm, 
                        block = args.block.split(","),
                        output_shape=[y_timestep, 2])

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

model = model.to(torch.float)

#--------Load the data----------------
# train_gen, val_gen, _, ds_names = getGens(args, model_type)

user_list = os.listdir(data_dir)
random.shuffle(user_list)

train_len = (int)(len(user_list) * train_size)
validation_len = (int)(len(user_list) * validation_size)

train_list      = user_list[:train_len]
validation_list = user_list[train_len:(train_len + validation_len)]
# test_list       = user_list[(train_len + validation_len):]
test_list       = user_list[(train_len + validation_len):(train_len + validation_len + 3)]
print(f"train_list:      {train_list}")
print(f"validation_list: {validation_list}")
print(f"test_list:       {test_list}")

training_data   = GeoLifeDataSet(data_dir, train_list, sample_s, sample_q, length, y_timestep)
validation_data = GeoLifeDataSet(data_dir, validation_list, sample_s, sample_q, length, y_timestep)
test_data       = GeoLifeDataSet(data_dir, test_list, sample_s, sample_q, length, y_timestep)

train_dataloader      = DataLoader(training_data, batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_data, batch_size, shuffle=False)
test_dataloader       = DataLoader(test_data, batch_size, shuffle=False)

randomNumber = int(np.random.rand(1) * 10000000)
print("-------ID Number is:", randomNumber)

#--------Define Losses annd metrics----------------
loss_object = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# https://sanghyu.tistory.com/87

#--------Define Callbacks----------------
lr_scheduler = None  #callback
if args.early_stopping:
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args.patience, verbose=True)

#------- Train the model -----------------
best_val_loss = float("inf")
no_improvement = 0

# http://www.gisdeveloper.co.kr/?p=8615 - loss 값 확인 블로그
loss_train_list = []
loss_val_list = []

for epoch in range(args.num_epochs):
    print(f"epoch: {epoch}")
    
    loss_train = 0.0
    for X_train, y_train in train_dataloader:
        if len(X_train) < 2:
            continue
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_object(outputs, y_train)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    loss_train_list.append(loss_train)
    
    model.eval()
    loss_val = 0.0
    with torch.no_grad():
        for X_val, y_val in validation_dataloader:
            if len(X_val) < 2:
                continue
            val_outputs = model(X_val)
            val_loss = loss_object(val_outputs, y_val)
            loss_val += val_loss.item()
        loss_val_list.append(loss_val)
        print(f"train loss {loss_train}, validation loss: {loss_val}")
        # print(f"train loss {loss_train_list}") 
        # print(f"validation loss: {loss_val_list}")
    
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        no_improvement = 0
        print('save best weight and bias')
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improvement += 1
    
    if no_improvement == args.patience:
        print(f"Early stopping after {epoch+1} epochs.")
        break
    
    lr_scheduler.step(val_loss)

#------- Test the model -----------------
mse = nn.MSELoss()
best_model = TimeHetNet(dims_inf = ast.literal_eval(args.dims),
                        dims_pred = ast.literal_eval(args.dims_pred), 
                        activation="relu", 
                        time=args.tmax_length,
                        batchnorm=args.batchnorm, 
                        block = args.block.split(","),
                        output_shape=[y_timestep, 2])
best_model.load_state_dict(torch.load('best_model.pth'))

for test_X, test_y in test_dataloader:
    if len(test_X) < 2:
        continue
    pred_y = best_model(test_X)
    acc = mse(pred_y, test_y)
    print(acc)

# ts_final = run_experiment(args, best_model, ds_names, mse)

# for data in ts_final:
#     print(data)

# https://pytorch.org/docs/stable/tensorboard.html - Visualize training history
# https://lynnshin.tistory.com/54 - pytorch summary library - 모델 구조 파악

#------ Save results ---------------------
# key = args.key
# if key == "":
#     key = None

# result_path =   save_results(args        = args,
#                              history     = history,
#                              key         = key,
#                              ts_loss     = ts_final,
#                              randomnumber= randomNumber)

# os.system(f"mkdir -p {result_path}_model")
# if args.save_weights:
#     het_net.save_weights(f"{result_path}_model/model")