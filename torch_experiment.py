import os
import gc
import numpy as np
import time
import datetime
import sys
import ast
import itertools

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
print("##################################################################")

model_type = 'gru'
print("Building Network ...")

model      = TimeHetNet(dims_inf = ast.literal_eval(args.dims),
                        dims_pred = ast.literal_eval(args.dims_pred), 
                        activation="relu", 
                        time=args.tmax_length,
                        batchnorm=args.batchnorm, 
                        block = args.block.split(","))

model_type = "TimeHetNet"
print("Using Time Hetnet")


# EncM = SEncModel(control=args.control_steps)
# # EncM = SEncModel(control=80)
# model = HetNet(EncM, "slice",
#                dims = ast.literal_eval(args.dims),
#                acti ='relu',
#                drop1 = 0.01,
#                drop2 = 0.01,
#                share_qs = False)
# print("Using Hetnet")

model = model.to(torch.float)

#--------Load the data----------------
train_gen, val_gen, _, ds_names = getGens(args, model_type)
randomNumber = int(np.random.rand(1)*10000000)
print("-------ID Number is:", randomNumber)

#--------Define Losses annd metrics----------------
loss_object = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# args.grad_clip will be used in the process of train.
# https://sanghyu.tistory.com/87

#--------Define Callbacks----------------
lr_scheduler = None  #callback
if args.early_stopping:
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args.patience, verbose=True)

#------- Train the model -----------------
# The example of train_gen
#   query set X:(5, 3, 3, 3)
# support set X:(5, 7, 3, 3)
# support set y:(5, 7, 1)
#   query set y:(5, 3, 1)
# query_set_X = train_gen[0][0]
# query_set_y = train_gen[1]
# support_set_X = train_gen[0][1]
# support_set_y = train_gen[0][2]
best_val_loss = float("inf")
no_improvement = 0

# X_train = train_gen[0] # (query_x, support_x, support_y)
# y_train = train_gen[1] # (query_y)

# http://www.gisdeveloper.co.kr/?p=8615 - loss 값 확인 블로그
loss_train_list = []
loss_val_list = []

train_step = 1
validation_step = 1

for epoch in range(args.num_epochs):
    print(f"epoch: {epoch}")
    
    loss_train = 0.0
    count = 0
    for _ in range(train_step):
        train_set = next(train_gen)
        X_train = train_set[0]
        y_train = torch.from_numpy(train_set[1]).float()

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
        for _ in range(validation_step):
            val_set = next(val_gen)
            X_val = val_set[0]
            y_val = torch.from_numpy(val_set[1]).float()
            val_outputs = model(X_val)
            val_loss = loss_object(val_outputs, y_val)
            loss_val += val_loss.item()
        loss_val_list.append(loss_val)
        print(f"train loss {loss_train}, validation loss: {loss_val}")
    
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

del(train_gen)
del(val_gen)
gc.collect()

#------- Test the model -----------------
mse = nn.MSELoss()
best_model = TimeHetNet(dims_inf = ast.literal_eval(args.dims),
                        dims_pred = ast.literal_eval(args.dims_pred), 
                        activation="relu", 
                        time=args.tmax_length,
                        batchnorm=args.batchnorm, 
                        block = args.block.split(","))
best_model.load_state_dict(torch.load('best_model.pth'))
ts_final = run_experiment(args, best_model, ds_names, mse)

for data in ts_final:
    print(data)

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