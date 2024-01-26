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
    res = nn.mse(y_true, y_pred)
    return torch.std(res, unbiased=False)

def metricMSE(y_true, y_pred):
    res = mse(y_true, y_pred)
    return torch.mean(res)


def make_Tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
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
sample_s = 5
sample_q = 5

args_epoch = 20
args_patience = 10

y_timestep = 5 # must be less than length
length = 100
gap = 10 # 5mins

train_size = 0.2
validation_size = 0.1
batch_size = 1

is_train = True

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
                        batchnorm=args.batchnorm,
                        block = args.block.split(","),
                        output_shape=[y_timestep, 2],
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

model = model.to(torch.float)

#--------Load the data----------------
# train_gen, val_gen, _, ds_names = getGens(args, model_type)

# user_list = os.listdir(data_dir)
locationPreprocessor = LocationPreprocessor('data/geolife/')
user_df = locationPreprocessor.get_valid_user_list()
user_list = []
for user in user_df['valid_user_list']:
    user_list += [locationPreprocessor.getUserId(user)]

random.shuffle(user_list)

train_len       = (int)(len(user_list) * train_size)
validation_len  = (int)(len(user_list) * validation_size)

train_list      = user_list[:train_len]
validation_list = user_list[train_len:(train_len + validation_len)]
test_list       = user_list[(train_len + validation_len):(train_len + validation_len + 10)]
# test_list       = user_list[(train_len + validation_len):]
print(f"train_list:      {train_list}")
print(f"validation_list: {validation_list}")
print(f"test_list:       {test_list}")

training_data   = GeoLifeDataSet(data_dir, train_list, sample_s, sample_q, length, y_timestep, gap)
validation_data = GeoLifeDataSet(data_dir, validation_list, sample_s, sample_q, length, y_timestep, gap)
test_data       = GeoLifeDataSet(data_dir, test_list, sample_s, sample_q, length, y_timestep, gap)

train_dataloader      = DataLoader(training_data, batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_data, batch_size, shuffle=False)
test_dataloader       = DataLoader(test_data, batch_size, shuffle=False)

randomNumber = int(np.random.rand(1) * 10000000)
print("-------ID Number is:", randomNumber)
# best_model_path = 'best_model/' + str(randomNumber) + '.pth'
best_model_path = 'best_model.pth'

if is_train == True:
    print('Start Train')
    #--------Define Tensorboard--------
    writer_dir = make_Tensorboard_dir(writer_dir_name)
    writer = SummaryWriter(writer_dir)

    best_model_path = writer_dir + "/" + best_model_path

    #--------Define Losses annd metrics----------------
    loss_object = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # https://sanghyu.tistory.com/87

    #--------Define Callbacks----------------
    lr_scheduler = None  #callback
    if args.early_stopping:
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args_patience, verbose=True)

    #------- Train the model -----------------
    best_val_loss = float("inf")
    no_improvement = 0

    # http://www.gisdeveloper.co.kr/?p=8615 - loss 값 확인 블로그
    loss_train_list = []
    loss_val_list = []
    
    # for epoch in range(args.num_epochs):
    for epoch in range(args_epoch):
        print(f"epoch: {epoch}")
        
        model.train()
        loss_train = 0.0
        for idx, train_data in enumerate(train_dataloader, 0):
            X_train, y_train = train_data
            if len(X_train) < 2:
                continue
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = loss_object(outputs, y_train)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            if idx % 10 == 9:
                writer.add_scalar('training loss',
                                  loss_train / 10,
                                  epoch * len(train_dataloader) + idx)
        loss_train_list.append(loss_train)
        
        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for idx, val_data in enumerate(validation_dataloader, 0):
                X_val, y_val = val_data
                if len(X_val) < 2:
                    continue
                val_outputs = model(X_val)
                val_loss = loss_object(val_outputs, y_val)
                loss_val += val_loss.item()
                if idx % 10 == 9:
                    writer.add_scalar('validation loss',
                                      loss_val / 10,
                                      epoch * len(validation_dataloader) + idx)
            loss_val_list.append(loss_val)
            print(f"train loss: {loss_train}, validation loss: {loss_val}")
        
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            no_improvement = 0
            print('save best weight and bias')
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improvement += 1
        
        if no_improvement == args_patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break
print('Finish Train')
#------- Test the model -----------------
print('Start Test')
mse = nn.MSELoss()
best_model = TimeHetNet(dims_inf = ast.literal_eval(args.dims),
                        dims_pred = ast.literal_eval(args.dims_pred), 
                        activation="relu", 
                        time=args.tmax_length,
                        batchnorm=args.batchnorm, 
                        block = args.block.split(","),
                        output_shape=[y_timestep, 2],
                        length=length)
print(f'best_model_path: {best_model_path}')
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()

class_pred_y = []
class_test_y = []
que_x_list = []

for idx, test_data in enumerate(test_dataloader, 0):
    test_X, test_y = test_data
    if len(test_X) < 2:
        continue
    que_x, _, _ = test_X
    pred_y = best_model(test_X)
    acc = mse(pred_y, test_y)
    writer.add_scalar('accuracy', acc, test_list[idx])
    print(acc)

writer.close()
print('Finish Test')