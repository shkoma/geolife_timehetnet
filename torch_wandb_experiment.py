import ast
import os
import numpy as np
import pandas as pd
import datetime
import wandb

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch_mlp import MLP
from torch_mlp_dataset import MlpDataset
from torch_segment_dataset import SegmentDataset
from data.geolife.convert_minmax_location import LocationPreprocessor

from args            import argument_parser
from torch_time_het import TimeHetNet
from torch_hetnet import HetNet

def mlp_metricMenhattan(y_true, y_pred):
    row = torch.abs(y_pred[:,:,0] - y_true[:,:,0])
    col = torch.abs(y_pred[:,:,1] - y_true[:,:,1])
    return torch.sum(row + col)

def metricMenhattan(y_true, y_pred):
    row = torch.abs(y_pred[:,:,:,0] - y_true[:,:,:,0])
    col = torch.abs(y_pred[:,:,:,1] - y_true[:,:,:,1])
    return torch.sum(row + col)

def metricEuclidean(y_true, y_pred):
    row = (y_pred[:,1] - y_true[:,1])**2
    col = (y_pred[:,2] - y_true[:,2])**2
    return torch.mean((row+col)**0.5)

def make_Tensorboard_dir(dir_name, dir_format):
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime(dir_format)
    return os.path.join(root_logdir, sub_dir_name)

##### CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
##################################################

##### args
args = argument_parser()

model_type = 'mlp'
model_type = 'time-hetnet'
# model_type = 'hetnet'

user_list_type = 'single'

loss_method = 'euclidean'
loss_method = 'mse'

hidden_layer = 5
cell = 256

file_mode = 'min'

# min
round_min = 60 # 60, 120, 180
day = int(24/int(round_min/60)) #8#24 # 6*24
day_divide = day

# sec
round_sec = 30 # (seconds) per 10s
min_length = 2
time_delta = 20 # (minutes) 1 segment length
# length = min_length * time_delta

how_many = 3
length = 20 * min_length#day * how_many
y_timestep = 5 * min_length

x_attribute = 9
label_attribute = 2

sample_s = 4
sample_q = 1

batch_size = 150

args_early_stopping = True
args_epoch = 1500000
args_lr = 0.0015

# be careful to control args_patience, it can be stucked in a local minimum point.
args_patience = 1500000

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
dir_format = '[' + model_type + '_' + user_list_type + ']_%Y%m%d-%H%M%S'
configuration_file = 'configuration.csv'

writer_dir = make_Tensorboard_dir(writer_dir_name, dir_format)
##################################################

##### Train Phase
is_train = True
best_model_path = 'best_model.pth'
best_train_model = 'best_train_model.pth'
##################################################

##### Time grid User list
time_grid_csv = 'data/geolife/time_grid_sample.csv'
user_df = pd.read_csv(time_grid_csv)
ratio_var = 'ratio_' + str(round_min) + 'min'
user_df = user_df.loc[user_df[ratio_var] > 10, :]
locationPreprocessor = LocationPreprocessor('data/geolife/')
user_list = []
for user in user_df['user_id'].to_list():
    user_list += [locationPreprocessor.getUserId(user)]
print(f"user_list: {user_list}")
##################################################

##### Train - Validation user list
train_len       = (int)(len(user_list) * train_size)
validation_len  = (int)(len(user_list) * validation_size)

train_list      = user_list[0:train_len]
validation_list = user_list[train_len:(train_len + validation_len)]
test_list       = user_list[(train_len + validation_len):]

train_list = user_list[2:3]
validation_list = user_list[2:3]
test_list       = user_list[2:3]
##################################################

config = {
    'model_type':model_type,
    'user_list_type':user_list_type,
    'epochs':args_epoch,
    'learning_rate':args_lr,
    'batch_size':batch_size,
    'sample_size':sample_s,
    'day':day,
    'day_divide':day_divide,
    'y_timestep':y_timestep,
    'length':length,
    'writer_dir':writer_dir,
    'train_list':train_list,
    'test_list':test_list
}

def write_configruation(conf_file):
    #--------Write Configration--------
    import pandas as pd
    conf_df = pd.DataFrame({'device':[device],
                            'user_list_type':[user_list_type],
                            'model_type':[model_type],
                            'args_dims':[ast.literal_eval(args.dims)],
                            'round_min': [round_min],
                            'round_sec': [round_sec],
                            'day': [day],
                            'day_divide': [day_divide],
                            'length': [length],
                            'y_timestep': [y_timestep],
                            'loss_method':[loss_method],
                            'batch_size':[batch_size],
                            'hidden_layer':[hidden_layer],
                            'cell':[cell],
                            'label_attribute':[label_attribute],
                            'sample_s':[sample_s],
                            'sample_q':[sample_q],
                            'epoch':[args_epoch],
                            'patience':[args_patience],
                            'x_attribute':[x_attribute],
                            'file_mode':[file_mode],
                            'time_delta':[time_delta],
                            'train_list':[train_list],
                            'val_list':[validation_list],
                            'test_list':[test_list],
                            'train_columns':[training_data.get_train_columns()]})
    conf_df.to_csv(conf_file, index=False)

print("##################################################################")
print(f"use_cuda: {use_cuda}, device: {device}")
print(f"model_type: {model_type}")

print(f"train len: {train_len}")
print(f"validation len: {validation_len}")
print(f"test len: {len(test_list)}")

print(f"train: [{train_list}]")
print(f"validation: [{validation_list}]")
print(f"test: [{test_list}]")

print("Building Network ...")

# Dataset
if model_type == 'mlp':
    training_data           = MlpDataset('train', data_dir, train_list, y_timestep, day, day_divide, round_min, round_sec, label_attribute, length, device)
    validation_data         = MlpDataset('test', data_dir, validation_list, y_timestep, day, day_divide, round_min, round_sec, label_attribute, length, device)
    test_data               = MlpDataset('test', data_dir, test_list, y_timestep, day, day_divide, round_min, round_sec, label_attribute, length, device)
    train_dataloader        = DataLoader(training_data, batch_size, shuffle=False)
    validation_dataloader   = DataLoader(validation_data, batch_size, shuffle=False)
    test_dataloader         = DataLoader(test_data, batch_size, shuffle=False)
else:
    # training_data           = SegmentDataset('train', user_list_type, data_dir, train_list, device, day, day_divide, round_min, round_sec, y_timestep, length, label_attribute, sample_s, sample_q)
    # validation_data         = SegmentDataset('test', user_list_type, data_dir, validation_list, device, day, day_divide, round_min, round_sec, y_timestep, length, label_attribute, sample_s, sample_q)
    # test_data               = SegmentDataset('test', user_list_type, data_dir, test_list, device, day, day_divide, round_min, round_sec, y_timestep, length, label_attribute, sample_s, sample_q)
    from torch_trajectory_dataset import TrajectoryDataset
    train_list = ['035']
    validation_list = ['035']
    test_list       = ['035']
    training_data = TrajectoryDataset(data_mode='train',
                                        user_list_type=user_list_type, 
                                        data_dir=data_dir, 
                                        user_list=train_list, 
                                        device=device, 
                                        round_sec=round_sec, 
                                        y_timestep=y_timestep, 
                                        length=length, 
                                        label_attribute=label_attribute, 
                                        sample_s=sample_s, 
                                        sample_q=sample_q)
    validation_data = TrajectoryDataset(data_mode='test',
                                        user_list_type=user_list_type, 
                                        data_dir=data_dir, 
                                        user_list=validation_list, 
                                        device=device, 
                                        round_sec=round_sec, 
                                        y_timestep=y_timestep, 
                                        length=length, 
                                        label_attribute=label_attribute, 
                                        sample_s=sample_s, 
                                        sample_q=sample_q)
    test_data = TrajectoryDataset(data_mode='test',
                                        user_list_type=user_list_type, 
                                        data_dir=data_dir, 
                                        user_list=test_list, 
                                        device=device, 
                                        round_sec=round_sec, 
                                        y_timestep=y_timestep, 
                                        length=length, 
                                        label_attribute=label_attribute, 
                                        sample_s=sample_s, 
                                        sample_q=sample_q)
    train_dataloader        = DataLoader(training_data, batch_size, shuffle=False)
    validation_dataloader   = DataLoader(validation_data, batch_size, shuffle=False)
    test_dataloader         = DataLoader(test_data, batch_size, shuffle=False)

if is_train == True:
    print(f"train_len: {len(train_dataloader.dataset)}")
    print(f"test_len: {len(test_dataloader.dataset)}")
    print('Start Train')

    #------- Init Wandb -------
    # wandb.init(project='geolife_timehetnet', config=config)

    #--------Define Tensorboard--------
    # writer_dir = make_Tensorboard_dir(writer_dir_name, dir_format)
    writer = SummaryWriter(writer_dir)

    best_model_path = writer_dir + "/" + best_model_path
    best_train_model = writer_dir + "/" + best_train_model

    #--------Define a Model
    if model_type == 'mlp':
        model = MLP(input_shape=[length, x_attribute], y_timestep = y_timestep, loss_fn=loss_method, label_attribute=label_attribute, cell=cell, hidden_layer=hidden_layer)
        test_model = MLP(input_shape=[length, x_attribute], y_timestep = y_timestep, loss_fn=loss_method, label_attribute=label_attribute, cell=cell, hidden_layer=hidden_layer)

    elif model_type == 'hetnet':
        model = HetNet(dims = ast.literal_eval(args.dims),
                       output_shape=[y_timestep, label_attribute],
                       acti ='relu',
                       drop1 = 0.01,
                       drop2 = 0.01,
                       share_qs = False)

        print("Using Hetnet")
    elif model_type == 'time-hetnet':
        model = TimeHetNet(dims_inf = ast.literal_eval(args.dims),
                           dims_pred = ast.literal_eval(args.dims_pred),
                           activation="relu",
                           time=args.tmax_length,
                           batchnorm=False,
                           block = args.block.split(","),
                           output_shape=[y_timestep, label_attribute],
                           length = length)
        test_model = TimeHetNet(dims_inf = ast.literal_eval(args.dims),
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
    if loss_method == 'euclidean':
        criterion = metricEuclidean
    elif loss_method == 'menhattan':
        criterion = metricMenhattan
    else:
        criterion = nn.MSELoss()

    test_criterion = metricEuclidean
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

            if model_type == 'mlp':
                output = torch.concat([torch.unsqueeze(task_y[:,:,0], -1), output], axis=-1)
                loss = criterion(task_y[task_y[:, :, 0] == 1], output[output[:, :, 0] == 1])
            else:
                mask, y_true = task_y
                output = torch.cat([mask[:, :, :].unsqueeze(-1), output], axis=-1)
                y_true = torch.cat([mask[:, :, :].unsqueeze(-1), y_true], axis=-1)
                loss = criterion(y_true[y_true[:, :, :, 0] > 0.5], output[output[:, :, :, 0] > 0.5])

            loss_train += loss.item()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 1:
            # writer.add_scalar('training loss', loss_train, epoch)
            wandb.log({'training loss':loss_train}, step=epoch)

        model.eval()
        if epoch == 0:
            write_configruation(writer_dir + "/" + configuration_file)

        loss_val = 0.0
        with torch.no_grad():
            for val_idx, val_data in enumerate(validation_dataloader, 0):
                X_val, y_val = val_data
                output = model(X_val)

                if model_type == 'mlp':
                    output = torch.concat([torch.unsqueeze(y_val[:, :, 0], -1), output], axis=-1)
                    val_loss = test_criterion(y_val[y_val[:, :, 0] == 1], output[output[:, :, 0] == 1])
                else:
                    mask, y_true = y_val
                    output = torch.cat([mask[:, :, :].unsqueeze(-1), output], axis=-1)
                    y_true = torch.cat([mask[:, :, :].unsqueeze(-1), y_true], axis=-1)
                    val_loss = test_criterion(y_true[y_true[:, :, :, 0] > 0.5], output[output[:, :, :, 0] > 0.5])

                loss_val += val_loss.item()

            if epoch % 100 == 1:
                # writer.add_scalar('validation loss', loss_val, epoch)
                wandb.log({'validation loss':loss_val}, step=epoch)
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

        # Test mode
        if epoch % 100 == 1:
            test_model.load_state_dict(torch.load(best_model_path))
            test_model = test_model.to(device)
            test_model.eval()

            with torch.no_grad():
                loss_test = 0.0
                for idx, test_data in enumerate(test_dataloader, 0):
                    test_X, test_y = test_data
                    output = test_model(test_X)

                    if model_type == 'mlp':
                        output = torch.concat([torch.unsqueeze(test_y[:, :, 0], -1), output], axis=-1)
                        dist = test_criterion(test_y[test_y[:, :, 0] == 1], output[output[:, :, 0] == 1])
                    else:
                        mask, y_true = test_y
                        output = torch.cat([mask[:, :, :].unsqueeze(-1), output], axis=-1)
                        y_true = torch.cat([mask[:, :, :].unsqueeze(-1), y_true], axis=-1)
                        dist = test_criterion(y_true[y_true[:, :, :, 0] > 0.5], output[output[:, :, :, 0] > 0.5])

                    loss_test += dist
                    wandb.log({'Euclidean distance': loss_test}, step=epoch)
                    # writer.add_scalar('Euclidean distance', loss_test, epoch)
                    print(f"test loss: {loss_test}")
print('Finish Train')
