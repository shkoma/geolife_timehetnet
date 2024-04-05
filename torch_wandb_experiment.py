import ast
import os
import pandas as pd
import datetime
import wandb

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch_args import ArgumentSet, ArgumentMask, HetnetMask
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
device = torch.device("cuda:1" if use_cuda else "cpu")
##################################################

##### args
args = argument_parser()

model_type = 'mlp'
# model_type = 'time-hetnet'
# model_type = 'hetnet'
is_mask = True
# is_mask = False

user_list_type = 'single'
# user_list_type = 'multi'

loss_method = 'euclidean'
loss_method = 'mse'

file_mode = 'min'
if model_type == "mlp":
    x_attribute = 9
    length = ArgumentMask.total_day * ArgumentMask.time_stamp
    y_timestep = ArgumentMask.output_day * ArgumentMask.time_stamp
    hidden_layer = 7
    cell = 256

else: #time-hetnet
    if is_mask == True:
        sample_s = HetnetMask.sample_s
        sample_q = HetnetMask.sample_q
        length = HetnetMask.length
        y_timestep = HetnetMask.y_timestep
    else:
        sample_s = ArgumentSet.sample_s
        sample_q = ArgumentSet.sample_q
        length = ArgumentSet.length
        y_timestep = ArgumentSet.y_timestep

label_attribute = 2
batch_size = ArgumentSet.batch_size

args_early_stopping = True
args_epoch = 1500000
args_lr = 0.001

# be careful to control args_patience, it can be stucked in a local minimum point.
args_patience = 1500000

args_factor = 0.1

train_size = 0.9

from data.geolife.gps_grid_map_creator import GPSGridMapCreator
grid_len = 1
min_max_location = pd.read_csv('min_max_location.csv')
mapCreator = GPSGridMapCreator(grid_len) # 1m grid
mapCreator.create_grid_map(min_lat=min_max_location['min_lat'][0],
                           min_lon=min_max_location['min_lon'][0],
                           max_lat=min_max_location['max_lat'][0],
                           max_lon=min_max_location['max_lon'][0])
print(f"grid_map created")
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
writer = SummaryWriter(writer_dir)
##################################################

##### Train Phase
is_train = True
best_model_path = 'best_model.pth'
best_train_model = 'best_train_model.pth'
##################################################

##### Time grid User list
# time_grid_csv = 'data/geolife/time_grid_sample.csv'
if is_mask == True:
    time_grid_csv = 'mask_user_list.csv'
else:
    time_grid_csv = 'walk_speed_sample_user_list.csv'
user_df = pd.read_csv(time_grid_csv)
locationPreprocessor = LocationPreprocessor('data/geolife/')
user_list = []
for user in user_df['user_id'].to_list():
    user_list += [locationPreprocessor.getUserId(user)]
print(f"user_list: {user_list}")
##################################################

##### Train - Validation user list
train_len       = (int)(len(user_list) * train_size)

train_list      = user_list[0:train_len]
test_list       = user_list[train_len:]
#
# train_list = user_list#[:-num]
# validation_list = user_list#[:-num]
# test_list = user_list#[:-num]

# train_list = ['035']
# validation_list = ['035']
# test_list = ['035']
#
train_list = ['004']
validation_list = ['004']
test_list = ['004']
##################################################
def write_configruation(conf_file):
    #--------Write Configration--------
    import pandas as pd
    if model_type == 'mlp':
        conf_df = pd.DataFrame({'device': [device],
                                'user_list_type': [user_list_type],
                                'model_type': [model_type],
                                'random': [ArgumentSet.random],
                                'length': [length],
                                'y_timestep': [y_timestep],
                                'loss_method': [loss_method],
                                'batch_size': [batch_size],
                                'hidden_layer': [hidden_layer],
                                'cell': [cell],
                                'label_attribute': [label_attribute],
                                'epoch': [args_epoch],
                                'patience': [args_patience],
                                'x_attribute': [x_attribute],
                                'file_mode': [file_mode],
                                'train_list': [train_list],
                                'test_list': [test_list],
                                'write_dir': [writer_dir],
                                'train_columns': [training_data.get_train_columns()]})
    else:
        conf_df = pd.DataFrame({'device': [device],
                                'user_list_type': [user_list_type],
                                'model_type': [model_type],
                                'is_mask':[is_mask],
                                'random':[ArgumentSet.random],
                                'args_dims': [ast.literal_eval(args.dims)],
                                'length': [length],
                                'y_timestep': [y_timestep],
                                'loss_method': [loss_method],
                                'batch_size': [batch_size],
                                'label_attribute': [label_attribute],
                                'sample_s': [sample_s],
                                'sample_q': [sample_q],
                                'epoch': [args_epoch],
                                'patience': [args_patience],
                                'file_mode': [file_mode],
                                'train_list': [train_list],
                                'test_list': [test_list],
                                'write_dir': [writer_dir],
                                'train_columns': [training_data.get_train_columns()]})

    conf_df.to_csv(conf_file, index=False)

print("##################################################################")
print(f"use_cuda: {use_cuda}, device: {device}")
print(f"model_type: {model_type}")

print(f"train: [{train_list}]")
print(f"test: [{test_list}]")

print("Building Network ...")

# Dataset
if model_type == 'mlp':
    training_data           = MlpDataset(data_mode='train',
                                         data_dir=data_dir,
                                         writer_dir=writer_dir,
                                         user_list=train_list,
                                         label_attribute=label_attribute,
                                         device=device)
    test_data               = MlpDataset(data_mode='test',
                                         data_dir=data_dir,
                                         writer_dir=writer_dir,
                                         user_list=test_list,
                                         label_attribute=label_attribute,
                                         device=device)
    train_dataloader        = DataLoader(training_data, batch_size, shuffle=False)
    test_dataloader         = DataLoader(test_data, batch_size, shuffle=False)
else:
    # training_data           = SegmentDataset('train', user_list_type, data_dir, train_list, device, day, day_divide, round_min, round_sec, y_timestep, length, label_attribute, sample_s, sample_q)
    # test_data               = SegmentDataset('test', user_list_type, data_dir, test_list, device, day, day_divide, round_min, round_sec, y_timestep, length, label_attribute, sample_s, sample_q)
    from torch_trajectory_dataset import TrajectoryDataset

    training_data = TrajectoryDataset(data_mode='train',
                                        user_list_type=user_list_type, 
                                        data_dir=data_dir,
                                        writer_dir=writer_dir,
                                        user_list=train_list, 
                                        device=device,
                                        y_timestep=y_timestep, 
                                        length=length, 
                                        label_attribute=label_attribute, 
                                        sample_s=sample_s, 
                                        sample_q=sample_q,
                                        is_mask=is_mask)
    test_data = TrajectoryDataset(data_mode='test',
                                        user_list_type=user_list_type, 
                                        data_dir=data_dir,
                                        writer_dir=writer_dir,
                                        user_list=test_list, 
                                        device=device,
                                        y_timestep=y_timestep, 
                                        length=length, 
                                        label_attribute=label_attribute, 
                                        sample_s=sample_s, 
                                        sample_q=sample_q,
                                        is_mask=is_mask)
    train_dataloader        = DataLoader(training_data, batch_size, shuffle=False)
    test_dataloader         = DataLoader(test_data, batch_size, shuffle=False)

if is_train == True:
    if model_type == 'mlp':
        config = {
            'device': device,
            'model_type': model_type,
            'hidden_layer': hidden_layer,
            'cell': cell,
            'user_list_type': user_list_type,
            'epochs': args_epoch,
            'learning_rate': args_lr,
            'batch_size': batch_size,
            'input': ArgumentMask.input_day,
            'output': ArgumentMask.output_day,
            'time_stamp': ArgumentMask.time_stamp,
            'round_min': ArgumentMask.round_min,
            'start_time': ArgumentMask.start_time,
            'finish_time': ArgumentMask.finish_time,
            'train_list': train_list,
            'test_list': test_list,
            'train_len': len(train_dataloader.dataset),
            'test_len': len(test_dataloader.dataset),
            'write_dir': writer_dir,
            'random': ArgumentMask.random
        }
    else:
        config = {
            'device': device,
            'model_type': model_type,
            'is_mask': is_mask,
            'user_list_type': user_list_type,
            'epochs': args_epoch,
            'learning_rate': args_lr,
            'batch_size': batch_size,
            'length':length,
            'y_timestep':y_timestep,
            'support_set': sample_s,
            'train_list': train_list,
            'test_list': test_list,
            'train_len':len(train_dataloader.dataset),
            'test_len': len(test_dataloader.dataset),
            'write_dir': writer_dir,
            'random': ArgumentSet.random
        }

    print(f"train_len: {len(train_dataloader.dataset)}")
    print(f"test_len: {len(test_dataloader.dataset)}")
    print('Start Train')

    #------- Init Wandb -------
    wandb.init(project='geolife_timehetnet', config=config)

    #--------Define Tensorboard--------
    # writer_dir = make_Tensorboard_dir(writer_dir_name, dir_format)
    # writer = SummaryWriter(writer_dir)

    best_model_path = writer_dir + "/" + best_model_path
    best_train_model = writer_dir + "/" + best_train_model

    #--------Define a Model
    if model_type == 'mlp':
        model = MLP(input_shape=[ArgumentMask.total_day * ArgumentMask.time_stamp, training_data.get_x_attibutes()],
                    loss_fn=loss_method,
                    label_attribute=label_attribute,
                    cell=cell, hidden_layer=hidden_layer)
        test_model = MLP(input_shape=[ArgumentMask.total_day * ArgumentMask.time_stamp, test_data.get_x_attibutes()],
                         loss_fn=loss_method,
                         label_attribute=label_attribute,
                         cell=cell, hidden_layer=hidden_layer)

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
            nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

        if epoch % 100 == 1:
            wandb.log({'training loss':loss_train}, step=epoch)

        model.eval()
        if epoch == 0:
            write_configruation(writer_dir + "/" + configuration_file)

        loss_val = 0.0
        with torch.no_grad():
            for val_idx, val_data in enumerate(test_dataloader, 0):
                X_val, y_val = val_data
                output = model(X_val)

                if model_type == 'mlp':
                    # # reverse normalization
                    # output[:, :, 0] = output[:, :, 0] * mapCreator.get_num_lat()
                    # output[:, :, 1] = output[:, :, 1] * mapCreator.get_num_lon()
                    # y_val[:, :, 1] = y_val[:, :, 1] * mapCreator.get_num_lat()
                    # y_val[:, :, 2] = y_val[:, :, 2] * mapCreator.get_num_lon()

                    output = torch.concat([torch.unsqueeze(y_val[:, :, 0], -1), output], axis=-1)
                    val_loss = test_criterion(y_val[y_val[:, :, 0] == 1], output[output[:, :, 0] == 1])
                else:
                    mask, y_true = y_val
                    # # reverse normalization
                    # output[:, :, :, 0] = output[:, :, :, 0] * mapCreator.get_num_lat()
                    # output[:, :, :, 1] = output[:, :, :, 1] * mapCreator.get_num_lon()
                    # y_true[:, :, :, 0] = y_true[:, :, :, 0] * mapCreator.get_num_lat()
                    # y_true[:, :, :, 1] = y_true[:, :, :, 1] * mapCreator.get_num_lon()

                    output = torch.cat([mask[:, :, :].unsqueeze(-1), output], axis=-1)
                    y_true = torch.cat([mask[:, :, :].unsqueeze(-1), y_true], axis=-1)
                    val_loss = test_criterion(y_true[y_true[:, :, :, 0] > 0.5], output[output[:, :, :, 0] > 0.5])

                loss_val += val_loss.item()

            if epoch % 100 == 1:
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
                        # reverse normalization
                        output[:, :, 0] = output[:, :, 0] * mapCreator.get_num_lat()
                        output[:, :, 1] = output[:, :, 1] * mapCreator.get_num_lon()
                        test_y[:, :, 1] = test_y[:, :, 1] * mapCreator.get_num_lat()
                        test_y[:, :, 2] = test_y[:, :, 2] * mapCreator.get_num_lon()

                        output = torch.concat([torch.unsqueeze(test_y[:, :, 0], -1), output], axis=-1)
                        dist = test_criterion(test_y[test_y[:, :, 0] == 1], output[output[:, :, 0] == 1])
                    else:
                        mask, y_true = test_y

                        # reverse normalization
                        output[:, :, :, 0] = output[:, :, :, 0] * mapCreator.get_num_lat()
                        output[:, :, :, 1] = output[:, :, :, 1] * mapCreator.get_num_lon()
                        y_true[:, :, :, 0] = y_true[:, :, :, 0] * mapCreator.get_num_lat()
                        y_true[:, :, :, 1] = y_true[:, :, :, 1] * mapCreator.get_num_lon()

                        output = torch.cat([mask[:, :, :].unsqueeze(-1), output], axis=-1)
                        y_true = torch.cat([mask[:, :, :].unsqueeze(-1), y_true], axis=-1)
                        dist = test_criterion(y_true[y_true[:, :, :, 0] > 0.5], output[output[:, :, :, 0] > 0.5])

                    loss_test += dist
                    wandb.log({'Euclidean distance': loss_test}, step=epoch)
                    print(f"test loss: {loss_test}")
print('Finish Train')
