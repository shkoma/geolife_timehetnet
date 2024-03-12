import pandas as pd
from torch_mlp import MLP
from torch_segment_dataset import SegmentDataset

writer_dir_name = 'data/geolife/runs'
log_dir_name = '[segment_time-hetnet]_20240312-053012'
log_folder = writer_dir_name + '/' + log_dir_name
best_model_path = log_folder + '/best_model.pth'
# best_model_path = log_folder + '/best_train_model.pth'

config_df = pd.read_csv(log_folder + '/configuration.csv')

data_type = 'segment'
model_type = 'mlp'
model_type = 'time-hetnet'

data_dir = "data/geolife/Data/"
label_attribute = 2

sample_s = config_df['sample_s'][0]
sample_q = config_df['sample_q'][0]

args_epoch = config_df['epoch'][0]
args_patience = config_df['patience'][0]

gap_min = 12 # 1 min
gap = gap_min

hidden_layer = config_df['hidden_layer'][0]
cell = 256#config_df['cell'][0]

loss_method = 'mse'
# loss_method = 'cross'

# day = config_df['day'][0]
# week = config_df['week'][0]
x_attribute = config_df['x_attribute'][0]
y_timestep = config_df['y_timestep'][0]
length = config_df['length'][0]

round_sec = 10 # (seconds) per 10s
time_delta = 20 # (minutes) 1 segment length

train_list = config_df['train_list'][0]
validation_list = config_df['val_list'][0]
test_list = config_df['test_list'][0]

train_size = 0.4
validation_size = 0.1
batch_size = config_df['batch_size'][0] # each user
batch_size = 3

device = config_df['device'][0]

## Test Phase
is_train = False

print(f"y_timestep: {y_timestep}, length: {length}")
print(f"data_type: {data_type}, model_type: {model_type}")

# Time-hetnet graph
# https://jimmy-ai.tistory.com/30

# TensorBoard 설정
import ast
from torch_time_het import TimeHetNet

import torch
from torch.utils.data import DataLoader
from torch_geolife_dateset import GeoLifeDataSet
from data.geolife.convert_minmax_location import LocationPreprocessor
import random

from torch_mlp import MLP
from torch_segment_dataset import SegmentDataset
from torch_mlp_dataset import MlpDataset

data_dir = "data/geolife/Data/"

train_list = config_df['train_list'][0]
validation_list = config_df['val_list'][0]
test_list = config_df['val_list'][0]  # , '078']

test_list = ['068']

print(f"train_list:      {train_list}")
print(f"validation_list: {validation_list}")
print(f"test_list:       {test_list}")

test_data = SegmentDataset(model_type, data_dir, test_list, device, round_sec, time_delta, y_timestep, length,
                           label_attribute, sample_s, sample_q)
test_dataloader = DataLoader(test_data, batch_size, shuffle=False)

best_model = TimeHetNet(dims_inf=ast.literal_eval("[64,64,64]"),
                        dims_pred=ast.literal_eval("[64,64,64]"),
                        activation="relu",
                        time=100,
                        batchnorm=False,
                        block=str("gru,conv,conv,gru").split(","),
                        output_shape=[y_timestep, 2],
                        length=length)

# best_model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class_probs = []
class_label = []
que_x_list = []


def metricMenhattan(y_true, y_pred):
    row = torch.abs(y_pred[:, :, :, 0] - y_true[:, :, :, 0])
    col = torch.abs(y_pred[:, :, :, 1] - y_true[:, :, :, 1])
    return row + col


def metricEuclidean(y_true, y_pred):
    row = (y_pred[:, :, :, 0] - y_true[:, :, :, 0]) ** 2
    col = (y_pred[:, :, :, 1] - y_true[:, :, :, 1]) ** 2
    result = (row + col) ** 0.5
    return result


with torch.no_grad():
    for idx, test_data in enumerate(test_dataloader, 0):
        test_X, test_y = test_data

        class_probs = []
        class_label = []
        que_x_list = []
        loss_list = []

        que_x, _, _ = test_X
        output = best_model(test_X)
        que_x_list.append(que_x[:, :, -y_timestep:, 0])

        class_probs.append(output)
        class_label.append(test_y)

        loss = metricEuclidean(test_y, output)
        print(torch.sum(loss))
        test_loss = torch.unsqueeze(loss, -1)

        test_probs = torch.cat(class_probs).to(device)
        test_label = torch.cat(class_label).to(device)
        test_que_x = torch.cat(que_x_list).to(device)

        # print(test_probs.shape)
        # print(test_label.shape)
        print(test_label[0].shape)
        # print(test_que_x.shape)
        # print(test_loss.shape)

        # if idx % 10 != 1:
        #     continue

        for user_idx in range(test_label.shape[0]):
            if user_idx != 2:
                continue
            for test_idx in range(test_label.shape[1]):
                df = pd.DataFrame(data=test_que_x[user_idx][0], columns=['Time'])
                df_norm = MinMaxScaler().fit_transform(df)

                plt.figure(figsize=(10, 2))
                plt.subplot(1, 3, 1)
                plt.title('Row graph (0 ~ 117)')
                # plt.ylim(-1, 2)
                # plt.axis([xmin, xmax, ymin, ymax])
                # plt.scatter(df_norm, test_label[user_idx][test_idx][:, 0] , color='r', alpha=0.5)
                # plt.scatter(df_norm, test_probs[user_idx][test_idx][:, 0], color='g', alpha=0.3)
                plt.plot(df_norm, test_label[user_idx][test_idx][:, 0], color='r', alpha=0.5)
                plt.plot(df_norm, test_probs[user_idx][test_idx][:, 0], color='g', alpha=0.3)
                plt.xlabel('time')
                plt.ylabel('grid_row')

                # fig_col = plt.figure(figsize=(5, 3))
                plt.subplot(1, 3, 2)
                plt.title('Column graph (0 ~ 339)')
                # plt.ylim(-1, 2)
                # plt.scatter(df_norm, test_label[user_idx][test_idx][:, 1] , color='r', alpha=0.5)
                # plt.scatter(df_norm, test_probs[user_idx][test_idx][:, 1], color='g', alpha=0.3)
                plt.plot(df_norm, test_label[user_idx][test_idx][:, 1], color='r', alpha=0.5)
                plt.plot(df_norm, test_probs[user_idx][test_idx][:, 1], color='g', alpha=0.3)
                plt.xlabel('time')
                plt.ylabel('grid_col')
                # plt.show()

                plt.subplot(1, 3, 3)
                # plt.axis([xmin, xmax, ymin, ymax])
                # plt.axis([0, 1, 0, 1])
                plt.ylim(0, 10)
                plt.title('Euclidean')
                plt.plot(df_norm, (test_loss[user_idx][test_idx][:, 0]), color='b', alpha=0.5)
                plt.xlabel('time')
                plt.ylabel('loss')
