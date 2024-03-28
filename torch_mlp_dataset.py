import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MlpDataset(Dataset):
    def __init__(self, data_mode, data_dir, user_list, y_timestep, day, day_divide, round_min, round_sec, label_attribute, length, device):
        self.data_mode = data_mode
        self.data_dir = data_dir
        self.user_list = user_list
        self.y_timestep = y_timestep
        self.label_attribute = label_attribute
        self.day = day
        self.day_divide = day_divide
        self.round_min = round_min
        self.round_sec = round_sec
        self.length = length
        self.device = device
        self.samples = []

        # round_min
        if self.data_mode == 'train':
            self.min_csv = '_train_' + str(self.round_min) + 'min.csv'
        elif self.data_mode == 'valid':
            self.min_csv = '_valid_' + str(self.round_min) + 'min.csv'
        elif self.data_mode == 'test':
            self.min_csv = '_test_' + str(self.round_min) + 'min.csv'
        else:
            self.min_csv = '_origin_grid_' + str(self.round_min) + 'min.csv'

        for user_id in self.user_list:
            self.min_file = str(self.data_dir) + str(user_id) + '/csv/' + str(user_id) + self.min_csv
            self.df = pd.read_csv(self.min_file)
            self.df_1 = self.df[self.df.columns[2:].to_list()].copy()
            self.df_1 = pd.concat([self.df_1, self.df.iloc[:, 0:2]], axis=1)  # x, y
            self.columns = self.df_1.columns.to_list()
            self.sampleMinSet(self.df_1)

        self.samples = np.array(self.samples)

    def get_train_columns(self):
        return self.columns

    def sampleMinSet(self, dataset):
        user_df = dataset.copy()

        mins = int(self.day // self.day_divide)
        days = np.arange(int(user_df.shape[0] / mins) - int(self.length))
        random.shuffle(days)

        for idx in days:
            index = idx * mins
            cur_sample = user_df.iloc[index:(index + self.length), :]
            self.samples.append(cur_sample)
        return

    def sampleDaySet(self, dataset):
        user_df = dataset.copy()

        days = np.arange(int(user_df.shape[0] / self.day) - int(self.length/self.day))
        random.shuffle(days)

        for idx in days:
            index = idx * self.day
            cur_sample = user_df.iloc[index:(index + self.length), :]
            self.samples.append(cur_sample)
        return

    def __getitem__(self, idx):
        task_X = self.samples[idx, :, :].copy()
        task_y = task_X[-self.y_timestep:, -(self.label_attribute+1):].copy()

        if self.y_timestep > 0:
            task_X[-self.y_timestep:, -(self.label_attribute+1):] = 0

        task_X = torch.tensor(task_X, dtype=torch.double).to(self.device)
        task_y = torch.tensor(task_y, dtype=torch.double).to(self.device)

        return task_X, task_y

    def __len__(self):
        # batch를 구성할 수 있는 총 수
        # 이 수에서 batch를 조정할 수 있다.
        # 몇 명의 user 로 나눠서 할 지
        return len(self.samples)