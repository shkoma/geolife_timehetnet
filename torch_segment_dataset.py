import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class SegmentDataset(Dataset):
    def __init__(self, data_mode, user_list_type, data_dir, user_list, device, day, day_divide, round_min, round_sec, y_timestep, length, label_attribute, sample_s, sample_q):
        self.data_mode = data_mode
        self.user_list_type = user_list_type
        self.data_dir = data_dir
        self.user_list = user_list
        self.device = device
        self.day = day
        self.day_divide = day_divide
        self.round_min = round_min
        self.round_sec = round_sec
        self.y_timestep = y_timestep
        self.length = length
        self.label_attribute = label_attribute
        self.sample_s = sample_s
        self.sample_q = sample_q
        self.columns = []
        self.full_user_data_list = [] # user together, min mode

        if self.user_list_type == 'single':
            if self.data_mode == 'train':
                self.min_csv = '_train_' + str(self.round_min) + 'min.csv'
            elif self.data_mode == 'valid':
                self.min_csv = '_valid_' + str(self.round_min) + 'min.csv'
            elif self.data_mode == 'test':
                self.min_csv = '_test_' + str(self.round_min) + 'min.csv'
        else:
            # round_min
            self.min_csv = '_origin_grid_' + str(self.round_min) + 'min.csv'
        self.loadUserData()

    def get_train_columns(self):
        return self.columns

    def loadUserData(self):
        for user_id in self.user_list:
            # min
            min_file = str(self.data_dir) + str(user_id) + '/csv/' + str(user_id) + self.min_csv
            df = pd.read_csv(min_file)

            df_1 = df[df.columns[2:].to_list()].copy()
            df_1 = pd.concat([df_1, df.iloc[:, 0:2]], axis=1)

            self.columns = df_1.columns.to_list()
            self.sampleMinSet(df_1)
            # self.sampleDaySet(df_1)
        return

    def sampleFullSet(self, dataset):
        user_df = dataset.copy()
        total_samps = self.sample_s + self.sample_q

        grid = np.arange((user_df.shape[0]//self.length)//total_samps * total_samps)
        grid = grid.reshape[-1, total_samps]
        sample_list = np.arange(grid.shape[0])

        random.shuffle(sample_list)

        mini_batch = []
        count = 0
        for row_idx in sample_list:
            for col_idx in grid[row_idx,:]:
                count += 1
                index = col_idx * self.length
                cur_sample = user_df.iloc[index:index + self.length, :]
                if cur_sample.shape[0] != self.length:
                    continue
                mini_batch.append(cur_sample)

            if len(mini_batch) == total_samps:
                self.full_user_data_list.append(np.array(mini_batch))
                mini_batch = []
        return

    def sampleMinSet(self, dataset):
        user_df = dataset.copy()
        total_samps = self.sample_s + self.sample_q

        mins = int(self.day // self.day_divide)
        grid_mins = np.arange(int(user_df.shape[0]/mins) - int(self.length/mins * total_samps))
        grid_mins = grid_mins[:grid_mins.shape[0]//total_samps * total_samps]
        grid_mins = grid_mins.reshape(-1, total_samps)
        sample_list = np.arange(grid_mins.shape[0])

        random.shuffle(sample_list)

        mini_batch = []
        count = 0
        for row_idx in sample_list:
            for col_idx in grid_mins[row_idx,:]:
                count += 1
                index = col_idx * mins
                cur_sample = user_df.iloc[index:index + self.length, :]
                if cur_sample.shape[0] != self.length:
                    continue
                mini_batch.append(cur_sample)

            if len(mini_batch) == total_samps:
                self.full_user_data_list.append(np.array(mini_batch))
                mini_batch = []
        return

    def sampleDaySet(self, dataset):
        user_df = dataset.copy()
        total_samps = self.sample_s + self.sample_q

        grid_days = np.arange(int(user_df.shape[0] / self.day) - int(self.length/self.day * total_samps))
        grid_days = grid_days[0:(grid_days.shape[0]//total_samps) * total_samps]
        grid_days = grid_days.reshape(-1, total_samps)
        sample_list = np.arange(grid_days.shape[0])

        random.shuffle(sample_list)

        mini_batch = []
        count = 0
        for row_idx in sample_list:
            for col_idx in grid_days[row_idx,:]:
                count += 1
                index = row_idx*total_samps + col_idx*self.day
                cur_sample = user_df.iloc[index:index + self.length, :]
                if cur_sample.shape[0] != self.length:
                    continue
                mini_batch.append(cur_sample)

            if len(mini_batch) == total_samps:
                self.full_user_data_list.append(np.array(mini_batch))
                mini_batch = []
        return

    def __getitem__(self, index):
        # masking y_timestpe in task_X
        task_X = self.full_user_data_list[index].copy()
        task_y = task_X[:, -self.y_timestep:, -(self.label_attribute+1):].copy()

        if self.y_timestep > 0:
            task_X[:, -self.y_timestep:, -(self.label_attribute+1):] = 0

        sup_x = np.array(task_X[:self.sample_s, :, :])
        sup_y = np.array(task_y[:self.sample_s, :, 1:])
        que_x = np.array(task_X[self.sample_s:, :, :])
        que_y = np.array(task_y[self.sample_s:, :, 1:])
        mask  = np.array(task_y[self.sample_s:, :, 0])

        sup_x = torch.from_numpy(sup_x).double().to(self.device)
        sup_y = torch.from_numpy(sup_y).double().to(self.device)
        que_x = torch.from_numpy(que_x).double().to(self.device)
        que_y = torch.from_numpy(que_y).double().to(self.device)
        mask  = torch.from_numpy(mask).to(self.device)

        task_X = (que_x, sup_x, sup_y)
        task_y = (mask, que_y)

        return task_X, task_y
    
    def __len__(self):
        # batch를 구성할 수 있는 총 수
        # 이 수에서 batch를 조정할 수 있다.
        # 몇 명의 user 로 나눠서 할 지
        return len(self.full_user_data_list)