import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class SegmentDataset(Dataset):
    def __init__(self, model_type, data_dir, user_list, device, day, round_min, round_sec, time_delta, y_timestep, length, label_attribute, sample_s, sample_q, file_mode='min'):
        self.model_type = model_type
        self.data_dir = data_dir
        self.user_list = user_list
        self.device = device
        self.day = day
        self.round_min = round_min
        self.round_sec = round_sec
        self.time_delta = time_delta
        self.y_timestep = y_timestep
        self.length = length
        self.label_attribute = label_attribute
        self.sample_s = sample_s
        self.sample_q = sample_q
        self.file_mode = file_mode
        self.columns = []
        self.full_user_data_list = [] # user together, min mode

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
        return

    def sampleMinSet(self, dataset):
        user_df = dataset.copy()
        total_samps = self.sample_s + self.sample_q

        grid_mins = np.arange(user_df.shape[0] - self.length)
        grid_mins = grid_mins[0:(grid_mins.shape[0]//(total_samps*self.length)) * (total_samps*self.length)]
        grid_mins = grid_mins.reshape(-1, total_samps)
        sample_list = np.arange(grid_mins.shape[0])

        random.shuffle(sample_list)
        # print(f"day: {self.day}, grid_days: {grid_days.shape}, sample's len: {sample_list.shape}")

        mini_batch = []
        count = 0
        for row_idx in sample_list:
            for col_idx in grid_mins[row_idx,:]:
                count += 1
                # index = row_idx*total_samps + col_idx
                cur_sample = user_df.iloc[col_idx:col_idx + self.length, :]
                if cur_sample.shape[0] != self.length:
                    print(f'cur_sample is wrong')
                    continue
                mini_batch.append(cur_sample)

            if len(mini_batch) == total_samps:
                self.full_user_data_list.append(np.array(mini_batch))
                mini_batch = []
        return

    def sampleDaySet(self, dataset):
        user_df = dataset.copy()
        total_samps = self.sample_s + self.sample_q

        grid_days = np.arange(int(user_df.shape[0] / self.day) - int(self.length/self.day))
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
                    # print(f'cur_sample is wrong')
                    return
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