import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MlpDataset(Dataset):
    def __init__(self, data_dir, user_list, y_timestep, round_sec, time_delta, label_attribute, length, device):
        self.data_dir = data_dir
        self.user_list = user_list
        self.y_timestep = y_timestep
        self.label_attribute = label_attribute
        self.round_sec = round_sec
        self.time_delta = time_delta
        self.length = length
        self.device = device

        self.csv_file = '_origin_grid_' + str(self.round_sec) + 's.csv'
        self.user_file = str(self.data_dir) + str(self.user_list[0]) + '/csv/' + str(
            self.user_list[0]) + self.csv_file

        self.segment_csv = '_segment_list_' + str(self.time_delta) + 'min.csv'
        self.segment_file = str(self.data_dir) + str(self.user_list[0]) + '/csv/' + str(self.user_list[0]) + self.segment_csv

        self.label = pd.read_csv(self.user_file)
        self.label = self.label.drop(columns=['time_diff'])

        day_column = 10
        self.df_1 = self.label[self.label.columns[0:day_column].to_list()].copy()
        self.df_1 = pd.concat([self.df_1, self.label.iloc[:, day_column+4:day_column+6]], axis=1)  # 500m
        self.df_1 = pd.concat([self.df_1, self.label.iloc[:, day_column+2:day_column+4]], axis=1)  # 100m
        self.df_1 = pd.concat([self.df_1, self.label.iloc[:, day_column+6:day_column+8]], axis=1)  # 1000m
        self.columns = self.df_1.columns.to_list()

        self.samples = self.sampleSet(self.df_1)

    def get_train_columns(self):
        return self.columns
    def sampleSet(self, dataset):
        user_df = dataset.copy()
        segment_df = pd.read_csv(self.segment_file)
        segment_list = segment_df['segment_list'].to_list()

        mini_batch = []

        for seg_num in segment_list:
            seg_df = user_df.loc[user_df['segment'] == seg_num, :]
            if seg_df.shape[0] < self.length:
                # segment 길이가 짧다면, zero padding 진행
                fill_quota = np.abs(self.length - seg_df.shape[0])
                zeros_r = np.zeros([fill_quota, seg_df.shape[1]])
                cur_sample = np.concatenate([zeros_r, seg_df], axis=0)
                mini_batch.append(cur_sample)
            else:
                # segment 에서 요구된 길이만큼 추출
                cur_sample = seg_df.iloc[:self.length, :]
                mini_batch.append(cur_sample)

        return np.array(mini_batch)

    def __getitem__(self, idx):
        task_X = self.samples[idx, :, :].copy()
        task_y = task_X[-self.y_timestep:, -self.label_attribute:].copy()

        if self.y_timestep > 0:
            task_X[-self.y_timestep:, -self.label_attribute:] = 0

        task_X = torch.tensor(task_X, dtype=torch.double).to(self.device)
        task_y = torch.tensor(task_y, dtype=torch.double).to(self.device)

        return task_X, task_y

    def __len__(self):
        # batch를 구성할 수 있는 총 수
        # 이 수에서 batch를 조정할 수 있다.
        # 몇 명의 user 로 나눠서 할 지
        return len(self.samples)