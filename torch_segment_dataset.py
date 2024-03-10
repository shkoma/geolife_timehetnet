import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class SegmentDataset(Dataset):
    def __init__(self, model_type, data_dir, user_list, device, round_sec, time_delta, y_timestep, length, label_attribute, sample_s, sample_q):
        self.model_type = model_type
        self.data_dir = data_dir
        self.user_list = user_list
        self.device = device
        self.round_sec = round_sec
        self.time_delta = time_delta
        self.y_timestep = y_timestep
        self.length = length
        self.label_attribute = label_attribute
        self.sample_s = sample_s
        self.sample_q = sample_q
        self.columns = []

        self.csv_file = '_origin_grid_' + str(self.round_sec) + 's.csv'
        self.segment_file = '_segment_list_' + str(self.time_delta) +'min.csv'

        self.user_data_list = self.loadUserData()

    
    def get_train_columns(self):
        return self.columns

    def loadUserData(self):
        user_data_list = []

        day_column = 10
        for user_id in self.user_list:
            csv_file = str(self.data_dir) + str(user_id) + '/csv/' + str(user_id) + self.csv_file
            df = pd.read_csv(csv_file)
            df = df.drop(columns=['time_diff'])
            
            df_1 = df[df.columns[0:day_column].to_list()].copy()
            df_1 = pd.concat([df_1, df.iloc[:, day_column+4:day_column+6]], axis=1)  # 500m
            df_1 = pd.concat([df_1, df.iloc[:, day_column+2:day_column+4]], axis=1)  # 100m
            df_1 = pd.concat([df_1, df.iloc[:, day_column+6:day_column+8]], axis=1)  # 1000m

            user_data_list.append(self.sampleSet(df_1, user_id))
        return user_data_list

    def sampleSet(self, dataset, user_id):
        user_df = dataset.copy()
        segment_df = pd.read_csv(str(self.data_dir) + str(user_id) + '/csv/' + str(user_id) + self.segment_file)
        segment_list = segment_df['segment_list'].to_list()

        if len(self.columns) < 1:
            # just for log
            self.columns = user_df.columns.to_list()

        # segment_list 기반, sample 추출하여 mini-batch 형성
        mini_batch = []

        total_samps = self.sample_s + self.sample_q
        
        replace = False
        if total_samps > len(segment_list):
            replace = True

        segment_list = np.random.choice(segment_list, size=total_samps, replace=replace)

        for seg_num in segment_list:
            seg_df = user_df.loc[user_df['segment'] == seg_num, :]
            seg_df = seg_df.drop(columns=['segment'])
            if seg_df.shape[0] < self.length:
                # segment 길이가 짧다면, zero padding 진행
                fill_quota  = np.abs(self.length - seg_df.shape[0])
                zeros_r     = np.zeros([fill_quota, seg_df.shape[1]])
                cur_sample  = np.concatenate([zeros_r, seg_df], axis = 0)
                mini_batch.append(cur_sample)
            else:
                # segment 에서 요구된 길이만큼 추출
                cur_sample = seg_df.iloc[:self.length, :]
                mini_batch.append(cur_sample)

        return np.array(mini_batch)
    
    def __getitem__(self, index):
        # masking y_timestpe in task_X
        task_X = self.user_data_list[index].copy()
        task_y = task_X[:, -self.y_timestep:, -self.label_attribute:].copy()

        if self.y_timestep > 0:
            task_X[:, -self.y_timestep:, -self.label_attribute:] = 0

        ## model 에 따라 task_X, task_y 의 형태가 다름
        if self.model_type == 'time-hetnet' or self.model_type == 'hetnet':
            sup_x = np.array(task_X[:self.sample_s, :, :])
            sup_y = np.array(task_y[:self.sample_s, :, :])
            que_x = np.array(task_X[self.sample_s:, :, :])
            que_y = np.array(task_y[self.sample_s:, :, :])
            
            sup_x = torch.from_numpy(sup_x).double().to(self.device)
            sup_y = torch.from_numpy(sup_y).double().to(self.device)
            que_x = torch.from_numpy(que_x).double().to(self.device)
            que_y = torch.from_numpy(que_y).double().to(self.device)

            task_X = (que_x, sup_x, sup_y)
            task_y = que_y

        else: ## mlp
            # task_X, task_y 준비
            samples = self.user_data_list[index].copy()
            task_X = np.array(samples[:, :, :])
            task_y = np.array(samples[:, -self.y_timestep:, -self.label_attribute:])

            # task_X = torch.tensor(task_X, dtype=torch.double).to(self.device)
            # task_y = torch.tensor(task_y, dtype=torch.double).to(self.device)
            task_X = torch.from_numpy(task_X).double().to(self.device)
            task_y = torch.from_numpy(task_y).double().to(self.device)

        return task_X, task_y
    
    def __len__(self):
        # batch를 구성할 수 있는 총 수
        # 이 수에서 batch를 조정할 수 있다.
        # 몇 명의 user 로 나눠서 할 지
        return len(self.user_data_list)