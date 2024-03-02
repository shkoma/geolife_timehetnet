import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class SegmentDataset(Dataset):
    def __init__(self, data_dir, user_list, device, time_delta, y_timestep, length, label_attribute, sample_s, replace=False):
        self.data_dir = data_dir
        self.user_list = user_list
        self.device = device
        self.time_delta = time_delta
        self.y_timestep = y_timestep
        self.length = length
        self.label_attribute = label_attribute
        self.sample_s = sample_s
        self.replace = replace

        self.csv_file = '_origin_grid_10s.csv'
    
    def sampleSet(self, dataset):
        user_df = dataset.copy()
        user_df['datetime'] = pd.to_datetime(user_df['datetime'])
    
        # 세그먼트별 시작 시간과 끝 시간 추출 그리고, 시간 간격별 segment list 추출
        segment_info = user_df.groupby('segment')['datetime'].agg(['min', 'max'])
        segment_info = segment_info.reset_index()
        segment_info['time_gap'] = segment_info['max'] - segment_info['min']
        segment_info['over_time_delta'] = segment_info['time_gap'] >= pd.Timedelta(minutes=self.time_delta)
        segment_info = segment_info.loc[segment_info['over_time_delta'] == True, :]
        segment_list = segment_info['segment'].to_list()

        user_df = user_df.drop(columns=['datetime'])

        # segment_list 기반, sample 추출하여 mini-batch 형성
        mini_batch = []

        segment_list = np.random.choice(segment_list, size=self.sample_s, replace=self.replace)

        for seg_num in segment_list:
            seg_df = user_df.loc[user_df['segment'] == seg_num, :]
            if seg_df.shape[0] < length:
                # segment 길이가 짧다면, zero padding 진행
                fill_quota  = np.abs(length - seg_df.shape[0])
                zeros_r     = np.zeros([fill_quota, seg_df.shape[1]])
                cur_sample  = seg_df.copy()
                cur_sample  = np.concatenate([zeros_r, seg_df], axis = 0)
                mini_batch.append(cur_sample)
            else:
                # segment 에서 요구된 길이만큼 추출
                cur_sample = seg_df.iloc[:length, :]
                mini_batch.append(cur_sample)

        return np.array(mini_batch)
    
    def __getitem__(self, index):
        csv_file = str(self.data_dir) + str(self.user_list[index]) + '/csv/' + str(self.user_list[index]) + self.csv_file
        df = pd.read_csv(csv_file)
        
        df_1 = df[df.columns[0:13].to_list()].copy()
        # df_1 = pd.concat([df_1, df.iloc[:, 6:13]], axis=1) # ~ Hour + 100m
        # df_1 = pd.get_dummies(df_1, columns=['hour'], drop_first=True)

        df_1 = pd.concat([df_1, df.iloc[:, 13:15]], axis=1) # 100m
        df_1 = pd.concat([df_1, df.iloc[:, 21:]], axis=1) # 2000m
        df_1 = pd.concat([df_1, df.iloc[:, 17:19]], axis=1) # 1000m
        df_1 = pd.concat([df_1, df.iloc[:, 15:17]], axis=1) # 500m
        
        samples = self.sampleSet(df_1)
        
        # task_X, task_y 준비
        task_X = np.array(samples[:, :-self.y_timestep, 4:])
        task_y = np.array(samples[:, -self.y_timestep:, -self.label_attribute:])

        task_X = torch.from_numpy(task_X).double()
        task_y = torch.from_numpy(task_y).double()

        return task_X, task_y
    
    def __len__(self):
        # batch를 구성할 수 있는 총 수
        # 이 수에서 batch를 조정할 수 있다.
        # 몇 명의 user 로 나눠서 할 지
        return len(self.user_list)