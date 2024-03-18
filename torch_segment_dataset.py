import os
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

        if self.file_mode == 'sec':
            self.sec_csv = '_origin_grid_' + str(self.round_sec) + 's.csv'
            self.segment_file = '_segment_list_' + str(self.time_delta) +'min.csv'
        else:
            # round_min
            self.min_csv = '_origin_grid_' + str(self.round_min) + 'min.csv'
            self.min_file = str(self.data_dir) + str(self.user_list[0]) + '/csv/' + str(
                self.user_list[0]) + self.min_csv

        self.data_mode = 'full'
        self.user_data_list = []
        self.full_user_data_list = [] # user together, in sec mode and min mode
        self.loadUserData()

    def get_train_columns(self):
        return self.columns

    def loadUserData(self):
        for user_id in self.user_list:
            if self.file_mode == 'sec':
                csv_file = str(self.data_dir) + str(user_id) + '/csv/' + str(user_id) + self.sec_csv
                df = pd.read_csv(csv_file)
                df = df.drop(columns=['time_diff'])

                day_column = 10
                df_1 = df[df.columns[0:day_column].to_list()].copy()
                df_1 = pd.concat([df_1, df.iloc[:, day_column+4:day_column+6]], axis=1)  # 500m
                df_1 = pd.concat([df_1, df.iloc[:, day_column+2:day_column+4]], axis=1)  # 100m
                df_1 = pd.concat([df_1, df.iloc[:, day_column+6:day_column+8]], axis=1)  # 1000m
                df_1 = df_1.drop(columns=['x', 'y'])
                df_1 = pd.concat([df_1, df.iloc[:, 1:3]], axis=1)  # x, y
                self.columns = df_1.columns.to_list()

                if self.data_mode == 'full':
                    self.fullSampleSet(df_1, user_id)
                else:
                    self.user_data_list.append(self.sampleSet(df_1, user_id))
            else:
                # min
                min_file = str(self.data_dir) + str(user_id) + '/csv/' + str(user_id) + self.min_csv
                df = pd.read_csv(min_file)

                df_1 = df[df.columns[2:].to_list()].copy()
                self.columns = df_1.columns.to_list()
                self.sampleMinSet(df_1)
        return

    def sampleMinSet(self, dataset):
        user_df = dataset.copy()
        total_samps = self.sample_s + self.sample_q

        grid_days = np.arange(int(user_df.shape[0] / self.day) - int(self.length/self.day))
        grid_days = grid_days[0:(grid_days.shape[0]//total_samps) * total_samps]
        grid_days = grid_days.reshape(-1, total_samps)
        sample_list = np.arange(grid_days.shape[0])

        random.shuffle(sample_list)
        print(f"day: {self.day}, grid_days: {grid_days.shape}, sample's len: {sample_list.shape}")

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
        # count = 0
        # for idx in grid_days:
        #     count += 1
        #     index = idx * self.day
        #     cur_sample = user_df.iloc[index:index + self.length, :]
        #     mini_batch.append(cur_sample)
        #
        #     if count % total_samps == 0:
        #         self.full_user_data_list.append(np.array(mini_batch))
        #         mini_batch = []
        #         if len(grid_days) - count < total_samps:
        #             print(f"full_user_data_list: {len(self.full_user_data_list)}")
        #             return
        return

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

    def fullSampleSet(self, dataset, user_id):
        user_df = dataset.copy()
        segment_df = pd.read_csv(str(self.data_dir) + str(user_id) + '/csv/' + str(user_id) + self.segment_file)
        segment_list = segment_df['segment_list'].to_list()

        if len(self.columns) < 1:
            # just for log
            self.columns = user_df.columns.to_list()

        # segment_list 기반, sample 추출하여 mini-batch 형성
        mini_batch = []

        total_samps = self.sample_s + self.sample_q

        count = 0
        for seg_num in segment_list:
            if count != 0 and count % total_samps == 0:
                self.full_user_data_list.append(np.array(mini_batch))
                mini_batch = []
                if len(segment_list) - count < total_samps:
                    return
            count += 1

            seg_df = user_df.loc[user_df['segment'] == seg_num, :]
            seg_df = seg_df.drop(columns=['segment'])
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
        return

    def __getitem__(self, index):
        # masking y_timestpe in task_X
        if self.file_mode == 'sec':
            if self.data_mode == 'full':
                task_X = self.full_user_data_list[index].copy()
            else:
                task_X = self.user_data_list[index].copy()
        else:
            # min
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
        if self.file_mode == 'sec':
            if self.data_mode == 'full':
                length = len(self.full_user_data_list)
            else:
                length = len(self.user_data_list)
        else: # min
            length = len(self.full_user_data_list)
        return length