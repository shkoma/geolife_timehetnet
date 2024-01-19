import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset

# Data folder 중 숫자가 안되는 User folder는 삭제하고
# 남은 User data에서 train-test 폴더로 나눈 후
# train_set(dataset), test_set(dataset) 으로 진행 필요

class GeoLifeDataSet(Dataset):
    def __init__(self, data_dir, user_list, samples_s, samples_q, length, y_timestep, gap=1):
        self.data_dir   = data_dir
        self.csv_dir    = 'csv/'
        self.user_list  = user_list #os.listdir(data_dir)
        # user_list: all user
        self.samples_s  = samples_s
        # samples_s: the number of support set
        self.samples_q  = samples_q
        # samples_q: the number of query set
        self.length     = length 
        # length: the length of mini batch of a user
        self.y_timestep = y_timestep
        # y_time_step: the next time step to be predicted
        #              it must be less than length
        self.gap = gap
        # gap: the gap of time to check a user's location
    
    def sampleTime(self, dataset):
        cur_ds = dataset.copy()
        minibatch = []
        
        max_len = len(cur_ds)
        ###############################################
        # MAke sure samples from query and support 
        # do not intersect
        ##############################################
        # total_data_slice -> lenght 만큼 나눴을 때 총 slice 갯수
        total_data_slice = list(range(int(max_len/self.length)))
        total_samps = self.samples_q + self.samples_s
        
        slice_point = int(len(total_data_slice)*(self.samples_s/total_samps))
        # print(f"slice_point: {slice_point}")
        
        if slice_point == 0:
            return np.array(slice_point)

        s_s_list = total_data_slice[:slice_point]
        q_s_list = total_data_slice[slice_point:]

        replace = False
        if total_samps > len(total_data_slice):
            replace = True

        s_s_list = np.random.choice(s_s_list, size=self.samples_s, replace=replace)
        q_s_list = np.random.choice(q_s_list, size=self.samples_q, replace=replace)

        # print(f"s_list:{s_s_list}")
        # print(f"q_list:{q_s_list}")

        choice_list = np.concatenate([s_s_list, q_s_list])
        #################################################
        # print(f"choice_list: {choice_list}")
        
        for idx in choice_list:
            start_idx = idx * self.length
            if max_len - self.length >= 0:
                cur_sample = cur_ds.iloc[start_idx:(start_idx + self.length), :]
                minibatch.append(cur_sample)
            else:
                fill_quota  = np.abs(self.length - max_len)
                zeros_r     = np.zeros([fill_quota, cur_ds.shape[1]])
                cur_sample  = cur_ds[:, :]
                cur_sample  = np.concatenate([zeros_r, cur_sample], axis = 0)
                minibatch.append(cur_sample)
        return np.array(minibatch)
        
    def __getitem__(self, index):
        csv_path = os.path.join(self.data_dir, self.user_list[index], self.csv_dir)
        user_file = csv_path + self.user_list[index] + '_converted.csv'
        df = pd.read_csv(user_file)
        # df['realTime'] = df['days'] * 10000
        # df = df[['days', 'latitude', 'longitude']]
        df = df.drop_duplicates(subset=['days'])
        df_1 = df[['days', 'x', 'y']]
        
        idx_list = []
        for idx in range(df_1.shape[0]):
            if idx % self.gap == 0: # gap==60 -> 5 mins
                idx_list += [idx]
        user_df = df_1.iloc[idx_list, :].copy()

        samples = self.sampleTime(user_df)
        if samples.size < 2:
            return (torch.from_numpy(samples).float(), torch.from_numpy(samples).float())

        task_X = samples.copy()
        task_y = task_X[:, -self.y_timestep:, -2:].copy()
        
        if self.y_timestep > 0:
            task_X[:, -self.y_timestep:, -2:] = 0
        
        # sup_x = np.array(samples[:self.samples_s, :, [0]])
        # sup_y = np.array(samples[:self.samples_s, :, -2:])
        # que_x = np.array(samples[self.samples_s:, :, [0]])
        # que_y = np.array(samples[self.samples_s:, -self.y_timestep:, -2:])
        
        sup_x = np.array(task_X[:self.samples_s, :, :])
        sup_y = np.array(task_y[:self.samples_s, :, :])
        que_x = np.array(task_X[self.samples_s:, :, :])
        que_y = np.array(task_y[self.samples_s:, :, :])
        
        sup_x = torch.from_numpy(sup_x).float()
        sup_y = torch.from_numpy(sup_y).float()
        que_x = torch.from_numpy(que_x).float()
        que_y = torch.from_numpy(que_y).float()

        return (que_x, sup_x, sup_y), que_y
    
    def __len__(self):
        # batch를 구성할 수 있는 총 수
        # 이 수에서 batch를 조정할 수 있다.
        # 몇 명의 user 로 나눠서 할 지
        return len(self.user_list)


###################### Examples ######################
# data_dir = "data/geolife/Data/"
# sample_s = 5
# sample_q = 2
# length = 100
# y_timestep = 10

# user_list = os.listdir(data_dir)
# random.shuffle(user_list)
# train_size = 0.1
# train_list = user_list[:(int)(len(user_list)*train_size)]
# print(f"train_list: {len(train_list)}")

# training_data = GeoLifeDataSet(data_dir, train_list, sample_s, sample_q, length, y_timestep)
# test_data = GeoLifeDataSet(data_dir, train_list, sample_s, sample_q, length, y_timestep)

# train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)
# test_dataloader  = DataLoader(test_data, batch_size=1, shuffle=False)

# train_x, train_y = next(iter(train_dataloader))
# print(f"support_x: {train_x[0].shape}")
# print(f"support_y: {train_x[1].shape}")
# print(f"query_x: {train_x[2].shape}")
# print(f"query_y: {train_y.shape}")

# dataset = GeoLifeDataSet(data_dir, 
#                          train_list, 
#                          sample_s, 
#                          sample_q, 
#                          length, 
#                          y_timestep)
# dataset.__getitem__(0)
###################################################