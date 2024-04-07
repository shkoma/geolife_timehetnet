import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_args import ArgumentMask


class MlpDataset(Dataset):
    def __init__(self, data_mode, data_dir, writer_dir, user_list, label_attribute, device):
        self.data_mode = data_mode
        self.data_dir = data_dir
        self.writer_dir = writer_dir
        self.user_list = user_list
        self.label_attribute = label_attribute
        self.device = device
        self.x_attibutes = 0
        self.samples = []
        self.full_df = pd.DataFrame()

        # round_min
        if self.data_mode == 'train':
            self.csv_file = ArgumentMask.train_csv
        elif self.data_mode == 'test':
            self.csv_file = ArgumentMask.test_csv
        else:
            self.csv_file = ArgumentMask.output_csv

        if self.data_mode == 'train':
            self.csv_file = 'user_full_train.csv'
        elif self.data_mode == 'test':
            self.csv_file = 'user_full_test.csv'


        # for user_id in self.user_list:
        #     self.min_file = str(self.data_dir) + str(user_id) + '/csv/' + str(user_id) + self.csv_file
        #     copy_file = str(self.writer_dir) + '/' + str(user_id) + self.csv_file
        #     df = pd.read_csv(self.min_file)
        #     df.to_csv(copy_file, index=False)
        #
        #     df = df.set_index('mask').reset_index()
        #     df_1 = df[df.columns[3:].to_list()].copy()
        #     df_1 = pd.concat([df_1, df.iloc[:, 0:3]], axis=1)  # mask, x, y
        #     self.columns = df_1.columns.to_list()
        #     self.full_df = pd.concat([self.full_df, df_1], axis=0).reset_index(drop=True).copy()

        df = pd.read_csv(self.csv_file)
        self.copyConfiguration(df)

        df = df.set_index('mask').reset_index()
        df_1 = df[df.columns[3:].to_list()].copy()
        df_1 = pd.concat([df_1, df.iloc[:, 0:3]], axis=1)  # mask, x, y
        self.columns = df_1.columns.to_list()
        self.full_df = pd.concat([self.full_df, df_1], axis=0).reset_index(drop=True).copy()

        self.x_attibutes = len(self.full_df.columns)
        self.sampleSet(self.full_df)
        self.samples = np.array(self.samples)

    def copyConfiguration(self, df):
        copy_file = str(self.writer_dir) + '/' + self.csv_file
        df.to_csv(copy_file, index=False)

        train_user_list = pd.read_csv('train_user_list.csv')
        train_user_list.to_csv(str(self.writer_dir) + '/train_user_list.csv', index=False)

        test_user_list = pd.read_csv('test_user_list.csv')
        test_user_list.to_csv(str(self.writer_dir) + '/test_user_list.csv', index=False)

    def get_x_attibutes(self):
        return self.x_attibutes

    def get_train_columns(self):
        return self.columns

    def sampleSet(self, dataset):
        user_df = dataset.copy()

        dividen_len = ArgumentMask.total_day * ArgumentMask.time_stamp
        days = np.arange(user_df.shape[0]//dividen_len)
        random.shuffle(days)

        for idx in days:
            index = idx * dividen_len
            cur_sample = user_df.iloc[index:(index + dividen_len), :]
            self.samples.append(cur_sample)
        return

    def __getitem__(self, idx):
        task_X = self.samples[idx, :, :].copy()
        output_len = ArgumentMask.output_day * ArgumentMask.time_stamp
        task_y = task_X[-output_len:, -(self.label_attribute+1):].copy()

        if output_len > 0:
            task_X[-output_len:, -(self.label_attribute+1):] = 0

        task_X = torch.tensor(task_X, dtype=torch.double).to(self.device)
        task_y = torch.tensor(task_y, dtype=torch.double).to(self.device)

        return task_X, task_y

    def __len__(self):
        # batch를 구성할 수 있는 총 수
        # 이 수에서 batch를 조정할 수 있다.
        # 몇 명의 user 로 나눠서 할 지
        return len(self.samples)