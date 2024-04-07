# day grid file
# csv 파일 1hour 단위로 전처리하기 + masking (rounded)
# round_min 을 조절하여 round_min 간격으로 data를 전처리
from datetime import datetime, timedelta

import torch_args
from data.geolife.convert_minmax_location import LocationPreprocessor
from torch_args import ArgumentMask as args
from torch_args import HetnetMask as args_hetnet
import numpy as np
import pandas as pd

def getUserId(id):
    val = ""
    if id < 10:
        val += "00"
        val += str(id)
    elif id < 100:
        val += "0"
        val += str(id)
    else:
        val = str(id)
    return val

locationPreprocessor = LocationPreprocessor('data/geolife/')
valid_user_list = locationPreprocessor.get_valid_user_list()

round_time = args.round_time
output_csv = args.output_csv

train_csv = args.train_csv
test_csv = args.test_csv
train_ratio = args.train_ratio

local_folder = 'data/geolife/Data/'

# user_list = pd.read_csv('valid_user_list.csv')
user_list = pd.read_csv('mask_user_list.csv')
user_list = user_list.loc[user_list['ratio'] >= 10, :]
user_list = user_list['user_id'].to_list()

user_list = [8, 103, 35, 88, 4, 34, 64, 19, 22, 1, 11, 32, 38, 113, 9, 2, 3, 7, 167, 169, 17, 14, 0]

import random
random.shuffle(user_list)

# train_user_list = user_list[:int(len(user_list) * 0.9)]
# test_user_list = user_list[int(len(user_list) * 0.9):]

train_user_list = [8, 103, 88, 4, 34, 64, 19, 22, 1, 11, 32, 38, 113, 9, 2, 3, 7, 167, 169, 17, 14]
test_user_list = [35, 0]

# train_user_list = pd.read_csv('train_user_list.csv')['user_id'].to_list()
# test_user_list = pd.read_csv('test_user_list.csv')['user_id'].to_list()
# test_user_list = [35]#[32, 35]

mode = torch_args.global_model_type
# mode = 'mlp'

from data.geolife.gps_grid_map_creator import GPSGridMapCreator
grid_len = 1
min_max_location = pd.read_csv('min_max_location.csv')
mapCreator = GPSGridMapCreator(grid_len) # 1m grid
mapCreator.create_grid_map(min_lat=min_max_location['min_lat'][0],
                           min_lon=min_max_location['min_lon'][0],
                           max_lat=min_max_location['max_lat'][0],
                           max_lon=min_max_location['max_lon'][0])
print(f"grid_map created")

for cur_idx, user_list in enumerate([train_user_list, test_user_list]):
    if cur_idx == 0:
        cur_list = 'train'
    else:
        cur_list = 'test'

    user_full_df = pd.DataFrame()
    user_id_list = []
    total_sample_list = []
    time_grid_list = []
    begin_list = []
    end_list = []
    ratio_list = []
    for id in user_list:
        print(f"user_id: {id}")
        user_id = locationPreprocessor.getUserId(id)
        csv_file = local_folder + user_id + '/csv/' + user_id + '_quantile.csv'
        # csv_file = local_folder + user_id + '/csv/' + user_id + '.csv'
        output_file = local_folder + user_id + '/csv/' + user_id + output_csv

        train_file = local_folder + user_id + '/csv/' + user_id + train_csv
        test_file = local_folder + user_id + '/csv/' + user_id + test_csv

        df = pd.read_csv(csv_file)

        df['datetime'] = pd.to_datetime(df['date'] + " " + df['time'])
        df['datetime'] = df['datetime'].dt.round(round_time)

        df = df.set_index('datetime').reset_index()
        user_df = df.groupby('datetime')[df.columns[1:].to_list()].mean().reset_index()

        grid_row = 'row'
        grid_col = 'col'
        user_df[grid_row], user_df[grid_col] = mapCreator.find_grid_number(user_df['latitude'], user_df['longitude'])

        user_df['datetime'] = pd.to_datetime(user_df['datetime'])

        ## make data fully
        begin = user_df['datetime'][0]
        end = user_df['datetime'][user_df.shape[0] - 1]
        print(f'begin: {begin}')
        print(f'end: {end}')

        date_df = pd.DataFrame({'datetime': pd.date_range(begin.strftime('%Y-%m-%d 00:00:00'),
                                                          end.strftime('%Y-%m-%d 23:00:00'), freq=round_time)})
        origin_len = user_df.shape[0]

        user_df = pd.merge(date_df, user_df, how='outer', on='datetime')
        user_df = user_df.fillna(0)
        user_df['mask'] = np.where(user_df['longitude'] != 0, 1, 0)
        user_df['date'] = user_df['datetime'].dt.date

        ratio = int(round(origin_len / user_df.shape[0], 2) * 100)

        start_time = args.start_time
        finish_time = args.finish_time
        time_stamp = args.time_stamp
        input_day = args.input_day
        output_day = args.output_day
        total_day = args.total_day

        print(f"start_time : {start_time}, finish_time: {finish_time}")

        full_df = pd.DataFrame()
        total_date = user_df['date'].nunique()
        print(f"total_date: {total_date}")

        divide_len = 0
        for idx, date in enumerate(user_df['date'].unique()):
            if (idx + total_day) > total_date:
                break
            cond1 = (user_df['date'] >= date) # START DAY
            cond2 = (user_df['date'] < (date + timedelta(days=input_day)))
            input_df = user_df.loc[cond1 & cond2, :].reset_index(drop=True)

            cond3 = (user_df['date'] == (date + timedelta(days=input_day))) #output_day == 1
            output_df = user_df.loc[cond3, :].reset_index(drop=True)
            output_df = output_df.loc[(output_df['datetime'].dt.time >= start_time) & (output_df['datetime'].dt.time <= finish_time), :]
            cur_df = pd.concat([input_df, output_df], axis=0).reset_index(drop=True)

            full_df = pd.concat([full_df, cur_df], axis=0).reset_index(drop=True)

        if mode == 'time-hetnet':
            divided_len = args_hetnet.total_sample * args_hetnet.length
        else: # mlp
            divided_len = (total_day * time_stamp)

        if (full_df.shape[0] // divided_len) < 5:
            continue

        full_df = full_df.drop(columns=['days', 'date', 'datetime', 'latitude', 'longitude', 'what', 'altitude'])
        full_df = full_df.set_index('mask').reset_index()
        user_full_df = pd.concat([user_full_df, full_df], axis=0).reset_index(drop=True).copy()

        user_id_list += [user_id]
        print(f"time_stamp: {time_stamp}, total_day: {total_day}, sample length: {divided_len}")

    total_samples = user_full_df.shape[0] // divided_len
    train_samples = int(total_samples * train_ratio)

    print(f"mode: {mode}")
    print(f"{cur_list}_users: {len(user_list)}, user_list: {user_list}")

    if cur_list == 'train':
        user_full_df.to_csv('user_full_train.csv', index=False)
        print(f"user_full_train: {user_full_df.shape[0] // divided_len}")
    else: # test
        user_full_df.to_csv('user_full_test.csv', index=False)
        print(f"user_full_test: {user_full_df.shape[0] // divided_len}")

    time_df = pd.DataFrame({"user_id": user_id_list})
    time_df.to_csv(cur_list + '_user_list.csv', index=False)