# day grid file
# csv 파일 1hour 단위로 전처리하기 + masking (rounded)
# round_min 을 조절하여 round_min 간격으로 data를 전처리
from datetime import datetime, timedelta
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

lat1, lon1 = 39.975300, 116.452488  # Lower-left corner
lat2, lon2 = 41.367085, 122.651456  # Upper-right corner

round_time = args.round_time
output_csv = args.output_csv

train_csv = args.train_csv
test_csv = args.test_csv
train_ratio = args.train_ratio

user_id_list = []
total_sample_list = []
time_grid_list = []
begin_list = []
end_list = []
ratio = []

local_folder = 'data/geolife/Data/'

user_list = pd.read_csv('data/geolife/time_grid_sample.csv')
user_list = user_list.loc[user_list['ratio'] >= 10, :]
mode = 'time-hetnet'
mode = 'mlp'
print(f"mode: {mode}")
print(f"users: {user_list.shape[0]}, user_list: {user_list['user_id'].to_list()}")
# for id in valid_user_list['valid_user_list']:
for id in user_list['user_id']:
    # id = 4
    print(f"user_id: {id}")
    user_id = locationPreprocessor.getUserId(id)
    csv_file = local_folder + user_id + '/csv/' + user_id + '.csv'
    output_file = local_folder + user_id + '/csv/' + user_id + output_csv

    train_file = local_folder + user_id + '/csv/' + user_id + train_csv
    test_file = local_folder + user_id + '/csv/' + user_id + test_csv

    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['date'] + " " + df['time'])
    df['datetime'] = df['datetime'].dt.round(round_time)

    df = df.set_index('datetime').reset_index()
    user_df = df.groupby('datetime')[df.columns[1:].to_list()].mean().reset_index()
    user_df = locationPreprocessor.convert_coord_for_blender_for_user(user_df)

    user_df['datetime'] = pd.to_datetime(user_df['datetime'])
    user_df['year'] = user_df['datetime'].dt.year
    user_df['month'] = user_df['datetime'].dt.month
    user_df['week'] = user_df['datetime'].dt.weekday
    user_df['week'] += 1
    user_df['hour'] = user_df['datetime'].dt.hour
    user_df['hour'] += 1
    user_df['day'] = user_df['datetime'].dt.day
    user_df['day'] += 1

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
    user_df['time_grid'] = np.arange(1, user_df.shape[0] + 1)
    user_df['mask'] = np.where(user_df['x'] != 0, 1, 0)
    user_df['date'] = user_df['datetime'].dt.date

    start_time = args.start_time
    finish_time = args.finish_time
    time_stamp = args.time_stamp
    input_day = args.input_day
    output_day = args.output_day
    total_day = args.total_day

    print(f"start_time : {start_time}, finish_time: {finish_time}")

    user_df = user_df.loc[(user_df['datetime'].dt.time >= start_time) & (user_df['datetime'].dt.time <= finish_time), :]
    full_df = pd.DataFrame()
    total_date = user_df['date'].nunique()
    print(f"total_date: {total_date}")

    divide_len = 0
    for idx, date in enumerate(user_df['date'].unique()):
        if (idx + total_day) > total_date:
            break
        cond1 = (user_df['date'] >= date)
        cond2 = (user_df['date'] < (date + timedelta(days=total_day)))
        cur_df = user_df.loc[cond1 & cond2, :].reset_index(drop=True)
        full_df = pd.concat([full_df, cur_df], axis=0).reset_index(drop=True)

    if mode == 'time-hetnet':
        divided_len = args_hetnet.total_sample * args_hetnet.length
    else: # mlp
        divided_len = (total_day * time_stamp)

    if (full_df.shape[0] // divided_len) < 5:
        continue

    if args.random == 'random':
        import random
        samples = np.arange(full_df.shape[0] // divided_len)
        print(f"samples: {len(samples)}")
        random.shuffle(samples)

        random_df = pd.DataFrame()
        for sample in samples:
            idx = sample * divided_len
            cur_df = full_df.iloc[idx:(idx + divided_len), :].reset_index(drop=True).copy()
            random_df = pd.concat([random_df, cur_df], axis=0).reset_index(drop=True).copy()
        full_df = random_df.copy()

    full_df = full_df.drop(columns=['days', 'date', 'datetime', 'latitude', 'longitude', 'what', 'altitude'])
    full_df = full_df.set_index('mask').reset_index()
    # full_df = pd.get_dummies(full_df, columns=['hour', 'week'], drop_first=True).astype(np.int64)



    print(f"time_stamp: {time_stamp}, total_day: {total_day}, sample length: {divided_len}")
    print(f"full_df: {full_df.shape}")
    total_samples = full_df.shape[0] // divided_len
    train_samples = int(total_samples * train_ratio)
    train_df = full_df.iloc[:train_samples * divided_len, :].copy().reset_index(drop=True)
    test_df = full_df.iloc[train_samples * divided_len:, :].copy().reset_index(drop=True)

    user_id_list += [user_id]
    ratio += [int(round(origin_len / user_df.shape[0], 2) * 100)]
    begin_list += [begin.strftime('%Y-%m-%d')]
    end_list += [end.strftime('%Y-%m-%d')]
    time_grid_list += [user_df.shape[0]]
    total_sample_list += [total_samples]

    print(f'total sample: {total_samples}, {full_df.shape}')
    print(f'train sample: {train_samples}, {train_df.shape}')
    print(f'test sample: {test_df.shape[0] // divided_len}, {test_df.shape}')

    full_df.to_csv(output_file, index=False)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    # break

time_df = pd.DataFrame({"user_id": user_id_list,
                        "time_grid": time_grid_list,
                        "ratio_60min": ratio,
                        "total_samples": total_samples})
#
time_df.sort_values('total_samples', ascending=False)
time_df.to_csv('mask_user_list.csv')