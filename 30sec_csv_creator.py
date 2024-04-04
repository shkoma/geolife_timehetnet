# csv 파일 5분 단위로 전처리하기 (rounded)
# round_min 을 조절하여 round_min 간격으로 data를 전처리

from torch_args import ArgumentSet as args
from datetime import datetime, timedelta
from data.geolife.convert_minmax_location import LocationPreprocessor
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

days_min = 0.000696
lat1, lon1 = 39.975300, 116.452488  # Lower-left corner
lat2, lon2 = 41.367085, 122.651456  # Upper-right corner

round_time = args.round_time
output_csv = args.output_csv
train_csv = args.train_csv
test_csv = args.test_csv
train_ratio = args.train_ratio

user_id_list = []
begin_list = []
end_list = []
time_grid_list = []
ratio = []

local_folder = 'data/geolife/Data/'
for id in valid_user_list['valid_user_list']:
    id = 35
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
    user_df['prev_date'] = user_df['datetime'].shift(1)
    user_df['time_diff'] = user_df['datetime'] - user_df['prev_date']

    group_number = 1
    group_column = []

    for idx in range(user_df.shape[0]):
        if idx == 0:
            group_column.append(group_number)
            continue

        if user_df.iloc[idx, -1] != pd.Timedelta(seconds=30):
            group_number += 1
        group_column.append(group_number)

    user_df['group'] = group_column
    user_df['year'] = user_df['datetime'].dt.year
    user_df['month'] = user_df['datetime'].dt.month
    user_df['week'] = user_df['datetime'].dt.weekday
    user_df['week'] += 1
    user_df['hour'] = user_df['datetime'].dt.hour
    user_df['hour'] += 1
    user_df['day'] = user_df['datetime'].dt.day
    user_df['day'] += 1

    user_df = user_df.drop(columns=['days', 'datetime', 'latitude', 'longitude', 'what', 'altitude', 'time_diff', 'prev_date'])

    user_df['mask'] = np.where(user_df['x'] != 0, 1, 0)
    user_df = user_df.set_index('mask').reset_index()

    min_length = args.length
    total_samples = args.sample_s + args.sample_q
    divide = args.y_timestep
    minimun_dist = args.minimum_dist  # m
    max_speed = args.max_speed

    print(f"length: {min_length}, total_samples: {total_samples}")
    print(f"y_timetep: {divide}")

    df = user_df.copy()
    group_df = df.groupby(['group'])['mask'].sum().reset_index()
    group_df = group_df.loc[group_df['mask'] >= (min_length * total_samples), :].reset_index(drop=True)

    full_df = pd.DataFrame()
    slow_count = 0
    fast_count = 0

    for idx in range(group_df.shape[0]):
        group_id = group_df.iloc[idx, 0]
        df_temp = df.loc[df['group'] == group_id, :].reset_index(drop=True)
        for temp_idx in range(df_temp.shape[0] // divide):
            cur_idx = temp_idx * divide
            if cur_idx + min_length < df_temp.shape[0]:
                cur_df = df_temp.iloc[cur_idx:cur_idx + min_length, :].reset_index(drop=True)
                # check speed
                begin = cur_df.iloc[-args.y_timestep, 0:2]
                end = cur_df.iloc[-1, 0:2]
                x_dist = (begin[0] - end[0]) ** 2
                y_dist = (begin[1] - end[1]) ** 2
                dist = (x_dist + y_dist) ** 0.5

                # if dist < minimun_dist: # 30m
                #     slow_count += 1
                #     # print(f"slow ** x_dist: {x_dist**0.5}, y_dist: {y_dist**0.5}, dist: {dist}")
                #     continue
                # elif dist > (ArgumentSet.max_speed * ArgumentSet.y_timestep): # 0.25km/30sec, 0.5km/min -> 30km/h
                #     fast_count += 1
                #     # print(f"fast ** x_dist: {x_dist**0.5}, y_dist: {y_dist**0.5}, dist: {dist}")
                #     continue

                if dist > (args.max_speed * args.y_timestep):  # 0.25km/30sec, 0.5km/min -> 30km/h
                    fast_count += 1
                    # print(f"fast ** x_dist: {x_dist**0.5}, y_dist: {y_dist**0.5}, dist: {dist}")
                    continue

                full_df = pd.concat([full_df, cur_df], axis=0).reset_index(drop=True)

    ##### end of for statement
    sample_set_length = min_length * total_samples
    # print(f'before: {full_df.shape[0] // sample_set_length}')
    if (full_df.shape[0] // sample_set_length) < 5:
        continue

    if args.random == 'random':
        import random
        samples = np.arange(full_df.shape[0] // sample_set_length)
        print(f'num of samples: {len(samples)}')
        random.shuffle(samples)
        random_df = pd.DataFrame()
        for sample in samples:
            idx = sample * sample_set_length
            cur_df = full_df.iloc[idx:idx + sample_set_length, :].reset_index(drop=True).copy()
            random_df = pd.concat([random_df, cur_df], axis=0).reset_index(drop=True).copy()
        full_df = random_df.copy()

    # print(f'after: {full_df.shape[0] // sample_set_length}')
    # full_df = pd.get_dummies(full_df, columns=['hour', 'week'], drop_first=True).astype(np.int64)
    print(f"columns: {len(full_df.columns)}")

    time_grid_list += [full_df.shape[0] // sample_set_length]
    user_id_list += [user_id]
    full_df.to_csv(output_file, index=False)

    # Train-test set
    train_size = int((full_df.shape[0] // sample_set_length) * train_ratio) * sample_set_length
    train_df = full_df.iloc[:train_size, :]
    test_df = full_df.loc[train_size:, :]

    print(f"minimum_dist: {minimun_dist}, max_speed:{max_speed}")
    print(f"slow count: {slow_count}, fast_count: {fast_count}")
    print(f"total: {full_df.shape[0] // (min_length * total_samples)}, {full_df.shape}")
    print(f"train: {train_df.shape[0] // (min_length * total_samples)}, {train_df.shape}")
    print(f"test : {test_df.shape[0] // (min_length * total_samples)}, {test_df.shape}")
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    break

# full_list = pd.DataFrame({'user_id':user_id_list,
#                           'sample':time_grid_list})
# full_list = full_list.loc[full_list['sample'] > sample_set_length, :]
# full_list = full_list.sort_values('sample', ascending=False)
# full_list.to_csv('walk_speed_sample_user_list.csv', index=False)
# full_list

