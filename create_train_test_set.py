## 1sample: 20분에 data
## total_samples = support_set + query_set
## 총 참여 80분에 해당 하는 data만을 추출
from data.geolife.convert_minmax_location import LocationPreprocessor
from torch_args import ArgumentSet
import pandas as pd
import numpy as np
import random

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


locationPreprocessor = LocationPreprocessor()
valid_user_list = locationPreprocessor.get_valid_user_list()

days_min = 0.000696
gap = ArgumentSet.round_sec
round_min = str(gap) + 'min'
round_sec = str(gap) + 's'

round_time = round_sec

segment_csv = '_segment_time_grid_' + round_time + '.csv'
output_csv = '_segment_output_' + round_time + '.csv'

train_csv = '_train_output_' + round_time + '.csv'
test_csv = '_test_output_' + round_time + '.csv'

train_ratio = 0.9

user_id_list = []
time_grid_list = []
ratio = []

min = ArgumentSet.min
min_length = ArgumentSet.length
support_set = ArgumentSet.sample_s
query_set = ArgumentSet.sample_q
total_samples = support_set + query_set  # support_set: 4, query_set: 1

for id in valid_user_list['valid_user_list']:
    id = 35
    print(f"user_id: {id}")
    full_df = pd.DataFrame()
    user_id = locationPreprocessor.getUserId(id)
    user_id_list += [user_id]
    segment_file = './Data/' + user_id + '/csv/' + user_id + segment_csv
    output_file = './Data/' + user_id + '/csv/' + user_id + output_csv

    train_file = './Data/' + user_id + '/csv/' + user_id + train_csv
    test_file = './Data/' + user_id + '/csv/' + user_id + test_csv

    df = pd.read_csv(segment_file)

    group_df = df.groupby(['group'])['mask'].sum().reset_index()
    group_df = group_df.loc[group_df['mask'] >= (min_length * total_samples), :].reset_index(drop=True)

    mini_batch = []
    divide = (min // 2)
    if ArgumentSet.random != 'random':
        for idx in range(group_df.shape[0]):
            group_id = group_df.iloc[idx, 0]
            df_temp = df.loc[df['group'] == group_id, :].reset_index(drop=True)
            for temp_idx in range(df_temp.shape[0] // divide):
                cur_idx = temp_idx * divide
                if cur_idx + min_length < df_temp.shape[0]:
                    cur_df = df_temp.iloc[cur_idx:cur_idx + min_length, :].reset_index(drop=True)
                    full_df = pd.concat([full_df, cur_df], axis=0).reset_index(drop=True)
    else:
        for idx in range(group_df.shape[0]):
            group_id = group_df.iloc[idx, 0]
            df_temp = df.loc[df['group'] == group_id, :].reset_index(drop=True)
            for temp_idx in range(df_temp.shape[0] // divide):
                cur_idx = temp_idx * divide
                if cur_idx + min_length < df_temp.shape[0]:
                    cur_df = df_temp.iloc[cur_idx:cur_idx + min_length, :].reset_index(drop=True)
                    mini_batch += [cur_df]

        sample_list = np.arange(len(mini_batch))
        random.shuffle(sample_list)

        for idx in sample_list:
            cur_df = mini_batch[idx]
            full_df = pd.concat([full_df, cur_df], axis=0).reset_index(drop=True)

    print(full_df.shape[0] // (min_length * total_samples))
    time_grid_list += [full_df.shape[0] // (min_length * total_samples)]
    # full_df.to_csv(output_file, index=False)
    # Train-test set
    train_size = int(full_df.shape[0] * train_ratio)
    train_df = full_df.iloc[:train_size, :]
    test_df = full_df.loc[train_size:, :]

    print(f"train_valid: {train_df.shape}")
    print(f"test : {test_df.shape}")

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    break

# full_list = pd.DataFrame({'user_id':user_id_list,
#                           'sample':time_grid_list})
# full_list = full_list.loc[full_list['sample'] > min_length, :]
# full_list = full_list.sort_values('sample', ascending=False)
# full_list.to_csv('full_sample_user_list.csv', index=False)
# full_list