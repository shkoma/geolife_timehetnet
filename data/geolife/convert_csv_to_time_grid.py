# csv 파일 5분 단위로 전처리하기 (rounded)
# round_min 을 조절하여 round_min 간격으로 data를 전처리

from convert_minmax_location import LocationPreprocessor
from gps_grid_map_creator import GPSGridMapCreator
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


locationPreprocessor = LocationPreprocessor()
valid_user_list = locationPreprocessor.get_valid_user_list()

days_min = 0.000696
gap = 10
round_min = str(gap) + 'min'
round_sec = str(gap) + 's'

time_delta = 20
segment_delta = str(time_delta) + 'min'

lat1, lon1 = 39.975300, 116.452488  # Lower-left corner
lat2, lon2 = 41.367085, 122.651456  # Upper-right corner

# origin_grid_10min = just add grid in original csv file
# grid_10min        = from begin to end full data with fillna(ffill)

grid_csv = '_origin_grid_' + round_min + '.csv'
segment_csv = '_segment_list_' + segment_delta + '.csv'
grid_list = [1000]  # [50, 100, 500, 1000, 1500, 2000, 3000]

user_id_list = []
df_len_list = []
seg_list = []

for id in valid_user_list['valid_user_list']:
    # id = 68
    print(f"user_id: {id}")
    user_id = locationPreprocessor.getUserId(id)
    user_id_list += [user_id]
    csv_file = './Data/' + user_id + '/csv/' + user_id + '.csv'
    segment_file = './Data/' + user_id + '/csv/' + user_id + segment_csv
    grid_file = './Data/' + user_id + '/csv/' + user_id + grid_csv

    df = pd.read_csv(csv_file)
    df_len_list += [df.shape[0]]

    df['datetime'] = pd.to_datetime(df['date'] + " " + df['time'])
    df['datetime'] = df['datetime'].dt.round(round_min)

    df = df.set_index('datetime').reset_index()
    user_df = df.groupby('datetime')[df.columns[1:].to_list()].mean().reset_index()
    user_df = locationPreprocessor.convert_coord_for_blender_for_user(user_df)

    user_df['datetime'] = pd.to_datetime(user_df['datetime'])

    ## make data fully
    begin = user_df['datetime'][0]
    end = user_df['datetime'][user_df.shape[0] - 1]

    print(f'begin: {begin}')
    print(f'end: {end}')

    user_df['year'] = user_df['datetime'].dt.year
    user_df['month'] = user_df['datetime'].dt.month
    user_df['week'] = user_df['datetime'].dt.weekday + 1
    # user_df['weekend'] = np.where(user_df['week'] < 5, 0, 1)
    user_df['hour'] = user_df['datetime'].dt.hour + 1
    user_df['day'] = user_df['datetime'].dt.day

    date_df = pd.DataFrame({'datetime': pd.date_range(begin, end, freq=round_min)})
    user_df = pd.merge(date_df, user_df, how='outer', on='datetime')
    user_df['time_grid'] = np.arange(1, user_df.shape[0] + 1)
    user_df = user_df.fillna(0)
    user_df = user_df.drop(columns=['altitude', 'what', 'days'])
    user_df = user_df.set_index('datetime')

    for grid_len in grid_list:
        mapCreator = GPSGridMapCreator(grid_len)
        mapCreator.create_grid_map(lat1, lon1, lat2, lon2)
        grid_row = 'grid_row_' + str(grid_len) + 'm'  # row
        grid_col = 'grid_col_' + str(grid_len) + 'm'  # column
        grid_lat = 'grid_lat_' + str(grid_len) + 'm'  # lat
        grid_lon = 'grid_lon_' + str(grid_len) + 'm'  # lon
        grid_num = 'grid_num_' + str(grid_len) + 'm'  # num
        user_df[grid_row], user_df[grid_col], _, _, _ = mapCreator.find_grid_number(user_df['latitude'],
                                                                                    user_df['longitude'])

    user_df = user_df.drop(columns=['latitude', 'longitude'])

    ## remove 0 day
    df = user_df.copy()
    df_1 = df.head(1)
    day = 144  # 10min * 6 * 24
    full_day = int(df.shape[0] / day)
    # remove empty days
    for idx in range(0, full_day):
        x = df.iloc[day * idx:day * (idx + 1), 1:2].sum()[0]
        if x != 0.0:
            df_1 = pd.concat([df_1, df.iloc[day * idx:day * (idx + 1), :]], axis=0)

    df_1 = df_1.drop_duplicates().reset_index(drop=True)

    # check whether the user has any empty day
    full_day = int(df_1.shape[0] / day)
    for idx in range(0, full_day):
        x = df_1.iloc[day * idx:day * (idx + 1), 1:2].sum()[0]
        if x == 0.0:
            print(x)
    # save grid file
    df_1.to_csv(grid_file)