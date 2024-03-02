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

lat1, lon1 = 39.975300, 116.452488  # Lower-left corner
lat2, lon2 = 41.367085, 122.651456  # Upper-right corner

# origin_grid_10min = just add grid in original csv file
# grid_10min        = from begin to end full data with fillna(ffill)
round_csv = '_origin_round_' + round_sec + '.csv'
grid_csv = '_origin_grid_' + round_sec + '.csv'
grid_list = [50, 100, 500, 1000, 1500, 2000, 3000]

for id in valid_user_list['valid_user_list']:
    print(f"user_id: {id}")
    user_id = locationPreprocessor.getUserId(id)
    csv_file = './Data/' + user_id + '/csv/' + user_id + '.csv'
    rounded_file = './Data/' + user_id + '/csv/' + user_id + round_csv
    grid_file = './Data/' + user_id + '/csv/' + user_id + grid_csv

    df = pd.read_csv(csv_file)
    print(df.shape)
    df['datetime'] = pd.to_datetime(df['date'] + " " + df['time'])
    df['datetime'] = df['datetime'].dt.round(round_sec)

    df = df.set_index('datetime').reset_index()
    user_df = df.groupby('datetime')[df.columns[1:].to_list()].mean().reset_index()
    user_df = locationPreprocessor.convert_coord_for_blender_for_user(user_df)

    user_df['datetime'] = pd.to_datetime(user_df['datetime'])
    
    # begin = user_df['datetime'][0]
    # end = user_df['datetime'][user_df.shape[0]-1]

    # print(f'begin: {begin}')
    # print(f'end: {end}')

    # date_df = pd.DataFrame({'datetime':pd.date_range(begin, end, freq=round_min)})
    # user_df = pd.merge(date_df, user_df, how='outer', on='datetime')
    
    # # add days 
    # start_days = user_df['days'][0]
    # end_days = start_days + (date_df.shape[0] * (days_min * gap))
    # days_list = np.arange(start_days, end_days, (days_min * gap))

    # user_df['days'] = days_list[:user_df.shape[0]]
    user_df['year'] = user_df['datetime'].dt.year
    user_df['month'] = user_df['datetime'].dt.month
    user_df['week'] = user_df['datetime'].dt.weekday
    user_df['weekend'] = np.where(user_df['week'] < 5, 0, 1)
    user_df['hour'] = user_df['datetime'].dt.hour
    user_df['day'] = user_df['datetime'].dt.day
    # user_df = user_df.fillna(method='ffill')

    # 시간 간격 계산
    user_df['time_diff'] = user_df['datetime'].diff()

    # 1분 이상인 경우 세그먼트로 분류
    threshold = pd.Timedelta(minutes=1)
    user_df['segment'] = (user_df['time_diff'] > threshold).cumsum()
    user_df = user_df.drop(columns=['time_diff'])

    # save rounded file
    # user_df.to_csv(rounded_file, index=False)

    # grid process
    user_df = user_df.drop(columns=['datetime', 'altitude', 'what'])
    for grid_len in grid_list:
        mapCreator = GPSGridMapCreator(grid_len)
        mapCreator.create_grid_map(lat1, lon1, lat2, lon2)
        grid_row = 'grid_row_' + str(grid_len) + 'm' # row
        grid_col = 'grid_col_' + str(grid_len) + 'm' # column
        grid_lat = 'grid_lat_' + str(grid_len) + 'm' # lat
        grid_lon = 'grid_lon_' + str(grid_len) + 'm' # lon
        grid_num = 'grid_num_' + str(grid_len) + 'm' # num      
        user_df[grid_row], user_df[grid_col],_,_,_ = mapCreator.find_grid_number(user_df['latitude'], user_df['longitude'])
    # save grid file
    user_df.to_csv(grid_file, index=False)
