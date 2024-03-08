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

time_delta = 60
segment_delta = str(time_delta) + 'min'

lat1, lon1 = 39.975300, 116.452488  # Lower-left corner
lat2, lon2 = 41.367085, 122.651456  # Upper-right corner

# origin_grid_10min = just add grid in original csv file
# grid_10min        = from begin to end full data with fillna(ffill)

grid_csv = '_origin_grid_' + round_sec + '.csv'
segment_csv = '_segment_list_' + segment_delta + '.csv'
grid_list = [50, 100, 500, 1000, 1500, 2000, 3000]

user_id_list = []
df_len_list = []
seg_list = []

for id in valid_user_list['valid_user_list']:
    print(f"user_id: {id}")
    user_id = locationPreprocessor.getUserId(id)
    user_id_list += [user_id]
    csv_file = './Data/' + user_id + '/csv/' + user_id + '.csv'
    segment_file = './Data/' + user_id + '/csv/' + user_id + segment_csv
    grid_file = './Data/' + user_id + '/csv/' + user_id + grid_csv

    df = pd.read_csv(csv_file)
    df_len_list += [df.shape[0]]

    df['datetime'] = pd.to_datetime(df['date'] + " " + df['time'])
    df['datetime'] = df['datetime'].dt.round(round_sec)

    df = df.set_index('datetime').reset_index()
    user_df = df.groupby('datetime')[df.columns[1:].to_list()].mean().reset_index()
    user_df = locationPreprocessor.convert_coord_for_blender_for_user(user_df)

    user_df['datetime'] = pd.to_datetime(user_df['datetime'])
    
    user_df['year'] = user_df['datetime'].dt.year
    user_df['month'] = user_df['datetime'].dt.month
    user_df['week'] = user_df['datetime'].dt.weekday
    user_df['weekend'] = np.where(user_df['week'] < 5, 0, 1)
    user_df['hour'] = user_df['datetime'].dt.hour
    user_df['day'] = user_df['datetime'].dt.day

    # 시간 간격 계산
    user_df['time_diff'] = user_df['datetime'].diff()

    # 1분 이상인 경우 세그먼트로 분류
    threshold = pd.Timedelta(minutes=1)
    user_df['segment'] = (user_df['time_diff'] > threshold).cumsum()

    # save segment_list
    # 세그먼트별 시작 시간과 끝 시간 추출 그리고, 시간 간격별 segment list 추출
    segment_info = user_df.groupby('segment')['datetime'].agg(['min', 'max'])
    segment_info = segment_info.reset_index()
    segment_info['time_gap'] = segment_info['max'] - segment_info['min']
    segment_info['over_time_delta'] = segment_info['time_gap'] >= pd.Timedelta(minutes=time_delta)
    segment_info = segment_info.loc[segment_info['over_time_delta'] == True, :]
    segment_list = segment_info['segment'].to_list()

    segment_df = pd.DataFrame({'segment_list':segment_list})
    seg_list += [segment_df.shape[0]]
    segment_df.to_csv(segment_file, index=False)
    
    # grid process
    user_df = user_df.drop(columns=['altitude', 'what'])
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
    user_df = user_df.drop(columns=['datetime', 'latitude', 'longitude'])
    user_df.to_csv(grid_file, index=False)

segment_col = 'segment_list_' + segment_delta
data_volumn_df = pd.DataFrame({'user_id':user_id_list,
                               'data_vol':df_len_list,
                               segment_col:seg_list})
data_volumn_df = data_volumn_df.sort_values(['data_vol'], ascending=False)
data_volumn_df.to_csv('user_data_volumn.csv', index=False)

df = pd.read_csv('user_data_volumn.csv')
seg_list = []
for id in df['user_id']:
    print(f"user_id: {id}")
    user_id = getUserId(id)
    segment_file = './Data/' + user_id + '/csv/' + user_id + segment_csv
    seg_df = pd.read_csv(segment_file)
    seg_list += [seg_df.shape[0]]
    
df[segment_col] = seg_list
df.to_csv('user_data_volumn.csv', index=False)