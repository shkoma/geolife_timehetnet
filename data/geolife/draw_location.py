from convert_minmax_location import LocationPreprocessor
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as seaborn

import matplotlib.dates as dates
import datetime as dt

locationPreprocessor = LocationPreprocessor()
user_id = locationPreprocessor.getUserId(1)
csv_file = './Data/' + user_id + '/csv/' + user_id + '.csv'
csv_convert_file = './Data/' + user_id + '/csv/' + user_id + '_converted.csv'
user = pd.read_csv(csv_convert_file)

user_df = user[['latitude', 'longitude', 'x', 'y', 'days', 'time']].copy()

idx_list = []
for idx in range(user_df.shape[0]):
    if idx % 60 == 0: # 5 mins
        idx_list += [idx]
len(idx_list)

idx_list_partial = idx_list[-50:]
user_df_1 = user_df.iloc[idx_list_partial, :].copy()

plt.figure(figsize=(60, 12))
axes = plt.axes(projection='3d')
axes.view_init(elev=10, azim=-80)

axes.scatter3D(user_df_1['days'], user_df_1['x'], user_df_1['y'], s=20)
axes.plot3D(user_df_1['days'], user_df_1['x'], user_df_1['y'])
axes.set_xlabel('time')
axes.set_ylabel('latitude')
axes.set_zlabel('longitude')

user_df_1.head(5)