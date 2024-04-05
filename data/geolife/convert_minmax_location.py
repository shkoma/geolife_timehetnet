# location(Latitude, Logitude) 에 대한 최소, 최대값을 구해야 함.

import pandas as pd
import numpy as np
import os

class LocationPreprocessor():
    def __init__(self, data_path=""):
        self.data_path = data_path
        self.earth_radius = 6371000
        self.min_max_file = 'min_max_location.csv'
        self.valid_user_file = 'valid_user_list.csv'
        self.center_location = []
        return
    
    def getUserId(self, id):
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

    def get_valid_user_list(self):
        if os.path.isfile(self.data_path + self.valid_user_file):
            df = pd.read_csv(self.data_path + self.valid_user_file)
            return df
        else:
            print(f'{self.data_path + self.valid_user_file} is not valid')
            return pd.DataFrame()
        
    def set_valid_user_list(self):
        min_lat_list = []
        min_lon_list = []
        max_lat_list = []
        max_lon_list = []

        user_id_list = []
        for id in range(182):
            user_id = self.getUserId(id)
            csv_file = self.data_path + '/Data/' + user_id + '/csv/' + user_id + '.csv'
            df = pd.read_csv(csv_file)
            if df.shape[0] < 500:
                continue

            user0 = df[['days', 'latitude', 'longitude']].copy()
            user_id_list += [user_id]
            min_lat_list += [user0['latitude'].min()]
            max_lat_list += [user0['latitude'].max()]
            min_lon_list += [user0['longitude'].min()]
            max_lon_list += [user0['longitude'].max()]

        user_location_df = pd.DataFrame({'user_id':user_id_list,
                                        'min_lat':min_lat_list,
                                        'min_lon':min_lon_list,
                                        'max_lat':max_lat_list,
                                        'max_lon':max_lon_list})

        user_location_df_pre = user_location_df.copy()
        user_location_df_pre = user_location_df_pre.set_index('user_id', drop=True).copy()

        X = ['min_lat', 'min_lon', 'max_lat', 'max_lon']
        q1 = user_location_df_pre[X].quantile(0.25)
        q3 = user_location_df_pre[X].quantile(0.75)
        iqr = (q3-q1) * 1.5

        cond1 = user_location_df_pre[X] >= (q1 - iqr)
        user_location_df_pre = user_location_df_pre[cond1].dropna().copy()

        cond2 = user_location_df_pre[X] <= (q3 + iqr)
        user_location_df_pre = user_location_df_pre[cond2].dropna().copy()

        valid_user_list = user_location_df_pre.index
        print(f'valid user list: {valid_user_list}')

        df = pd.DataFrame({'valid_user_list':valid_user_list})
        df.to_csv(self.valid_user_file, index=False)


    def get_minmax_location(self, isForce = False):
        if isForce == True:
            self.set_valid_user_list()
        else:
            if os.path.isfile(self.min_max_file):
                df = pd.read_csv(self.min_max_file)
                return df
        
        min_lat = 1000000 
        min_lon = 1000000
        max_lat = 0
        max_lon = 0
        
        min_location = []
        max_location = []

        valid_user_list = self.get_valid_user_list()

        if valid_user_list.shape[0] < 2:
            self.set_valid_user_list()
            valid_user_list = self.get_valid_user_list()
            
        for id in valid_user_list['valid_user_list']:
            user_id = self.getUserId(id)
            csv_file = self.data_path  + '/Data/' + user_id + '/csv/' + user_id + '.csv'
            user0 = pd.read_csv(csv_file)
            
            if user0['latitude'].min() < min_lat:
                min_lat = user0['latitude'].min()
            if user0['latitude'].max() > max_lat:
                max_lat = user0['latitude'].max()
            if user0['longitude'].min() < min_lon:
                min_lon = user0['longitude'].min()
            if user0['longitude'].max() > max_lon:
                max_lon = user0['longitude'].max()

        print(f"min_lat: {min_lat}, min_lon: {min_lon}")
        print(f"max_lat: {max_lat}, max_lon: {max_lon}")
        min_location += [min_lat]
        min_location += [min_lon]
        max_location += [max_lat]
        max_location += [max_lon]
        
        df = pd.DataFrame({'min_location':min_location,
                           'max_location':max_location})
        df.to_csv(self.min_max_file, index=False)

        return df

    def get_center_location(self):
        if len(self.center_location) < 2:            
            df = self.get_minmax_location()
            self.center_location = (df['min_location'] + df['max_location'])/2
        return self.center_location
    
    def convert_coord_for_blender(self, lat, lon):
        center_location = self.get_center_location()
        delta_lat = lat - center_location[0]
        delta_lon = lon - center_location[1]
        
        x = delta_lon * self.earth_radius * (np.pi / 180) * np.cos(lat * (np.pi / 180))
        y = delta_lat * self.earth_radius * (np.pi / 180)
        return x, y
    
    def convert_coord_for_blender_for_user(self, user):
        center_location = self.get_center_location()
        delta_lat = user['latitude'] - center_location[0]
        delta_lon = user['longitude'] - center_location[1]
        
        user['x'] = delta_lon * self.earth_radius * (np.pi / 180) * np.cos(user['latitude'] * (np.pi / 180))
        user['y'] = delta_lat * self.earth_radius * (np.pi / 180)
        return user

### Convert user's location(Coordinate)
# locationPreprocessor = LocationPreprocessor()
# locationPreprocessor.get_minmax_location(True)
# valid_user_list = locationPreprocessor.get_valid_user_list()
# for id in valid_user_list['valid_user_list']:
#     print(f"user id: {id}")
#     user_id = locationPreprocessor.getUserId(id)
#     csv_file = './Data/' + user_id + '/csv/' + user_id + '.csv'
#     csv_convert_file = './Data/' + user_id + '/csv/' + user_id + '_converted.csv'
#     user_df = pd.read_csv(csv_file)
#     user_df = locationPreprocessor.convert_coord_for_blender_for_user(user_df)
    
#     user_df['datetime'] = pd.to_datetime(user_df['date'] + " " + user_df['time'])
#     user_df['month'] = user_df['datetime'].dt.month
#     user_df['week'] = user_df['datetime'].dt.weekday
#     user_df['weekend'] = np.where(user_df['week'] < 5, 0, 1)
#     user_df['hour'] = user_df['datetime'].dt.hour
#     user_df['day'] = user_df['datetime'].dt.day
#     user_df.to_csv(csv_convert_file, index=False)