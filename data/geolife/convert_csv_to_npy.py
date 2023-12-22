import pandas as pd
import numpy as np
import os

import torch
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).to(device)

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

for id in range(182):
    user_id = getUserId(id)
    csv_file = './Data/' + user_id + '/csv/' + user_id + '.csv'
    df = pd.read_csv(csv_file)

    user0 = df[['days', 'latitude', 'longitude']].copy()
    df_tensor = df_to_tensor(user0)
    batch = df_tensor.shape[0]//1500
    df_tensor = df_tensor[:batch*1500, :]
    df_tensor = df_tensor.reshape(-1, 500, 3)

    dataset = np.array(df_tensor)

    npy_dir = './Data/' + user_id + '/npy/'
    if os.path.isdir(npy_dir) == False:
        os.mkdir(npy_dir)
    np.save(os.path.join(npy_dir, user_id), dataset)
    print(f"user{user_id}: {df_tensor.shape}")