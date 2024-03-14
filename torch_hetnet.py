import os
import numpy as np
import torch
from torch import nn
# from torch.utils.data import DataLoader

class My_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, name=None):
        super(My_Linear, self).__init__()
        self.Linear = nn.Linear(in_features, out_features, bias, device)
        self.name = name
    
    def forward(self, x):
        # print(f"My_Linear - X.shape: {x.shape}, name: {self.name}")
        x = self.Linear(x)
        return x

def getSequential(dims=[32, 32, 1], name=None, activation=None, begin = True, middle = False, final=True, out_features = 1):
    final_list = []
    
    for idx, n in enumerate(dims):
        if final and idx == (len(dims) - 1):
            final_list.append(My_Linear(in_features=dims[idx-1], out_features=1, name=f"{name} - {idx}"))
        else:
            if begin and idx == 0:
                # begin 의 input shape 변경이 필요할 듯
                final_list.append(My_Linear(1, dims[idx], name=f"{name}-{idx}"))
            elif middle and idx == 0:
                final_list.append(My_Linear(dims[0] + 1, dims[idx], name=f"{name}-{idx}"))
            elif final and idx == 0:
                final_list.append(My_Linear(dims[idx-1]*2, dims[idx], name=f"{name}-{idx}"))
            else:    
                final_list.append(My_Linear(dims[idx-1], dims[idx], name=f"{name}-{idx}"))
            
            if activation == 'relu':
                final_list.append(nn.ReLU())
            else:
                final_list.append(nn.Sigmoid())
    return nn.Sequential(*final_list)

# getSequential(activation='relu')

class HetNet(nn.Module):
    def __init__(self, dims = [32, 32, 32],
                       output_shape=[1, 2],
                       acti = 'relu',
                       drop1 = 0.1,
                       drop2 = 0.1,
                       share_qs = False,
                       base = True):
        super(HetNet, self).__init__()
        # self.enc = enc
        # self.enc_type = enc_type
        self.base = base
        self.acti = acti
        self.out_features = output_shape[0] * output_shape[1]
        self.output_shape = output_shape
    
        # Prediction Network
        self.dense_fy = getSequential(dims=dims, activation=acti, out_features=self.out_features, begin = False, final=True, name='pred_fy')
        self.dense_fz = getSequential(dims=dims, activation=acti, begin = False, middle=True, final=False, name='pred_fz')
        self.dense_gz = getSequential(dims=dims, activation=acti, begin = False, final=False, name='pred_gz')
        
        # Support and Query Network (start with both same weight)
        self.dense_fv = getSequential(dims=dims, activation=acti, begin = False, middle=True, final=False, name='s_dense_fv')
        self.dense_gv = getSequential(dims=dims, activation=acti, begin = False, final=False, name='s_dense_gv')
        
        # U net
        self.dense_uf = getSequential(dims=dims, activation=acti, begin = False, middle = True, final=False, name='ux_dense_f')
        self.dense_ug = getSequential(dims=dims, activation=acti, begin = False, final=False, name='ux_dense_g')
        
        # Vbar network
        self.dense_v = getSequential(dims=dims, activation=acti, begin = True, final=False, name='vb_dense_v')
        self.dense_c = getSequential(dims=dims, activation=acti, begin = False, final=False, name='vb_dense_c')
        
        self.drop_layer1 = nn.Dropout(drop1)
        self.drop_layer2 = nn.Dropout(drop2)
        self.drop_layer3 = nn.Dropout(drop2)

    # input should be [samples X features] and [samples X labels]
    # def forward(self, inp, multi_task=True, training=False):
    def forward(self, inp):
        que_x, sup_x, sup_y = inp

        sup_x = sup_x.view(sup_x.shape[0], sup_x.shape[1], -1)
        # sup_x: (5, 10, 1440)
        sup_y = sup_y.view(sup_y.shape[0], sup_y.shape[1], -1)
        # sup_y: (5, 10, 12)
        que_x = que_x.view(que_x.shape[0], que_x.shape[1], -1)

        # sup_x: (5, 10, 120, 12) -> (5, 10, 1440)
        # sup_y: (5, 10, 6, 2) -> (5, 10, 12)
        ##### Inference Network #####
        ##### Vbar Network #####
        # Encode sup_x to FxK
        vs_bar = torch.unsqueeze(sup_x, axis=-1)  # Expand
        # sup_x: (5, 10, 1440, 1)
        vs_bar = self.dense_v(vs_bar)  # Fv
        # sup_x: (5, 10, 1440, 32)
        vs_bar = torch.mean(vs_bar, axis=1)  # mean(Fv)
        # sup_x: (5, 1440, 32)
        vs_bar = self.dense_c(vs_bar)  # Gv(mean(Fv))
        # sup_x: (5, 1440, 32)

        # Encode sup_y to FxK
        cs_bar = torch.unsqueeze(sup_y, axis=-1)
        # cs_bar = (5, 10, 12, 1)
        cs_bar = self.dense_v(cs_bar)  # Fc
        # cs_bar = (5, 10, 12, 32)
        cs_bar = torch.mean(cs_bar, axis=1)  # mean(Fc)
        # cs_bar = (5, 12, 32)
        cs_bar = self.dense_c(cs_bar)  # Gc(mean(Fc))
        # cs_bar = (5, 12, 32)

        ##### U Network #####
        # Tile FxK to NxFxK or NxJxK respectively
        vs_bar = torch.tile(torch.unsqueeze(vs_bar, axis=1), [1, sup_x.shape[1], 1, 1])
        # vs_bar = (5, 10, 1440, 32)
        cs_bar = torch.tile(torch.unsqueeze(cs_bar, axis=1), [1, sup_y.shape[1], 1, 1])
        # cs_bar = (5, 10, 12, 32)

        # Concatenate tiled to NxFxK+1 or NxJxK+1 respectively
        u_xs = torch.concat([torch.unsqueeze(sup_x, axis=-1), vs_bar], axis=-1)
        # u_xs = (5, 10, 1440, 33)
        u_ys = torch.concat([torch.unsqueeze(sup_y, axis=-1), cs_bar], axis=-1)
        # u_ys = (5, 10, 12, 33)

        # Embed latent
        u_xs = self.dense_uf(u_xs)  # Fu([sup_x, vs_bar])
        # u_xs = (5, 10, 1440, 32)
        u_ys = self.dense_uf(u_ys)  # Fu([sup_y, cs_bar])
        # u_ys = (5, 10, 12, 32)

        # attribute mean
        u_xs = torch.mean(u_xs, axis=2)  # mean(Fu([sup_x, vs_bar]))
        # u_xs = (5, 10, 32)
        u_ys = torch.mean(u_ys, axis=2)  # mean(Fu([sup_y, cs_bar]))
        # u_ys = (5, 10, 32)

        # Embed latent
        u_s = u_xs + u_ys  # mean(Fu([sup_x, vs_bar])) + mean(Fu([sup_y, cs_bar]))
        # u_s = (5, 10, 32)
        u_s = self.dense_ug(u_s)  # Gu(mean(Fu([sup_x, vs_bar])) + mean(Fu([sup_y, cs_bar])))
        # u_s = (5, 10, 32)

        ##### Support network #####
        # Tile u features from NxK to NxFxK / NxJxK
        u_s = torch.unsqueeze(u_s, axis=2)
        # u_s = (5, 10, 1, 32)
        u_xs = torch.tile(u_s, [1, 1, sup_x.shape[-1], 1])
        # u_xs = (5, 10, 1440, 32)
        u_ys = torch.tile(u_s, [1, 1, sup_y.shape[-1], 1])
        # u_ys = (5, 10, 12, 32)

        # Concatenate with original and embed to NxFxK
        in_xs = torch.concat([torch.unsqueeze(sup_x, axis=-1), u_xs], axis=-1)
        # in_xs = (5, 10, 1440, 33)
        in_ys = torch.concat([torch.unsqueeze(sup_y, axis=-1), u_ys], axis=-1)
        # in_ys = (5, 10, 12, 33)
        in_xs = self.dense_fv(in_xs)  # Fv([sup_x, u_xs])
        # in_xs = (5, 10, 1440, 32)
        in_ys = self.dense_fv(in_ys)  # Fc([sup_y, u_ys])
        # in_ys = (5, 10, 12, 32)

        # Aggregate and embed to FxK / JxK
        in_xs = torch.mean(in_xs, axis=1)  #
        # in_xs = (5, 1440, 32)
        in_ys = torch.mean(in_ys, axis=1)  # mean(Fc([sup_y, u_ys]))
        # in_ys = (5, 12, 32)
        in_xs = self.dense_gv(in_xs)  # in_xs = Gv(mean(Fc([sup_y, u_ys])))
        # in_xs = (5, 1440, 32)
        in_ys = self.dense_gv(in_ys)  # in_ys = Gv(mean(Fc([sup_y, u_ys])))
        # in_ys = (5, 12, 32)

        # Dropout
        in_xs = self.drop_layer1(in_xs)
        in_ys = self.drop_layer3(in_ys)

        ##### Prediction net #####
        # Tile support_net outputs to NxFxK / NxJxK
        p_xs = torch.tile(torch.unsqueeze(in_xs, axis=1), [1, que_x.shape[1], 1, 1])
        # p_xs = (5, 10, 1440, 32)
        p_ys = torch.tile(torch.unsqueeze(in_ys, axis=1), [1, que_x.shape[1], 1, 1])
        # p_ys = (5, 10, 12, 32)

        # [que_x, p_xs]
        z = torch.concat([torch.unsqueeze(que_x, axis=-1), p_xs], axis=-1)
        # z = (5, 10, 1440, 33)

        z = self.dense_fz(z)  # Fz([que_x, p_xs])
        # z = (5, 10, 1440, 32)
        z = torch.mean(z, axis=2)  # Nq x K  # mean(Fz([que_x, p_xs]))
        # z = (5, 10, 32)
        z = self.dense_gz(z)  # Gz(mean(Fz([que_x, p_xs])))
        # z = (5, 10, 32)

        z = torch.unsqueeze(z, axis=2)
        # z = (5, 10, 1, 32)

        z = torch.tile(z, [1, 1, sup_y.shape[-1], 1])
        # z = (5, 10, 12, 32)
        y = torch.concat([z, p_ys], axis=-1)
        # y = (5, 10, 12, 64)
        y = self.dense_fy(y)
        # y = (5, 10, 12)
        y = y.view(sup_y.shape[0], sup_y.shape[1], -1, 2)

        return y
