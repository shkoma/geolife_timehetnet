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
        print(f"My_Linear - X.shape: {x.shape}, name: {self.name}")
        x = self.Linear(x)
        return x

def getSequential(dims=[32, 32, 1], name=None, activation=None, begin = True, middle = False, final=True):
    final_list = []
    
    for idx, n in enumerate(dims):
        if final and idx == (len(dims) - 1):
            final_list.append(My_Linear(in_features=dims[idx-1], out_features=1, name=f"{name} - {idx}"))
        else:
            if begin and idx == 0:
                # begine 의 input shape 변경이 필요할 듯
                final_list.append(My_Linear(1, dims[idx], name=f"{name}-{idx}"))
            elif middle and idx == 0:
                final_list.append(My_Linear(33, dims[idx], name=f"{name}-{idx}"))
            elif final and idx == 0:
                final_list.append(My_Linear(64, dims[idx], name=f"{name}-{idx}"))
            else:    
                final_list.append(My_Linear(dims[idx], dims[idx], name=f"{name}-{idx}"))
            
            if activation == 'relu':
                final_list.append(nn.ReLU())
            else:
                final_list.append(nn.Sigmoid())
    return nn.Sequential(*final_list)

# getSequential(activation='relu')

class HetNet(nn.Module):
    def __init__(self, enc,
                       enc_type = 'slice',
                       dims = [32, 32, 32],
                       acti = 'relu',
                       drop1 = 0.1,
                       drop2 = 0.1,
                       share_qs = False,
                       base = True):
        super(HetNet, self).__init__()
        self.enc = enc
        self.enc_type = enc_type
        self.base = base
        self.acti = acti
    
        # Prediction Network
        self.dense_fy = getSequential(dims=dims, activation=acti, begin = False, final=True, name='pred_fy')
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

    def sub_call2(self, inp, training=False):
        que_x, sup_x, sup_y = inp
        print(f"sub_call2")
        print(f"que_x:{que_x.shape}, sup_x:{sup_x.shape}, sup_y:{sup_y.shape}")

        ##### Inference Network #####
        ##### Vbar Network #####
        # Encode sup_x to FxK
        vs_bar = torch.unsqueeze(sup_x, axis=-1) # Expand
        vs_bar = self.dense_v(vs_bar)            # Fv
        print(f"vs_bar:{vs_bar.shape}")
        vs_bar = torch.mean(vs_bar, axis=1)      # mean(Fv)
        vs_bar = self.dense_c(vs_bar)            # Gv(mean(Fv))
        print('Encode sup_x to FxK - vs_bar.shape:', vs_bar.shape)
        
        if not self.base:
            # Encode que_x to FxK
            vq_bar = torch.unsqueeze(que_x, axis=-1)
            vq_bar = self.dense_v(vq_bar)
            vq_bar = torch.mean(vq_bar, axis=1)
            vq_bar = self.dense_c(vq_bar)
            print('Encode que_x to FxK - vq_bar.shape', vq_bar.shape)
        
        # Encode sup_y to FxK
        cs_bar = torch.unsqueeze(sup_y, axis=-1)
        cs_bar = self.dense_v(cs_bar)            # Fc 
        cs_bar = torch.mean(cs_bar, axis=1)      # mean(Fc)
        cs_bar = self.dense_c(cs_bar)            # Gc(mean(Fc))
        print('Encode sup_y to FxK - cs_bar.shape:', cs_bar.shape)
        
        ##### U Network #####
        # Tile FxK to NxFxK or NxJxK respectively
        vs_bar = torch.tile(torch.unsqueeze(vs_bar, axis=1), [1, torch._shape_as_tensor(sup_x)[1], 1, 1])
        if not self.base:
            vq_bar = torch.tile(torch.unsqueeze(vq_bar,axis=1), [1, torch._shape_as_tensor(que_x)[1], 1, 1])
        cs_bar = torch.tile(torch.unsqueeze(cs_bar,axis=1), [1, torch._shape_as_tensor(sup_y)[1], 1, 1])
        print('Tile FxK to NxFxK or NxJxK respectively - vs_bar.shape, cs_bar.shape')
        print(vs_bar.shape, cs_bar.shape)
        
        # Concatenate tiled to NxFxK+1 or NxJxK+1 respectively
        # [sup_x, vs_bar]
        u_xs = torch.concat([torch._cast_Float(torch.unsqueeze(sup_x, axis=-1)), vs_bar], axis=-1)
        if not self.base:
            u_xq = torch.concat([torch._cast_Float(torch.unsqueeze(que_x, axis=-1)), vq_bar], axis=-1)
        # [sup_y, cs_bar]
        u_ys = torch.concat([torch._cast_Float(torch.unsqueeze(sup_y, axis=-1)), cs_bar], axis=-1)
        print('Concatenate tiled to NxFxK+1 or NxJxK+1 respectively - u_xs.shape, u_ys.shape')
        print(u_xs.shape, u_ys.shape)
        
        # Embed latent
        u_xs = self.dense_uf(u_xs)          # Fu([sup_x, vs_bar])
        if not self.base:
            u_xq = self.dense_uf(u_xq)
        u_ys = self.dense_uf(u_ys)          # Fu([sup_y, cs_bar])   
        u_xs = torch.mean(u_xs, axis=2)     # mean(Fu([sup_x, vs_bar]))
        if not self.base:
            u_xq = torch.mean(u_xq, axis=2)
        u_ys = torch.mean(u_ys, axis=2)     # mean(Fu([sup_y, cs_bar]))
        print('Embed latent - u_xs.shape, u_ys.shape')
        print(u_xs.shape, u_ys.shape)
        
        u_s  = u_xs + u_ys                  # mean(Fu([sup_x, vs_bar])) + mean(Fu([sup_y, cs_bar])) 
        u_s  = self.dense_ug(u_s)           # Gu(mean(Fu([sup_x, vs_bar])) + mean(Fu([sup_y, cs_bar])))
        if not self.base:
            u_q  = self.dense_ug(u_xq)
        print('u_s.shape:', u_s.shape)
        
        ##### Support network #####
        # Tile u features from NxK to NxFxK / NxJxK
        u_xs = torch.tile(torch.unsqueeze(u_s, axis=2), [1, 1, torch._shape_as_tensor(sup_x)[2], 1])
        if not self.base:
            u_xq = torch.tile(torch.unsqueeze(u_q,axis=2),[1, 1, torch._shape_as_tensor(que_x)[2], 1])
        u_ys = torch.tile(torch.unsqueeze(u_s, axis=2), [1, 1, torch._shape_as_tensor(sup_y)[2], 1])
        print('# Tile u features from NxK to NxFxK / NxJxK')
        print("u_xs.shape, u_ys.shape:", u_xs.shape, u_ys.shape)
        
        # Concatenate with original and embed to NxFxK
        # [sup_x, u_xs]
        in_xs = torch.concat([torch._cast_Float(torch.unsqueeze(sup_x, axis=-1)), u_xs], axis=-1)
        if not self.base:
            in_xq = torch.concat([torch._cast_Float(torch.unsqueeze(que_x, axis=-1)), u_xq], axis=-1)
        # [sup_y, u_ys]
        in_ys = torch.concat([torch._cast_Float(torch.unsqueeze(sup_y, axis=-1)), u_ys], axis=-1)
        in_xs = self.dense_fv(in_xs)          # Fv([sup_x, u_xs])
        if not self.base:
            in_xq = self.dense_fv(in_xq)
        in_ys = self.dense_fv(in_ys)          # Fc([sup_y, u_ys])

        # Aggregate and embed to FxK / JxK
        in_xs = torch.mean(in_xs, axis=1)     # mean(Fv([sup_x, u_xs]))
        if not self.base:
            in_xq = torch.mean(in_xq, axis=1)
        in_ys = torch.mean(in_ys, axis=1)     # mean(Fc([sup_y, u_ys]))
        in_xs = self.dense_gv(in_xs)          # in_xs = Gv(mean(Fc([sup_y, u_ys])))
        if not self.base:
            in_xq = self.dense_gv(in_xq)
        in_ys = self.dense_gv(in_ys)          # in_ys = Gv(mean(Fc([sup_y, u_ys])))

        # Dropout
        in_xs = self.drop_layer1(in_xs)
        if not self.base:
            in_xq = self.drop_layer2(in_xq)
        in_ys = self.drop_layer3(in_ys)
        
        ##### Prediction net ##### 
        # Tile support_net outputs to NxFxK / NxJxK
        #if True:
        p_xs = torch.tile(torch.unsqueeze(in_xs, axis=1), [1, torch._shape_as_tensor(que_x)[1], 1, 1])
        if not self.base:
            p_xq = torch.tile(torch.unsqueeze(in_xq, axis=1), [1, torch._shape_as_tensor(que_x)[1], 1, 1])
        p_ys = torch.tile(torch.unsqueeze(in_ys, axis=1), [1, torch._shape_as_tensor(que_x)[1], 1, 1]) 

        if not self.base:
            z = torch.concat([torch._cast_Float(torch.unsqueeze(que_x, axis=-1)), p_xs, p_xq], axis=-1)
        else:
        # [que_x, p_xs]
            z = torch.concat([torch._cast_Float(torch.unsqueeze(que_x, axis=-1)), p_xs], axis=-1)

        z = self.dense_fz(z)                # Fz([que_x, p_xs])
        z = torch.mean(z, axis=2) # Nq x K  # mean(Fz([que_x, p_xs]))
        z = self.dense_gz(z)                # Gz(mean(Fz([que_x, p_xs])))

        z = torch.tile(torch.unsqueeze(z, axis=2),[1, 1, torch._shape_as_tensor(sup_y)[2], 1])
        y = torch.concat([z, p_ys], axis=-1)
        y = self.dense_fy(y)
        # return y (que_y)
        return torch.squeeze(y, axis=-1)
        
    # input should be [samples X features] and [samples X labels]
    def forward(self, inp, multi_task=True, training=False):
        que_x, sup_x, sup_y = inp
        print(f"torch_hetnet forward")
        print(f"que_x:{que_x.shape}, sup_x:{sup_x.shape}, sup_y:{sup_y.shape}")
        
        que_x = self.enc(que_x)
        sup_x = self.enc(sup_x)
        sup_y = torch.from_numpy(sup_y).float()
        
        print(f"after encode")
        print(f"que_x:{que_x.shape}, sup_x:{sup_x.shape}, sup_y:{sup_y.shape}")
        
        inp = (que_x, sup_x, sup_y)
        
        if multi_task:
            return self.sub_call2(inp)
        return None