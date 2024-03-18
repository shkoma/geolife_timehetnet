import ast
import os
import numpy as np
import torch
from torch import nn
from args            import argument_parser
import time

# args = argument_parser()
# global_first_feature = ast.literal_eval(args.dims)[0]

global_first_feature =  ast.literal_eval("[32, 32, 32]")[0]
class My_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, name=None):
        super(My_Linear, self).__init__()
        self.Linear = nn.Linear(in_features, out_features, bias, device, dtype=torch.double)
        self.name = name
    
    def forward(self, x):
        # print(f"My_Linear - X.shape: {x.shape}, name: {self.name}")
        x = self.Linear(x)
        return x
    
class My_Conv1d(nn.Module):
    def __init__(self, in_features, out_features, 
                 kernel_size=1, stride=1, padding=1, dilation=1,
                 groups=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None, name=None):
        super(My_Conv1d, self).__init__()
        self.Conv1d = nn.Conv1d(in_features, out_features, kernel_size,
                                stride, padding, dilation, groups, bias,
                                padding_mode, device, dtype=torch.double)
        self.name = name
    
    def forward(self, x):
        # print(f"My_Conv1d - X.shape: {x.shape}, name: {self.name}")
        x = self.Conv1d(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, dims=[32,32,1], 
                 input_shape=None,
                 activation=None,
                 name=None,
                 final=True,
                 batchnorm=False,
                 dilate=False,
                 first_features=global_first_feature):
        super(ConvBlock, self).__init__()
        self.batchnorm = batchnorm
        self.final = final
        self.dilation = [1, 1, 1]
        
        
        # https://wikidocs.net/194947
        # in_channels: 입력 데이터의 채널 개수입니다. 예를 들어, 입력 데이터가 RGB 이미지인 경우 in_channels는 3이 됩니다.
        # out_channels: 출력 데이터의 채널 개수입니다. 이는 컨볼루션 필터의 개수를 의미하며, 출력 데이터가 몇 개의 특징 맵으로 변환되는지를 결정합니다.
        self.c1 = My_Conv1d(in_features=first_features, out_features=dims[0],
                            kernel_size=3, name=f"{name}-0",
                            padding='same', dilation=self.dilation[0])
        # self.relu1 = nn.ReLU()
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.c2 = My_Conv1d(in_features=dims[0], out_features=dims[1],
                            kernel_size=3, name=f"{name}-1",
                            padding='same', dilation=self.dilation[1])
        # self.relu2 = nn.ReLU()
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.c3 = My_Conv1d(in_features=dims[1], out_features=dims[2],
                            kernel_size=3, name=f"{name}-1",
                            padding='same', dilation=self.dilation[2]) 
        if not self.final:
            self.relu3 = nn.LeakyReLU(negative_slope=0.1) #nn.ReLU()
        
        if self.batchnorm:
            self.bn1 = nn.BatchNorm1d()
            self.bn2 = nn.BatchNorm1d()
            self.bn3 = nn.BatchNorm1d()
    
    def forward(self, inp): # inp: (3, 20, 9, 100, 33)
        shape = torch._shape_as_tensor(inp)
        x = torch.reshape(inp, [-1, shape[-2], shape[-1]])
        x = x.transpose(-2, -1)
    
        out = self.c1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.c2(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.c3(out)
        if self.batchnorm:
            out = self.bn3(out)
        if not self.final:
            out = self.relu3(out)
        
        out = out.transpose(-1, -2)
        
        out_shape = torch._shape_as_tensor(out)
        new_shape = shape[:-1].tolist() + [out_shape[-1]]
        out = torch.reshape(out, new_shape)
        # print(f"out_shape:{out.shape}")
        return out

class My_GRU(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, batch_first=True,
                 bidirectional=False, name=None):
        super(My_GRU, self).__init__()
        self.GRU = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first, bidirectional, dtype=torch.double)
        self.name = name
    # input_size:입력크기로 훈련 데이터셋의 칼럼 갯수
    # hidden_size: 은닉층의 뉴런 갯수
    # num_layers: GRU 계층의 갯수
    
    # example of gru
    # gru = nn.GRU(input_size=10, 
    #              hidden_size=32, 
    #              num_layers=1, batch_first=True, bidirectional=False)
    # inputs = torch.zeros(1, 35, 10)
    # outputs, hidden_state = gru(inputs)
    # print(outputs.shape, hidden_state.shape)
    # # torch.Size([1, 35, 32]) torch.Size([1, 1, 32])

    def forward(self, x):
        # print(f"My_Gru - X.shape: {x.shape}, name: {self.name}")
        x = self.GRU(x)
        return x
    
class GruBlock(nn.Module):
    def __init__(self, dims=[32, 32, 1], input_shape=None, 
                 activation = None, name=None, final=False):
        super(GruBlock, self).__init__()
        self.final = final
        # input_shape - (batch, input_features)
        # input_size 는 input_features만 필요
        self.gru = My_GRU(input_size=input_shape[1],
                          num_layers=len(dims),
                          hidden_size=dims[0],
                          name=f"{name}")

    def forward(self, inp):
        # input is TASKS x SAMPLES x FEATURES x TIME x Latent vector
        shape = torch._shape_as_tensor(inp)
        # (3, 20, 6, 100, 1)
        x = torch.reshape(inp, [-1, shape[-2], shape[-1]])
        # (300, 100, 1)
        x, f = self.gru(x)
        # x:(300, 100, 32)
        # f:(3, 100, 32)
        
        if self.final:
            new_shape = shape[:-2].tolist() + [-1]
            out = torch.reshape(f, new_shape)
        else:
            new_shape = shape[:-1].tolist() + [-1]
            # (3, 20, 6, 100, -1)
            out = torch.reshape(x, new_shape)
            # (3, 20, 6, 100, 32)
        return out


def getSequential(dims=[32, 32, 1], name=None, activation=None,
                  begin = True, middle = False, y_middle = False, final=True,
                  out_features = 2, length = 0):
    final_list = []
    for idx, n in enumerate(dims):
        if final and idx == (len(dims) - 1):
            final_list.append(My_Linear(in_features=dims[idx-1], out_features=out_features, name=f"{name} - {idx}"))
        else:
            if begin and idx == 0: # begin 의 input_shape 변경이 필요할 듯
                final_list.append(My_Linear(1, dims[idx], name=f"{name}-{idx}"))
            elif middle and idx == 0:
                final_list.append(My_Linear(dims[idx] + 1, dims[idx], name=f"{name}-{idx}"))
            elif y_middle and idx == 0:
                final_list.append(My_Linear((dims[idx] + out_features), dims[idx], name=f"{name}-{idx}"))
            elif final and idx == 0:
                final_list.append(My_Linear(dims[idx]*2, dims[idx], name=f"{name}-{idx}"))
            else:    
                final_list.append(My_Linear(dims[idx], dims[idx], name=f"{name}-{idx}"))
            
            if activation == 'relu':
                # final_list.append(nn.ReLU())
                final_list.append(nn.LeakyReLU(negative_slope=0.1))
            else:
                final_list.append(nn.Sigmoid())
    return nn.Sequential(*final_list)

def getTimeBlock(block = 'conv', dims=[32, 32, 1], input_shape=None, activation=None, name=None, final=True, batchnorm=False, dilate=False, first_features=global_first_feature):
    if block == 'conv':
        return ConvBlock(dims=dims,input_shape=input_shape,activation=activation,name=name,final=final,batchnorm=batchnorm,dilate=dilate, first_features=first_features)
    elif block == 'gru':
        return GruBlock(dims=dims,input_shape=input_shape,activation=activation,name=name,final=final)
    else:
        raise ValueError(f"Block type {block} not defined.")
    
class TimeHetNet(nn.Module):
    def __init__(self,
                dims_inf = [32,32,32],
                dims_pred = [32,32,32],
                activation = "relu",
                time=100,
                batchnorm = False,
                block = "conv",
                merge = False,
                dropout = 0.0,
                output_shape = [1, 2],
                length = 0):
        super(TimeHetNet, self).__init__()
        self.enc_type = 'None'
        
        if len(block) == 1:
            block = f"{block},{block},{block},{block}"
        self.block = block
        self.output_shape = output_shape
        self.out_features = (output_shape[0] * output_shape[1])
        self.length = length #* output_shape[1]

        ## Prediction network
        self.dense_fz = getSequential(dims=dims_pred, 
                                      activation=activation,
                                      begin=False,
                                      middle=False, 
                                      final=True, 
                                      name="pred_dense_fz",
                                      out_features=self.out_features,
                                      length=self.length)
        self.time_fz = getTimeBlock(block=block[-1], 
                                    dims=dims_pred, 
                                    activation=activation,
                                    final=False,
                                    name='pred_time_fz',
                                    input_shape=(time, dims_inf[-1]+1),
                                    batchnorm=batchnorm)

        self.time_gz = getTimeBlock(block=block[-1],
                                    dims=dims_pred,
                                    activation=activation,
                                    final=False,name="pred_time_gz",
                                    input_shape=(time, dims_pred[-1]),
                                    batchnorm=batchnorm)
        
        ## Support and Query network (start with both same weights)
        self.time_fv = getTimeBlock(block=block[2],
                                    dims=dims_inf,
                                    activation=activation,
                                    final=False,
                                    name="s_time_fv",
                                    input_shape=(time, dims_inf[-1]+1),
                                    batchnorm=batchnorm,
                                    first_features=dims_inf[-1]+1)
        self.time_gv = getTimeBlock(block=block[2],
                                    dims=dims_inf,
                                    activation=activation,
                                    final=False,name="s_time_gv",
                                    input_shape=(time, dims_inf[-1]),
                                    batchnorm=batchnorm)
        self.dense_fv = getSequential(dims=dims_inf,
                                      activation=activation,
                                      begin=False,
                                      middle=False,
                                      y_middle=True,
                                      final=False,
                                      name="s_dense_fv",
                                      out_features=self.out_features,
                                      length=self.length)
        # # U net
        self.dense_uf = getSequential(dims=dims_inf,
                                      activation=activation,
                                      begin=False,
                                      middle=True,
                                      final=False,
                                      name="ux_dense_f",
                                      out_features=self.out_features,
                                      length=self.length)
        self.time_uf = getTimeBlock(block=block[1],
                                    dims=dims_inf,
                                    activation=activation,
                                    final=False,
                                    name="ux_time_f",
                                    input_shape=(time, dims_inf[-1]+1),
                                    batchnorm=batchnorm,
                                    first_features=dims_inf[-1]+1)
        self.time_ug = getTimeBlock(block=block[1],
                                    dims=dims_inf,
                                    activation=activation,
                                    final=False,
                                    name="ux_time_g",
                                    input_shape=(time, dims_inf[-1]),
                                    batchnorm=batchnorm)

        # # Vbar network
        self.time_v  = getTimeBlock(block=block[0],
                                    dims=dims_inf,
                                    activation=activation,
                                    final=False,
                                    name="vb_time_v, fv_bar",
                                    input_shape=(time, 1), # input_shape 의 1 수정이 필요할 듯?
                                    batchnorm=batchnorm)
        self.time_c  = getTimeBlock(block=block[0],
                                    dims=dims_inf,
                                    activation=activation,
                                    final=False,
                                    name="vb_time_c, gv_bar", 
                                    input_shape=(time, dims_inf[-1]),
                                    batchnorm=batchnorm)

        self.dense_v  = getSequential(dims=dims_inf,
                                      activation=activation,
                                      begin=True,
                                      middle=False,
                                      final=False,
                                      name="vb_dense_v, fc_bar",
                                      out_features=self.out_features,
                                      length=self.length)
        self.dense_c  = getSequential(dims=dims_inf,
                                      activation=activation,
                                      begin=False,
                                      middle=False,
                                      final=False,
                                      name="vb_dense_c, gc_bar",
                                      out_features=self.out_features,
                                      length=self.length)
    
    # input should be [Metabatch x samples X Time X features] and 
    #                 [Metabatch samples X labels]
    def forward(self, inp):
        que_x, sup_x, sup_y = inp
        # sup_y = sup_y[:, :, :, 1:].reshape(sup_y.shape[0], sup_y.shape[1], -1)
        sup_y = sup_y.view(sup_y.shape[0], sup_y.shape[1], -1)
        
        M = torch._shape_as_tensor(sup_x)[0] # Batch (user_id)
        N = torch._shape_as_tensor(sup_x)[1] # Mini-Batch
        T = torch._shape_as_tensor(sup_x)[2] # Time (row)
        F = torch._shape_as_tensor(sup_x)[3] # Channels/Features (column)

        zero_count = torch.sum(torch.concat([que_x, sup_x], axis=1), axis=[1, 3])
        zero_count = torch.count_nonzero(zero_count, dim=1)
        zero_count = torch.unsqueeze(zero_count, -1)
        zero_count = torch.unsqueeze(zero_count, -1)

        ##### Vbar network #####
        # Encode sup_x MxNxTxF to MxFxTxK (DS over Instances)
        vs_bar = torch.transpose(sup_x, 3, 2)
        # vs_bar: (5, 10, 12, 120)
        vs_bar = torch.unsqueeze(vs_bar, -1) # 각 채널당 Time의 변화를 볼 수 있게 만듬
        # vs_bar: (5, 10, 12, 120, 1)
        vs_bar = self.time_v(vs_bar) # gru (fv bar)
        # vs_bar: (5, 10, 12, 120, 64)
        
        vs_bar = torch.mean(vs_bar, axis=1)
        # vs_bar: (5, 12, 120, 64)
        vs_bar = self.time_c(vs_bar) # conv (gv bar)
        # vs_bar: (5, 12, 120, 64)
        vs_bar = torch.transpose(vs_bar, 2, 1)
        # vs_bar: (5, 120, 12, 64)
        
        # Encode sup_y MxNx1 to Mx1xK, T' = 1 인 상태, T'= # 일 경우, #의 y'을 예측
        cs_bar = torch.unsqueeze(sup_y, axis=-1) 
        # cs_bar: (5, 10, 12, 1)
        cs_bar = self.dense_v(cs_bar) # fc bar
        # cs_bar: (5, 10, 12, 32)
        cs_bar = torch.mean(cs_bar, axis=1) 
        # cs_bar: (5, 12, 32)
        cs_bar = self.dense_c(cs_bar) # gc bar
        # cs_bar: (5, 12, 32)
        
        ##### U network ##### (DS over Channels)
        vs_bar = torch.tile(torch.unsqueeze(vs_bar, axis=1),[1,sup_x.shape[1],1,1,1]) # M,N,T,F,K
        # vs_bar: (5, 10, 120, 12, 64)
        u_xs = torch.concat([torch.unsqueeze(sup_x, axis=-1), vs_bar], -1)
        # u_xs: (5, 10, 120, 12, 65)
        
        u_xs = torch.transpose(u_xs, 3, 2)
        # u_xs: (5, 10, 12, 120, 65)
        # 각 channel(feature) 별 시간의 변화량으로 data을 읽을 수 있음
        u_xs = self.time_uf(u_xs) # conv, Fu
        # u_xs: (5, 10, 12, 120, 64)

        u_xs = torch.mean(u_xs, axis=2) 
        # u_xs:(3, 20, 100, 32), 각 sample(20)별 channel의 평균
        
        cs_bar = torch.tile(torch.unsqueeze(cs_bar, axis=1), [1,N,1,1]) 
        # cs_bar: (3, 20, 1, 32) # MxNx1xK 
        u_ys = torch.concat([torch.unsqueeze(sup_y, axis=-1), cs_bar], axis=-1) 
        # u_ys: (3, 20, 1, 33)              # MxNx1x(K+1) 
        u_ys = self.dense_uf(u_ys)          # MxNx1xK # u_ys(3, 20, 1, 32)
        u_ys = torch.mean(u_ys, axis=2)     # MxNxK # u_ys:(3, 20, 32)
        u_ys = torch.unsqueeze(u_ys, 2)     # MxNxK # u_ys:(3, 20, 1, 32)
        u_ys = torch.tile(u_ys, [1,1,T,1])  # MxNxTxK # u_ys(3, 20, 100, 32)

        u_s  = u_xs + u_ys         # MxNxTxK # u_s:(3, 20, 100, 32)
        u_s = self.time_ug(u_s)    # Conv-Gu MxNxTxK # u_s:(3, 20, 100, 32) (batch, N, T, K)
        
        #### Inference Network #### (DS over Instances)
        in_xs = torch.tile(torch.unsqueeze(u_s, axis=3), [1, 1, 1, F, 1]) 
        # in_xs: (3, 20, 100, 6, 32)
        in_xs = torch.concat([torch.unsqueeze(sup_x, axis=-1) , in_xs], -1)
        # in_xs: (3, 20, 100, 6, 33)
        
        in_xs = torch.transpose(in_xs, 3, 2) # in_xs: (3, 20, 6, 100, 33)
        in_xs = self.time_fv(in_xs) # fv, in_xs: (3, 20, 6, 100, 32)
        in_xs = torch.mean(in_xs, axis=1) # in_xs: (3, 6, 100, 32)
        in_xs = self.time_gv(in_xs) # gv, in_xs: (3, 6, 100, 32)
        in_xs = torch.transpose(in_xs, 2, 1) # in_xs: (3, 100, 6, 32)
        
        # Label encoding
        in_ys = torch.mean(u_s, axis=2) # in_ys: (3, 20, 32)
        in_ys = torch.concat([sup_y, in_ys], axis=-1) # in_ys: (3, 20, 33)
        in_ys = self.dense_fv(in_ys) # gw, in_ys: (3, 20, 32)
        
        #### Prediction Network ####
        p_xs = torch.tile(torch.unsqueeze(in_xs, axis=1), [1, N, 1, 1, 1]) # p_xs: (3, 20, 100, 6, 32)
        que_x_1 = torch.unsqueeze(que_x, axis=-1) # que_x_1:(3, 20, 100, 6, 1)
        # que_x_1 = torch.tile(torch.unsqueeze(que_x, axis=-1), [1, N, 1, 1, 1]) 
        
        z = torch.concat([p_xs, que_x_1], axis=-1) # z: (3, 20, 100, 6, 33)
        z = torch.transpose(z, 3, 2) # z: (3, 20, 6, 100, 33)
        z = self.time_fz(z) # z: (3, 20, 6, 100, 32)
        
        # (Ds over channels)
        z = torch.mean(z, axis=2) # z: (3, 20, 100, 33)
        z = self.time_gz(z) # z: (3, 20, 100, 33)
        
        if self.block[-1] == 'gru':
            out = z[:, :, -1, :] # out: (3, 20, 32)
        else:
            # reduce time array
            if self.zero_div:
                out = torch.sum(z, axis=-2)
                if zero_count == 0:
                    out = 0
                else:
                    out = torch.div(out, zero_count)
                    
        out = torch.concat([out, in_ys], -1) # out: (3, 20, 64)
        out = self.dense_fz(out) # out: (3, 20, 1) # output 의 1의 값이 변경될 수 있어야 한다. T'의 y' 값
        out = out.view(sup_y.shape[0], sup_y.shape[1], -1, 2)
        return out
