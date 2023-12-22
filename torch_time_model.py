import torch
from torch import nn

class SliceEncoderModel(nn.Module):
    def __init__(self, control=80):
        super(SliceEncoderModel, self).__init__()
        self.control = control
        self.mul     = 1.0
        if control == 100:
            self.control = 1
            self.mul     = 0.0
    
    def forward(self, x):
        # input is TASKS x SAMPLES x TIME X FEATURES
        # ex, time == 100, 
        # t1의 time 은 99, feature 0 ~ n-1
        t1 = x[:,:,-1,:-1]
        # t2의 time은 20, feature n -> 모두 실수 처리
        t2 = x[:,:,-self.control-1,-1:]*self.mul
        # t2 = tf.expand_dims(t2,-1)
        
        # reshaped to  TASKS X SAMPLES X FEATURES
        return torch.concat([torch.from_numpy(t1).float(), torch.from_numpy(t2).float()], axis=-1)