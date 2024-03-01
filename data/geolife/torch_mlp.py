import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(MLP, self).__init__()
        self.input_shape = input_shape[0] * input_shape[1]
        
        self.final_shape = output_shape
        self.output_shape = output_shape[0] * output_shape[1]
        print(f"input_size: {self.input_shape}")
        print(f"output_size: {self.output_shape}")

        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.output_shape)
        
    def forward(self, x):
        x = x.view(-1, self.input_shape)
        # x = torch.reshape(x, in_shape_new)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # out_shape = torch._shape_as_tensor(x)
        # out_shape_new = out_shape[:-1].() + [-1] + [self.output_shape[1]]
        out = torch.reshape(x, self.final_shape)
        return out