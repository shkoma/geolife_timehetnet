import torch
import torch.nn as nn

# MLP 모델
class MLP(nn.Module):
    def __init__(self, input_shape, y_timestep, label_attribute):
        super(MLP, self).__init__()
        
        self.input_shape = input_shape[-2] * input_shape[-1]
        self.y_timestep = y_timestep
        self.label_attribute = label_attribute
        self.output_shape = y_timestep * label_attribute
        
        self.fc1 = nn.Linear(self.input_shape, 256, dtype=torch.double)
        self.fc2 = nn.Linear(256, 256, dtype=torch.double)
        self.fc3 = nn.Linear(256, self.output_shape, dtype=torch.double)
        
    def forward(self, x):
        batch = torch._shape_as_tensor(x)[0]
        mini_batch = torch._shape_as_tensor(x)[1]
        time_step = torch._shape_as_tensor(x)[2]
        attribute = torch._shape_as_tensor(x)[3]

        in_shape_new = [-1] + [time_step * attribute]
        x = torch.reshape(x, in_shape_new)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        out_shape_new = [batch] + [-1] + [self.y_timestep] + [self.label_attribute]
        out = torch.reshape(x, out_shape_new)
        return out
