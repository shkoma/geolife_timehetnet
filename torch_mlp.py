import torch
import torch.nn as nn

# MLP 모델
class MLP(nn.Module):
    def __init__(self, input_shape, y_timestep, label_attribute, cell=256, num_of_hidden=2):
        super(MLP, self).__init__()
        
        self.input_shape = input_shape[-2] * input_shape[-1]
        self.y_timestep = y_timestep
        self.label_attribute = label_attribute
        self.output_shape = y_timestep * label_attribute
        self.cell = cell
        self.num_of_hidden = num_of_hidden
        self.fc_list= self.getSequential()
    
    def getSequential(self):
        final_list = []
        final_list.append(nn.Linear(self.input_shape, self.cell, dtype=torch.double))
        final_list.append(nn.LeakyReLU(negative_slope=0.1))

        for _ in range(self.num_of_hidden):
            final_list.append(nn.Linear(self.cell, self.cell, dtype=torch.double))
            final_list.append(nn.LeakyReLU(negative_slope=0.1))

        final_list.append(nn.Linear(self.cell, self.output_shape, dtype=torch.double))
        final_list.append(nn.LeakyReLU(negative_slope=0.1))
        return nn.Sequential(*final_list)

    def forward(self, x):
        batch = torch._shape_as_tensor(x)[0]
        mini_batch = torch._shape_as_tensor(x)[1]
        time_step = torch._shape_as_tensor(x)[2]
        attribute = torch._shape_as_tensor(x)[3]

        in_shape_new = [-1] + [time_step * attribute]
        x = torch.reshape(x, in_shape_new)

        x  = self.fc_list(x)
        
        out_shape_new = [batch] + [-1] + [self.y_timestep] + [self.label_attribute]
        out = torch.reshape(x, out_shape_new)
        return out