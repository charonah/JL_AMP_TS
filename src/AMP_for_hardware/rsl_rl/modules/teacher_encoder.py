import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import LSTM

class Teacher_encoder(nn.Module):
    def __init__(self, 
                        num_privileged_obs,
                        num_terrain_obs,
                        num_privileged_output,
                        num_terrain_output,
                        privileged_hidden_dims=[64,32],
                        terrain_hidden_dims=[256,128],
                        activation='elu',
                        **kwargs):
        
        if kwargs:
            print("Teacher_encoder.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(Teacher_encoder, self).__init__()

        
        mlp_input_dim_p = num_privileged_obs
        mlp_input_dim_t = num_terrain_obs

        activation = get_activation(activation)
        
        privileged_encoder_layers = []
        print(privileged_hidden_dims[0])
        privileged_encoder_layers.append(nn.Linear(mlp_input_dim_p, privileged_hidden_dims[0]))
        privileged_encoder_layers.append(activation)
        for l in range(len(privileged_hidden_dims)):
            if l == len(privileged_hidden_dims) - 1:
                privileged_encoder_layers.append(nn.Linear(privileged_hidden_dims[l], num_privileged_output))
            else:
                privileged_encoder_layers.append(nn.Linear(privileged_hidden_dims[l], privileged_hidden_dims[l + 1]))
                privileged_encoder_layers.append(activation)
        self.privileged_encoder = nn.Sequential(*privileged_encoder_layers)


        terrain_encoder_layers = []
        terrain_encoder_layers.append(nn.Linear(mlp_input_dim_t, terrain_hidden_dims[0]))
        terrain_encoder_layers.append(activation)
        for l in range(len(terrain_hidden_dims)):
            if l == len(terrain_hidden_dims) - 1:
                terrain_encoder_layers.append(nn.Linear(terrain_hidden_dims[l], num_terrain_output))
            else:
                terrain_encoder_layers.append(nn.Linear(terrain_hidden_dims[l], terrain_hidden_dims[l + 1]))
                terrain_encoder_layers.append(activation)
        self.terrain_encoder  = nn.Sequential(*terrain_encoder_layers)

        print(f"privileged_encoder MLP: {self.privileged_encoder}")
        print(f"terrain_encoder MLP: {self.terrain_encoder}")
        # Setting the Model to Training Mode:
        self.privileged_encoder.train()
        self.terrain_encoder.train()

    def forward(self, x1, x2):
        y1 = self.privileged_encoder(x1)
        y2 = self.terrain_encoder(x2)
        return y1, y2

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
        




    