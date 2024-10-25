import torch.nn as nn
from rsl_rl.modules.mlp import MLP
    
class StmEncoder(nn.Module):
    def __init__(self, 
        num_proprio_obs = 225,
        device = 'cpu',
        stm_encoder_hidden_dims=[256,128],
        privileged_encoder_output_dims = 8,
        terrain_encoder_output_dims = 16,
        activation='elu',
        **kwargs):
        
        if kwargs:
            print("Dagger.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(StmEncoder, self).__init__()

        #student_encoder
        output_dim = privileged_encoder_output_dims + terrain_encoder_output_dims
        self.student_encoder = MLP(num_input=num_proprio_obs,num_output=output_dim,hidden_dims=stm_encoder_hidden_dims,activation=activation)

        print(f"stm_encoder MLP: {self.student_encoder}")

    def forward(self, input):
        student_latent = self.student_encoder(input)
        return student_latent  