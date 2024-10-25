from rsl_rl.utils import unpad_trajectories, split_and_pad_trajectories
import torch
import torch.nn as nn
from torch.nn.modules import LSTM
    
class LstmEncoder(nn.Module):
    def __init__(self, 
        num_proprio_obs = 45,
        device = 'cpu',
        rnn_hidden_size=256,
        rnn_num_layers=3,
        lstm_encoder_hidden_dims=[256,128],
        privileged_encoder_output_dims = 8,
        terrain_encoder_output_dims = 16,
        activation='elu',
        **kwargs):
        
        if kwargs:
            print("Dagger.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(LstmEncoder, self).__init__()
        
        activation = get_activation(activation)
        self.device = device
        
        # LSTM
        self.lstm = Lstm(num_proprio_obs, num_layers=rnn_num_layers,hidden_size=rnn_hidden_size)

        #student_encoder
        student_encoder_output_dim = privileged_encoder_output_dims + terrain_encoder_output_dims
        student_encoder_layers_input =  rnn_hidden_size
        student_encoder_layers = []
        student_encoder_layers.append(nn.Linear(student_encoder_layers_input, lstm_encoder_hidden_dims[0]))
        student_encoder_layers.append(activation)
        for l in range(len(lstm_encoder_hidden_dims)):
            if l == len(lstm_encoder_hidden_dims) - 1:
                student_encoder_layers.append(nn.Linear(lstm_encoder_hidden_dims[l], student_encoder_output_dim))
            else:
                student_encoder_layers.append(nn.Linear(lstm_encoder_hidden_dims[l], lstm_encoder_hidden_dims[l + 1]))
                student_encoder_layers.append(activation)
        self.student_encoder = nn.Sequential(*student_encoder_layers)

        print(f"lstm MLP: {self.lstm}")
        print(f"student_encoder MLP: {self.student_encoder}")
      
    def hidden_cell_init(self, dones=None):
        self.lstm.hidden_cell_init(dones)

    def forward(self, input, masks=None, hidden_states=None):
        input_encoder = self.lstm(input, masks, hidden_states)
        student_latent = self.student_encoder(input_encoder.squeeze(0))
        return student_latent
     
    def get_hidden_states(self):
        return self.lstm.hidden_cell
    
class Lstm(nn.Module):
    def __init__(self, input_size, num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_cell = None
    
    def forward(self, input, masks=None, hidden_cell=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_cell is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.lstm(input, hidden_cell)
            out = unpad_trajectories(out, masks)           
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_cell = self.lstm(input, self.hidden_cell)
        return out

    def hidden_cell_init(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_cell in self.hidden_cell:
            hidden_cell[..., dones, :] = 0.0
        
       
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