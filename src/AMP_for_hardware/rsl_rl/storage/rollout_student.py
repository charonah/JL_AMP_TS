import torch
import numpy as np


class RolloutStudent:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, num_envs, obs_dim, sequence_size = 50, device = "cpu"):
        """Initialize a ReplayBuffer object.
        Arguments:
            sequence_size (int): maximum size of buffer
        """
        self.states = torch.zeros(sequence_size, num_envs, obs_dim).to(device)
        self.sequence_size = sequence_size
        self.device = device
        self.obs_dim = obs_dim
    
    def insert(self, states):
        """Add new states to memory."""
        self.states[:self.sequence_size-1, :, :] = self.states[1:self.sequence_size, :, :].clone()
        
        self.states[-1, :, :] = states[:,:self.obs_dim].unsqueeze(0) 

    def get_sequence_obs(self):
        return self.states