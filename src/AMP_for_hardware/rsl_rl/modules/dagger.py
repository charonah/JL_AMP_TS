from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from rsl_rl.algorithms import AMPPPO, PPO
from rsl_rl.modules import ActorCritic, Teacher_encoder
from rsl_rl.storage import RolloutStudent
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from legged_gym.envs import *
from datetime import datetime
from rsl_rl.runners import *
from legged_gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params, get_collect_path
from rsl_rl.modules import ActorCritic
from rsl_rl.modules.teacher_encoder import Teacher_encoder
from rsl_rl.modules.lstm_encoder import LstmEncoder
from rsl_rl.modules.stm_encoder import StmEncoder
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import LSTM
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm import tqdm
from rsl_rl.utils import unpad_trajectories, split_and_pad_trajectories

class PolicyMlp(nn.Module):
    def __init__(self, 
                        num_proprio_obs = 45,
                        num_actions = 12,
                        privileged_encoder_output_dims = 8,
                        terrain_encoder_output_dims = 16,
                        actor_hidden_dims = [256,128,64],
                        activation='elu',
                        tanh_actor_output = False,
                        **kwargs):
        
        if kwargs:
            print("Dagger.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(PolicyMlp, self).__init__()   
        
        activation = get_activation(activation)
        activation_tanh = get_activation('tanh')
        mlp_input_dim_a = privileged_encoder_output_dims + terrain_encoder_output_dims + num_proprio_obs
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(activation)
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                if tanh_actor_output:
                    actor_layers.append(activation_tanh)
            else:
                actor_layers.append(activation)
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
        self.actor = nn.Sequential(*actor_layers)
        
        print(f"policy_ MLP: {self.actor}")
        
    def forward(self,actor_input):
        actions_mean = self.actor(actor_input)
        return actions_mean     

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
class Dagger:
    def __init__(self,
                env: VecEnv,
                train_cfg,
                lstm_encoder: LstmEncoder,
                stm_encoder: StmEncoder,
                policy_mlp: PolicyMlp,
                actor_critic: ActorCritic,
                collect_dir=None,
                log_dir=None,
                device='cpu'
                ):   
        
        self.env = env
        self.train_cfg = train_cfg
        self.lstm_encoder = lstm_encoder
        self.stm_encoder = stm_encoder
        self.policy_mlp = policy_mlp
        self.actor_critic = actor_critic
        self.log_dir = log_dir
        self.collect_dir = collect_dir
        self.device = device
        self.dagger_cfg = self.train_cfg["dagger"]
        self.save_sequence = train_cfg["dagger"]["save_sequence"]
        self.runner_cfg = self.train_cfg["runner"]
        self.num_mini_batches = self.dagger_cfg["num_mini_batches"]
        self.num_epochs = self.dagger_cfg["num_learning_epochs"]
        self.save_interval = self.dagger_cfg["save_interval"]
        self.learning_rate = self.train_cfg["algorithm"]["learning_rate"]
        self.max_grad_norm = self.train_cfg["algorithm"]["max_grad_norm"]
        self.num_envs = self.env.num_envs
        self.num_proprio_obs = self.env.num_proprio_obs
        
    def init_data_rollout_lstm(self,num_proprio_obs,num_privileged_output,num_terrain_output,hidden_cell_init):
        self.proprio_obs = torch.zeros(self.save_sequence, self.num_envs, num_proprio_obs, device=self.device)
        self.privileged_latent = torch.zeros(self.save_sequence,self.num_envs, num_privileged_output, device=self.device)
        self.terrain_latent = torch.zeros(self.save_sequence,self.num_envs, num_terrain_output, device=self.device)
        self.actions_teacher = torch.zeros(self.save_sequence, self.num_envs, self.env.num_actions, device=self.device)
        self.dones = torch.zeros(self.save_sequence, self.num_envs, device=self.device)
        self.hidden_cell = [torch.zeros(self.save_sequence, *hidden_cell_init[i].shape, device=self.device) for i in range(len(hidden_cell_init))]
        self.hidden_cell_init = hidden_cell_init
    
    def add_data_rollout_lstm_1(self, num_step, proprio_obs, privileged_latent, terrain_latent, actions_teacher,hidden_cell):
        self.proprio_obs[num_step].copy_(proprio_obs)
        self.privileged_latent[num_step].copy_(privileged_latent)
        self.terrain_latent[num_step].copy_(terrain_latent)
        self.actions_teacher[num_step].copy_(actions_teacher)
        if hidden_cell == None:
            for i in range(len(self.hidden_cell_init)):
                self.hidden_cell[i][num_step].copy_(torch.zeros_like(self.hidden_cell_init[i]))
        else:
            for i in range(len(hidden_cell)):
                self.hidden_cell[i][num_step].copy_(hidden_cell[i])

    def add_data_rollout_lstm_2(self, num_step, dones):
        self.dones[num_step].copy_(dones)
        
    def data_clear_lstm(self):
        self.proprio_obs.zero_()
        self.privileged_latent.zero_()
        self.terrain_latent.zero_()
        self.actions_teacher.zero_()
        self.dones.zero_()
        for i in range(len(self.hidden_cell)):
            self.hidden_cell[i].zero_()
            
    def mini_batch_generator_lstm(self):
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.proprio_obs, self.dones)
        mini_batch_size = self.num_envs // self.num_mini_batches
        for epoch in range(self.num_epochs):
            first_traj = 0
            for i in range(self.num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                proprio_obs_batch = padded_obs_trajectories[:, first_traj:last_traj]

                privileged_latent_batch = self.privileged_latent[:, start:stop]
                terrain_latent_batch = self.terrain_latent[:, start:stop]
                actions_teacher_batch = self.actions_teacher[:, start:stop]

                last_was_done = last_was_done.permute(1, 0)
                hidden_cell_batch = [ saved_hidden_cell.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_cell in self.hidden_cell ]
                yield proprio_obs_batch,privileged_latent_batch,terrain_latent_batch,actions_teacher_batch,masks_batch,hidden_cell_batch
                first_traj = last_traj
    
    def load_dagger_lstm(self, path):
        loaded_dict = torch.load(path)
        self.lstm_encoder.load_state_dict(loaded_dict['lstm_encoder_state_dict'])
        self.policy_mlp.load_state_dict(loaded_dict['policy_mlp_state_dict'])
        print(f"Loading dagger from: {path}")
    
    def save_dagger_lstm(self, path):
        torch.save({'lstm_encoder_state_dict': self.lstm_encoder.state_dict(),
                   'policy_mlp_state_dict': self.policy_mlp.state_dict()},
                   path)   
    
    def prepare_dagger_lstm(self, load_run=-1, checkpoint=-1, dagger_resume=False):
        # save path
        current_date_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_root_dagger = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.runner_cfg["experiment_name"], 'exported_dagger')       
        self.log_dir_dagger = os.path.join(log_root_dagger, current_date_time_str +self.runner_cfg["run_name"])
        os.makedirs(self.log_dir_dagger, exist_ok=True)
        
        #load lstm_encoder and mlp for init if dagger_resume=True
        if dagger_resume:
            dagger_path = get_load_path(log_root_dagger, load_run, checkpoint)
            self.load_dagger_lstm(dagger_path)

        # init policy_mlp
        if not dagger_resume: 
            self.policy_mlp.actor.load_state_dict(self.actor_critic.actor.state_dict())
        
        # Optimizer for policy_mlp
        self.params_dagger = [{'params': self.lstm_encoder.parameters(), 'name': 'lstm_encoder'},
                              {'params': self.policy_mlp.parameters(), 'name': 'policy'}]   
        self.optimizer_dagger = optim.Adam(self.params_dagger, lr=self.learning_rate)
        
        # SummaryWriter
        self.writer_dagger = SummaryWriter(log_dir=self.log_dir_dagger, flush_secs=10)
    
    def update_dagger_lstm(self,iteration):                                
        mse_loss = nn.MSELoss()
        generator = self.mini_batch_generator_lstm()
        writer_counter = 0
        writer_start = iteration * self.num_epochs * self.num_mini_batches
        for sample in generator:
            self.proprio_obs_batch, self.privileged_latent_batch, self.terrain_latent_batch, self.actions_teacher_batch, self.masks_batch, self.hidden_cell_batch = sample
            latent_ref = torch.cat((self.privileged_latent_batch,
                                    self.terrain_latent_batch),dim=-1)
            latent_policy= self.lstm_encoder.forward(self.proprio_obs_batch, self.masks_batch, self.hidden_cell_batch)
            actions_ref = self.actions_teacher_batch
            proprio_obs = unpad_trajectories(self.proprio_obs_batch, self.masks_batch)
            policy_actor_input = torch.cat((proprio_obs,
                                    latent_policy),dim=-1)
            actions_policy = self.policy_mlp.forward(policy_actor_input)
            
            #compute loss
            loss_latent = mse_loss(latent_policy,latent_ref.detach())
            loss_policy = mse_loss(actions_policy,actions_ref)
            loss_dagger = loss_latent + loss_policy

            # Gradient step for dagger
            self.optimizer_dagger.zero_grad()
            loss_dagger.backward()
            nn.utils.clip_grad_norm_(self.policy_mlp.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.lstm_encoder.parameters(), self.max_grad_norm)
            self.optimizer_dagger.step()
            
            self.writer_dagger.add_scalar('Policy_Loss/', loss_policy, writer_start + writer_counter)
            self.writer_dagger.add_scalar('Encoder_Loss/', loss_latent, writer_start + writer_counter)
            writer_counter += 1
        # save policy_mlp
        if iteration % self.save_interval == 0:
            self.save_dagger_lstm(os.path.join(self.log_dir_dagger, 'model_{}.pt'.format(iteration)))
        return loss_dagger
    
    def init_data_rollout_stm(self,num_proprio_obs,num_privileged_output,num_terrain_output):
        self.proprio_obs = torch.zeros(self.save_sequence, self.num_envs, num_proprio_obs, device=self.device)
        self.privileged_latent = torch.zeros(self.save_sequence,self.num_envs, num_privileged_output, device=self.device)
        self.terrain_latent = torch.zeros(self.save_sequence,self.num_envs, num_terrain_output, device=self.device)
        self.actions_teacher = torch.zeros(self.save_sequence, self.num_envs, self.env.num_actions, device=self.device)
    
    def add_data_rollout_stm(self, num_step, proprio_obs, privileged_latent, terrain_latent, actions_teacher):
        self.proprio_obs[num_step].copy_(proprio_obs)
        self.privileged_latent[num_step].copy_(privileged_latent)
        self.terrain_latent[num_step].copy_(terrain_latent)
        self.actions_teacher[num_step].copy_(actions_teacher)
        
    def data_clear_stm(self):
        self.proprio_obs.zero_()
        self.privileged_latent.zero_()
        self.terrain_latent.zero_()
        self.actions_teacher.zero_()
            
    def mini_batch_generator_stm(self):
        mini_batch_size = self.num_envs * self.save_sequence // self.num_mini_batches
        indices = torch.randperm(self.num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)
        proprio_obs = self.proprio_obs.flatten(0, 1)
        privileged_latent = self.privileged_latent.flatten(0, 1)
        terrain_latent = self.terrain_latent.flatten(0, 1)
        actions_teacher = self.actions_teacher.flatten(0, 1)
        for epoch in range(self.num_epochs):
            for i in range(self.num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size
                batch_idx = indices[start:stop]
                proprio_obs_batch = proprio_obs[batch_idx]

                privileged_latent_batch = privileged_latent[batch_idx]
                terrain_latent_batch = terrain_latent[batch_idx]
                actions_teacher_batch = actions_teacher[batch_idx]
                yield proprio_obs_batch,privileged_latent_batch,terrain_latent_batch,actions_teacher_batch
                
    def load_dagger_stm(self, path):
        loaded_dict = torch.load(path)
        self.stm_encoder.load_state_dict(loaded_dict['stm_encoder_state_dict'])
        self.policy_mlp.load_state_dict(loaded_dict['policy_mlp_state_dict'])
        print(f"Loading dagger from: {path}")
    
    def save_dagger_stm(self, path):
        torch.save({'stm_encoder_state_dict': self.stm_encoder.state_dict(),
                   'policy_mlp_state_dict': self.policy_mlp.state_dict()},
                   path)   
    
    def prepare_dagger_stm(self, load_run=-1, checkpoint=-1, dagger_resume=False):
        # save path
        current_date_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_root_dagger = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.runner_cfg["experiment_name"], 'exported_dagger')       
        self.log_dir_dagger = os.path.join(log_root_dagger, current_date_time_str +self.runner_cfg["run_name"])
        os.makedirs(self.log_dir_dagger, exist_ok=True)
        
        #load lstm_encoder and mlp for init if dagger_resume=True
        if dagger_resume:
            dagger_path = get_load_path(log_root_dagger, load_run, checkpoint)
            self.load_dagger_stm(dagger_path)

        # init policy_mlp
        if not dagger_resume: 
            self.policy_mlp.actor.load_state_dict(self.actor_critic.actor.state_dict())
        
        # Optimizer for policy_mlp
        self.params_dagger = [{'params': self.stm_encoder.parameters(), 'name': 'stm_encoder'},
                              {'params': self.policy_mlp.parameters(), 'name': 'policy'}]   
        self.optimizer_dagger = optim.Adam(self.params_dagger, lr=self.learning_rate)
        
        # SummaryWriter
        self.writer_dagger = SummaryWriter(log_dir=self.log_dir_dagger, flush_secs=10)
        
    def update_dagger_stm(self,iteration):                             
        mse_loss = nn.MSELoss()
        generator = self.mini_batch_generator_stm()
        writer_counter = 0
        writer_start = iteration * self.num_epochs * self.num_mini_batches
        for sample in generator:
            self.proprio_obs_batch, self.privileged_latent_batch, self.terrain_latent_batch, self.actions_teacher_batch = sample
            latent_ref = torch.cat((self.privileged_latent_batch,
                                    self.terrain_latent_batch),dim=-1)
            latent_policy= self.stm_encoder.forward(self.proprio_obs_batch)
            actions_ref = self.actions_teacher_batch
            policy_actor_input = torch.cat((self.proprio_obs_batch[:,-self.num_proprio_obs:],
                                    latent_policy),dim=-1)
            actions_policy = self.policy_mlp.forward(policy_actor_input)
            
            #compute loss
            loss_latent = mse_loss(latent_policy,latent_ref)
            loss_policy = mse_loss(actions_policy,actions_ref)
            loss_dagger = loss_latent + loss_policy

            # Gradient step for dagger
            self.optimizer_dagger.zero_grad()
            loss_dagger.backward()
            nn.utils.clip_grad_norm_(self.policy_mlp.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.stm_encoder.parameters(), self.max_grad_norm)
            self.optimizer_dagger.step()
            
            self.writer_dagger.add_scalar('Policy_Loss/', loss_policy, writer_start + writer_counter)
            self.writer_dagger.add_scalar('Encoder_Loss/', loss_latent, writer_start + writer_counter)
            writer_counter += 1
        # save policy_mlp
        if iteration % self.save_interval == 0:
            self.save_dagger_stm(os.path.join(self.log_dir_dagger, 'model_{}.pt'.format(iteration)))
        return loss_dagger