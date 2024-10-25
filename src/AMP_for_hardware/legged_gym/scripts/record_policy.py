# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import copy

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger
from legged_gym.utils.helpers import get_load_path, class_to_dict
from datetime import datetime
from rsl_rl.runners import *

from rsl_rl.modules.dagger import PolicyMlp,LstmEncoder,StmEncoder
from rsl_rl.modules.mlp import MLP

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import LSTM
import time

class Lstm(nn.Module):
    def __init__(self, 
        num_proprio_obs = 45,
        device = 'cpu',
        rnn_hidden_size=256,
        rnn_num_layers=3,
        activation='elu',
        **kwargs):
        
        if kwargs:
            print("Dagger.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(Lstm, self).__init__()
        
        # LSTM
        self.lstm_input_size = num_proprio_obs
        self.lstm_hidden_size = rnn_hidden_size
        self.lstm_num_layers = rnn_num_layers 
        self.lstm = LSTM(self.lstm_input_size, self.lstm_hidden_size, self.lstm_num_layers)

        print(f"lstm MLP: {self.lstm}")

    def forward(self,input,hidden_cell):
        lstm_output, h_c = self.lstm(input,hidden_cell)
        return lstm_output, h_c

class Encoder(nn.Module):
    def __init__(self, 
        device = 'cpu',
        rnn_hidden_size=256,
        student_encoder_hidden_dims=[256,128],
        privileged_encoder_output_dims = 8,
        terrain_encoder_output_dims = 16,
        activation='elu',
        **kwargs):
        
        if kwargs:
            print("Dagger.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(Encoder, self).__init__()
        
        self.device = device

        #student_encoder
        student_encoder_output_dim = privileged_encoder_output_dims + terrain_encoder_output_dims
        student_encoder_layers_input =  rnn_hidden_size
        self.student_encoder = MLP(student_encoder_layers_input,student_encoder_output_dim,student_encoder_hidden_dims,activation)

        print(f"student_encoder MLP: {self.student_encoder}")
        print('.............',(student_encoder_output_dim))
    
    def forward(self,input):
        student_latent = self.student_encoder(input)
        return student_latent   
    
class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, lstm, encoder):
        super().__init__()
        self.lstm = copy.deepcopy(lstm.lstm)
        self.lstm.cpu()
        self.encoder = copy.deepcopy(encoder.student_encoder)
        self.encoder.cpu()

    def forward(self, x, hidden, cell):
        hidden_cell = (hidden, cell)
        out, (hidden, cell) = self.lstm(x, hidden_cell)
        return self.encoder(out.squeeze(0)), hidden, cell

    def export(self, path):
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)    

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 6)
    env_cfg.env.episode_length_s = 60    #modify to change the running time 
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False
    lin_vel_scale=env_cfg.normalization.obs_scales.lin_vel
    ang_vel_scale=env_cfg.normalization.obs_scales.ang_vel
    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    train_cfg_dict = class_to_dict(train_cfg)
    policy_cfg = train_cfg_dict["policy"]
    dagger_load_run = train_cfg.dagger.dagger_load_run
    dagger_cp = train_cfg.dagger.dagger_checkpoint
    # load policy
    log_root_encoder = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported_dagger')
    student_encoder_path = get_load_path(log_root_encoder, load_run=dagger_load_run, checkpoint=dagger_cp)
    log_root_policy = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported_dagger')
    student_policy_path = get_load_path(log_root_policy, load_run=dagger_load_run, checkpoint=dagger_cp)
    if env.include_history_steps == None:
        lstm_encoder = LstmEncoder(env.num_proprio_obs,
                                    env.device,
                                    **policy_cfg).to(env.device)
        loaded_dict = torch.load(student_encoder_path)
        lstm_encoder.load_state_dict(loaded_dict['lstm_encoder_state_dict'])
        print("Load lstm_encoder from:",student_encoder_path)
        lstm = Lstm( env.num_proprio_obs,
                    env.device,
                    **policy_cfg).to(env.device)
        
        encoder = Encoder( env.device,
                    **policy_cfg).to(env.device)
        lstm.lstm.load_state_dict(lstm_encoder.lstm.lstm.state_dict())
        encoder.student_encoder.mlp.load_state_dict(lstm_encoder.student_encoder.state_dict())

    else:
        stm_encoder = StmEncoder(env.num_proprio_obs * env.include_history_steps,
                            env.device,
                            **policy_cfg).to(env.device)
        loaded_dict = torch.load(student_encoder_path)
        stm_encoder.load_state_dict(loaded_dict['stm_encoder_state_dict'])
        print("Load stm_encoder from:",student_encoder_path)
    
    policy_mlp = PolicyMlp( env.num_proprio_obs,
                    env.num_actions,
                    **policy_cfg).to(env.device)

    loaded_dict = torch.load(student_policy_path)
    policy_mlp.load_state_dict(loaded_dict['policy_mlp_state_dict'])
    print("Load policy_mlp from:",student_policy_path)
    
    # export policy as a jit module (used to run it from C++)
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H-%M-%S')
    
    def export_policy_as_jit(mlp, path, dir_name):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, dir_name)
        model = copy.deepcopy(mlp).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)
    
    def export_lstm_encoder_as_jit(lstm, encoder, path, dir_name):
        # assumes LSTM: TODO add GRU
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, dir_name)
        exporter = PolicyExporterLSTM(lstm, encoder)
        exporter.export(path)
  
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 
                            train_cfg.runner.experiment_name, 'exported_policies',
                            current_date_str, current_time_str)
        if env.include_history_steps == None:
            export_lstm_encoder_as_jit(lstm, encoder, path, 'lstm_encoder.jit')
        else:
            export_policy_as_jit(stm_encoder, path, 'stm_encoder.jit')
        export_policy_as_jit(policy_mlp.actor, path, 'policy_mlp.jit')
        print('Exported policy as jit script to: ', path)

if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    play(args)