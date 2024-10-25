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
import pickle
import time
import os
from collections import deque
import statistics
import json
from tqdm import tqdm

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import AMPPPO, PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.modules.teacher_encoder import Teacher_encoder
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.modules.dagger import LstmEncoder, PolicyMlp, Dagger, StmEncoder
from rsl_rl.utils import split_and_pad_trajectories

class AMPTSDAggerOnPolicyRunner:
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 collect_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.max_episode_length = int(self.env.max_episode_length)
        
        self.num_envs = self.env.num_envs
        self.num_proprio_obs = self.env.num_proprio_obs
        num_privileged_obs = self.env.num_privileged_obs
        num_terrain_obs = self.env.num_terrain_obs
        self.num_privileged_output = self.policy_cfg["privileged_encoder_output_dims"]
        self.num_terrain_output = self.policy_cfg["terrain_encoder_output_dims"]
        self.beta = train_cfg["dagger"]["beta"]
        self.save_sequence = train_cfg["dagger"]["save_sequence"]    
        self.max_dagger_iter = train_cfg["dagger"]["max_dagger_iterations"]
        self.dagger_load_run = train_cfg["dagger"]["dagger_load_run"]
        self.dagger_checkpoint = train_cfg["dagger"]["dagger_checkpoint"]
        self.dagger_resume = train_cfg["dagger"]["dagger_resume"]
        
        if self.env.num_privileged_obs is not None:
            self.num_critic_obs = self.num_proprio_obs + num_privileged_obs + num_terrain_obs
        else:
            self.num_critic_obs = self.num_proprio_obs

        self.num_actor_obs = (self.num_proprio_obs + self.num_privileged_output + self.num_terrain_output) 
            
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic        
        actor_critic: ActorCritic = actor_critic_class( num_actor_obs=self.num_actor_obs,
                                                        num_critic_obs=self.num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        
        teacher_encoder: Teacher_encoder = Teacher_encoder(  num_privileged_obs=num_privileged_obs,
                                            num_terrain_obs=num_terrain_obs,
                                            num_privileged_output=self.num_privileged_output,
                                            num_terrain_output=self.num_terrain_output,
                                            **self.policy_cfg).to(self.device)
        
        if self.env.include_history_steps == None:
            lstm_encoder: LstmEncoder = LstmEncoder(self.num_proprio_obs,
                                self.device,
                                **self.policy_cfg).to(self.device)
            stm_encoder = None
        else:
            stm_encoder: StmEncoder = StmEncoder(self.num_proprio_obs * self.env.include_history_steps,
                        self.device,
                        **self.policy_cfg).to(self.device)
            lstm_encoder = None
        
        policy_mlp: PolicyMlp = PolicyMlp( self.num_proprio_obs,
                    self.env.num_actions,
                    **self.policy_cfg).to(self.device)
        
        amp_data = AMPLoader(
            device, time_between_frames=self.env.dt, preload_transitions=True,
            num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"])
        
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator: AMPDiscriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            train_cfg['runner']['amp_reward_coef'],
            train_cfg['runner']['amp_discr_hidden_dims'], device,
            train_cfg['runner']['amp_task_reward_lerp']).to(self.device)


        alg_class = eval(self.cfg["algorithm_class_name"]) # AMPPPO
        min_std = (
            torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
            (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))
        self.alg: AMPPPO = alg_class(actor_critic, teacher_encoder, discriminator, amp_data, amp_normalizer, device=self.device, min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]  #num_steps_per_env means num_transitions_per_env
        self.save_interval = self.cfg["save_interval"]

        # Log
        self.log_dir = log_dir
        self.collect_dir = collect_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.current_collecting_iteration = 0
        self.it = 0

        self.dagger = Dagger(self.env,
                        train_cfg,
                        lstm_encoder,
                        stm_encoder,
                        policy_mlp,
                        self.alg.actor_critic,
                        self.collect_dir,
                        log_dir,
                        self.device)
        _, _, _ = self.env.reset()
        self.env.episode_length_buf.zero_()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.num_actor_obs], [self.num_critic_obs], [self.env.num_actions])
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        terrain_obs = self.env.get_terrain_observations()
        privileged_latent, terrain_latent = self.alg.teacher_encoder.forward(privileged_obs,terrain_obs)
        actor_obs = torch.cat((obs, privileged_latent, terrain_latent),dim=-1)
        # amp_obs
        amp_obs = self.env.get_amp_observations()
        critic_obs = torch.cat((obs, privileged_obs, terrain_obs),dim=-1)
        actor_obs, critic_obs, amp_obs = actor_obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.discriminator.train()
        self.alg.teacher_encoder.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rewdiscbuffer = deque(maxlen=100)
        rewtaskbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_disc_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_task_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            self.it = it
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs, amp_obs)
                    obs, privileged_obs, terrain_obs, rewards_task, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions)
                    next_amp_obs = self.env.get_amp_observations()
                    privileged_latent, terrain_latent = self.alg.teacher_encoder.forward(privileged_obs,terrain_obs)
                    actor_obs = torch.cat((obs, privileged_latent, terrain_latent),dim=-1)
                    critic_obs = torch.cat((obs, privileged_obs, terrain_obs),dim=-1)
                    actor_obs, critic_obs, next_amp_obs, rewards_task, dones = actor_obs.to(self.device), critic_obs.to(self.device), next_amp_obs.to(self.device), rewards_task.to(self.device), dones.to(self.device)

                    # Account for terminal states.
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    rewards, rewards_disc, _  = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards_task, normalizer=self.alg.amp_normalizer)
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_reward_disc_sum += rewards_disc
                        cur_reward_task_sum += rewards_task
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rewdiscbuffer.extend(cur_reward_disc_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rewtaskbuffer.extend(cur_reward_task_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_reward_disc_sum[new_ids] = 0
                        cur_reward_task_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'teacher_encoder_state_dict': self.alg.teacher_encoder.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.it,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.teacher_encoder.load_state_dict(loaded_dict['teacher_encoder_state_dict'])
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def dataset_aggergation_lstm(self):
        self.dagger.prepare_dagger_lstm(self.dagger_load_run, self.dagger_checkpoint, self.dagger_resume)
        
        proprio_obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        terrain_obs = self.env.get_terrain_observations()
        privileged_latent, terrain_latent = self.alg.teacher_encoder.forward(privileged_obs,terrain_obs)
        actor_obs = torch.cat((proprio_obs, privileged_latent, terrain_latent),dim=-1)
        hidden_cell = self.dagger.lstm_encoder.get_hidden_states()
        latent_student = self.dagger.lstm_encoder.forward(proprio_obs.unsqueeze(0))
        student_policy_obs = torch.cat((proprio_obs, latent_student),dim=-1)
        actor_obs, student_policy_obs = actor_obs.to(self.device), student_policy_obs.to(self.device)
        
        self.hidden_cell_init = self.dagger.lstm_encoder.get_hidden_states()
        
        self.dagger.init_data_rollout_lstm(self.num_proprio_obs,self.num_privileged_output,self.num_terrain_output,self.hidden_cell_init)

        self.alg.actor_critic.eval()
        self.alg.teacher_encoder.eval()

        tot_iter = self.max_dagger_iter
        with tqdm(total=tot_iter,ncols=120,desc='Dagger Progress') as overall_bar:
            for it in range(tot_iter):
                # Rollout
                with torch.inference_mode():
                    self.dagger.data_clear_lstm()
                    for i in range(self.save_sequence):
                        actions_teacher = self.alg.act_dagger(actor_obs)
                        actions_student = self.dagger.policy_mlp.forward(student_policy_obs).detach()
                        # actions = self.beta ** it * actions_teacher + (1 - self.beta ** it) * actions_student
                        actions = actions_student
                        self.dagger.add_data_rollout_lstm_1(i, proprio_obs, privileged_latent, terrain_latent, actions_teacher, hidden_cell)
                        proprio_obs, privileged_obs, terrain_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions)
                        self.dagger.add_data_rollout_lstm_2(i, dones)
                        privileged_latent, terrain_latent = self.alg.teacher_encoder.forward(privileged_obs,terrain_obs)
                        actor_obs = torch.cat((proprio_obs, privileged_latent, terrain_latent),dim=-1)
                        self.dagger.lstm_encoder.hidden_cell_init(dones=dones)
                        hidden_cell = self.dagger.lstm_encoder.get_hidden_states()
                        latent_student = self.dagger.lstm_encoder.forward(proprio_obs.unsqueeze(0))
                        student_policy_obs = torch.cat((proprio_obs, latent_student),dim=-1)
                        actor_obs,  student_policy_obs = actor_obs.to(self.device), student_policy_obs.to(self.device)
                    overall_bar.update()   
                loss_dagger = self.dagger.update_dagger_lstm(it) 
                
    def dataset_aggergation_stm(self):
        self.dagger.prepare_dagger_stm(self.dagger_load_run, self.dagger_checkpoint, self.dagger_resume)
        
        proprio_obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        terrain_obs = self.env.get_terrain_observations()
        privileged_latent, terrain_latent = self.alg.teacher_encoder.forward(privileged_obs,terrain_obs)
        actor_obs = torch.cat((proprio_obs[:,-self.num_proprio_obs:], privileged_latent, terrain_latent),dim=-1)
        latent_student = self.dagger.stm_encoder.forward(proprio_obs)
        student_policy_obs = torch.cat((proprio_obs[:,-self.num_proprio_obs:], latent_student),dim=-1)
        actor_obs, student_policy_obs = actor_obs.to(self.device), student_policy_obs.to(self.device)
        
        self.dagger.init_data_rollout_stm(self.num_proprio_obs * self.env.include_history_steps,self.num_privileged_output,self.num_terrain_output)

        self.alg.actor_critic.eval()
        self.alg.teacher_encoder.eval()

        tot_iter = self.max_dagger_iter
        with tqdm(total=tot_iter,ncols=120,desc='Dagger Progress') as overall_bar:
            for it in range(tot_iter):
                # Rollout
                with torch.inference_mode():
                    self.dagger.data_clear_stm()
                    for i in range(self.save_sequence):
                        actions_teacher = self.alg.act_dagger(actor_obs)
                        actions_student = self.dagger.policy_mlp.forward(student_policy_obs).detach()
                        # actions = self.beta ** it * actions_teacher + (1 - self.beta ** it) * actions_student
                        actions = actions_student
                        self.dagger.add_data_rollout_stm(i, proprio_obs, privileged_latent, terrain_latent, actions_teacher)
                        proprio_obs, privileged_obs, terrain_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions)
                        privileged_latent, terrain_latent = self.alg.teacher_encoder.forward(privileged_obs,terrain_obs)
                        actor_obs = torch.cat((proprio_obs[:,-self.num_proprio_obs:], privileged_latent, terrain_latent),dim=-1)
                        latent_student = self.dagger.stm_encoder.forward(proprio_obs)
                        student_policy_obs = torch.cat((proprio_obs[:,-self.num_proprio_obs:], latent_student),dim=-1)
                        actor_obs,  student_policy_obs = actor_obs.to(self.device), student_policy_obs.to(self.device)
                    overall_bar.update()   
                loss_dagger = self.dagger.update_dagger_stm(it)
    
    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward_disc', statistics.mean(locs['rewdiscbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward_task', statistics.mean(locs['rewtaskbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            # self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # self.writer.add_scalar('Train/mean_reward_disc/time', statistics.mean(locs['rewdiscbuffer']), self.tot_time)
            # self.writer.add_scalar('Train/mean_reward_task/time', statistics.mean(locs['rewtaskbuffer']), self.tot_time)
            # self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)
        
        