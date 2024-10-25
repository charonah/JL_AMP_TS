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
import cv2
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger
from legged_gym.utils.helpers import get_load_path, class_to_dict
from datetime import datetime
from rsl_rl.runners import *
from legged_gym.envs.base import observation_buffer
from rsl_rl.modules.dagger import PolicyMlp, LstmEncoder, StmEncoder
from isaacgym.torch_utils import *
import numpy as np
import torch
import time
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
import cv2

     
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 5)
    env_cfg.env.episode_length_s = 60    #modify to change the running time 
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False   
    env_cfg.domain_rand.randomize_torque = False  
    lin_vel_scale=env_cfg.normalization.obs_scales.lin_vel
    ang_vel_scale=env_cfg.normalization.obs_scales.ang_vel
    
    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _, _ = env.reset()
    proprio_obs = env.get_observations()

    # load policy
    # train_cfg.runner.resume = True
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    train_cfg_dict = class_to_dict(train_cfg)
    policy_cfg = train_cfg_dict["policy"]

    log_root_encoder = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported_dagger')
    student_encoder_path = get_load_path(log_root_encoder, load_run=-1, checkpoint=-1)
    log_root_policy = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported_dagger')
    student_policy_path = get_load_path(log_root_policy, load_run=-1, checkpoint=-1)
    if env.include_history_steps == None:
        lstm_encoder = LstmEncoder(env.num_proprio_obs,
                                    env.device,
                                    **policy_cfg).to(env.device)
        loaded_dict = torch.load(student_encoder_path)
        lstm_encoder.load_state_dict(loaded_dict['lstm_encoder_state_dict'])
        print("Load lstm_encoder from:",student_encoder_path)
        latent_obs = lstm_encoder.forward(proprio_obs.unsqueeze(0))
        obs = torch.cat((proprio_obs, latent_obs),dim=-1)
    else:
        stm_encoder = StmEncoder(env.num_proprio_obs * env.include_history_steps,
                            env.device,
                            **policy_cfg).to(env.device)
        loaded_dict = torch.load(student_encoder_path)
        stm_encoder.load_state_dict(loaded_dict['stm_encoder_state_dict'])
        print("Load stm_encoder from:",student_encoder_path)
        latent_obs = stm_encoder.forward(proprio_obs)
        obs = torch.cat((proprio_obs[:,-env_cfg.env.num_proprio_obs:], latent_obs),dim=-1)
    policy_mlp = PolicyMlp( env.num_proprio_obs,
                    env.num_actions,
                    **policy_cfg).to(env.device)

    loaded_dict = torch.load(student_policy_path)
    policy_mlp.load_state_dict(loaded_dict['policy_mlp_state_dict'])
    print("Load policy_mlp from:",student_policy_path)

      
    # export policy as a jit module (used to run it from C++)
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H-%M-%S')

    logger = Logger(env.dt)
    np.random.seed(int(time.time())) 
    robot_index = np.random.randint(0,env_cfg.env.num_envs) # which robot is used for logging
    # print('*******',robot_index)
    joint_index = 2 # which joint is used for logging
    stop_state_log = 2000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    def compute_vel_command(idx, vel_x_max, vel_y_max, vel_yaw_max, heading_max, tolstep = 4*int(env.max_episode_length)):
        num_step = 4
        step_interval = int(tolstep / num_step)
        num_interval = 6
        idx_interval = np.floor(step_interval /num_interval)
        vel_x = np.linspace(-vel_x_max, vel_x_max, num_interval)
        vel_y = np.linspace(-vel_y_max, vel_y_max, num_interval)
        vel_yaw = np.linspace(-vel_yaw_max, vel_yaw_max, num_interval)
        head_c = np.linspace(-heading_max, heading_max, num_interval)
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head = 0.0, 0.0, 0.0, 0.0
        if env_cfg.commands.heading_command:
            forward = quat_apply(env.base_quat, env.forward_vec)
            for j in range(num_step):
                for k in range(num_interval):
                    if j< num_step-1:
                        if idx >= k * idx_interval + j * step_interval and idx < (k +1) * idx_interval + j * step_interval:
                            heading = torch.atan2(forward[:, 1], forward[:, 0])
                            x_vel_cmd, y_vel_cmd, head_cmd = vel_x[k]*int(j==0), vel_y[k]*int(j==1), head_c[k]*int(j==2)
                            head_cmd = to_torch(head_cmd)
                            yaw_vel_cmd = (torch.clip(0.8*wrap_to_pi(head_cmd - heading), -1., 1.))
                            break
                    else:
                        if idx >= k * idx_interval + j * step_interval and idx < (k +1) * idx_interval + j * step_interval:
                            x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head_cmd = vel_x[k], vel_y[k], vel_yaw[k], head_c[k]
                            heading = torch.atan2(forward[:, 1], forward[:, 0])
                            head_cmd = to_torch(head_cmd)
                            yaw_vel_cmd = (torch.clip(0.5*wrap_to_pi(head_cmd - heading), -1., 1.))
                            break
        else:
            for j in range(num_step):
                for k in range(num_interval):
                    if j< num_step-1:
                        if idx >= k * idx_interval + j * step_interval and idx < (k +1) * idx_interval + j * step_interval:
                            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = vel_x[k]*int(j==0), vel_y[k]*int(j==1), vel_yaw[k]*int(j==2)
                            break
                    else:
                        if idx >= k * idx_interval + j * step_interval and idx < (k +1) * idx_interval + j * step_interval:
                            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = vel_x[k], vel_y[k], vel_yaw[k], vel_yaw[k]
                            break
        return x_vel_cmd, y_vel_cmd, yaw_vel_cmd

    video_duration = 100
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')
    video = None
    
    if env.include_history_steps != None:
        obs_buf_history = observation_buffer.ObservationBuffer(
        env.num_envs, env.num_proprio_obs,
        env.include_history_steps, env.device)

    range_commands = True
    vel_x_m, vel_y_m, vel_yaw_m, head_m = 1.0, 0.6, 1.0, 3.14
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd, _ = 1.0, 0.0, 0.0, 0.0
    obs[:,42:45]=torch.tensor([x_vel_cmd*lin_vel_scale, y_vel_cmd*lin_vel_scale, yaw_vel_cmd*ang_vel_scale])
    for i in range(10*int(env.max_episode_length)):
        # print("obs", obs)
        actions = policy_mlp.forward(obs.detach())
        # print("actions", actions)
        if range_commands == True:
            range_time = 4*int(env.max_episode_length)
            stop_state_log = range_time 
            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = compute_vel_command(
                idx = i, vel_x_max=vel_x_m, vel_y_max=vel_y_m, vel_yaw_max=vel_yaw_m, heading_max=head_m, tolstep = range_time)
        # print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)
        #获取proprio_obs  
            proprio_obs, privileged_obs,terrain_obs, rews, dones, infos, _, _ = env.step(actions.detach())
            if env_cfg.commands.heading_command:
                if env.include_history_steps == None:
                    proprio_obs[:,42:44] = torch.tensor([x_vel_cmd*lin_vel_scale, y_vel_cmd*lin_vel_scale])
                    proprio_obs[:, 44] = yaw_vel_cmd*ang_vel_scale
                else:
                    wyj = (env.include_history_steps - 1) * env_cfg.env.num_proprio_obs
                    proprio_obs[:,wyj+42:wyj+44] = torch.tensor([x_vel_cmd*lin_vel_scale, y_vel_cmd*lin_vel_scale])
                    proprio_obs[:,wyj+44] = yaw_vel_cmd*ang_vel_scale
            else:
                if env.include_history_steps == None:
                    proprio_obs[:,42:45]=torch.tensor([x_vel_cmd*lin_vel_scale, y_vel_cmd*lin_vel_scale, yaw_vel_cmd*ang_vel_scale])
                else:
                    wyj = (env.include_history_steps - 1) * env_cfg.env.num_proprio_obs
                    proprio_obs[:,wyj+42:wyj+45]=torch.tensor([x_vel_cmd*lin_vel_scale, y_vel_cmd*lin_vel_scale, yaw_vel_cmd*ang_vel_scale])
                    obs_buf_history.insert(proprio_obs[:,-env_cfg.env.num_proprio_obs:])
                    proprio_obs = obs_buf_history.get_obs_vec(np.arange(env.include_history_steps))
                    
            #获取latent_obs       
            if env.include_history_steps == None:
                latent_obs = lstm_encoder.forward(proprio_obs.unsqueeze(0))
                obs = torch.cat((proprio_obs, latent_obs),dim=-1)
            else:
                latent_obs = stm_encoder.forward(proprio_obs)
                obs = torch.cat((proprio_obs[:,-env_cfg.env.num_proprio_obs:], latent_obs),dim=-1)
            
        if RECORD_FRAMES:
            frames_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                                                            train_cfg.runner.experiment_name, 'exported_videos',
                                                            current_date_str, current_time_str)
            if not os.path.isdir(frames_path):
                    os.makedirs(frames_path,exist_ok=True)
            filename = os.path.join(frames_path, f'{img_idx}.png')
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img = cv2.imread(filename)
            if video is None:
                    video = cv2.VideoWriter(os.path.join(frames_path,'record_student.mp4'), cv2.VideoWriter_fourcc(*'MP4V'),
                                                                    int(1 / env.dt), (img.shape[1], img.shape[0]))
            video.write(img)         
            img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': x_vel_cmd,
                    'command_y': y_vel_cmd,
                    'command_yaw': obs[robot_index,44].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    # 'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        
    if RECORD_FRAMES:
        video.release()
        file_list = os.listdir(frames_path)
        for file in file_list:
            if file.endswith(".png"):
                os.remove(os.path.join(frames_path,file))

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)