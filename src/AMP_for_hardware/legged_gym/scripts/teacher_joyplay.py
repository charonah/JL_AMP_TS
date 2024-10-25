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

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from datetime import datetime
from rsl_rl.runners import *

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.teacher_encoder import Teacher_encoder
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import time
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float

import math

import sys
import threading


from tqdm import tqdm
import pygame
from threading import Thread

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
# x_scale, y_scale, yaw_scale = 2.5, 2.0, 0.0
joystick_use = True
joystick_opened = False
HEADING_COMMAND = False
global reset_pos

if joystick_use:

    pygame.init()

    try:
        # 获取手柄
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")

    # 用于控制线程退出的标志
    exit_flag = False


# 处理手柄输入的线程
    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head_vel_cmd, CAMERA_MODE, reset_pos

        X_button_was_pressed = False 
        A_button_was_pressed = False 

        for i in range(joystick.get_numbuttons()):
            button = joystick.get_button(i)
            print(f"Button {i}: {button}")

        if HEADING_COMMAND:
            while not exit_flag:
                # 获取手柄输入
                pygame.event.get()

                X_button_pressed = joystick.get_button(2)
                A_button_pressed = joystick.get_button(0)

                if X_button_pressed and not X_button_was_pressed:
                    CAMERA_MODE = "follow" if CAMERA_MODE == "free" else "free"
                    print(f"Camera mode switched to:{CAMERA_MODE}")
                X_button_was_pressed = X_button_pressed

                if A_button_pressed and not A_button_was_pressed:
                    reset_pos = True
                A_button_was_pressed = A_button_pressed

                # 更新机器人命令
                x_vel_cmd = -joystick.get_axis(1) * 1.
                y_vel_cmd = -joystick.get_axis(0) * 1
                head_vel_cmd = -joystick.get_axis(3) * 3.14

                # print("AAAAA",x_vel_cmd, y_vel_cmd, head_vel_cmd)

                # 等待一小段时间，可以根据实际情况调整
                pygame.time.delay(100)
        else:
            while not exit_flag:
                # 获取手柄输入
                pygame.event.get()

                X_button_pressed = joystick.get_button(2)
                A_button_pressed = joystick.get_button(0)

                if X_button_pressed and not X_button_was_pressed:
                    CAMERA_MODE = "follow" if CAMERA_MODE == "free" else "free"
                    print(f"Camera mode switched to:{CAMERA_MODE}")
                X_button_was_pressed = X_button_pressed

                if A_button_pressed and not A_button_was_pressed:
                    reset_pos = True
                    print("press A")
                A_button_was_pressed = A_button_pressed
                
                # 更新机器人命令
                x_vel_cmd = -joystick.get_axis(1) * 2
                y_vel_cmd = -joystick.get_axis(0) * 1.5
                yaw_vel_cmd = -joystick.get_axis(3) * 6

                # print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)

                # 等待一小段时间，可以根据实际情况调整
                pygame.time.delay(100)

        # 启动线程

    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()


def play(args):
    global reset_pos
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.episode_length_s = 1000    #modify to change the running time 
    # env_cfg.terrain.step_height = 0.10
    # env_cfg.terrain.terrain_length = 8.
    # env_cfg.terrain.terrain_width = 8.
    env_cfg.terrain.terrain_name = 2
    env_cfg.terrain.num_rows = 5   #10
    env_cfg.terrain.num_cols = 5    #20
    env_cfg.domain_rand.friction_range = [1.0, 1.5]

    # env_cfg.terrain.terrain_proportions = [0., 0., 0.5, 0.5, 0.0]
    env_cfg.terrain.curriculum = True 

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False   
    env_cfg.domain_rand.randomize_torque = False    
    env_cfg.asset.dagger = True    
    lin_vel_scale=env_cfg.normalization.obs_scales.lin_vel
    ang_vel_scale=env_cfg.normalization.obs_scales.ang_vel

    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.commands.heading_command = HEADING_COMMAND
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env_cfg.terrain.curriculum = False
    _, _, _ = env.reset()
    proprio_obs = env.get_observations()
    privileged_obs = env.get_privileged_observations()
    terrain_obs = env.get_terrain_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    privileged_latent, terrain_latent = ppo_runner.alg.teacher_encoder.forward(privileged_obs,terrain_obs)
    obs = torch.cat((proprio_obs, privileged_latent, terrain_latent),dim=-1)

    # export policy as a jit module (used to run it from C++)
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H-%M-%S')
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 
                            train_cfg.runner.experiment_name, 'exported_policies',
                            current_date_str, current_time_str)
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    np.random.seed(int(time.time())) 
    robot_index = np.random.randint(0,env_cfg.env.num_envs) # which robot is used for logging
    # print('*******',robot_index)
    joint_index = 2 # which joint is used for logging
    stop_state_log = 5000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    env_ids = torch.tensor([0], device=env.device)

    # obs[:,42:45]=torch.tensor([x_vel_cmd*lin_vel_scale, y_vel_cmd*lin_vel_scale, yaw_vel_cmd*ang_vel_scale])
    try:
        # for i in range(10*int(env.max_episode_length)):
        for i in range(10*int(env.max_episode_length)):

        # print("obs", obs)
            actions = policy(obs.detach())
            # print("actions", actions)
            # print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)
            proprio_obs, privileged_obs,terrain_obs, rews, dones, infos, _, _ = env.step(actions.detach())
            privileged_latent, terrain_latent = ppo_runner.alg.teacher_encoder.forward(privileged_obs,terrain_obs)
            obs = torch.cat((proprio_obs, privileged_latent, terrain_latent),dim=-1)
            # print(type(x_vel_cmd), type(y_vel_cmd), type(yaw_vel_cmd))
            if env_cfg.commands.heading_command:
                obs[:,42:44] = torch.tensor([x_vel_cmd*lin_vel_scale, y_vel_cmd*lin_vel_scale])
                forward = quat_apply(env.base_quat, env.forward_vec)
                heading = torch.atan2(forward[:, 1], forward[:, 0])
                head_cmd = to_torch(head_vel_cmd)
                obs[:,44] = (torch.clip(0.8*wrap_to_pi(head_cmd - heading), -1., 1.))
            else:
                obs[:,42:45]=torch.tensor([x_vel_cmd*lin_vel_scale, y_vel_cmd*lin_vel_scale, yaw_vel_cmd*ang_vel_scale])
            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            
            print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd, env.root_states[:, 2] - torch.mean(env.measured_heights, dim=1))

            if CAMERA_MODE == "follow":
                # camera_position += camera_vel * env.dt
                # env.set_camera(camera_position, camera_position + camera_direction)
                look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
                # camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
                yaw = env.quaternion_to_euler(env.root_states[:, 3:7])[:,2]
                camera_rot = yaw.squeeze(0).cpu().numpy() + np.pi
                camera_relative_position = 2.0 * np.array(
                        [np.cos(camera_rot), np.sin(camera_rot), 0.45])
                env.set_camera(look_at + camera_relative_position, look_at)

            if reset_pos == True:    
                last_root_states =  env.root_states.clone( )  
                env.root_states[0,:] = env.base_init_state
                env.root_states[:, :2] += last_root_states[:,:2]
                env.root_states[:, 2:3] += last_root_states[:,2:3] + 2 * env_cfg.rewards.base_height_target
                env_ids_int32 = env_ids.to(dtype=torch.int32)
                env.gym.set_actor_root_state_tensor_indexed(
                        env.sim, gymtorch.unwrap_tensor(env.root_states),
                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
                print("reset_pos:",reset_pos)
                reset_pos = False

            if i < stop_state_log:
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': x_vel_cmd,
                        'command_y': y_vel_cmd,
                        'command_yaw': obs[robot_index,44].item()/ang_vel_scale,
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        # 'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    }
                )
            elif i==stop_state_log:
                print("sssssssssssssssss")
                logger.plot_states()
            if  0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes>0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i==stop_rew_log:
                logger.print_rewards()

    except KeyboardInterrupt:
        # 在接收到 Ctrl+C 时，设置退出标志
        exit_flag = True
        if joystick_opened and joystick_use:
            joystick_thread.join()  # 等待线程退出

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    CAMERA_MODE = "free" # 自由模式/跟随模式状态码
    reset_pos = False
    args = get_args()
    play(args)