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

# 在键盘控制时需要先用鼠标点击弹出的黑色窗口，这个窗口用来捕获键盘输入

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import pygame
import threading

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
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

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
KEYBOARD_CONTROL = True
HEADING_COMMAND = False
global reset_pos

if KEYBOARD_CONTROL:
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Robot Keyboard Control")

    # 用于控制线程退出的标志
    exit_flag = False

    # 处理键盘输入的线程
    def handle_keyboard_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, reset_pos, CAMERA_MODE

        while not exit_flag:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit_flag = True

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        x_vel_cmd = 0.8
                    elif event.key == pygame.K_s:
                        x_vel_cmd = -0.8
                    elif event.key == pygame.K_a:
                        y_vel_cmd = 0.8
                    elif event.key == pygame.K_d:
                        y_vel_cmd = -0.8
                    elif event.key == pygame.K_LEFT:
                        yaw_vel_cmd = 0.8
                    elif event.key == pygame.K_RIGHT:
                        yaw_vel_cmd = -0.8
                    elif event.key == pygame.K_x:
                        CAMERA_MODE = "follow" if CAMERA_MODE == "free" else "free"
                        print(f"Camera mode switched to: {CAMERA_MODE}")
                    elif event.key == pygame.K_r:
                        reset_pos = True
                        print("Reset position triggered")

                elif event.type == pygame.KEYUP:
                    if event.key in [pygame.K_w, pygame.K_s]:
                        x_vel_cmd = 0.0
                    if event.key in [pygame.K_a, pygame.K_d]:
                        y_vel_cmd = 0.0
                    if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        yaw_vel_cmd = 0.0

            pygame.time.delay(100)

    # 启动线程
    keyboard_thread = threading.Thread(target=handle_keyboard_input)
    keyboard_thread.start()

def play(args):
    global reset_pos
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.episode_length_s = 1000 
    env_cfg.terrain.terrain_name = 0
    env_cfg.terrain.num_rows = 5   #10
    env_cfg.terrain.num_cols = 5    #20
    env_cfg.domain_rand.friction_range = [1.0, 1.5]

    env_cfg.terrain.terrain_proportions = [0., 1, 0., 0.,]
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

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env_cfg.terrain.curriculum = False
    _, _, _ = env.reset()
    proprio_obs = env.get_observations()
    privileged_obs = env.get_privileged_observations()
    terrain_obs = env.get_terrain_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    privileged_latent, terrain_latent = ppo_runner.alg.teacher_encoder.forward(privileged_obs, terrain_obs)
    obs = torch.cat((proprio_obs, privileged_latent, terrain_latent), dim=-1)

    logger = Logger(env.dt)
    np.random.seed(int(time.time())) 
    robot_index = np.random.randint(0, env_cfg.env.num_envs)
    joint_index = 2  
    stop_state_log = 5000  
    stop_rew_log = env.max_episode_length + 1 
    env_ids = torch.tensor([0], device=env.device)

    try:
        for i in range(10 * int(env.max_episode_length)):
            actions = policy(obs.detach())
            proprio_obs, privileged_obs, terrain_obs, rews, dones, infos, _, _ = env.step(actions.detach())
            privileged_latent, terrain_latent = ppo_runner.alg.teacher_encoder.forward(privileged_obs, terrain_obs)
            obs = torch.cat((proprio_obs, privileged_latent, terrain_latent), dim=-1)
            
            if env_cfg.commands.heading_command:
                obs[:,42:44] = torch.tensor([x_vel_cmd, y_vel_cmd])
                forward = quat_apply(env.base_quat, env.forward_vec)
                heading = torch.atan2(forward[:, 1], forward[:, 0])
                obs[:,44] = (torch.clip(0.8 * wrap_to_pi(head_vel_cmd - heading), -1., 1.))
            else:
                obs[:,42:45] = torch.tensor([x_vel_cmd, y_vel_cmd, yaw_vel_cmd])
            
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
                logger.log_states({
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': x_vel_cmd,
                        'command_y': y_vel_cmd,
                        'command_yaw': yaw_vel_cmd,
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        # 'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    })
            elif i == stop_state_log:
                logger.plot_states()
            if 0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes > 0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i == stop_rew_log:
                logger.print_rewards()

    except KeyboardInterrupt:
        exit_flag = True
        keyboard_thread.join()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    CAMERA_MODE = "free" 
    reset_pos = False
    args = get_args()
    play(args)