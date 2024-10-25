"""Replay AMP trajectories."""
import cv2
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import matplotlib.pyplot as plt
from isaacgym import gymapi, gymtorch
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import time
import numpy as np
import torch
import math
from datetime import datetime
from rsl_rl.datasets.motion_loader import AMPLoader

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    for k in env_cfg.control.stiffness.keys():
            env_cfg.control.stiffness[k] = 0.0
    for k in env_cfg.control.damping.keys():
            env_cfg.control.damping[k] = 0.0
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
#     train_cfg.runner.resume = True
    train_cfg.algorithm.amp_replay_buffer_size = 2
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(
                    env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H-%M-%S')
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 
                            train_cfg.runner.experiment_name, 'exported_policies',
                            current_date_str, current_time_str)
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    camera_rot = 0
    camera_rot_per_sec = np.pi / 6
    SET_CAMERA  = False
    img_idx = 0
    farame_duration = 0.2  # zero for default frame_duration, positive for frame_duration = farame_duration
    env.amp_loader = AMPLoader(motion_files=env_cfg.env.amp_motion_files, device=env.device, time_between_frames=env.dt, frame_dura=farame_duration)

    video_duration = 100
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')
    video = None

    t = 0.0
    traj_idx = 0

    joint_velocities = [[] for _ in range(len(env.amp_loader.trajectory_lens))]
    joint_positions = [[] for _ in range(len(env.amp_loader.trajectory_lens))]
    angular_velocities = [[] for _ in range(len(env.amp_loader.trajectory_lens))]
    linear_velocities = [[] for _ in range(len(env.amp_loader.trajectory_lens))]
    heights = [[] for _ in range(len(env.amp_loader.trajectory_lens))]

    while traj_idx < len(env.amp_loader.trajectory_lens):
        actions = torch.zeros((env_cfg.env.num_envs, env.num_actions),
                                device=env.sim_device)

        if (t + env.amp_loader.time_between_frames +env.dt) >= env.amp_loader.trajectory_lens[traj_idx]:
                traj_idx += 1
                if traj_idx == len(env.amp_loader.trajectory_lens):
                    break
                t = 0
        else:
                t += env.dt
        # print(env.amp_loader.trajectory_names[traj_idx])
        env_ids = torch.tensor([0], device=env.device)
        root_pos = torch.tensor([[0.0, 0.0, 0.42]], device=env.device)
        env.root_states[env_ids, :3] = root_pos
        root_orn = torch.tensor([[0.0, 0.0, 0, 1]], device=env.device)
        env.root_states[env_ids, 3:7] = root_orn

        a = env.amp_loader.get_full_frame_at_time_batch(
                                                        np.array([traj_idx]), np.array([t]))

        env.root_states[env_ids, 2] = env.amp_loader.get_height_batch(a)  #设定高度

        env.root_states[env_ids, 7:10] = quat_rotate(
                        root_orn, 
                        env.amp_loader.get_linear_vel_batch(a))
        
        env.root_states[env_ids, 10:13] = quat_rotate(
                        root_orn,
                        env.amp_loader.get_angular_vel_batch(
                                        env.amp_loader.get_full_frame_at_time_batch(
                                                        np.array([traj_idx]), np.array([t]))))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(
                        env.sim, gymtorch.unwrap_tensor(env.root_states),
                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        env.dof_pos[env_ids] = env.amp_loader.get_joint_pose_batch(
                        env.amp_loader.get_full_frame_at_time_batch(
                                        np.array([traj_idx]), np.array([t])))
        env.dof_vel[env_ids] = env.amp_loader.get_joint_vel_batch(
                        env.amp_loader.get_full_frame_at_time_batch(
                                        np.array([traj_idx]), np.array([t])))
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # print(env.dof_state)
        env.gym.set_dof_state_tensor_indexed(env.sim,
                                                gymtorch.unwrap_tensor(env.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32),
                                                len(env_ids_int32))
        # print('---')
        # foot_pos_amp = env.amp_loader.get_tar_toe_pos_local_batch(
        #     env.amp_loader.get_full_frame_at_time_batch(
        #         np.array([traj_idx]), np.array([t])))
        # print(env.get_amp_observations()[0, 12:24])
        # print(foot_pos_amp[0])

        joint_velocities[traj_idx].append(env.dof_vel[env_ids].cpu().numpy())  
        joint_positions[traj_idx].append(env.dof_pos[env_ids].cpu().numpy())   
        angular_velocities[traj_idx].append(env.root_states[env_ids, 10:13].cpu().numpy())  
        linear_velocities[traj_idx].append(env.root_states[env_ids, 7:10].cpu().numpy())    
        heights[traj_idx].append(env.amp_loader.get_height_batch(a).cpu().numpy()) 
        env.step(actions.detach())
        
        if RECORD_FRAMES:
            frames_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                                                            train_cfg.runner.experiment_name, 'exported',
                                                            'frames')
            if not os.path.isdir(frames_path):
                    os.mkdir(frames_path)
            filename = os.path.join('logs', train_cfg.runner.experiment_name,
                                                            'exported', 'frames', f'{img_idx}.png')
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img = cv2.imread(filename)
            if video is None:
                    video = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                                                                    int(1 / env.dt), (img.shape[1], img.shape[0]))
            video.write(img)
            img_idx += 1

        # Reset camera position.
        if SET_CAMERA == True:            
            look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
            camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
            camera_relative_position = 2.0 * np.array(
                    [np.cos(camera_rot), np.sin(camera_rot), 0.45])
            env.set_camera(look_at + camera_relative_position, look_at)

                
    if RECORD_FRAMES:
            video.release()

    for k in range(len(env.amp_loader.trajectory_lens)):
        fig, axs = plt.subplots(5, 1, figsize=(15, 12))   
        
        joint_velocities_array = np.array(joint_velocities[k]).squeeze()
        if joint_velocities_array.ndim > 1:
                for i in range(joint_velocities_array.shape[1]):
                        axs[0].plot(joint_velocities_array[:, i], label=f'Joint Velocity {i}')
        else:
                axs[0].plot(joint_velocities_array, label='Joint Velocities')
        axs[0].set_title('Joint Velocities')
        axs[0].legend()
        
        joint_positions_array = np.array(joint_positions[k]).squeeze()
        list_joint = [1, 4, 7, 10]
        if joint_positions_array.ndim > 1:
                for i in range(joint_positions_array.shape[1]):
                        axs[1].plot(joint_positions_array[:, i], label=f'Joint Position {i}')
        else:
                axs[1].plot(joint_positions_array, label='Joint Positions')
        axs[1].set_title('Joint Positions')
        axs[1].legend()

        angular_velocities_array = np.array(angular_velocities[k]).squeeze()
        if angular_velocities_array.ndim > 1:
                for i in range(angular_velocities_array.shape[1]):
                        axs[2].plot(angular_velocities_array[:, i], label=f'Angular Velocity {i}')
        else:
                axs[2].plot(angular_velocities_array, label='Angular Velocities')
        axs[2].set_title('Angular Velocities')
        axs[2].legend()

        linear_velocities_array = np.array(linear_velocities[k]).squeeze()
        if linear_velocities_array.ndim > 1:
                for i in range(linear_velocities_array.shape[1]):
                        axs[3].plot(linear_velocities_array[:, i], label=f'Linear Velocity {i}')
        else:
                axs[3].plot(linear_velocities_array, label='Linear Velocities')
        axs[3].set_title('Linear Velocities')
        axs[3].legend()

        heights_array = np.array(heights[k]).squeeze()
        axs[4].plot(heights_array)
        axs[4].set_title('Heights')

    plt.show()
            
    time.sleep(600)
                
if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    args = get_args()
    play(args)