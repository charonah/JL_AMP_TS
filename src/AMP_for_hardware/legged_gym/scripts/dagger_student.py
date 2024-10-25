from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from datetime import datetime
from rsl_rl.runners import *

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.teacher_encoder import Teacher_encoder

import numpy as np
import torch

def dagger(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.resume = True
    train_cfg.runner.amp_num_preload_transitions = 10
    env_cfg.terrain.num_cols = 16
    env_cfg.commands.curriculum = False
    # max_curriculum_linx = env_cfg.commands.max_curriculum_linx
    # max_curriculum_liny = env_cfg.commands.max_curriculum_liny
    # max_curriculum_yaw = env_cfg.commands.max_curriculum_yaw
    max_curriculum_linx = 1.25
    max_curriculum_liny = 0.6
    max_curriculum_yaw = 1.0
    env_cfg.asset.dagger = False
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    # env_cfg.terrain.terrain_proportions = [0.4, 0.2, 0.2, 0.2]
    if env_cfg.terrain.mesh_type == "plane":
        print("平坦地形")
        env_cfg.commands.ranges.lin_vel_x = [-max_curriculum_linx, max_curriculum_linx]
        env_cfg.commands.ranges.lin_vel_y = [-max_curriculum_linx, max_curriculum_liny]
        env_cfg.commands.ranges.ang_vel_yaw = [-max_curriculum_yaw, max_curriculum_yaw]
    else:
        env_cfg.commands.ranges_onflat.lin_vel_x = [-max_curriculum_linx, max_curriculum_linx]
        env_cfg.commands.ranges_onflat.lin_vel_y = [-max_curriculum_liny, max_curriculum_liny]
        env_cfg.commands.ranges_onflat.ang_vel_yaw = [-max_curriculum_yaw, max_curriculum_yaw]

    env, env_cfg = task_registry.make_env(name=args.task, args=args,env_cfg=env_cfg)
    if env_cfg.terrain.mesh_type == "plane":
        env.step_out_flat_flag = False
    else:
        env.step_out_flat_flag = True
    env_cfg.terrain.curriculum = False


    ppo_runner, train_cfg, log_dir, collect_dir  = task_registry.make_dagger_alg_runner(env=env, name=args.task, args=args,train_cfg=train_cfg)
    if env_cfg.env.include_history_steps == None:
        ppo_runner.dataset_aggergation_lstm() 
    else:
        ppo_runner.dataset_aggergation_stm() 

if __name__ == '__main__':
    args = get_args()
    dagger(args)