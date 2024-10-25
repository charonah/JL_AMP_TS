import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner, AMPOnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# def record_config(env_config, train_config, log_root):

#     env_cfg = env_config
#     train_cfg = train_config
#     log_dir=log_root

#     file_path=os.path.join(log_dir,'Config.txt')

#     os.makedirs(log_dir, exist_ok=True)

#     env_dict = class_to_dict(env_cfg.env)
#     init_state_dict = class_to_dict(env_cfg.init_state)
#     control_dict = class_to_dict(env_cfg.control)
#     terrain_dict = class_to_dict(env_cfg.terrain)
#     asset_dict = class_to_dict(env_cfg.asset)
#     normalization_dict = class_to_dict(env_cfg.normalization)
#     domain_rand_dict = class_to_dict(env_cfg.domain_rand)
#     noise_dict = class_to_dict(env_cfg.noise)
#     rewards_dict = class_to_dict(env_cfg.rewards)
#     commands_dict = class_to_dict(env_cfg.commands)
#     sim_dict = class_to_dict(env_cfg.sim)

#     policy_dict = class_to_dict(train_cfg.policy)
#     algorithm_dict = class_to_dict(train_cfg.algorithm)
#     runner_dict = class_to_dict(train_cfg.runner)
   
#     dict_list = {
#                 'env': env_dict, 
#                 'init_state': init_state_dict,
#                 'control': control_dict,
#                 'terrain': terrain_dict,
#                 'asset': asset_dict,
#                 'normalization': normalization_dict,
#                 'domain_rand': domain_rand_dict,
#                 'noise': noise_dict,
#                 'rewards': rewards_dict,
#                 'commands': commands_dict,
#                 'sim': sim_dict,
#                 'policy': policy_dict,
#                 'algorithm': algorithm_dict,
#                 'runner': runner_dict
#                 }

#     def write_dicts_to_txt(named_dicts, file_path, indent_level=0):
#         with open(file_path, 'w') as file:
#             def write_dict(dictionary, indent_level):
#                 for key, value in dictionary.items():
#                     if isinstance(value, dict):
#                         file.write('\t' * indent_level + f'{key}:\n')
#                         write_dict(value, indent_level + 1)
#                     else:
#                         file.write('\t' * indent_level + f'{key}: {value}\n')

#             for name, dictionary in named_dicts.items():
#                 file.write(f'class {name} :\n')
#                 write_dict(dictionary, indent_level + 1)
#                 file.write('\n')

#     # 使用函数将多个字典写入文件
#     write_dicts_to_txt(dict_list, file_path)
def record_config(log_root, name="a1_amp"):
    log_dir=log_root
    os.makedirs(log_dir, exist_ok=True)

    str_config = name + '_config.txt'
    file_path1=os.path.join(log_dir, str_config)
    file_path2=os.path.join(log_dir, 'legged_robot_config.txt')
    
    root1 = name.split('_')[0]

    root_path1 = os.path.join(LEGGED_GYM_ENVS_DIR, root1, name + '_config.py')
    root_path2 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'legged_robot_config.py')

    with open(root_path1, 'r', encoding='utf-8') as file:
        content = file.read()

    with open(file_path1, 'w', encoding='utf-8') as file:
        file.write(content)

    with open(root_path2, 'r',encoding='utf-8') as file:
        content = file.read()

    with open(file_path2, 'w', encoding='utf-8') as file:
        file.write(content)