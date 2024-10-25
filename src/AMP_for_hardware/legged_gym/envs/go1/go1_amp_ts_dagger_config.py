      
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
import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

MOTION_FILES = glob.glob('datasets/new_go1_31_gazebo/*')


class GO1AMPTSDAggerCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include. default:None
        num_observations = 262        
        num_proprio_obs = 45
        num_privileged_obs = 30
        num_terrain_obs = 187
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        episode_length_s = 20 # episode length in seconds

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.3] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,  # [rad]
            'RL_hip_joint': 0.0,  # [rad]
            'FR_hip_joint': 0.0,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.9,     # [rad]
            'RL_thigh_joint': 0.9,   # [rad]
            'FR_thigh_joint': 0.9,     # [rad]
            'RR_thigh_joint': 0.9,   # [rad]

            'FL_calf_joint': -1.8,   # [rad]
            'RL_calf_joint': -1.8,    # [rad]
            'FR_calf_joint': -1.8,  # [rad]
            'RR_calf_joint': -1.8,    # [rad]
        }
        randomize_start_xy = False
        rand_xy_range = 1
        randomize_start_z = False
        rand_z_range = 0.02

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        # mesh_type = 'plane'
        measure_heights = True
        # curriculum = False
        terrain_name = 1     # 0: normal_small  1: range_stepwidth_small

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["base", "thigh", "calf"]
        terminate_after_contacts_on = [
            "base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        fix_base_link = False
        dagger = False

    class normalization( LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            feet_cont_force = 0.002
            mass = 0.025
        clip_observations = 100.
        clip_actions = 100.

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_com = True
        com_displacement_range = [[-0.05, 0.15],
                                  [-0.1, 0.1],
                                  [-0.05, 0.05]]

        randomize_friction = True
        friction_range = [0.25, 1.75] # [0.3, 1.75]

        randomize_restitution = True
        restitution_range = [0.0, 1.0]

        randomize_base_mass = True
        added_mass_range = [-5.0, 8.0]  #kg

        randomize_link_mass = True
        added_link_mass_range = [0.7, 1.3]  # # Factor
        push_robots = True
        push_interval_s = 4
        push_duration = 0
        max_push_vel_xy = 1.0
        max_push_vel_ang = 0.4

        randomize_motor_offset = True
        motor_offset_range = [-0.1, 0.1] # Offset to add to the motor angles [-0.06, 0.06]

        randomize_gains = True
        stiffness_multiplier_range = [0.7, 1.3]  # Factor
        damping_multiplier_range = [0.7, 1.3]  # Factor

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]   # Factor

        randomize_joint_friction = True
        joint_friction_range = [0.0, 0.06]

        randomize_joint_damping = True
        joint_damping_range = [0.0, 0.02]

        randomize_joint_armature = False
        joint_armature_range = [0.0,0.0]     # Factor

        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [2, 6]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            friction = 0.05
            restitution = 0.05
            contact_force = 0.05
            com_pos = 0.05
            com_mass = 0.05
            height_measurements = 0.1

    class rewards( LeggedRobotCfg.rewards ):
        
        tracking_sigma = 0.15 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9
        base_height_target = 0.30
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0 
            tracking_ang_vel = 0.5 
            torques = -1e-4
            dof_acc = -2.5e-7
            feet_air_time = 0.3    # 1.0--->0.3
            collision = -0.1
            action_rate = -0.1
            stand_still = 0.0
            dof_pos_limits = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            base_height = 0.0 
            feet_stumble = 0.0 
            termination = 0.0
            dof_vel = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = True       # commands.curriculum与terrain.curriculum不能同时为True
        using_new_curr = True   # 是否用新的课程
        max_curriculum_linx = 2.0
        max_curriculum_liny = 1.0
        max_curriculum_yaw = 3.0
        max_curriculum_heading = 6.28
        num_update_step = 4     # 更新 num_update_step 次达到最大速度
        coe_curriculum_lin = 0.8   # coefficient of curriculum_vel
        coe_curriculum_yaw = 0.8   # coefficient of curriculum_yaw
        coe_envs_num = 0.8  
        update_command_counter = 20
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]
        class ranges_onflat(LeggedRobotCfg.commands.ranges_onflat):
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

class GO1AMPTSDAggerCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPTSDAggerOnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4
        num_proprio_obs = GO1AMPTSDAggerCfg.env.num_proprio_obs
        num_privileged_obs = GO1AMPTSDAggerCfg.env.num_privileged_obs
        num_terrain_obs = GO1AMPTSDAggerCfg.env.num_terrain_obs
    
    class policy( LeggedRobotCfgPPO.policy ):
        tanh_actor_output = True
        
    class dagger:
        dagger_resume = False
        num_mini_batches = 8
        num_learning_epochs = 5
        max_dagger_iterations = 20000  #迭代次数
        save_interval = 100   #存储pt文件的频率
        save_sequence = 50 #lstm 的序列长度
        beta = 0.9    #dagger 衰减系数beta**i
        dagger_load_run = -1   #指定dagger路径，默认-1 表示exported_dagger 下的最后一个文件夹
        dagger_checkpoint = -1 #指定模型文件，默认-1 表示最后一个模型;若要指定model_1000.py，则设为1000
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go1_amp_ts_dagger'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 500000 # number of policy updates 
        max_collecting_iterations = 1000
        amp_reward_coef = 1.3  # used in AMP 
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.0  # amp大于0，teacher_student等于0
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4

  

    