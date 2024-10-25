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

# MOTION_FILES = glob.glob('datasets/mocap_motions_jl/*')
# MOTION_FILES = glob.glob('datasets/mocap_motions_jl/single.txt')
MOTION_FILES = glob.glob('datasets/mocap_motions_jl/jl_31_data.txt')
# MOTION_FILES = glob.glob('datasets/test/*')

class JLAMPTSDAggerCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        include_history_steps = None                            # Number of steps of history to include.
        num_observations = 262       
        num_proprio_obs = 45
        num_privileged_obs = 30
        num_terrain_obs = 187
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        episode_length_s = 20                                   # episode length in seconds

    class init_state( LeggedRobotCfg.init_state ): 
        pos = [0.0, 0.0, 0.25]                                  # x,y,z [m] base_height:0.23,fall from 0.27, 换urdf记得改!!!!
        default_joint_angles = {                                # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.,                                 # [rad]
            "RL_hip_joint": 0.,                                 # [rad]
            "FR_hip_joint": -0.,                                # [rad]
            "RR_hip_joint": -0.,                                # [rad]

            "FL_thigh_joint": 0.8,                              # [rad]
            "RL_thigh_joint": 0.8,                              # [rad]
            "FR_thigh_joint": 0.8,                              # [rad]
            "RR_thigh_joint": 0.8,                              # [rad]

            "FL_calf_joint": -1.3,                              # [rad]
            "RL_calf_joint": -1.3,                              # [rad]
            "FR_calf_joint": -1.3,                              # [rad]
            "RR_calf_joint": -1.3,                              # [rad]
        }
        randomize_start_xy = False
        rand_xy_range = 1
        randomize_start_z = False
        rand_z_range = 0.02

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 18.}                              # [N*m/rad]
        # damping = {'joint': 0.4}                               # [N*m*s/rad]
        stiffness = {'hip_joint': 8., 'thigh_joint': 8, 'calf_joint': 9.2}                              # [N*m/rad]
        damping = {'hip_joint': 0.2, 'thigh_joint': 0.2, 'calf_joint': 0.23}                               # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        # mesh_type = 'plane'
        measure_heights = True
        curriculum = True                                       # 复杂地形才为True
        max_init_terrain_level = 4                              # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        platform = 3.
        # terrain types: [rough_flat, slope, stairs up, stairs down, discrete, wave]
        terrain_proportions = [0.2, 0.2, 0.3, 0.3]          # 切记第一个是rough_flat
        # terrain_proportions = [0.0, 0.0, 1.0, 0.0]          # play_test
        terrain_name = 0                                        # 0: normal_small ,1: range_stepwidth_small
        step_width = 0.25                                       # step width for the stairs
        slope_treshold = 0.4 # slopes above this threshold will be corrected to vertical surfaces

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/jl/urdf/jl.urdf"
        name = "jl"
        foot_name = "foot"
        penalize_contacts_on = ["base", "thigh", "calf"]
        terminate_after_contacts_on = []#["base", "thigh"]
        self_collisions = 1                                     # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        fix_base_link = False
        dagger = False

    class normalization( LeggedRobotCfg.normalization):         # 换urdf记得改!!!!
        class obs_scales(LeggedRobotCfg.normalization):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            feet_cont_force = 0.02
            mass = 0.025
        clip_observations = 100.
        clip_actions = 100.

    class domain_rand(LeggedRobotCfg.domain_rand):              # 换urdf记得改!!!!
        randomize_com = True
        com_displacement_range = [[-0.03, 0.03],
                                  [-0.03, 0.03],
                                  [-0.03, 0.03]]

        randomize_friction = True
        friction_range = [0.01, 1.75]                           # [0.3, 1.75]

        randomize_restitution = True
        restitution_range = [0.0, 1.0]

        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]                          # kg

        randomize_link_mass = True
        added_link_mass_range = [0.8, 1.2]                       # multiplier

        push_robots = True
        push_interval_s = 4
        push_duration = 0
        max_push_vel_xy = 1.0
        max_push_vel_ang = 0.4

        randomize_motor_offset = True
        motor_offset_range = [-0.05, 0.05]                      # Offset to add to the motor angles [-0.06, 0.06]

        randomize_gains = True
        stiffness_multiplier_range = [0.7, 1.3]                 # multiplier
        damping_multiplier_range = [0.7, 1.3]                   # multiplier

        randomize_torque = True
        torque_multiplier_range = [0.9, 1.1]                    # multiplier

        randomize_joint_friction = False
        joint_friction_range = [0.0, 0.06]

        randomize_joint_damping = False
        joint_damping_range = [0.0, 0.02]

        randomize_joint_armature = False
        joint_armature_range = [0.0,0.0]                        # multiplier

        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [4, 6]

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0                                       # scales other values
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05                                      # 0.05
            friction = 0.05
            restitution = 0.05
            contact_force = 0.05
            com_pos = 0.05
            com_mass = 0.05
            height_measurements = 0.1

    class rewards( LeggedRobotCfg.rewards ):
        tracking_sigma = 0.10                                  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25                              # 换urdf记得改!!!!
        toe_height_target = 0.1                                # 换urdf记得改!!!! 0.08 0.10 0.12
        class scales( LeggedRobotCfg.rewards.scales ):     
            upward = 0.5
            foot_slide_up = -0.05
            stand_nice = -0.1
            termination = -0.0
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.5
            lin_vel_z_up = -4
            ang_vel_xy_up = -0.25
            orientation_up = -1.5                                 # -0.2
            collision_up = -1.                                    # -0.1 -0.2 
            base_height_up = -0.15                                 # -100, -80 -0.2 
            feet_collision_onstair_up = -0.2                       # -0.1
            foot_clearance_up = 0.15                              # -0.5 -1.0 -2.0 -1.0 -5.
            foot_discrepancy = 0
            feet_air_time_up = 1.5                                # 0.5 0.6 0.5 0.8 0.2 0.3 0.6 0.3
            stand_still = -1
            hip_pos_limits_up = -0.2
            torques = -7e-4                                      # -1e-4(go1), -7e-4(effort比), -4e-4(质量比)
            dof_acc = -2.5e-7                                        # -2.5e-7       smooth_scale
            base_acc_up = -0.15
            action_rate = -0.02                                 # -0.15  -0.1      smooth_scale
            dof_vel = -2.5e-4
            dof_pos_limits = -5.
            dof_vel_limits = -0.
            torque_limits = -0.   # -7e-2
            move_feet_up = -5
            stumble_up = -3                                     # -0.5, -0.3, -0.8 -0.1 -1.0       
            # from locomotion
            # foot_clearance_up =  -0.5    #-0.5
            # foot_mirror_up = -0.005
            hip_pos_up = -0.05
            smoothness = -0.01
            # joint_power = -2e-5

    class commands(LeggedRobotCfg.commands):
        curriculum = False                                  # commands.curriculum与terrain.curriculum不能同时为True
        using_new_curr = True                               # 是否用新的课程
        max_curriculum_linx = 2.0
        max_curriculum_liny = 1.0
        max_curriculum_yaw = 3.0
        max_curriculum_heading = 6.28
        num_update_step = 4                                 # 更新 num_update_step 次达到最大速度
        coe_curriculum_lin = 0.8                            # coefficient of curriculum_vel
        coe_curriculum_yaw = 0.8                            # coefficient of curriculum_yaw
        coe_envs_num = 0.8  
        update_command_counter = 15                         # 15
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 21.                               # time before command are changed[s]
        heading_command = True                              # if true: compute ang vel command from heading error
        add_stand_phase = True                             # learn stand 
        stand_time = 16
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]                         # min max [m/s]
            lin_vel_y = [-0.6, 0.6]                         # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]                       # min max [rad/s]
            heading = [-3.14, 3.14]
        class ranges_onflat(LeggedRobotCfg.commands.ranges_onflat):
            lin_vel_x = [-2.0, 2.0]                         # min max [m/s]
            lin_vel_y = [-0.6, 0.6]                         # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]                       # min max [rad/s]
            heading = [-3.14, 3.14]

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]                           # [m/s^2]
        up_axis = 1                                         # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1                                 # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01                           # [m]
            rest_offset = 0.0                               # [m]
            bounce_threshold_velocity = 0.5                 # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23                   # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2                          # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            
class JLAMPTSDAggerCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPTSDAggerOnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 6
        max_grad_norm = 0.8 # 1.0
        num_proprio_obs = JLAMPTSDAggerCfg.env.num_proprio_obs
        num_privileged_obs = JLAMPTSDAggerCfg.env.num_privileged_obs
        num_terrain_obs = JLAMPTSDAggerCfg.env.num_terrain_obs
        amp_loss_coef = 0#1                    # 1.4

    class policy( LeggedRobotCfgPPO.policy ):
        tanh_actor_output = False

    class dagger:
        dagger_resume = False
        num_mini_batches = 8
        num_learning_epochs = 5
        max_dagger_iterations = 20000                       # 迭代次数
        save_interval = 100                                 # 存储pt文件的频率
        save_sequence = 50                                  # lstm 的序列长度
        beta = 0.9                                          # dagger 衰减系数beta**i
        dagger_load_run = -1                                # 指定dagger路径，默认-1 表示exported_dagger 下的最后一个文件夹
        dagger_checkpoint = -1                              # 指定模型文件，默认-1 表示最后一个模型;若要指定model_1000.py，则设为1000

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'jl_amp_ts_dagger'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 500000                             # number of policy updates
        max_collecting_iterations = 1000
        amp_reward_coef =  0.#1.0                           # 1.0 2.0 1.5 0.5 1.3
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.0
        amp_discr_hidden_dims = [1024, 512]
        min_normalized_std = [0.05, 0.02, 0.05] * 4