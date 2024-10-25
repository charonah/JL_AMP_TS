from ..base.legged_robot import LeggedRobot
import torch
import numpy as np
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from isaacgym.torch_utils import *
from isaacgym import gymapi

class JLAMP(LeggedRobot):
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum, 适用WJZ论文
        if self.cfg.terrain.curriculum:
            if not self.step_out_flat_flag:
                self._update_terrain_curriculum_wjz(env_ids)
            else:
                if self.set_flat_env_origin:
                    self.terrain_levels[self.flat_idx] = torch.randint(0, self.cfg.terrain.num_rows, (self.flat_idx.shape[0],), device=self.device)
                    self.env_origins[self.flat_idx] = self.terrain_origins[self.terrain_levels[self.flat_idx], self.terrain_types[self.flat_idx]]
                    self.set_flat_env_origin = False    
                mask_flat = torch.isin(env_ids,self.flat_idx)
                flat_env_ids = env_ids[mask_flat]
                if self.cfg.commands.curriculum:
                    if self.cfg.commands.using_new_curr:
                        self.update_command_curriculum_onflat_counter(flat_env_ids)
                    else:
                        if (self.common_step_counter % self.max_episode_length==0):
                            self.update_command_curriculum_onflat(flat_env_ids)
                mask_noflat = ~torch.isin(env_ids,self.flat_idx)
                noflat_env_ids = env_ids[mask_noflat]
                self._update_terrain_curriculum_wjz(noflat_env_ids)
        
        # 更新速度指令,适用于平面地形和非wjz版本
        if self.cfg.terrain.mesh_type == 'plane':    
            if self.cfg.commands.curriculum:
                if self.cfg.commands.using_new_curr:
                    self.update_command_curriculum_counter(env_ids)
                else:
                    if (self.common_step_counter % self.max_episode_length==0):
                        self.update_command_curriculum(env_ids)
        
        # reset robot states,wjz版本用                
        if not self.step_out_flat_flag:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)
        else:
            self._reset_dofs(env_ids)
            
            mask_flat = torch.isin(env_ids,self.flat_idx)
            flat_env_ids = env_ids[mask_flat]
            self._reset_root_states_flat(flat_env_ids)
            
            mask_noflat = ~torch.isin(env_ids,self.flat_idx)
            noflat_env_ids = env_ids[mask_noflat]
            self._reset_root_states(noflat_env_ids)
            
        # reset robot states,平面地形和其他情况用
        if self.cfg.terrain.mesh_type == 'plane':
            if self.cfg.env.reference_state_initialization:
                frames = self.amp_loader.get_full_frame_batch(len(env_ids))
                self._reset_dofs_amp(env_ids, frames)
                self._reset_root_states(env_ids)
                # self._reset_root_states_amp(env_ids, frames)
            else:
                self._reset_dofs(env_ids)
                self._reset_root_states(env_ids)

        # reset command
        if self.step_out_flat_flag:
            mask_flat = torch.isin(env_ids,self.flat_idx)
            flat_env_ids = env_ids[mask_flat]
            self._resample_commands_onflat(flat_env_ids)
            
            mask_noflat = ~torch.isin(env_ids,self.flat_idx)
            noflat_env_ids = env_ids[mask_noflat]
            self._resample_commands(noflat_env_ids)
        else:
            self._resample_commands(env_ids)
            
        # if self.cfg.domain_rand.randomize_gains:
        #     new_randomized_gains = self.compute_randomized_gains(len(env_ids))
        #     self.randomized_p_gains[env_ids] = new_randomized_gains[0]
        #     self.randomized_d_gains[env_ids] = new_randomized_gains[1]
       
        # Randomise the rigid body parameters (ground friction, restitution, etc.):
        # self.randomize_rigid_body_props(env_ids)
        # self._refresh_actor_rigid_shape_props(env_ids)

        # Randomize joint parameters:
        self.randomize_dof_props(env_ids)
        self._refresh_actor_dof_props(env_ids)
        
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.still_buffer[env_ids] = 0
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_euler_xyz[env_ids] = self.quaternion_to_euler(self.base_quat[env_ids])
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum and (self.cfg.terrain.mesh_type == 'plane'):
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["max_command_y"] = self.command_ranges["lin_vel_y"][1]
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]
        if self.flat_idx != None:
            self.extras["episode"]["max_command_x_flat"] = self.command_ranges_onflat["lin_vel_x"][1]
            self.extras["episode"]["max_command_y_flat"] = self.command_ranges_onflat["lin_vel_y"][1]
            self.extras["episode"]["max_command_yaw_flat"] = self.command_ranges_onflat["ang_vel_yaw"][1]

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
        if self.cfg.domain_rand.add_lag:   
            self.lag_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_lag_timesteps:
                self.lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.lag_timesteps_range[0],self.cfg.domain_rand.lag_timesteps_range[1]+1,(len(env_ids),),device=self.device) 
                if self.cfg.domain_rand.randomize_lag_timesteps_perstep:
                    self.last_lag_timestep[env_ids] = self.cfg.domain_rand.lag_timesteps_range[1]
            else:
                self.lag_timestep[env_ids] = self.cfg.domain_rand.lag_timesteps_range[1]

        if self.cfg.commands.add_stand_phase:
            stand_envs_ids = (self.episode_length_buf > int(self.cfg.commands.stand_time / self.dt)==0).nonzero(as_tuple=False).flatten()        
            self.commands[stand_envs_ids, :] = 0.0