from ..base.legged_robot import LeggedRobot
import torch
import numpy as np
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from isaacgym.torch_utils import *
from isaacgym import gymapi
from legged_gym.utils.terrain import Terrain

class JLAMPTS(LeggedRobot):

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        reward =  torch.square(self.base_lin_vel[:, 2])
        reward[self.noflat_idx] = 0
        return reward
    
    def _reward_base_acc(self):
        # Penalize base velocity accelerations
        last_base_lin_vel = self.last_base_vel[:,:3]
        reward = torch.square(self.base_lin_vel - last_base_lin_vel).sum(dim=1)
        return reward
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        reward = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return reward
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward =  torch.sum(torch.square(self.projected_gravity[:, :2])*torch.tensor([self.cfg.rewards.pitch_roll_factor], device=self.device), dim=1)
        # reward[self.noflat_idx] = 0
        reward[self.stair_idx] = 0
        return reward

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        height_diff = base_height - self.cfg.rewards.base_height_target
    
        # Penalize only if height is less than the target
        penalty = torch.clip(height_diff, max=0).square() * 1000
    
        return penalty
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        delta = (self.last_dof_vel - self.dof_vel) / self.dt
        # delta = torch.where(torch.abs(delta) < 10, torch.zeros_like(delta), delta)
        return torch.sum(torch.square(delta), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        delta = self.last_actions - self.actions
        # delta = torch.where(torch.abs(delta) < 30, torch.zeros_like(delta), delta)
        return torch.sum(torch.square(delta), dim=1)
        # return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_feet_collision_onstair(self):
        # Penalize feet vertical collisions on stairs
        reward = torch.zeros(self.num_envs,device=self.device)
        contact_force_onstair = self.contact_forces[self.stair_idx, :, :]
        reward[self.stair_idx] = torch.sum(1.*(torch.norm(contact_force_onstair[:, self.feet_indices, :2], dim=-1) > 0.1), dim=1)
        return reward
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    
    def _reward_hip_pos_limits(self):
        # Penalize hip dof pos too far from default value
        hip_pos = self.dof_pos[:,[0,3,6,9]]
        reward = torch.sum(torch.square(hip_pos - self.default_dof_pos[:,[0,3,6,9]]),dim=1,keepdim=True).squeeze(1)
        reward *= ~(torch.norm(self.commands[:, 2], dim=-1) > 0.08)
        # reward[self.rotate_idx] = 0
        return reward
    
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # lin_vel_error[self.stair_idx] *= 0.3 
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # ang_vel_error[self.stair_idx] *= 0.3
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(torch.clip(self.feet_air_time - 0.5, max=0) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        # for plant:
        # reward_stand_still = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        # for clamp:
        reward_stand_still = torch.sum(torch.abs(self.dof_pos - self.default_start_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        # reward_stand_still[self.stair_idx] = 0
        return reward_stand_still

    def _reward_stand_still_all(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) 

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_move_feet(self):
        # Penalize the behavior of not moving when a command is given.
        feet_vel = self.rigid_body_state[:, self.feet_indices, 7:9]
        cond = (torch.norm(self.commands[:,:3],dim=-1) > 0.2).int()
        return (torch.sum(torch.norm(feet_vel,dim=-1),dim=-1) < 0.1).int() * cond
    
    def _reward_move_feet_now(self):
        # Penalize the behavior of not moving when a command is given.
        feet_vel = self.rigid_body_state[:, self.feet_indices, 7:9]
        cond = (torch.norm(self.commands[:,:3],dim=-1) > 0.1).int()
        return (torch.sum(torch.norm(feet_vel,dim=-1),dim=-1) < 0.01).int() * cond

    def _reward_foot_clearance(self):
        """A reward term to encourage larger feet clearances."""
        # Need to filter the contacts because the contact reporting of PhysX is
        # unreliable on meshes.
        feet_positions = self.rigid_body_state[:, self.feet_indices, 0:3]
        base_position = self.root_states[:, :3].unsqueeze(1)
        local_feet_positions = feet_positions - base_position
        tensor_shape = local_feet_positions.shape

        # Transform from [batch, num_feet, 3] to [batch x num_feet, 3].
        local_feet_positions = local_feet_positions.reshape(-1, 3)
        quat = self.base_quat.repeat(1, 4).reshape(-1, 4)
        local_feet_positions = quat_rotate_inverse(quat, local_feet_positions).reshape(tensor_shape)        
        
        # We assume that the local feet positions are negative in the base frame
        # The clearance reward is larger when the swing legs are higher.
        rew_clearance = (local_feet_positions[:, :, 2] + self.cfg.rewards.base_height_target - self.cfg.rewards.toe_height_target)

        self.xy_contact_state = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=-1) > 0.1
        xy_contact_filt = self.xy_contact_state * self.last_xy_contact_state
        self.num_xy_contact += xy_contact_filt
        self.num_xy_contact *= xy_contact_filt
        rew_clearance[self.stair_idx] -= 0.05 * self.num_xy_contact[self.stair_idx] + 0.1 * self.xy_contact_state[self.stair_idx]
        rew_clearance[self.stair_idx] = torch.clip(rew_clearance[self.stair_idx], min=-self.cfg.rewards.base_height_target * 1.3) 
        self.last_xy_contact_state = self.xy_contact_state
        rew_clearance = torch.clip(rew_clearance, max=0)

        # Only apply to swing legs.
        # contact_filt is a [batch, 4] array. A foot is on the ground only if the
        # contact force exceed 10 N in the z-direction.
        contact = self.contact_forces[:, self.feet_indices, 2] > 2.0  # Newton
        contact_filt = torch.logical_or(contact, self.last_clearance_contacts)
        contact_filt[self.stair_idx] *= ~self.xy_contact_state[self.stair_idx]
        self.last_clearance_contacts = contact
        rew_clearance = torch.where(contact_filt,
                                    1000 * torch.ones(self.num_envs, 4, device=self.device, requires_grad=False),
                                    rew_clearance)
        rew_clearance = torch.exp(-(torch.square(rew_clearance)) * 100).sum(dim=1)
        rew_clearance *= torch.norm(self.commands[:, :3], dim=-1) > 0.1  # no reward for zero command

        return rew_clearance
    
    def _reward_foot_discrepancy(self):
        """A reward term to encourage lower foot discrepancy."""
        feet_positions = self.rigid_body_state[:, self.feet_indices, 0:3]
        base_position = self.root_states[:, :3].unsqueeze(1)
        local_feet_positions = feet_positions - base_position
        tensor_shape = local_feet_positions.shape

        # Transform from [batch, num_feet, 3] to [batch x num_feet, 3].
        local_feet_positions = local_feet_positions.reshape(-1, 3)
        quat = self.base_quat.repeat(1, 4).reshape(-1, 4)
        local_feet_positions = quat_rotate_inverse(
            quat, local_feet_positions).reshape(tensor_shape)    
        
        lf_rb_discrepancy = torch.abs(local_feet_positions[:,0,2] - local_feet_positions[:, 3, 2])
        rf_lb_discrepancy = torch.abs(local_feet_positions[:,1,2] - local_feet_positions[:, 2, 2])
        rew_foot_discrepancy = lf_rb_discrepancy + rf_lb_discrepancy

        rew_foot_discrepancy = torch.exp(-(torch.square(rew_foot_discrepancy)) * 50)
        rew_foot_discrepancy[self.noflat_idx] = 0

        return rew_foot_discrepancy
    

    ## reward from locomotionwithnp3o
    # def _reward_foot_clearance_up(self):
    #     cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    #     footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
    #     footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     for i in range(len(self.feet_indices)):
    #         footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
    #         footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
    #     height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
    #     foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
    #     #no_contact = 1.*(self.contact_filt == 0)

    #     clearance_reward = height_error * foot_leteral_vel 
        
    #     return torch.sum(clearance_reward, dim=1)*torch.clamp(-self.projected_gravity[:,2],0,1)
    
    def _reward_foot_mirror(self):
        diff1 = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[9,10,11]]),dim=-1)
        diff2 = torch.sum(torch.square(self.dof_pos[:,[3,4,5]] - self.dof_pos[:,[6,7,8]]),dim=-1)
        return 0.5*(diff1 + diff2)
    
    def _reward_hip_pos(self):
        #return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        flag = 1.*(torch.abs(self.commands[:,1]) == 0)
        reward = flag * torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:, [0, 3, 6, 9]])), dim=1)
        reward[self.stair_idx] = 0
        return reward
        #return flag * 1.*(torch.abs(torch.sum(self.dof_pos[:, [0, 3, 6, 9]],dim=-1)) > 0.0)

    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
   
    def _reward_joint_power(self):
        # Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)
    
    def _reward_foot_slide(self):
        cur_footvel_translated = self.rigid_body_state[:, self.feet_indices, 7:10] - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        cost_slide = torch.sum(contact_filt * foot_leteral_vel, dim=1)
        return cost_slide

    def _reward_feet_velocity(self):
        foot_v = self.rigid_body_state[:, self.feet_indices, 7:10]
        feet_v_slip = self.contact_filt * torch.norm(foot_v[:, :, :3], dim=2)

        return torch.sum(feet_v_slip, dim=1)
    
 