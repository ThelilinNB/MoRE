
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.datasets.motion_loader_g1 import G1_AMPLoader
import torch
import cv2
import numpy as np
import torch.nn.functional as F

class ym1_16Dof_MoE_Resi_Robot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if getattr(self.cfg.env, "amp_motion_files_list", None) is not None:
            self.amp_motion_files_list = self.cfg.env.amp_motion_files_list
        self.num_amp_obs = self.cfg.env.num_amp_obs

        # Body mask 支持 - 预加载数据，但根据 iteration 动态启用
        self.body_mask_enabled = False  # 初始禁用
        if hasattr(self.cfg.depth, 'body_mask_path'):
            try:
                body_mask_data = np.load(self.cfg.depth.body_mask_path, allow_pickle=True)
                self.body_masks = body_mask_data['body_masks']
                body_masks_gait = body_mask_data['gait']
                self.gait_to_indices = {}
                for i in range(self.cfg.env.num_gait):
                    gait_indices = np.where(body_masks_gait == i)[0]
                    self.gait_to_indices[i] = gait_indices
                print(f"✅ Body mask 数据已加载: {self.body_masks.shape[0]} 个 masks")
                if hasattr(self.cfg.depth, 'body_mask_start_step'):
                    print(f"   将在 {self.cfg.depth.body_mask_start_step} iterations 后启用")
            except Exception as e:
                print(f"⚠️  Body mask 加载失败: {e}")
                self.body_masks = None
                self.gait_to_indices = {i: [] for i in range(self.cfg.env.num_gait)}
        else:
            self.body_masks = None
            self.gait_to_indices = {i: [] for i in range(self.cfg.env.num_gait)}
    
    def check_and_enable_body_mask(self, current_iter):
        """
        检查当前 iteration 并动态启用 body mask
        
        Args:
            current_iter: 当前训练 iteration
        """
        if self.body_masks is not None and not self.body_mask_enabled:
            if hasattr(self.cfg.depth, 'body_mask_start_step') and current_iter >= self.cfg.depth.body_mask_start_step:
                self.body_mask_enabled = True
                self.cfg.depth.add_body_mask = True  # 同时启用配置标志
                print(f"\n{'='*60}")
                print(f"🎭 Body Mask 已启用 (iteration {current_iter})")
                print(f"   路径: {self.cfg.depth.body_mask_path}")
                print(f"{'='*60}\n")

    def get_amp_observations(self):
        return self.dof_pos
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        num_gait = self.cfg.env.num_gait
        noise_vec[:num_gait+3] = 0. # gait cmd + lin cmd
        noise_vec[num_gait+3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[num_gait+6:9] = noise_scales.gravity * noise_level
        noise_vec[num_gait+9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[num_gait+9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[num_gait+9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        return noise_vec
        
    def _init_buffers(self):
        super()._init_buffers()
        # action smooth reward
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_feet_contact_force = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.gait_commands = torch.zeros(self.num_envs, self.cfg.env.num_gait, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_rew_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.gait_choices = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        self.feet_indicator_offset = torch.tensor(self.cfg.asset.feet_indicator_offset, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_indicator_pos = torch.zeros(self.num_envs, len(self.feet_indices), *self.feet_indicator_offset.shape,dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_collision_indicator_offset = torch.tensor(self.cfg.asset.feet_collision_indicator_offset, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_collision_indicator_pos = torch.zeros(self.num_envs, len(self.feet_indices), *self.feet_collision_indicator_offset.shape,dtype=torch.float, device=self.device, requires_grad=False)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_last_actions[env_ids] = 0.
        self.last_feet_contact_force[env_ids] = 0.

    def _create_envs(self):
        super()._create_envs()
        shoulder_penalty_names = []
        for name in self.cfg.asset.extra_shoulder_penalty_name:
            shoulder_penalty_names.extend([s for s in self.dof_names if name in s])
        self.shoulder_pelvic_indices = torch.zeros(len(shoulder_penalty_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(shoulder_penalty_names)):
            self.shoulder_pelvic_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], shoulder_penalty_names[i])
        
        self.shoulder_ry_limits = self.dof_pos_limits[self.shoulder_pelvic_indices]
    
    def _reset_dofs(self, env_ids):
        if self.cfg.init_state.random_default_pos:
            rand_default_pos = self.motion_reference[np.random.randint(0, self.motion_reference.shape[0], size=(env_ids.shape[0], )), :]
            self.dof_pos[env_ids] = rand_default_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def compute_both_feet_info(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        for i in range(len(self.feet_indices)):
            self.footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            self.footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])    

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.knee_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.knee_indices, 0:3]

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact


        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        terminal_obs, terminal_critic_obs = self.compute_observations()
        self.reset_idx(env_ids)

        self.update_depth_buffer()
        self.warp_update_depth_buffer()

        self._post_physics_step_callback()
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_feet_contact_force[:] = self.contact_forces[:, self.feet_indices]

        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            if self.cfg.depth.use_camera and self.cfg.depth.warp_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
                window_name = "Warp Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Warp Depth Image", self.warp_depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
            elif self.cfg.depth.warp_camera:
                window_name = "Warp Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Warp Depth Image", self.warp_depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
            elif self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
            # self._draw_foot_indicator()

        cur_knee_pos_trans = self.knee_pos - self.root_states[:, 0:3].unsqueeze(1)
        for i in range(len(self.knee_indices)):
            self.knee_pos_in_body[:, i, :] = quat_rotate_inverse(self.base_quat, cur_knee_pos_trans[:, i, :])
    
        return env_ids, terminal_amp_states, terminal_obs[env_ids], terminal_critic_obs[env_ids]

    def _post_physics_step_callback(self):
        self.compute_both_feet_info()
        self.compute_feet_indicator_pos()
        self.compute_feet_collision_indicator_pos()
        
        return super()._post_physics_step_callback()
    
    def _draw_foot_indicator(self):
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.01, 10, 10, None, color=(1, 0, 0))
        indicator_pos = self.feet_collision_indicator_pos.reshape(-1, 3)
        for i, point in enumerate(indicator_pos):
            pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)
            gymutil.draw_lines(
                sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose
            )

    def compute_feet_indicator_pos(self):
        num_dot = self.feet_indicator_offset.shape[0]
        ankle_quat = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        feet_offset = self.feet_indicator_offset.view(1, 1, num_dot, 3).expand(self.num_envs, 2, num_dot, 3)
        quat_expanded = ankle_quat.unsqueeze(2).expand(-1, -1, num_dot, -1)  # (num_envs, 2, num_dot, 4)
        rotated_points = quat_apply(quat_expanded.reshape(-1, 4), feet_offset.reshape(-1, 3))
        rotated_points = rotated_points.view(self.num_envs, 2, num_dot, 3)
        self.feet_indicator_pos = rotated_points + self.feet_pos.unsqueeze(2)  # (num_envs, 2, num_dot, 3)
        
    def compute_feet_collision_indicator_pos(self):
        # collision indicator
        num_dot = self.feet_collision_indicator_offset.shape[0]
        ankle_quat = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        feet_offset = self.feet_collision_indicator_offset.view(1, 1, num_dot, 3).expand(self.num_envs, 2, num_dot, 3)
        quat_expanded = ankle_quat.unsqueeze(2).expand(-1, -1, num_dot, -1)  # (num_envs, 2, num_dot, 4)
        rotated_points = quat_apply(quat_expanded.reshape(-1, 4), feet_offset.reshape(-1, 3))
        rotated_points = rotated_points.view(self.num_envs, 2, num_dot, 3)
        self.feet_collision_indicator_pos = rotated_points + self.feet_pos.unsqueeze(2)  # (num_envs, 2, num_dot, 3)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1000., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)

        self.reset_buf |= (self._get_base_heights() < 0.4)
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.gait_commands,
                                    self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ),dim=-1)
        
        self.privileged_obs_buf = torch.cat((self.gait_commands,  
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ),dim=-1)

        if self.cfg.env.feet_info:  # 6 * 2 = 12
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.footpos_in_body_frame.reshape(self.num_envs, -1), 
                                                 self.footvel_in_body_frame.reshape(self.num_envs, -1)), dim=-1)
        
        if self.cfg.env.foot_force_info:  # 6
            contact_force = self.sensor_forces.flatten(1) * self.obs_scales.contact_force
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, contact_force), dim=-1)
        
        if self.cfg.env.priv_info:  # 32 + 1 + 1 + 1 + 3 = 38
            self.privileged_obs_buf= torch.cat((self.privileged_obs_buf, self.root_states[:, 2].unsqueeze(-1)), dim=-1)

            if self.cfg.domain_rand.randomize_friction:  # 1
                self.privileged_obs_buf= torch.cat((self.privileged_obs_buf, self.randomized_frictions), dim=-1)

            if (self.cfg.domain_rand.randomize_base_mass):  # 1
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.randomized_added_masses), dim=-1)

            if (self.cfg.domain_rand.randomize_com_pos):  # 3
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.randomized_com_pos * self.obs_scales.com_pos), dim=-1)

            if (self.cfg.domain_rand.randomize_gains):  # 16 * 2
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, (self.randomized_p_gains / self.p_gains - 1) * self.obs_scales.pd_gains), dim=-1)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, (self.randomized_d_gains / self.d_gains - 1) * self.obs_scales.pd_gains), dim=-1)
        
        if self.cfg.terrain.measure_heights:  # 187
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
            
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        return self.obs_buf, self.privileged_obs_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.gait_rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if name in self.cfg.rewards.all_gait_rewards:
                for gait_idx in range(self.cfg.env.num_gait):
                    if name in self.cfg.rewards.gait_rewards[gait_idx]:
                        self.gait_rew_buf[:, gait_idx] += rew 
                        num_gait = self.gait_commands[:, gait_idx].sum()
                        self.episode_sums[name] += ((rew * self.gait_commands[:, gait_idx]) / num_gait) * self.num_envs
            else:
                self.rew_buf += rew
                self.episode_sums[name] += rew
        self.rew_buf += (self.gait_rew_buf * self.gait_commands).sum(dim=1)
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        super()._resample_commands(env_ids)

        # exclude_mode = "exclude_100"
        exclude_mode = None
        if exclude_mode == "exclude_010":
            valid_indices = torch.tensor([0, 2], device=self.device)
        elif exclude_mode == "exclude_100": 
            valid_indices = torch.tensor([1, 2], device=self.device)
        elif exclude_mode == "exclude_001":
            valid_indices = torch.tensor([0, 1], device=self.device)
        else: 
            valid_indices = torch.arange(3, device=self.device)

        indices = torch.randint(0, len(valid_indices), size=(len(env_ids),))
        selected_indices = valid_indices[indices]
        self.gait_commands[env_ids] = self.gait_choices[selected_indices]

        if self.cfg.terrain.mesh_type == "trimesh":
            only_forward_env = (self.env_class!=0)
            self.commands[only_forward_env, 3] = 0
            self.commands[only_forward_env, 2] = 0
            self.commands[only_forward_env, 1] = 0
            self.commands[only_forward_env, 0] = torch.abs(self.commands[only_forward_env, 0])
    
    """ Locomotion Reward Functions"""
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
   
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        # 每个关节的一阶差分（动作变化）
        term_1_per_joint = torch.square(self.last_actions - self.actions)
        # 每个关节的二阶差分（加速度变化）
        term_2_per_joint = torch.square(self.actions + self.last_last_actions - 2 * self.last_actions)
        # 每个关节的动作幅度
        term_3_per_joint = 0.05 * torch.abs(self.actions)
        
        # 每个关节的总惩罚
        per_joint_penalty = term_1_per_joint + term_2_per_joint + term_3_per_joint
        
        # 初始化关节名称
        if not hasattr(self, '_smoothness_joint_names'):
            self._smoothness_joint_names = self.dof_names
            self._smoothness_step_counter = 0
            print(f"\n监控动作平滑度关节: {self._smoothness_joint_names}")
        
        self._smoothness_step_counter += 1
        
        # 每 24 步打印一次
        if self._smoothness_step_counter % 24 == 0:
            # 计算每个关节的平均惩罚
            joint_penalties = per_joint_penalty.mean(dim=0).cpu().numpy()
            total_penalty = per_joint_penalty.sum(dim=1).mean().item()
            
            # 找出惩罚最大的前5个关节
            sorted_indices = joint_penalties.argsort()[::-1][:5]
            
            penalty_info = []
            for idx in sorted_indices:
                name = self._smoothness_joint_names[idx]
                penalty = joint_penalties[idx]
                # 简化关节名称
                short_name = name.replace('_joint', '').replace('left_', 'L_').replace('right_', 'R_')
                penalty_info.append(f"{short_name}:{penalty:.2f}")
            
            print(f"[平滑度] 总:{total_penalty:.1f} | Top5: {' | '.join(penalty_info)}")
        
        return per_joint_penalty.sum(dim=1)   

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_joint_power(self):
        # Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             3 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_arm_joint_deviation(self):
        return torch.square(torch.norm(torch.abs(self.dof_pos[:, 12:] - self.default_dof_pos[:, 12:]), dim=1))
    
    def _reward_arm_symmetry(self):
        """
        惩罚左右手臂姿势不对称
        关节索引: 12=left_shoulder_pitch, 13=left_elbow, 14=right_shoulder_pitch, 15=right_elbow
        """
        # 左手臂关节 (shoulder_pitch, elbow)
        left_arm = self.dof_pos[:, 12:14]
        # 右手臂关节 (shoulder_pitch, elbow)
        right_arm = self.dof_pos[:, 14:16]
        
        # 计算左右手臂的差异
        arm_diff = torch.abs(left_arm - right_arm)
        
        # 初始化计数器
        if not hasattr(self, '_arm_symmetry_step_counter'):
            self._arm_symmetry_step_counter = 0
            print(f"\n✅ 手臂对称性奖励已启用")
        
        self._arm_symmetry_step_counter += 1
        
        # 每 24 步打印一次
        if self._arm_symmetry_step_counter % 24 == 0:
            mean_diff = arm_diff.mean(dim=0).cpu().numpy()
            max_diff = arm_diff.max(dim=0)[0].cpu().numpy()
            left_mean = left_arm.mean(dim=0).cpu().numpy()
            right_mean = right_arm.mean(dim=0).cpu().numpy()
            print(f"[手臂对称] 左臂:[{left_mean[0]:.3f}, {left_mean[1]:.3f}] | "
                  f"右臂:[{right_mean[0]:.3f}, {right_mean[1]:.3f}] | "
                  f"差异:[{mean_diff[0]:.3f}, {mean_diff[1]:.3f}]")
        
        # 返回平方差异的和
        return torch.sum(torch.square(arm_diff), dim=1)

    def _reward_hip_joint_deviation(self):
        return torch.square(torch.norm(torch.abs(self.dof_pos[:, [1, 2, 7, 8]]), dim=1))
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_feet_lateral_distance(self):
        # Penalize feet lateral distance deviating from target
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        # 计算实际横向距离（取绝对值）
        actual_lateral_distance = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        
        # 计算前后间距（X方向）
        actual_longitudinal_distance = torch.abs(footpos_in_body_frame[:, 0, 0] - footpos_in_body_frame[:, 1, 0])
        
        # 初始化计数器
        if not hasattr(self, '_lateral_dist_step_counter'):
            self._lateral_dist_step_counter = 0
        
        self._lateral_dist_step_counter += 1
        
        # 每 24 步打印一次
        if self._lateral_dist_step_counter % 24 == 0:
            mean_dist = actual_lateral_distance.mean().item()
            min_dist = actual_lateral_distance.min().item()
            max_dist = actual_lateral_distance.max().item()
            target = self.cfg.rewards.feet_min_lateral_distance_target
            mean_long = actual_longitudinal_distance.mean().item()
            print(f"[脚间距] 横向目标:{target:.3f}m | 横向:{mean_dist:.3f}m | 前后:{mean_long:.3f}m")
        
        # 惩罚偏离目标的距离（平方惩罚）
        rew = torch.square(actual_lateral_distance - self.cfg.rewards.feet_min_lateral_distance_target)
        return rew
    
    def _reward_feet_slippage(self):
        return torch.sum(torch.norm(self.feet_vel, dim=-1) * (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    
    def _reward_feet_contact_force(self):
        # penalize high contact forces
        return torch.sum(F.relu(self.contact_forces[:, self.feet_indices, 2] - self.cfg.rewards.feet_contact_force_range[0]), dim=-1)
    
    def _reward_feet_force_rate(self):
        return torch.sum(F.relu(self.contact_forces[:, self.feet_indices, 2] - self.last_feet_contact_force[..., 2]), dim=-1)
    
    def _reward_feet_contact_momentum(self):
        """
        Penalizes the momentum of the feet contact forces, encouraging a more stable and controlled motion.
        foot vel * contact force
        """
        feet_contact_force = self.contact_forces[:, self.feet_indices, 2]
        feet_vertical_vel = self.feet_vel[:, :, 2]
        rew = torch.sum(torch.abs(feet_contact_force * feet_vertical_vel), dim=-1)
        return rew
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        contact_forces_norm = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
        has_collision = (contact_forces_norm > 0.1)
        
        # 初始化
        if not hasattr(self, '_collision_body_names'):
            self._collision_body_names = []
            body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
            for idx in self.penalised_contact_indices:
                self._collision_body_names.append(body_names[idx])
            print(f"\n监控碰撞部位: {self._collision_body_names}")
            self._step_counter = 0
        
        self._step_counter += 1
        
        # 每 24 步（一个 iteration）打印一次
        if self._step_counter % 24 == 0:
            collision_counts = has_collision.sum(dim=0).cpu().numpy()
            total_collisions = has_collision.sum().item()
            
            if total_collisions > 0:
                collision_info = []
                for i, (name, count) in enumerate(zip(self._collision_body_names, collision_counts)):
                    if count > 0:
                        percentage = count / self.num_envs * 100
                        collision_info.append(f"{name}:{count}({percentage:.0f}%)")
                print(f"[碰撞] 总:{total_collisions:.0f} | {' | '.join(collision_info)}")
        
        return torch.sum(1. * has_collision, dim=1)
    def _reward_stuck(self):
        # Penalize stuck
        return (torch.abs(self.base_lin_vel[:, 0]) < 0.1) * (torch.abs(self.commands[:, 0]) > 0.1)
    
    def _reward_cheat(self):
        # penalty cheating to bypass the obstacle
        no_cheat_env = (self.env_class != 0)
        forward = quat_apply(self.base_quat[no_cheat_env], self.forward_vec[no_cheat_env])
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        cheat = (heading > 1.0) | (heading < -1.0)
        cheat_penalty = torch.zeros(self.num_envs, device=self.device)
        cheat_penalty[no_cheat_env] = cheat.float()
        return cheat_penalty
    
    def _reward_feet_edge(self):
        foot_indicators_pos_xy = ((self.feet_indicator_pos[..., :2]+self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()
        foot_indicators_pos_xy[..., 0] = torch.clip(foot_indicators_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        foot_indicators_pos_xy[..., 1] = torch.clip(foot_indicators_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)

        feet_at_edge = self.x_edge_mask[foot_indicators_pos_xy[..., 0], foot_indicators_pos_xy[..., 1]]
        feet_at_edge = torch.sum(feet_at_edge, dim=-1) >= 3
        feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(feet_at_edge, dim=1)
        return rew

    def _reward_y_offset_pen(self):
        # Penalize y offset from the origin (except roughness terrain)
        pen = torch.abs(self.root_states[:, 1] - self.origin_y) * (self.env_class != 0)
        return pen

    """ Gait Reward Functions"""
    def _reward_squat_height(self):
        # Encourages the robot to maintain a specific base height during squats.
        root_height = self.root_states[:, 2]
        feet_height = self.feet_pos[:, :, 2]
        base_height = root_height - torch.min(feet_height, dim=1).values
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.squat_height_target)/self.cfg.rewards.tracking_sigma)
    
    def _reward_knee_height(self):
        # Encourages the robot to maintain a specific knee height during high-knees gait.
        knee_height_diff = torch.abs(self.knee_pos_in_body[..., 2].amax(dim=1) - self.cfg.rewards.high_knees_target)
        rew = torch.exp(-knee_height_diff / self.cfg.rewards.tracking_sigma)
        return rew * (torch.abs(self.commands[:, 0]) > 0.2)
    
    """ Terrain Specific Rewards"""
    # Encourages the robot successfully traverse the terrain
    def _reward_knee_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalised_contact_indices[[3, 7]], :], dim=-1), dim=1)
    
    def _reward_feet_collision_pen(self):
        # Penalize feet hitting vertical surfaces
        foot_indicators_pos_xy = ((self.feet_collision_indicator_pos[..., :2]+self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()
        foot_indicators_pos_xy[..., 0] = torch.clip(foot_indicators_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        foot_indicators_pos_xy[..., 1] = torch.clip(foot_indicators_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        # stair
        up_stair_feet_collision = self.stair_pen_mask[0][foot_indicators_pos_xy[:, :, 0, 0], foot_indicators_pos_xy[:, :, 0, 1]] * self.contact_filt
        down_stair_feet_collision = self.stair_pen_mask[1][foot_indicators_pos_xy[:, :, 1, 0], foot_indicators_pos_xy[:, :, 1, 1]] * self.contact_filt
        stair_num_indicator_in_pen_area = torch.sum(up_stair_feet_collision, dim=-1) + torch.sum(down_stair_feet_collision, dim=-1)
        # pit
        pit_feet_collision = self.x_edge_mask[foot_indicators_pos_xy[:, :, 0, 0], foot_indicators_pos_xy[:, :, 0, 1]] * self.contact_filt
        pit_num_indicator_in_pen_area = torch.sum(pit_feet_collision, dim=-1)
        return pit_num_indicator_in_pen_area * (self.env_class == 1) + stair_num_indicator_in_pen_area * (self.env_class == 3)


    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
