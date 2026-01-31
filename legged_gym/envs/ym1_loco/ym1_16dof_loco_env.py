
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.datasets.motion_loader_g1 import G1_AMPLoader
import torch
import cv2
import torch.nn.functional as F
import numpy as np

class ym1_16Dof_Loco_Robot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.amp_motion_files = self.cfg.env.amp_motion_files
        self.num_amp_obs = self.cfg.env.num_amp_obs
        if self.cfg.env.reference_state_initialization: # NOTE only for visualize reference motion
            self.amp_loader = G1_AMPLoader(motion_dir=self.amp_motion_files, device=self.device, time_between_frames=self.dt)
            self.motion_reference = self.amp_loader.get_joint_pose_batch_16dof(torch.cat(self.amp_loader.trajectories_full, dim=0))
        
        # Body mask 支持 - 预加载数据，但根据 iteration 动态启用
        self.body_mask_enabled = False  # 初始禁用
        if hasattr(self.cfg.depth, 'body_mask_path'):
            try:
                body_mask_data = np.load(self.cfg.depth.body_mask_path, allow_pickle=True)
                self.body_masks = body_mask_data['body_masks']
                print(f"✅ Body mask 数据已加载: {self.body_masks.shape[0]} 个 masks")
                print(f"   将在 {self.cfg.depth.body_mask_start_iter} iterations 后启用")
            except Exception as e:
                print(f"⚠️  Body mask 加载失败: {e}")
                self.body_masks = None
        else:
            self.body_masks = None
        
    def get_amp_observations(self):
        return self.dof_pos
    
    def check_and_enable_body_mask(self, current_iter):
        """
        检查当前 iteration 并动态启用 body mask
        
        Args:
            current_iter: 当前训练 iteration
        """
        if self.body_masks is not None and not self.body_mask_enabled:
            if current_iter >= self.cfg.depth.body_mask_start_iter:
                self.body_mask_enabled = True
                self.cfg.depth.add_body_mask = True  # 同时启用配置标志
                print(f"\n{'='*60}")
                print(f"🎭 Body Mask 已启用 (iteration {current_iter})")
                print(f"   路径: {self.cfg.depth.body_mask_path}")
                print(f"{'='*60}\n")

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
        noise_vec[:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_feet_contact_force = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_feet_contact_force = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_indicator_offset = torch.tensor(self.cfg.asset.feet_indicator_offset, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_indicator_pos = torch.zeros(self.num_envs, len(self.feet_indices), *self.feet_indicator_offset.shape,dtype=torch.float, device=self.device, requires_grad=False)
        
        self.feet_collision_indicator_offset = torch.tensor(self.cfg.asset.feet_collision_indicator_offset, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_collision_indicator_pos = torch.zeros(self.num_envs, len(self.feet_indices), *self.feet_collision_indicator_offset.shape,dtype=torch.float, device=self.device, requires_grad=False)


        # 步态相位变量 - 用于强制交替步态
        self.gait_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_frequency = 1.5  # 步态频率 Hz，约0.67秒一个完整周期
        self.last_swing_foot = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0=左脚, 1=右脚
        
        # 追踪每只脚的最远前进位置（用于防止并步）
        self.feet_max_forward_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)  # [左脚, 右脚]

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_feet_contact_force[env_ids] = 0.
        self.last_last_feet_contact_force[env_ids] = 0.
        # 重置双脚着地计时器
        if hasattr(self, 'both_feet_contact_time'):
            self.both_feet_contact_time[env_ids] = 0.
        # 重置步态相位
        if hasattr(self, 'gait_phase'):
            self.gait_phase[env_ids] = 0.
            self.last_swing_foot[env_ids] = 0
        # 重置脚的最远前进位置
        if hasattr(self, 'feet_max_forward_pos'):
            self.feet_max_forward_pos[env_ids] = 0.
        
        # === 新增 [2026-01-23]: 重置地形记忆变量 ===
        # 修改动机: 解决楼顶盲区问题引入了记忆机制，机器人看到楼梯后会保持高抬腿一段时间
        # 但重置时必须清空记忆，否则机器人重生到平地时会继续高抬腿
        # if hasattr(self, 'avg_obstacle_height'):
        #     self.avg_obstacle_height[env_ids] = 0.

    def _draw_foot_indicator(self):
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.01, 10, 10, None, color=(1, 0, 0))
        indicator_pos = self.feet_indicator_pos.reshape(-1, 3)
        for i, point in enumerate(indicator_pos):
            pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)
            gymutil.draw_lines(
                sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose
            )

    def _reset_dofs(self, env_ids):
        if self.cfg.init_state.random_default_pos:
            rand_default_pos = self.motion_reference[np.random.randint(0, self.motion_reference.shape[0], size=(env_ids.shape[0], )), :]
            self.dof_pos[env_ids] = rand_default_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        else:
            # 减小初始随机范围，从 0.5~1.5 改为 0.95~1.05，让机器人更稳定地开始
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.95, 1.05, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

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

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.contact_over = torch.logical_and(~contact, self.last_contacts)
        self.last_contacts = contact

        # [Modified] Move callback up to ensure measured_heights is updated before reward computation
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        terminal_obs, terminal_critic_obs = self.compute_observations()
        self.reset_idx(env_ids)

        self.update_depth_buffer()
        self.warp_update_depth_buffer()
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_last_feet_contact_force[:] = torch.clone(self.last_feet_contact_force[:])
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
    
        return env_ids, terminal_amp_states, terminal_obs[env_ids], terminal_critic_obs[env_ids]

    def _post_physics_step_callback(self):
        self.compute_both_feet_info()
        self.compute_feet_indicator_pos()
        self.compute_feet_collision_indicator_pos()
        
        return super()._post_physics_step_callback()
    
    def compute_both_feet_info(self):
        # compute both feet swing length
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        for i in range(len(self.feet_indices)):
            self.footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            self.footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
    
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
        # self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)  # 已禁用: 不再因roll/pitch角度过大而重置
        self.reset_buf |= (self._get_base_heights() < 0.4)

        if self.cfg.terrain.mesh_type == "trimesh":
            offset_y = torch.abs(self.root_states[:, 1] - self.origin_y)
            only_forward_env = torch.logical_and(self.env_class != 0, self.env_class != 1)
            self.reset_buf |= torch.logical_and(only_forward_env, offset_y>1.0)
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ),dim=-1)
        
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
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
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
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

        only_forward_env = torch.logical_and(self.env_class != 0, self.env_class != 1)
        self.commands[only_forward_env, 3] = 0
        self.commands[only_forward_env, 2] = 0
        self.commands[only_forward_env, 1] = 0
        self.commands[only_forward_env, 0] = torch.abs(self.commands[only_forward_env, 0])
    
    #------------ reward functions----------------
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
    
    def _reward_base_height(self):
        # Penalize base height deviating from target (relative to feet, not world frame)
        # 计算 base 到脚底的相对高度，而不是世界坐标系的绝对高度
        base_height = self.root_states[:, 2]  # base 的 Z 坐标
        
        # 获取两只脚的平均高度作为地面参考
        feet_height = self.feet_pos[:, :, 2].mean(dim=1)  # 两只脚的平均 Z 坐标
        
        # 计算 base 相对于脚底的高度
        relative_height = base_height - feet_height
        
        target_height = self.cfg.rewards.base_height_target
        
        # 初始化计数器
        if not hasattr(self, '_base_height_step_counter'):
            self._base_height_step_counter = 0
        
        self._base_height_step_counter += 1
        
        # 每 24 步打印一次
        if self._base_height_step_counter % 24 == 0:
            mean_rel_height = relative_height.mean().item()
            min_rel_height = relative_height.min().item()
            max_rel_height = relative_height.max().item()
            mean_abs_height = base_height.mean().item()
            print(f"[Base高度] 目标:{target_height:.3f}m | 相对高度:{mean_rel_height:.3f}m | 绝对高度:{mean_abs_height:.3f}m")
        
        # 惩罚相对高度偏离目标
        rew = torch.square(relative_height - target_height)
        return rew
    
    def _reward_joint_power(self):
        # Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_feet_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
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

    def _reward_hip_joint_deviation(self):
        return torch.square(torch.norm(torch.abs(self.dof_pos[:, [1, 2, 7, 8]]), dim=1))
    
    def _reward_leg_joint_deviation(self):
        """
        惩罚腿部关节（hip_pitch, knee）偏离默认位置太多，保持自然站姿
        关节索引: 0=left_leg_pitch, 3=left_knee, 6=right_leg_pitch, 9=right_knee
        """
        leg_indices = [0, 3, 6, 9]  # leg_pitch 和 knee 关节
        deviation = self.dof_pos[:, leg_indices] - self.default_dof_pos[:, leg_indices]
        return torch.sum(torch.square(deviation), dim=1)
    
    def _reward_knee_hyperextension(self):
        """
        严厉惩罚膝关节反屈（负角度），防止膝盖向后弯
        关节索引: 3=left_knee, 9=right_knee
        """
        knee_indices = [3, 9]
        knee_pos = self.dof_pos[:, knee_indices]
        # 膝关节角度小于0.05弧度时惩罚
        hyperextension = torch.clamp(0.05 - knee_pos, min=0)
        return torch.sum(torch.square(hyperextension) * 100, dim=1)
    
    def _reward_ankle_deviation(self):
        """
        惩罚脚踝关节偏离默认位置太多，防止过度背屈/跖屈
        关节索引: 4=left_ankle_pitch, 10=right_ankle_pitch
        """
        ankle_pitch_indices = [4, 10]
        ankle_pos = self.dof_pos[:, ankle_pitch_indices]
        default_ankle = self.default_dof_pos[:, ankle_pitch_indices]
        
        # 允许一定范围的偏离（±0.5 rad ≈ ±30°），超出则惩罚
        deviation = torch.abs(ankle_pos - default_ankle)
        excess = torch.clamp(deviation - 0.5, min=0)
        return torch.sum(torch.square(excess), dim=1)

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

    def _reward_no_fly(self):
        is_jump =  torch.all(self.contact_forces[:, self.feet_indices, 2] < 1, dim=1)
        return is_jump.float()
    
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
    
    def _reward_feet_longitudinal_distance(self):
        """
        惩罚前后脚（两脚在X方向的间距过大），鼓励双脚保持平行
        只在零速度命令时生效
        """
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        # 计算前后间距（X方向），目标是尽量小
        actual_longitudinal_distance = torch.abs(footpos_in_body_frame[:, 0, 0] - footpos_in_body_frame[:, 1, 0])
        
        # 只在双脚都着地时惩罚前后脚（摆动时允许前后差异）
        both_contact = torch.all(self.contact_filt, dim=1)
        
        # 只在零速度命令时生效
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        no_cmd = cmd_norm < 0.1
        
        # 平方惩罚，目标是0（双脚平行）
        target = getattr(self.cfg.rewards, 'feet_longitudinal_distance_target', 0.05)
        rew = torch.square(actual_longitudinal_distance - target) * both_contact.float() * no_cmd.float()
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
        
        # return torch.sum(1. * has_collision, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~self.contact_filt
        return rew_airTime
    
    # def _reward_single_foot_contact(self):
    #     """
    #     奖励单脚着地（交替步态），惩罚双脚同时着地或同时离地
    #     """
    #     # 检测每只脚是否着地
    #     contact = self.contact_filt  # shape: (num_envs, 2) 左脚和右脚
        
    #     # 只有一只脚着地时奖励（异或操作）
    #     single_contact = contact[:, 0] ^ contact[:, 1]  # 左脚 XOR 右脚
        
    #     # 只在有速度命令时才奖励
    #     has_command = torch.norm(self.commands[:, :2], dim=1) > 0.1
        
    #     return single_contact.float() * has_command.float()
    
    def _reward_single_foot_contact(self):
        """
        速度自适应的步态奖励：
        - 站立 (<0.15 m/s)：奖励双脚着地，保持稳定
        - 慢走 (0.15~0.4 m/s)：允许双支撑期，轻微奖励单脚
        - 快走 (>0.4 m/s)：强制单脚支撑，交替步态
        """
        contact = self.contact_filt  # shape: (num_envs, 2) 左脚和右脚
        single_contact = contact[:, 0] ^ contact[:, 1]  # 只有一只脚着地
        both_contact = torch.all(contact, dim=1)  # 双脚都着地
        
        cmd_vel = torch.norm(self.commands[:, :2], dim=1)
        
        # 三个速度区间
        is_standing = cmd_vel < 0.15       # 站立
        is_slow_walk = (cmd_vel >= 0.15) & (cmd_vel < 0.4)  # 慢走
        is_fast_walk = cmd_vel >= 0.4      # 快走
        
        # 站立：奖励双脚着地，保持稳定
        stand_reward = both_contact.float() * is_standing.float()
        
        # 慢走：轻微奖励单脚，但也接受双脚（允许更长双支撑期）
        slow_walk_reward = (single_contact.float() * 0.5 + both_contact.float() * 0.3) * is_slow_walk.float()
        
        # 快走：强烈奖励单脚，鼓励交替步态
        fast_walk_reward = single_contact.float() * is_fast_walk.float()
        
        return stand_reward + slow_walk_reward + fast_walk_reward

    def _reward_alternating_gait(self):
        """
        强制交替步态：惩罚双脚同时着地时间过长
        鼓励机器人快速切换支撑脚，而不是双脚同时站在同一台阶
        """
        # 双脚都着地
        both_contact = torch.all(self.contact_filt, dim=1)
        
        # 初始化双脚着地计时器
        if not hasattr(self, 'both_feet_contact_time'):
            self.both_feet_contact_time = torch.zeros(self.num_envs, device=self.device)
        
        # 更新计时器
        self.both_feet_contact_time = torch.where(
            both_contact,
            self.both_feet_contact_time + self.dt,
            torch.zeros_like(self.both_feet_contact_time)
        )
        
        # 只在有速度命令时惩罚
        has_command = torch.norm(self.commands[:, :2], dim=1) > 0.1
        
        # 双脚着地超过0.15秒开始惩罚（允许短暂的双支撑相）
        penalty = F.relu(self.both_feet_contact_time - 0.15) * has_command.float()
        
        return penalty
    
    def _reward_step_length(self):
        """
        奖励较大的步长，鼓励机器人迈大步而不是小碎步
        """
        # 计算两脚在前进方向的距离差
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        # 前后脚距离（X方向）
        step_length = torch.abs(footpos_in_body_frame[:, 0, 0] - footpos_in_body_frame[:, 1, 0])
        
        # 只在单脚着地时奖励步长（摆动相）
        single_contact = self.contact_filt[:, 0] ^ self.contact_filt[:, 1]
        
        # 只在有速度命令时奖励
        has_command = torch.norm(self.commands[:, :2], dim=1) > 0.1
        
        # 奖励步长，目标约0.3m
        rew = torch.clamp(step_length, 0, 0.4) * single_contact.float() * has_command.float()
        
        return rew
    
    def _reward_gait_phase(self):
        """
        基于相位的步态奖励 - 强制左右腿交替
        使用正弦波相位来指定哪只脚应该在摆动相
        """
        # 更新步态相位
        has_command = torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.gait_phase = torch.where(
            has_command,
            (self.gait_phase + self.dt * self.gait_frequency * 2 * 3.14159) % (2 * 3.14159),
            torch.zeros_like(self.gait_phase)
        )
        
        # 相位 0~π: 左脚应该摆动 (离地)
        # 相位 π~2π: 右脚应该摆动 (离地)
        left_should_swing = (self.gait_phase < 3.14159)  # 0~π
        right_should_swing = ~left_should_swing  # π~2π
        
        # 实际接触状态
        left_contact = self.contact_filt[:, 0]
        right_contact = self.contact_filt[:, 1]
        
        # 奖励：当相位指示某脚应该摆动时，该脚确实离地
        left_correct = left_should_swing & ~left_contact  # 左脚应摆动且确实离地
        right_correct = right_should_swing & ~right_contact  # 右脚应摆动且确实离地
        
        # 惩罚：当相位指示某脚应该摆动时，该脚却着地
        left_wrong = left_should_swing & left_contact  # 左脚应摆动但着地
        right_wrong = right_should_swing & right_contact  # 右脚应摆动但着地
        
        reward = (left_correct.float() + right_correct.float()) * has_command.float()
        penalty = (left_wrong.float() + right_wrong.float()) * has_command.float() * 0.5
        
        return reward - penalty
    
    def _reward_foot_swing_symmetry(self):
        """
        惩罚不对称的摆动模式 - 防止总是同一只脚先迈步
        追踪哪只脚最后摆动，惩罚连续同一只脚摆动
        """
        # 检测哪只脚刚从着地变为离地（开始摆动）
        left_start_swing = self.contact_over[:, 0]  # 左脚刚离地
        right_start_swing = self.contact_over[:, 1]  # 右脚刚离地
        
        # 更新最后摆动的脚
        self.last_swing_foot = torch.where(left_start_swing, torch.zeros_like(self.last_swing_foot), self.last_swing_foot)
        self.last_swing_foot = torch.where(right_start_swing, torch.ones_like(self.last_swing_foot), self.last_swing_foot)
        
        # 惩罚连续同一只脚摆动
        # 如果左脚开始摆动，但上次也是左脚摆动 -> 惩罚
        left_repeat = left_start_swing & (self.last_swing_foot == 0)
        right_repeat = right_start_swing & (self.last_swing_foot == 1)
        
        has_command = torch.norm(self.commands[:, :2], dim=1) > 0.1
        penalty = (left_repeat.float() + right_repeat.float()) * has_command.float()
        
        return -penalty
    
    def _reward_stuck(self):
        # Penalize stuck
        return (torch.abs(self.base_lin_vel[:, 0]) < 0.1) * (torch.abs(self.commands[:, 0]) > 0.1)
    
    def _reward_cheat(self):
        # penalty cheating to bypass the obstacle
        no_cheat_env = torch.logical_and(self.env_class != 0, self.env_class != 1)
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
        feet_at_edge = torch.sum(feet_at_edge, dim=-1) >= 2
        feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(feet_at_edge, dim=1)
        return rew

    # def _reward_feet_edge(self):
    #     feet_pos_xy = ((self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
    #     feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
    #     feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
    #     feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
    #     self.feet_at_edge = self.contact_filt & feet_at_edge
    #     rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
    #     return rew
    
    def _reward_y_offset_pen(self):
        """
        惩罚机器人偏离直线路径（Y方向偏移）
        这对于保持直线行走非常重要
        """
        # 计算相对于起始Y位置的偏移
        y_offset = torch.abs(self.root_states[:, 1] - self.origin_y)
        
        # 只在有前进速度命令时惩罚横向偏移
        forward_cmd = torch.abs(self.commands[:, 0]) > 0.1
        
        # 对于非平地环境（env_class != 0 和 != 1），强制直线行走
        non_flat_env = torch.logical_and(self.env_class != 0, self.env_class != 1)
        
        # 组合条件：有前进命令 或 非平地环境
        should_penalize = torch.logical_or(forward_cmd, non_flat_env)
        
        pen = y_offset * should_penalize.float()
        
        # 初始化打印计数器
        if not hasattr(self, '_y_offset_print_counter'):
            self._y_offset_print_counter = 0
        
        self._y_offset_print_counter += 1
        
        # 每 100 步打印一次统计
        if self._y_offset_print_counter % 100 == 0:
            mean_offset = y_offset.mean().item()
            max_offset = y_offset.max().item()
            penalized_envs = should_penalize.sum().item()
            print(f"[Y偏移] 平均:{mean_offset:.3f}m | 最大:{max_offset:.3f}m | 惩罚环境数:{penalized_envs}/{self.num_envs}")
        
        return pen

    def _reward_stand_still(self):
        """
        奖励在零速度指令时保持静止站立
        当速度指令接近零时，惩罚任何身体运动
        """
        # 检测是否是零速度指令
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        no_cmd = cmd_norm < 0.1  # 速度指令小于0.1m/s视为零速度
        
        # 惩罚身体线速度
        lin_vel_penalty = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        
        # 惩罚脚部速度（防止原地踏步）
        feet_vel_penalty = torch.sum(torch.norm(self.feet_vel, dim=-1), dim=1)
        
        # 只在零速度指令时应用惩罚
        penalty = (lin_vel_penalty + 0.5 * feet_vel_penalty) * no_cmd.float()
        
        return penalty
    
    def _reward_feet_still_when_stand(self):
        """
        零速度时奖励双脚同时着地且保持静止
        """
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        no_cmd = cmd_norm < 0.1
        
        # 双脚都着地
        both_contact = torch.all(self.contact_filt, dim=1)
        
        # 脚部速度很小
        feet_vel_norm = torch.sum(torch.norm(self.feet_vel, dim=-1), dim=1)
        feet_still = feet_vel_norm < 0.1
        
        # 奖励：零速度 + 双脚着地 + 脚静止
        reward = (no_cmd & both_contact & feet_still).float()
        
        return reward

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


    def _reward_both_feet_same_height(self):
        """
        惩罚双脚在同一高度（并步行为）
        只在上坡/楼梯地形时惩罚，平地不惩罚
        """
        # 获取两只脚的高度
        left_foot_height = self.feet_pos[:, 0, 2]
        right_foot_height = self.feet_pos[:, 1, 2]
        
        # 计算高度差
        height_diff = torch.abs(left_foot_height - right_foot_height)
        
        # 双脚都着地时
        both_contact = torch.all(self.contact_filt, dim=1)
        
        # 有前进速度命令时
        has_forward_cmd = self.commands[:, 0] > 0.2  # 只在前进时
        
        # 检测是否在楼梯/斜坡地形（env_class > 1 通常是非平地）
        # 或者通过脚的绝对高度判断：如果脚高于初始高度，说明在爬坡
        on_elevated_terrain = (self.feet_pos[:, :, 2].mean(dim=1) > 0.20)  # 脚平均高度 > 10cm
        
        # 高度差小于阈值（比如8cm）时惩罚 - 说明是并步
        same_height = height_diff < 0.08
        
        # 只在：有前进命令 + 双脚着地 + 高度差小 + 在高地形时惩罚
        penalty = (both_contact & has_forward_cmd & same_height & on_elevated_terrain).float()
        
        return penalty

    def _reward_step_forward_alternating(self):
        """
        奖励交替向前迈步 - 防止并步行为
        检测：如果一只脚着地，但位置在另一只脚后面，说明是并步
        """
        # 获取脚在身体坐标系中的位置
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        # 当前脚的前进位置（X方向）
        left_foot_x = footpos_in_body_frame[:, 0, 0]
        right_foot_x = footpos_in_body_frame[:, 1, 0]
        
        # 检测脚刚着地（从离地变为着地）
        left_just_landed = self.contact_filt[:, 0] & (~self.last_contacts[:, 0])
        right_just_landed = self.contact_filt[:, 1] & (~self.last_contacts[:, 1])
        
        # 有前进速度命令时
        has_forward_cmd = self.commands[:, 0] > 0.2
        
        # 当左脚着地时，检查它是否在右脚前面（正常）还是后面（并步）
        # 正常：左脚在右脚前面至少 5cm
        left_ahead_of_right = left_foot_x > (right_foot_x + 0.10)
        left_behind_right = left_foot_x < (right_foot_x - 0.10)  # 并步：左脚在右脚后面
        
        # 当右脚着地时，检查它是否在左脚前面（正常）还是后面（并步）
        right_ahead_of_left = right_foot_x > (left_foot_x + 0.10)
        right_behind_left = right_foot_x < (left_foot_x - 0.10)  # 并步：右脚在左脚后面
        
        # 奖励：着地时在对侧脚前面
        left_reward = (left_just_landed & has_forward_cmd & left_ahead_of_right).float() * torch.clamp(left_foot_x - right_foot_x, 0, 0.5)
        right_reward = (right_just_landed & has_forward_cmd & right_ahead_of_left).float() * torch.clamp(right_foot_x - left_foot_x, 0, 0.5)
        
        # 惩罚：着地时在对侧脚后面（并步行为）
        left_penalty = (left_just_landed & has_forward_cmd & left_behind_right).float()
        right_penalty = (right_just_landed & has_forward_cmd & right_behind_left).float()
        
        # 更新最远位置（用于调试，但不用于奖励计算）
        self.feet_max_forward_pos[:, 0] = torch.maximum(self.feet_max_forward_pos[:, 0], left_foot_x)
        self.feet_max_forward_pos[:, 1] = torch.maximum(self.feet_max_forward_pos[:, 1], right_foot_x)
        
        return left_reward + right_reward - 2.0 * (left_penalty + right_penalty)

    # === 新增 [2026-01-23]: 楼梯攀爬核心奖励函数组合拳 ===
    
    def _reward_height_scan_gradient_clearance(self):
        """
        [核心奖励 1] 基于地形扫描的自适应抬脚 + 记忆功能
        修改动机: 解决上楼梯抬腿高度不足 & 楼梯顶部盲区导致的摔倒。
                 引入记忆机制(avg_obstacle_height)，即使视觉看到平地，也能保持一段时间高抬腿，确保后腿安全上岸。
        """
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 1. 初始化记忆变量
        if not hasattr(self, 'avg_obstacle_height'):
            self.avg_obstacle_height = torch.zeros(self.num_envs, device=self.device)
        
        # 2. 获取地形信息
        # measured_heights 是 (num_envs, num_points)，通常定义为 base_z - terrain_z
        # 如果是正值并且很大，说明地面很低（悬崖）。如果是负值，说明地面高于预期（但这取决于 specifically implementation）。
        # 通常 legged_gym 中 measured_heights = clip(root_z - cfg.base_height - terrain_height, -1, 1)
        # 所以 terrain_height = root_z - cfg.base_height - measured_heights
        # 如果前方有台阶，terrain_height 变高， measured_heights 变小 (甚至为负)。
        # 你的代码中: heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_heights, -1, 1.)
        # 等等，通常 self.measured_heights 是从地形采样的原始高度值吗？或者已经是相对值？
        # 在 _get_heights() 中： 
        # points = quat_apply_yaw_inverse(self.base_quat, self.measured_points) + (self.root_states[:, :3]).unsqueeze(1)
        # heights = self.terrain.height_field_raw[x, y] * self.terrain.vertical_scale
        # 所以 self.measured_heights 通常是世界坐标系下的地形绝对高度。
        
        # 我们需要计算障碍物相对于脚的高度。
        # 简化算法：使用 measured_heights (世界绝对高度) -脚下地面高度
        
        # 为了兼容性，我们直接观测 measured_heights 的变化梯度
        # 或者使用先前逻辑：如果measured_heights中前方点比当前脚下高
        
        # 假设 self.measured_heights 存储的是地形高度采样值 (scalar)
        
        # 注意：你在之前的分析中提到 "heights = -self.measured_heights # 取负值，正值表示有凸起"
        # 这取决于 measured_heights 在 observations 中的处理方式。
        # 原生 legged_gym 中：
        # self.measured_heights 是 update_height_scanning 得到的 absolute z values of terrain.
        # 在 compute_observations 中：
        # heights = clip(root_z - 0.5 - measured_heights, -1, 1) * scale
        # 所以 obs 里的 height 是 "base 高出地面的量"。值越小，说明地面越高（离base越近）。
        
        # 这里为了稳健，直接使用观测到的绝对地形高度 self.measured_heights
        heights = self.measured_heights # (num_envs, num_points)
        
        # 这是一个绝对高度值。我们需要计算"相对于当前地面"的凸起高度。
        # 取脚部高度作为参考地平面
        feet_height_min = torch.min(self.feet_pos[:, :, 2], dim=1)[0] # (num_envs,)
        
        # 计算地形相对于脚的高度
        relative_terrain_height = heights - feet_height_min.unsqueeze(1)
        
        # 过滤：只关心比脚高的地方
        positive_heights = torch.clamp(relative_terrain_height, min=0)
        
        # 取扫描范围内最高的 10% 点的平均值，作为"有效障碍物高度"
        k = max(1, int(heights.shape[1] * 0.1))
        top_heights, _ = torch.topk(positive_heights, k, dim=1)
        effective_obstacle_height = torch.mean(top_heights, dim=1)
        
        # 3. 更新记忆 (关键逻辑：解决顶部盲区)
        # diff > 0 (看到台阶): alpha = 0.5 (快速反应，立刻抬腿)
        # diff < 0 (看到平地): alpha = 0.02 (极慢衰减，保持高抬腿状态约1-2秒)
        diff = effective_obstacle_height - self.avg_obstacle_height
        alpha = torch.where(diff > 0, 
                            torch.ones_like(diff) * 0.5, 
                            torch.ones_like(diff) * 0.02)
        self.avg_obstacle_height = self.avg_obstacle_height + alpha * diff
        
        # 4. 计算动态目标高度
        base_clearance = self.cfg.rewards.clearance_height_target  # 例如 -0.49
        # 障碍物高度限制，防止过高
        obstacle_level = torch.clamp(self.avg_obstacle_height, min=0, max=0.30)
        # 目标高度 = 基础 + 障碍物高度 * 1.2 (留出安全余量)
        # 注意: base_clearance 是负数 (foot_z - base_z)，obstacle_level 是正数
        # 我们希望脚抬得更高，即 (foot_z - base_z) 变大 (更接近0，或者正数)
        # 比如 base_z=0.5, foot_z=0 -> clearance=-0.5
        # 遇到0.2m台阶，希望 foot_z=0.25 -> clearance=-0.25
        # 所以 target = base_clearance + obstacle_level
        target_clearance = base_clearance + obstacle_level * 1.2
        
        # 5. 计算惩罚
        # 获取脚相对于 base 的 Z 坐标
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        foot_z_body = footpos_in_body_frame[:, :, 2]  # 脚相对于 base 的 Z 坐标
        
        # 只在摆动相计算
        is_swing = ~self.contact_filt
        
        # 误差计算：只惩罚低于目标高度的情况 (即脚太低了)
        # target_clearance 是期望的最低高度 (例如 -0.25)
        # foot_z_body 是实际高度 (例如 -0.4)
        # 我们希望 foot_z_body >= target_clearance
        # 所以 error = clamp(target - actual, min=0) => clamp(-0.25 - (-0.4), 0) = 0.15
        height_error = torch.clamp(target_clearance.unsqueeze(1) - foot_z_body, min=0.0)
        
        # 速度加权：下落阶段 (vz < 0) 如果高度不够，极其危险，给予重罚
        foot_vz = self.feet_vel[:, :, 2]
        vel_penalty = torch.where(foot_vz < 0, 
                                torch.ones_like(foot_vz) * 2.0, 
                                torch.ones_like(foot_vz) * 1.0)
        
        return torch.sum(torch.square(height_error) * vel_penalty * is_swing.float(), dim=1)


    def _reward_feet_toe_collision(self):
        """
        [核心奖励 2] 惩罚脚部受到水平撞击
        修改动机: 解决上楼步幅过大，导致脚尖踢到台阶垂直面。这是“痛觉”反馈。
        通过检测水平接触力是否远大于垂直力，判断是否发生了踢撞。
        """
        # 获取接触力 (num_envs, 2, 3)
        contact_forces = self.contact_forces[:, self.feet_indices, :]
        
        # 计算水平力和垂直力
        horizontal_force = torch.norm(contact_forces[..., :2], dim=-1)  # XY 平面
        vertical_force = torch.abs(contact_forces[..., 2])  # Z 轴
        
        # 判定逻辑：
        # 1. 必须有显著的接触力 (> 10N)
        # 2. 水平力显著大于垂直力的一定比例 (例如 > 0.5 * Fz)
        # 正常站立时 Fz 很大，Fxy 很小
        # 踢到台阶时 Fxy 很大，Fz 可能较小或中等
        total_force = torch.norm(contact_forces, dim=-1)
        is_collision = (horizontal_force > vertical_force * 0.5) & (total_force > 10.0)
        
        # 返回碰撞次数
        return torch.sum(is_collision.float(), dim=1)


    def _reward_stair_reach_penalty(self):
        """
        [核心奖励 3] 楼梯上的步幅限制
        修改动机: 辅助防止步幅过大，避免机器人为了追速度而强行迈大步。
        根据地形记忆变量判断是否在楼梯上，如果在，则限制脚离身体过远。
        """
        # 1. 判断是否在楼梯/复杂地形上
        if hasattr(self, 'avg_obstacle_height'):
            # 如果记忆中的障碍高度 > 5cm，认为在楼梯模式
            on_stair = self.avg_obstacle_height > 0.05
        else:
            # 备用方案：使用脚的实际高度
            feet_height = self.feet_pos[:, :, 2].max(dim=1)[0]
            on_stair = feet_height > 0.15
        
        # 2. 计算脚相对于 Base 的水平距离
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        # 只看 XY 平面距离
        dist = torch.norm(footpos_in_body_frame[..., :2], dim=-1)
        
        # 3. 设定阈值：平地步幅允许 0.4m+，但在楼梯上建议限制在 0.32m
        limit = 0.32
        
        # 4. 计算超出部分的惩罚
        over_reach = torch.clamp(dist - limit, min=0)
        
        # 只有在楼梯上才惩罚步幅过大
        return torch.sum(over_reach * on_stair.unsqueeze(1).float(), dim=1)


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
