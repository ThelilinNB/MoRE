"""
YM1 16DOF Loco Sim2Sim Deployment Script for MuJoCo
对应 ym1_16dof_loco_config.py 的策略 (num_obs=57, 无 gait_cmd)

Usage: python deploy/deploy_mujoco/deploy_mujoco_ym1.py ym1_16dof.yaml

键盘控制:
  W/S: 前进/后退 (每次 ±0.1 m/s)
  A/D: 左移/右移 (每次 ±0.1 m/s)
  Q/E: 左转/右转 (每次 ±0.1 rad/s)
  R: 重置速度为0
"""
import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import cv2
import torch.nn.functional as F
import threading


def get_gravity_orientation(quaternion):
    """Convert quaternion to projected gravity vector in body frame"""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands using PD control"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class KeyboardController:
    """键盘控制器，使用 pynput 监听按键"""
    def __init__(self, cmd, step_size=0.1):
        self.cmd = cmd  # [vx, vy, vyaw]
        self.step_size = step_size
        self.running = True
        
        # 速度限制
        self.vx_range = [-1.0, 1.5]
        self.vy_range = [-1.0, 1.0]
        self.vyaw_range = [-1.0, 1.0]
        
        try:
            from pynput import keyboard
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()
            self.use_pynput = True
            print("键盘控制已启用 (pynput)")
        except ImportError:
            self.use_pynput = False
            print("警告: pynput 未安装，使用 OpenCV 窗口按键控制")
            print("请确保 depth image 窗口处于焦点状态")
    
    def on_press(self, key):
        try:
            k = key.char.lower()
            self.process_key(k)
        except AttributeError:
            pass
    
    def process_key(self, k):
        """处理按键"""
        if k == 'w':
            self.cmd[0] = np.clip(self.cmd[0] + self.step_size, *self.vx_range)
        elif k == 's':
            self.cmd[0] = np.clip(self.cmd[0] - self.step_size, *self.vx_range)
        elif k == 'a':
            self.cmd[1] = np.clip(self.cmd[1] + self.step_size, *self.vy_range)
        elif k == 'd':
            self.cmd[1] = np.clip(self.cmd[1] - self.step_size, *self.vy_range)
        elif k == 'q':
            self.cmd[2] = np.clip(self.cmd[2] + self.step_size, *self.vyaw_range)
        elif k == 'e':
            self.cmd[2] = np.clip(self.cmd[2] - self.step_size, *self.vyaw_range)
        elif k == 'r':
            self.cmd[:] = 0.0
        
        print(f"[CMD] vx={self.cmd[0]:.2f}, vy={self.cmd[1]:.2f}, vyaw={self.cmd[2]:.2f}")
    
    def check_opencv_key(self):
        """检查 OpenCV 窗口的按键（备用方案）"""
        if not self.use_pynput:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.process_key(chr(key))
    
    def stop(self):
        self.running = False
        if self.use_pynput:
            self.listener.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YM1 16DOF Loco Sim2Sim Deployment")
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)
        init_qpos = np.array(config["init_qpos"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]  # 57 for loco (no gait_cmd)
        obs_history_len = config["obs_history_len"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        # depth camera configs
        depth_far_clip = config["depth_far_clip"]
        depth_near_clip = config["depth_near_clip"]
        depth_buffer_len = config["depth_buffer_len"]
        depth_size = config["depth_size"]
        cam_update_interval = config["cam_update_interval"]
        crop_image = config["crop_image"]
        crop_size = config["crop_size"]
        gaussian_filter = config["gaussian_filter"]
        gaussian_filter_kernel = config["gaussian_filter_kernel"]
        gaussian_filter_sigma = config["gaussian_filter_sigma"]
        gaussian_noise = config["gaussian_noise"]
        gaussian_noise_std = config["gaussian_noise_std"]
        depth_dis_noise = config["depth_dis_noise"]

    def process_depth_image(depth_image):
        """Process raw depth image with noise and normalization"""
        depth_image = depth_image + depth_dis_noise * 2 * (np.random.rand(1) - 0.5)
        if gaussian_noise:
            depth_image = depth_image + gaussian_noise_std * np.random.randn(*depth_image.shape)
        depth_image = np.clip(depth_image, depth_near_clip, depth_far_clip)
        depth_image = (depth_image - depth_near_clip) / (depth_far_clip - depth_near_clip) - 0.5
        return depth_image
    
    def crop_resize_depth(depth_image):
        """Crop and resize depth image"""
        clip_left, clip_top, clip_right, clip_bottom = crop_size
        depth_image = F.interpolate(
            depth_image[clip_top:-clip_bottom, clip_left:depth_size[1]-clip_right].unsqueeze(0).unsqueeze(0), 
            size=(64, 64), mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)
        return depth_image
    
    def adaptive_gaussian_filter(depth_image, kernel_size=gaussian_filter_kernel, sigma=gaussian_filter_sigma):
        """Apply Gaussian filter to depth image"""
        imgs = cv2.GaussianBlur(depth_image.numpy(), (kernel_size, kernel_size), sigma)
        return torch.from_numpy(imgs).to(depth_image.device)
    
    def update_depth_cam(depth_image_buffer):
        """Update depth camera and return processed depth image"""
        depth_renderer.update_scene(d, camera=depth_cam_id)
        depth_image = depth_renderer.render()
        depth_image = np.rot90(depth_image, k=1)
        depth_image = process_depth_image(depth_image)
        depth_image = torch.tensor(depth_image, dtype=torch.float32)
        
        if crop_image:
            depth_image = crop_resize_depth(depth_image)
        if gaussian_filter:
            depth_image = adaptive_gaussian_filter(depth_image, kernel_size=gaussian_filter_kernel, sigma=gaussian_filter_sigma)

        cv2.namedWindow('depth image', cv2.WINDOW_NORMAL)
        cv2.imshow("depth image", depth_image_buffer[0, -1].detach().numpy() + 0.5)
        cv2.waitKey(1)
        return depth_image

    # Initialize context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    # trajectory_history 不包含 gait_cmd，所以维度就是 num_obs (57)
    trajectory_history = torch.zeros(size=(1, obs_history_len, num_obs), dtype=torch.float32)
    depth_image_buffer = torch.zeros(1, depth_buffer_len, 64, 64, dtype=torch.float32)
    counter = 0
    cam_update_counter = 0

    # Load robot model
    print(f"Loading MuJoCo model from: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Set initial dof positions
    d.qpos[3:] = init_qpos

    # Load policy
    print(f"Loading policy from: {policy_path}")
    policy = torch.jit.load(policy_path)
    print("Policy loaded successfully!")

    # Setup depth renderer
    depth_renderer = mujoco.Renderer(m, width=depth_size[1], height=depth_size[0])
    depth_renderer.enable_depth_rendering()
    depth_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")

    print(f"\nStarting simulation...")
    print(f"  Observation dim: {num_obs} (no gait_cmd)")
    print(f"  Command: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, vyaw={cmd[2]:.2f}")
    print(f"  Duration: {simulation_duration}s")
    print(f"  Control freq: {1.0/(simulation_dt*control_decimation):.1f}Hz")
    print("\n键盘控制:")
    print("  W/S: 前进/后退 (±0.1 m/s)")
    print("  A/D: 左移/右移 (±0.1 m/s)")
    print("  Q/E: 左转/右转 (±0.1 rad/s)")
    print("  R: 重置速度为0")
    print("\nPress Ctrl+C or close the viewer to stop.\n")

    # 初始化键盘控制器
    keyboard_ctrl = KeyboardController(cmd, step_size=0.1)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # Apply PD control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            
            # Step physics
            mujoco.mj_step(m, d)
            counter += 1
            
            if (counter > 0) and (counter % control_decimation) == 0:
                # Get robot state
                qj = d.qpos[7:]  # joint positions
                dqj = d.qvel[6:]  # joint velocities
                quat = d.qpos[3:7]  # base quaternion
                omega = d.qvel[3:6]  # base angular velocity

                # Scale observations
                qj_scaled = (qj - default_angles) * dof_pos_scale
                dqj_scaled = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega_scaled = omega * ang_vel_scale

                # Build observation vector (57 dims for loco, NO gait_cmd)
                # obs = [cmd(3), ang_vel(3), gravity(3), dof_pos(16), dof_vel(16), actions(16)]
                obs[:3] = cmd * cmd_scale  
                obs[3:6] = omega_scaled
                obs[6:9] = gravity_orientation
                obs[9:9+num_actions] = qj_scaled
                obs[9+num_actions:9+2*num_actions] = dqj_scaled
                obs[9+2*num_actions:9+3*num_actions] = action

                # Update depth camera
                if (cam_update_counter) % cam_update_interval == 0:
                    depth_imgs = update_depth_cam(depth_image_buffer)
                    if (depth_image_buffer == 0).all():  # first image
                        depth_image_buffer = torch.stack([depth_imgs] * depth_buffer_len, dim=0).unsqueeze(0)
                    else:
                        depth_image_buffer = torch.cat([depth_image_buffer[:, 1:, ...], depth_imgs.unsqueeze(0).unsqueeze(1)], dim=1)
                cam_update_counter += 1
                
                # 检查 OpenCV 按键（备用方案）
                keyboard_ctrl.check_opencv_key()

                # Update trajectory history (全部 57 维 obs)
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                trajectory_history = torch.cat([trajectory_history[:, 1:], obs_tensor.unsqueeze(1)], dim=1)

                # Policy inference
                # 对于 loco 策略: policy(obs, trajectory_history, depth_buffer)
                # depth_buffer 使用最后两帧 (buffer_len=3, 取 [1:3] 即最后2帧)
                action = policy(obs_tensor, trajectory_history, depth_image_buffer[:, 1:3, ...]).detach().numpy().squeeze()
                
                target_dof_pos = action * action_scale + default_angles

            # Sync viewer
            viewer.sync()

            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    keyboard_ctrl.stop()
    print("\nSimulation finished!")
