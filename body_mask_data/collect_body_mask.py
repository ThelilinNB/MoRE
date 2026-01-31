"""
Script to collect body mask data for training residual policy.
This script runs the trained policy and collects depth images with body self-occlusion.
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import numpy as np
from datetime import datetime
from scipy import ndimage


def collect_body_mask(args):
    # ===== Configuration =====
    NUM_EXPERT_GAITS = 3  # Number of expert policies (walk/run, high-knees, squat)
    RANDOM_ASSIGN_GAIT = True  # If True, randomly assign gait labels; if False, use default 0
    
    # Collection mode
    STATIC_COLLECTION = False  # Set to True to collect standing still data
    # =========================
    
    # Get environment and training configs
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Override parameters for data collection
    env_cfg.env.num_envs = 10  # 增加到10，确保能被num_cols整除
    env_cfg.env.episode_length_s = 40  # 增加 episode 长度来采集更多数据（原来20秒）
    
    # 地形配置 - 确保 num_rows * num_cols >= num_envs
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10  # num_rows * num_cols = 50 >= 10 envs
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5  # 最大初始地形等级
    
    # 确保 difficulty_level 是有效的
    if hasattr(env_cfg.terrain, 'difficulty_level'):
        env_cfg.terrain.difficulty_level = 1  # 使用整数而不是浮点数
    
    # Disable randomization for consistent data collection
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.push_robots = True
    
    # CRITICAL: Enable both cameras
    # use_camera captures robot body (with self-occlusion)
    # warp_camera can be used for comparison
    env_cfg.depth.use_camera = True
    env_cfg.depth.warp_camera = True
    env_cfg.depth.y_angle = [55, 55]
    env_cfg.depth.x_angle = [0, 0]
    env_cfg.depth.z_angle = [0, 0]
    
    # Set command ranges (you can modify based on gait types)
    if STATIC_COLLECTION:
        # Standing still - zero velocity commands
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        env_cfg.commands.heading_command = False
        
        # Optional: Add some joint randomization for varied standing poses
        env_cfg.noise.add_noise = True  # 保持噪声获得轻微姿态变化
        
        print("📍 STATIC COLLECTION MODE: Robot will stand still")
    else:
        # Normal locomotion
        env_cfg.commands.ranges.lin_vel_x = [-0.5, 1.0]
        env_cfg.commands.ranges.lin_vel_y = [-0.5, 0.5]
        env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
        env_cfg.commands.heading_command = True
        print("🚶 DYNAMIC COLLECTION MODE: Robot will move")
    
    env_cfg.commands.resampling_time = 10
    
    # Terrain types - 确保所有值都是浮点数
    env_cfg.terrain.terrain_dict = {
        "roughness": 0., 
        "slope": 0.,
        "pit": 1.,
        "gap": 1.,
        "stair": 1.,
    }
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    
    # Create environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Process trajectory history
    num_gait = env.cfg.env.num_gait if hasattr(env.cfg.env, 'num_gait') else 0
    trajectory_history = torch.zeros(size=(env.num_envs, env.obs_history_len, env.num_obs-num_gait), device=env.device)
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs[:, num_gait:].unsqueeze(1)), dim=1)
    
    # Data collection containers
    body_masks_list = []
    gait_list = []
    
    print("=" * 80)
    print("Starting body mask data collection...")
    print(f"Number of environments: {env.num_envs}")
    print(f"Episode length: {env.cfg.env.episode_length_s}s")
    print("=" * 80)
    
    infos = {}
    total_samples = 0
    sample_interval = 2  # 减少采样间隔（原来5，现在3）- 更密集地采集数据
    
    # Get initial depth
    if env.cfg.depth.use_camera:
        infos["depth"] = env.depth_buffer.clone().to(env.device)
    else:
        raise ValueError("use_camera must be True for body mask collection!")
    
    # Main collection loop
    num_steps = int(env.max_episode_length)
    for i in range(num_steps):
        # Get depth image
        if infos["depth"] is not None:
            depth_image = infos['depth']
        
        if env.cfg.depth.use_camera:
            obs = (obs, depth_image)
        
        # Run policy
        if isinstance(obs, tuple):
            actions = policy(obs[0].detach(), trajectory_history.detach(), obs[1][:, :2, ...].detach())
        else:
            actions = policy(obs.detach(), trajectory_history)
        
        # Step environment
        obs, _, _, dones, infos, *_ = env.step(actions.detach())
        
        # Process trajectory history
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        trajectory_history = torch.concat((trajectory_history[:, 1:], obs[:, num_gait:].unsqueeze(1)), dim=1)
        
        # Collect data at specified interval
        if i % sample_interval == 0 and i > 0:
            # ===== KEY INSIGHT =====
            # depth_buffer (IsaacGym camera): terrain + robot body
            # warp_depth_buffer (Warp camera): terrain ONLY (no robot body)
            # body_mask = where robot body occludes the terrain
            
            # Get depth from both cameras
            isaac_depth = env.depth_buffer[:, 1, :, :].cpu().numpy()      # (num_envs, H, W) - with body
            warp_depth = env.warp_depth_buffer[:, 1, :, :].cpu().numpy()  # (num_envs, H, W) - without body
            
            # Debug: print depth range on first collection
            if i == sample_interval:
                print(f"\n[DEBUG] Depth buffer statistics:")
                print(f"  IsaacGym (with body)  - Min: {isaac_depth.min():.4f}, Max: {isaac_depth.max():.4f}, Mean: {isaac_depth.mean():.4f}")
                print(f"  Warp (terrain only)   - Min: {warp_depth.min():.4f}, Max: {warp_depth.max():.4f}, Mean: {warp_depth.mean():.4f}")
                print(f"  Config - near_clip: {env.cfg.depth.near_clip}, far_clip: {env.cfg.depth.far_clip}")
                print()
            
            # Get gait information
            # Since we're collecting data for a single locomotion type but need to 
            # support multiple experts, we randomly assign gait labels
            if RANDOM_ASSIGN_GAIT:
                # Randomly assign each sample to one of the expert gaits
                current_gait = np.random.randint(0, NUM_EXPERT_GAITS, size=env.num_envs, dtype=np.int32)
            else:
                # Use environment gait if available, otherwise default to 0
                if hasattr(env, 'gait'):
                    current_gait = env.gait.cpu().numpy()
                else:
                    current_gait = np.zeros(env.num_envs, dtype=np.int32)
            
            if i == sample_interval:
                print(f"[DEBUG] Gait assignment mode: {'Random' if RANDOM_ASSIGN_GAIT else 'Sequential/Default'}")
                print(f"  Number of expert gaits: {NUM_EXPERT_GAITS}")
                print()
            
            # ===== Create body mask according to README =====
            # Where robot body is present: use isaac_depth (normalized to [-0.5, 0.5])
            # Where robot body is NOT present: fill with 1.0
            
            # Strategy: Detect body occlusion by comparing IsaacGym and Warp depths
            # Robot body should cause a SIGNIFICANT depth difference
            
            # 1. Calculate depth difference
            depth_diff = warp_depth - isaac_depth  # Positive if isaac is closer
            
            # 2. Identify potential body regions
            # Use a threshold that filters out small terrain variations
            min_depth_diff = 0.1  # Minimum 0.1 normalized depth units (~0.2m at 2m range)
            body_mask_candidate = (depth_diff > min_depth_diff)
            
            # 3. Additional filtering: remove small isolated regions (noise)
            # Only keep reasonably sized connected components
            
            # Process each environment's mask separately
            body_mask = np.zeros_like(body_mask_candidate)
            for env_idx in range(body_mask_candidate.shape[0]):
                # Label connected components
                labeled, num_features = ndimage.label(body_mask_candidate[env_idx])
                
                # Keep only regions with sufficient size
                min_region_size = 20  # pixels (adjust based on your needs)
                for region_id in range(1, num_features + 1):
                    region_mask = (labeled == region_id)
                    if region_mask.sum() >= min_region_size:
                        body_mask[env_idx] |= region_mask
            
            # 4. Optional: morphological operations to clean up mask
            # Uncomment if you want smoother masks
            # from scipy.ndimage import binary_closing, binary_opening
            # for env_idx in range(body_mask.shape[0]):
            #     body_mask[env_idx] = binary_closing(body_mask[env_idx], iterations=1)
            #     body_mask[env_idx] = binary_opening(body_mask[env_idx], iterations=1)
            
            # Create the final body mask array
            # Body regions: keep isaac_depth values (already in [-0.5, 0.5])
            # Non-body regions: fill with 1.0
            normalized_depth = isaac_depth.copy()
            normalized_depth[~body_mask] = 1.0
            
            # Debug: print mask statistics on first collection
            if i == sample_interval:
                print(f"[DEBUG] Body mask statistics:")
                print(f"  Body pixels: {body_mask.sum()} ({body_mask.mean()*100:.1f}%)")
                print(f"  Non-body pixels: {(~body_mask).sum()} ({(~body_mask).mean()*100:.1f}%)")
                print(f"  Final depth range: [{normalized_depth.min():.4f}, {normalized_depth.max():.4f}]")
                print(f"  Values = 1.0: {(normalized_depth == 1.0).sum()} ({(normalized_depth == 1.0).mean()*100:.1f}%)")
                print()
            
            # Add to collection
            body_masks_list.append(normalized_depth)
            gait_list.append(current_gait)
            
            total_samples += env.num_envs
            
            if (i // sample_interval) % 10 == 0:
                print(f"Step {i}/{num_steps} - Collected {total_samples} samples")
    
    # Convert lists to arrays
    body_masks_array = np.concatenate(body_masks_list, axis=0)  # (total_samples, H, W)
    gait_array = np.concatenate(gait_list, axis=0)  # (total_samples,)
    
    print("=" * 80)
    print(f"Collection completed!")
    print(f"Total samples collected: {len(body_masks_array)}")
    print(f"Body masks shape: {body_masks_array.shape}")
    print(f"Gait array shape: {gait_array.shape}")
    
    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'body_mask_data')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f'body_masks_{args.task}_{timestamp}.npz')
    
    np.savez(save_path,
             body_masks=body_masks_array,
             gait=gait_array)
    
    print(f"Data saved to: {save_path}")
    print("=" * 80)
    
    # Print statistics
    print("\nData Statistics:")
    print(f"  Depth range: [{body_masks_array.min():.3f}, {body_masks_array.max():.3f}]")
    print(f"  Depth mean: {body_masks_array.mean():.3f}")
    print(f"  Depth std: {body_masks_array.std():.3f}")
    print(f"  Unique gait types: {np.unique(gait_array)}")
    
    return save_path


if __name__ == '__main__':
    args = get_args()
    save_path = collect_body_mask(args)
    
    # Optional: visualize some samples
    print("\nTo visualize collected data, you can use:")
    print(f"  python legged_gym/scripts/visualize_body_mask.py --data_path {save_path}")
