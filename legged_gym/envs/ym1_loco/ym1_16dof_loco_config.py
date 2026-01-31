from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class ym1_16Dof_Loco_Cfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.78] # x,y,z [m] - 提高5cm

        random_default_pos = False
        default_joint_angles = { # = target angles [rad] when action = 0.0

            'left_leg_pitch_joint' : -0.1,         
            'left_leg_roll_joint' : 0,               
            'left_leg_yaw_joint' : 0. ,   
            'left_knee_pitch_joint' : 0.35,       
            'left_ankle_pitch_joint' : -0.25,     
            'left_ankle_roll_joint' : 0,  

            'right_leg_pitch_joint' : -0.1,                                       
            'right_leg_roll_joint' : 0, 
            'right_leg_yaw_joint' : 0., 
            'right_knee_pitch_joint' : 0.35,                                             
            'right_ankle_pitch_joint': -0.25,                              
            'right_ankle_roll_joint' : 0,

            'left_shoulder_pitch_joint': 0.46,
            'left_elbow_pitch_joint': 0.33,

            'right_shoulder_pitch_joint': 0.46,
            'right_elbow_pitch_joint': 0.33,
        }

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh  # 先用平面测试稳定性
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast

        edge_width_thresh = 0
        horizontal_scale = 0.1 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True
        min_difficulty_level = 0.5
        test_mode = False
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 14
        terrain_width = 4
        num_rows = 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 30 # number of terrain cols (types)
        
        terrain_dict = {"roughness": 1., 
                        "slope": 1.,
                        "pit": 1,
                        "gap": 1,
                        "stair": 1,}
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 0.75# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False
        num_goals = 8
        difficulty_level = 1
        test_mode = False
        
    class env(LeggedRobotCfg.env):
        num_observations = 57
        num_actions = 16
        amp_motion_files = './resources/g1_amp_data/lafan_walk+run_50FPS' 
        num_amp_obs = num_actions
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        feet_info = True
        priv_info = True
        foot_force_info = True
        scan_dot = True
        num_privileged_obs = 60
        if feet_info:
            num_privileged_obs += 12
        if priv_info:
            num_privileged_obs += 38
        if foot_force_info:
            num_privileged_obs += 6
        if scan_dot:
            num_privileged_obs += 187

    class depth(LeggedRobotCfg.depth):
        use_camera = False
        warp_camera = True
        warp_device = 'cuda:0'

        position = [0.0576235, 0.01753, 0.42987]
        original = (64, 64)
        resized = (64, 64)
        near_clip = 0
        far_clip = 2
        update_interval = 5
        buffer_len = 2 + 1  # 匹配模型训练时的设置
        fovy_range = [79.3, 79.3]

        # ----- camera randomization -----
        # 从头开始训练就打开相机噪声
        y_angle = [42, 48]  # 相机俯仰角随机范围
        z_angle = [-1, 1]   # 相机偏航角随机范围
        x_angle = [-1, 1]   # 相机横滚角随机范围

        rand_position = True  # 启用位置随机
        x_pos_range = [-0.01, 0.01]
        y_pos_range = [-0.01, 0.01]
        z_pos_range = [-0.01, 0.01]

        dis_noise = 0.03  # 深度噪声

        gaussian_noise = True  # 启用高斯噪声
        gaussian_noise_std = 0.05

        gaussian_filter = True  # 启用高斯滤波
        gaussian_filter_kernel = [1, 3, 5]
        gaussian_filter_sigma = 1.2

        random_cam_delay = False
        # --------------------------------

        # ----- augmentation for deployment -----
        # body mask - 在 2000 iterations 后自动启用
        add_body_mask = False  # 初始关闭
        body_mask_path = '/home/ps/lpl/MoRE/body_mask_data/body_masks_ym1_16dof_loco_20260128_193552.npz'
        body_mask_start_iter = 5000  # 2k iterations 后启用
        # crop depth
        crop_depth = True  # 深度裁剪始终启用
        crop_pixels = [10, 20, 10, 5] # left top right bottom
        # ---------------------------------------


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 2]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True  
        push_interval_s = 8; push_interval_min_s = 8
        min_push_vel_xy = 1; max_push_vel_xy = 1
        stair_no_push = False

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        
        randomize_com_pos = True
        com_x_pos_range = [-0.03, 0.03]
        com_y_pos_range = [-0.03, 0.03]
        com_z_pos_range = [-0.03, 0.03]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_restitution = True
        restitution_range = [0.0, 1.0]

        randomize_actuation_offset = True
        actuation_offset_range = [-0.1, 0.1]

        delay_update_global_steps = 24 * 5000
        action_delay = True
        action_curr_step = [0, 1, 2]
        action_buf_len = 5

    class noise:
        add_noise = False
        noise_level = 0.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {
            'leg_yaw_joint': 100.,
            'leg_roll_joint': 100.,
            'leg_pitch_joint': 100.,
            'knee_pitch_joint': 150.,
            'ankle_pitch_joint': 40.,
            'ankle_roll_joint': 40.,    
            'shoulder_pitch_joint': 60,
            'elbow_pitch_joint': 40 
        }  # [N*m/rad]
        
        damping = {
            'leg_yaw_joint': 3,
            'leg_roll_joint': 3,
            'leg_pitch_joint': 3,
            'knee_pitch_joint': 5,
            'ankle_pitch_joint': 2.5,
            'ankle_roll_joint': 2.5,
            'shoulder_pitch_joint': 2,
            'elbow_pitch_joint': 1
        } # [N*m/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '/home/ps/lpl/MoRE/resources/robots/ym1/ym1_16dof.urdf'

        name = "ym1"
        foot_name = "ankle_roll"
        knee_name = "knee"
        penalize_contacts_on = ["leg", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        feet_indicator_offset = [ [-0.071, 0, -0.048], [-0.042, 0, -0.048], [-0.013, 0, -0.048], [0.016, 0, -0.048], [0.046, 0, -0.048], [0.075, 0, -0.048], [0.104, 0, -0.048], [0.133, 0, -0.048]]
        feet_collision_indicator_offset = [[-0.090, 0, -0.048], [0.16, 0, -0.048]]


    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        feet_min_lateral_distance_target = 0.30
        feet_longitudinal_distance_target = 0.02  # 前后脚目标间距，越小越平行
        clearance_height_target = -0.49
        base_height_target = 0.63  # 目标 base 高度
        class scales:
            tracking_lin_vel = 3.0  # [修改] 降低，防止为了追速度而忽略地形
            tracking_ang_vel = 2.0  # [修改] 降低

            dof_acc = -2.5e-6
            dof_vel = -1e-3
            action_rate = -0.03
            action_smoothness = -0.15 # -0.15

            ang_vel_xy = -0.1  # 增加，惩罚 roll/pitch 旋转
            orientation = -4.0 #-5.0  # 增大躯干姿态惩罚，保持直立
            joint_power = -2.5e-5
            feet_clearance = -0.25 
            feet_stumble = -1.0
            torques = -1e-5
            arm_joint_deviation = -0.5
            hip_joint_deviation = -0.5

            dof_pos_limits = -2.0
            dof_vel_limits = -1.0
            torque_limits = -1.0
            no_fly = 0.2
            feet_lateral_distance = -2.0 # -5.0
            feet_longitudinal_distance = -5.0  # 惩罚前后脚
            feet_slippage = -2.0 # -2.0  # 增大滑动惩罚，从 -0.25 改为 -1.0
            feet_contact_force = -2.5e-4
            feet_force_rate = -2.5e-4
            feet_contact_momentum = -2.5e-4
            collision = -15.
            feet_air_time = 1.5  # 增大抬脚奖励，从 1.0 改为 2.0
            single_foot_contact = 1.0  # 增大单脚着地奖励，强制交替步态
            
            step_length = 1.0  # 奖励较大步长

            # === 新增 [2026-01-23]: 楼梯攀爬组合拳权重 ===
            # height_scan_gradient_clearance = -1.0   # 自适应抬脚高度 (解决盲区摔倒)
            # feet_toe_collision = -2.0               # 脚尖碰撞惩罚 (解决踢台阶，痛感反馈)
            # stair_reach_penalty = -1.0              # 楼梯步幅限制 (辅助约束)

            # foot_swing_symmetry = 2.0  # 新增：惩罚不对称摆动，防止总是同一只脚先迈
            
            # both_feet_same_height = -1.0  # 惩罚双脚在同一高度（并步行为）
            # step_forward_alternating = 1.0  # 奖励交替向前迈步，每步必须超越上一步

            stuck = -1
            cheat = -2
            feet_edge = -0.5
            # y_offset_pen = -0.5  # 增加！强化直线行走（从 -0.5 改为 -3.0）
            base_height = -10.0
            
            # 零速度站立奖励
            stand_still = -2.0  # 惩罚零速度时的身体和脚部运动
            feet_still_when_stand = 1.0  # 奖励零速度时双脚着地静止

            default_joint_pos = 0.4

            # feet_collision_pen = -1

        feet_contact_force_range = [200. , 600.]
        # only_positive_rewards = False

class ym1_16Dof_Loco_CfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunnerMulti'

    class policy:
        init_noise_std = 0.5
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        his_latent_dim = 64

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        use_amp = False
        amp_loader_type = 'lafan_16dof_multi'
        amp_loader_class_name = "G1_AMPLoader"
        
        entropy_coef = 0.01
        policy_learning_rate = 5e-4

    class runner( LeggedRobotCfgPPO.runner ):
        num_amp_frames = 5
        policy_class_name = "ActorCriticDepth"
        algorithm_class_name = "AMPPPOMulti"

        use_lerp = False
        max_iterations = 120000
        run_name = ''
        experiment_name = 'ym1_16dof_loco'
        save_interval = 400
        amp_reward_coef = 5
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.5
        amp_discr_hidden_dims = [1024, 512]

  
