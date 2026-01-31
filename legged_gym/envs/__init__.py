from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

from legged_gym.envs.g1_loco.g1_16dof_loco_config import G1_16Dof_Loco_Cfg, G1_16Dof_Loco_CfgPPO
from legged_gym.envs.g1_loco.g1_16dof_loco_env import G1_16Dof_Loco_Robot 
task_registry.register( "g1_16dof_loco", G1_16Dof_Loco_Robot, G1_16Dof_Loco_Cfg(), G1_16Dof_Loco_CfgPPO())

from legged_gym.envs.g1_loco.g1_16dof_moe_residual_config import G1_16Dof_MoE_Residual_Cfg, G1_16Dof_MoE_Residual_CfgPPO
from legged_gym.envs.g1_loco.g1_16dof_moe_residual_env import G1_16Dof_MoE_Resi_Robot
task_registry.register( "g1_16dof_resi_moe", G1_16Dof_MoE_Resi_Robot, G1_16Dof_MoE_Residual_Cfg(), G1_16Dof_MoE_Residual_CfgPPO())


from legged_gym.envs.ym1_loco.ym1_16dof_moe_residual_config import ym1_16Dof_MoE_Residual_Cfg, ym1_16Dof_MoE_Residual_CfgPPO
from legged_gym.envs.ym1_loco.ym1_16dof_moe_residual_env import ym1_16Dof_MoE_Resi_Robot
task_registry.register( "ym1_16dof_resi_moe", ym1_16Dof_MoE_Resi_Robot, ym1_16Dof_MoE_Residual_Cfg(), ym1_16Dof_MoE_Residual_CfgPPO())

from legged_gym.envs.yme_loco.yme_16dof_loco_config import yme_16Dof_Loco_Cfg, yme_16Dof_Loco_CfgPPO
from legged_gym.envs.yme_loco.yme_16dof_loco_env import yme_16Dof_Loco_Robot 
task_registry.register( "yme_16dof_loco", yme_16Dof_Loco_Robot, yme_16Dof_Loco_Cfg(), yme_16Dof_Loco_CfgPPO())

from legged_gym.envs.ym1_loco.ym1_16dof_loco_config import ym1_16Dof_Loco_Cfg, ym1_16Dof_Loco_CfgPPO
from legged_gym.envs.ym1_loco.ym1_16dof_loco_env import ym1_16Dof_Loco_Robot 
task_registry.register( "ym1_16dof_loco", ym1_16Dof_Loco_Robot, ym1_16Dof_Loco_Cfg(), ym1_16Dof_Loco_CfgPPO())