"""AME PPO configuration for the Go2 robot."""

from isaaclab.utils import configclass

from ..go2 import stage_cfg
from .ame_rsl_rl_ppo_cfg import AMEPPORunnerCfg


@configclass
class Go2AMEPPORunnerCfg(AMEPPORunnerCfg):
    """Use the same AME PPO settings in both Go2 training stages."""

    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    def __post_init__(self):
        self.experiment_name = stage_cfg.experiment_name()
        self.policy.map_scan_dim = stage_cfg.MAP_SCAN_DIM
        self.diagnostics.enabled = True
