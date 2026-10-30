"""Register the two Go2 AME entry points; stage selection lives in stage_cfg."""

import gymnasium as gym
from ame_locomotion.tasks.manager_based.ame_locomotion import agents

##
# Register Gym environments.
##

gym.register(
    id="AME-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2AMEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.go2_ame_rsl_rl_ppo_cfg:Go2AMEPPORunnerCfg",
    },
)

gym.register(
    id="AME-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2AMEEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.go2_ame_rsl_rl_ppo_cfg:Go2AMEPPORunnerCfg",
    },
)
