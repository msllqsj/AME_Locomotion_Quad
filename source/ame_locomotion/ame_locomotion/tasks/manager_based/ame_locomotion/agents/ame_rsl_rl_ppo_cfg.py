# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AMEActorCriticCfg(RslRlPpoActorCriticCfg):
    """AME encoder options exposed to Isaac Lab and Hydra."""

    attach_global: bool = False
    map_scan_dim: tuple[int, int, int] = (33, 21, 3)
    mha_dim: int = 64
    num_heads: int = 16
    cnn_downsample: bool = True


@configclass
class AMEDiagnosticsCfg:
    """Read-only module attribution, attention and terrain logging."""

    enabled: bool = False
    scalar_interval: int = 10
    attribution_interval: int = 50
    image_interval: int = 200
    max_images: int = 4


@configclass
class AMEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Common AME PPO settings shared by robot-specific configurations."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "ame"
    diagnostics: AMEDiagnosticsCfg = AMEDiagnosticsCfg()
    policy = AMEActorCriticCfg(
        class_name="ActorCriticEncoder",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1AMEPPORunnerCfg(AMEPPORunnerCfg):
    """AME PPO configuration for the G1 robot."""

    experiment_name = "g1_ame"
