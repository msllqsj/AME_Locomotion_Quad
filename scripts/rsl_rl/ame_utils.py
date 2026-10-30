"""AME configuration checks and checkpoint initialization shared by entry points."""

import math

import torch


def validate_ame_config(env_cfg, agent_cfg):
    if agent_cfg.policy.class_name != "ActorCriticEncoder":
        return
    pattern = env_cfg.scene.height_scanner.pattern_cfg
    if pattern.resolution <= 0:
        raise ValueError("AME scanner resolution must be positive.")
    # Match GridPattern's sampling rule, including its endpoint tolerance.
    expected = tuple(
        torch.arange(
            start=-size / 2,
            end=size / 2 + 1.0e-9,
            step=pattern.resolution,
            device="cpu",
        ).numel()
        for size in pattern.size
    ) + (3,)
    actual = tuple(agent_cfg.policy.map_scan_dim)
    if actual != expected or pattern.ordering != "xy":
        raise ValueError(
            f"AME map mismatch: policy={actual}, scanner={expected}, ordering={pattern.ordering}; use xy ordering."
        )
    if (
        actual[2] != 3
        or agent_cfg.policy.num_heads <= 0
        or agent_cfg.policy.mha_dim <= 0
        or agent_cfg.policy.mha_dim % agent_cfg.policy.num_heads
    ):
        raise ValueError("AME needs xyz map coordinates and mha_dim divisible by num_heads.")
    if type(env_cfg).__name__.startswith("Go2AME"):
        from ame_locomotion.tasks.manager_based.ame_locomotion.go2 import stage_cfg

        if env_cfg.finetune != stage_cfg.FINETUNE:
            raise ValueError(
                "Select the Go2 stage in go2/stage_cfg.py (FINETUNE), then restart; do not override env.finetune."
            )


def validate_go2_observations(env, agent_cfg):
    if not type(env.unwrapped.cfg).__name__.startswith("Go2AME"):
        return
    obs = env.get_observations()
    map_size = math.prod(agent_cfg.policy.map_scan_dim)
    expected = {"policy": 45 + map_size, "critic": 48 + map_size}
    for key, width in expected.items():
        if obs[key].shape[-1] != width:
            raise ValueError(f"Go2 {key}: expected {width} observations, got {obs[key].shape[-1]}.")
    if env.num_actions != 12:
        raise ValueError(f"Go2 needs 12 actions, got {env.num_actions}.")


def load_training_checkpoint(runner, path, *, warm_start=False):
    """Warm start weights only; resume retains the optimizer and iteration."""
    try:
        infos = runner.load(path, load_optimizer=not warm_start, map_location=runner.device)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint {path} does not match the selected robot/AME network. "
            "Check robot, map_scan_dim and attach_global."
        ) from exc
    if warm_start:
        runner.current_learning_iteration = 0
    return infos
