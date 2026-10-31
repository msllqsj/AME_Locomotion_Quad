"""Per-reward terrain, curriculum-level and PPO-iteration gates for Go2."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, is_dataclass

import torch
from isaaclab.utils import configclass

CONTINUOUS_TERRAINS = (
    "flat",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "boxes",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
)
GAIT_TERMS = ("feet_gait", "joint_mirror", "joint_deviation_hip_l1")
AIR_TERMS = ("feet_air_time", "feet_air_time_variance_penalty", "feet_air_time_excess_penalty")
STYLE_TERMS = (
    "base_height_l2",
    "feet_slide",
    "feet_slippage_3d",
    "feet_contact_forces",
    "feet_height_body",
    "standing_joint_positions",
    "standing_joint_velocities",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "flat_orientation_l2",
)
GATED_TERMS = GAIT_TERMS + AIR_TERMS + STYLE_TERMS


@configclass
class RewardGateRuleCfg:
    scale: float = 1.0
    terrain_scales: dict[str, float] = {"continuous": 1.0, "discrete": 1.0, "unknown": 1.0}
    max_level: int | None = None
    above_level_scale: float = 0.0
    end_iteration: int | None = None
    after_iteration_scale: float = 0.0


@configclass
class RewardGatesCfg:
    enabled: bool = True
    continuous_terrain_names: tuple[str, ...] = CONTINUOUS_TERRAINS
    groups: dict[str, RewardGateRuleCfg] = {
        "gait_guidance": RewardGateRuleCfg(max_level=5, end_iteration=3000),
        "air_contact": RewardGateRuleCfg(),
        "default_style": RewardGateRuleCfg(),
    }
    term_groups: dict[str, str] = {
        **dict.fromkeys(GAIT_TERMS, "gait_guidance"),
        **dict.fromkeys(AIR_TERMS, "air_contact"),
        **dict.fromkeys(STYLE_TERMS, "default_style"),
    }
    overrides: dict[str, dict] = {
        "base_height_l2": {"terrain_scales": {"discrete": 0.0, "unknown": 0.0}},
        "feet_height_body": {"terrain_scales": {"continuous": 0.0, "unknown": 0.0}},
        "lin_vel_z_l2": {"terrain_scales": {"discrete": 0.25}},
    }


def _as_dict(value):
    return asdict(value) if is_dataclass(value) else copy.deepcopy(value)


def _terrain_labels(env, continuous_names):
    """Use the generator's exact curriculum column allocation, never random row IDs."""
    terrain = getattr(env.scene, "terrain", None)
    cfg = getattr(terrain, "cfg", None)
    unknown = ["unknown"] * env.num_envs
    if getattr(cfg, "terrain_type", None) == "plane":
        return ["flat"] * env.num_envs, ["continuous"] * env.num_envs, True
    generator = getattr(cfg, "terrain_generator", None)
    if generator is None:
        return unknown, unknown, False
    entries = [(name, term) for name, term in generator.sub_terrains.items() if term.proportion > 0]
    if not entries:
        return unknown, unknown, False
    categories = ["continuous" if name in continuous_names else "discrete" for name, _ in entries]
    if not generator.curriculum:
        names = [entries[0][0]] * env.num_envs if len(entries) == 1 else unknown
        groups = [categories[0]] * env.num_envs if len(set(categories)) == 1 else unknown
        return names, groups, False
    types = getattr(terrain, "terrain_types", None)
    if types is None:
        return unknown, unknown, False
    total = sum(term.proportion for _, term in entries)
    cumulative, value = [], 0.0
    for _, term in entries:
        value += term.proportion / total
        cumulative.append(value)
    columns = []
    for col in range(generator.num_cols):
        index = next(i for i, bound in enumerate(cumulative) if col / generator.num_cols + 0.001 < bound)
        columns.append(index)
    ids = types.long().cpu().tolist()
    return [entries[columns[i]][0] for i in ids], [categories[columns[i]] for i in ids], True


class RewardGates:
    """Compiled independent rules; state is environment-local and never checkpointed."""

    def __init__(self, env):
        from ..terrains.finetune_terrain_cfg import FINETUNE_ROUGH_TERRAINS_CFG
        from ..terrains.terrain_cfg import ROUGH_TERRAINS_CFG

        self.env = env
        cfg = env.cfg.reward_gates
        self.enabled = cfg.enabled
        self.iteration = 0
        self._step = None
        self._factors = {}
        known_terrains = (
            set(CONTINUOUS_TERRAINS)
            | set(ROUGH_TERRAINS_CFG.sub_terrains)
            | set(FINETUNE_ROUGH_TERRAINS_CFG.sub_terrains)
        )
        generator = getattr(getattr(getattr(env.scene, "terrain", None), "cfg", None), "terrain_generator", None)
        if generator is not None:
            known_terrains.update(generator.sub_terrains)
        if set(cfg.continuous_terrain_names) - known_terrains:
            raise ValueError("reward_gates: unknown continuous terrain name.")
        if set(cfg.term_groups) != set(GATED_TERMS):
            raise ValueError(f"reward_gates.term_groups must map exactly these rewards: {GATED_TERMS}")
        if set(cfg.overrides) - set(GATED_TERMS):
            raise ValueError("reward_gates.overrides contains an unknown reward name.")
        self.rules = {}
        fields = set(_as_dict(RewardGateRuleCfg()))

        def validate(rule):
            if set(rule) != fields:
                raise ValueError(f"reward_gates: invalid rule fields {set(rule) ^ fields}")
            if not isinstance(rule["terrain_scales"], dict):
                raise ValueError("reward_gates.terrain_scales must be a dictionary.")
            if set(rule["terrain_scales"]) - known_terrains - {"continuous", "discrete", "unknown"}:
                raise ValueError("reward_gates: unknown terrain scale key.")
            for value in [rule[k] for k in ("scale", "above_level_scale", "after_iteration_scale")] + list(
                rule["terrain_scales"].values()
            ):
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("reward_gates coefficients must be finite and nonnegative.")
            for name in ("max_level", "end_iteration"):
                value = rule[name]
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"reward_gates.{name} must be a nonnegative integer or None.")

        groups = {name: _as_dict(rule) for name, rule in cfg.groups.items()}
        for rule in groups.values():
            validate(rule)
        for name, group in cfg.term_groups.items():
            if group not in groups:
                raise ValueError(f"reward_gates: unknown group {group!r} for {name}.")
            if not hasattr(env.cfg.rewards, name):
                raise ValueError(f"reward_gates: unknown reward {name!r}.")
            rule = copy.deepcopy(groups[group])
            override = copy.deepcopy(cfg.overrides.get(name, {}))
            if set(override) - fields:
                raise ValueError(f"reward_gates: unknown override fields for {name}.")
            terrain_scales = override.pop("terrain_scales", {})
            if not isinstance(terrain_scales, dict):
                raise ValueError("reward_gates terrain override must be a dictionary.")
            rule["terrain_scales"].update(terrain_scales)
            rule.update(override)
            validate(rule)
            self.rules[name] = rule
        names, categories, self.levels_known = _terrain_labels(env, cfg.continuous_terrain_names)
        self._terrain = {
            name: torch.tensor(
                [rule["terrain_scales"].get(n, rule["terrain_scales"].get(c, 1.0)) for n, c in zip(names, categories)],
                device=env.device,
                dtype=torch.float32,
            )
            for name, rule in self.rules.items()
        }
        self._ones = torch.ones(env.num_envs, device=env.device)

    def set_iteration(self, iteration):
        self.iteration = int(iteration)
        self._step = None

    def factors(self):
        step = (getattr(self.env, "common_step_counter", None), self.iteration)
        if step[0] is not None and self._step == step:
            return self._factors
        self._step = step
        terrain = getattr(self.env.scene, "terrain", None)
        plane = getattr(getattr(terrain, "cfg", None), "terrain_type", None) == "plane"
        levels = torch.zeros_like(self._ones) if plane else getattr(terrain, "terrain_levels", None)
        self._factors = {}
        for name, rule in self.rules.items():
            category = self._terrain[name] if self.enabled else self._ones
            level = self._ones
            time = 1.0
            scale = rule["scale"] if self.enabled else 1.0
            if self.enabled:
                if rule["max_level"] is not None:
                    level = (
                        torch.where(levels <= rule["max_level"], 1.0, rule["above_level_scale"])
                        if self.levels_known and levels is not None
                        else torch.zeros_like(self._ones)
                    )
                if rule["end_iteration"] is not None and self.iteration >= rule["end_iteration"]:
                    time = rule["after_iteration_scale"]
            self._factors[name] = (category, level, time, category * level * time * scale)
        return self._factors

    def apply(self, value, name):
        factor = self.factors()[name][3]
        # Do not short-circuit the underlying stateful reward or hide active NaNs.
        return torch.where(factor != 0, value * factor, torch.zeros_like(value))

    def metrics(self):
        """Last reward-control-step factors, including zero-weight terms, without extra forwards."""
        if not self._factors:
            return {}
        values = {}
        for name, (category, level, time, total) in self._factors.items():
            prefix = f"RewardGates/{name}"
            weight = getattr(self.env.cfg.rewards, name).weight
            for key, value in (
                ("category", category.mean()),
                ("level", level.mean()),
                ("time", time),
                ("total", total.mean()),
                ("enabled_fraction", (total != 0).float().mean()),
                ("base_weight", weight),
                ("effective_weight", total.mean() * weight),
            ):
                values[f"{prefix}/{key}"] = torch.as_tensor(value, device=self.env.device)
        values["RewardGates/last_control_step"] = torch.as_tensor(self._step[0], device=self.env.device)
        manager = getattr(self.env, "reward_manager", None)
        if manager is not None:
            for name in self.rules:
                index = manager.active_terms.index(name)
                values[f"RewardGates/{name}/weighted_reward_per_second"] = manager._step_reward[:, index].mean()
        packed = torch.stack(list(values.values())).detach().cpu().tolist()
        return dict(zip(values, packed))


def initialize_reward_gates(env, env_ids=None):
    """Startup event: validate final post-Hydra settings before the first environment step."""
    env._reward_gates = RewardGates(env)


def apply_reward_gate(env, value, name):
    gates = getattr(env, "_reward_gates", None)
    if gates is None:
        # Also support direct reward evaluation outside a running simulator.
        initialize_reward_gates(env)
        gates = env._reward_gates
    return gates.apply(value, name)
