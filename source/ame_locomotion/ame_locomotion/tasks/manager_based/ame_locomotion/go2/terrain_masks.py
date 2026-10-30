"""Shared terrain-group masks for Go2 reward terms."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass


DEFAULT_CONTINUOUS_TERRAINS = (
    "flat",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "boxes",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
)


@configclass
class TerrainMaskCfg:
    """Terrain names treated as continuous support in both training stages."""

    continuous_terrain_names: tuple[str, ...] = DEFAULT_CONTINUOUS_TERRAINS


def _build_terrain_masks(env):
    all_envs = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    continuous = torch.zeros_like(all_envs)
    known = torch.zeros_like(all_envs)
    terrain = getattr(env.scene, "terrain", None)
    terrain_cfg = getattr(terrain, "cfg", None)
    options = getattr(getattr(env, "cfg", None), "terrain_masks", None)
    names = getattr(options, "continuous_terrain_names", DEFAULT_CONTINUOUS_TERRAINS)
    if getattr(terrain_cfg, "terrain_type", None) == "plane":
        continuous[:] = True
        known[:] = True
    else:
        generator = getattr(terrain_cfg, "terrain_generator", None)
        if generator is not None:
            terrain_names = list(generator.sub_terrains)
            groups = [name in names for name in terrain_names]
            if terrain_names and not generator.curriculum:
                # Random mixed cells cannot be inferred from terrain_types.
                # A homogeneous group is still known even without a type label.
                if all(groups) or not any(groups):
                    known[:] = True
                    continuous[:] = all(groups)
            elif terrain_names:
                types = getattr(terrain, "terrain_types", None)
                proportions = [float(generator.sub_terrains[name].proportion) for name in terrain_names]
                total = sum(proportions)
                if types is not None and total > 0:
                    cumulative, value = [], 0.0
                    for proportion in proportions:
                        value += proportion / total
                        cumulative.append(value)
                    columns = []
                    for col in range(generator.num_cols):
                        index = next(
                            i for i, bound in enumerate(cumulative) if col / generator.num_cols + 0.001 < bound
                        )
                        columns.append(groups[index])
                    continuous = torch.tensor(columns, dtype=torch.bool, device=env.device)[types.long()]
                    known[:] = True
    return {"all": all_envs, "continuous": continuous, "discrete": known & ~continuous, "known": known}


def get_terrain_mask(env, terrain_mask: str | None = "continuous") -> torch.Tensor:
    """Return one cached mask per group, shared by all terms in this environment.

    Classification uses assigned curriculum columns, not local ray geometry.
    Unknown random mixtures are excluded from both continuous/discrete groups.
    Environment resets/level changes keep type assignments, so the cache survives.
    """
    name = "all" if terrain_mask is None else terrain_mask
    if name not in ("all", "continuous", "discrete", "known"):
        raise ValueError(f"Unknown terrain_mask {terrain_mask!r}; use all, continuous, discrete or known.")
    masks = getattr(env, "_go2_terrain_masks", None)
    if masks is None:
        masks = _build_terrain_masks(env)
        env._go2_terrain_masks = masks
    return masks[name]


def apply_terrain_mask(env, value: torch.Tensor, terrain_mask: str | None = None) -> torch.Tensor:
    """Apply an optional reward gate; excluded nonfinite values become zero."""
    if terrain_mask is None or terrain_mask == "all":
        return value
    return torch.where(get_terrain_mask(env, terrain_mask), value, torch.zeros_like(value))


def clear_terrain_mask_cache(env) -> None:
    """Invalidate after explicit terrain-layout/config changes outside normal resets."""
    if hasattr(env, "_go2_terrain_masks"):
        del env._go2_terrain_masks
