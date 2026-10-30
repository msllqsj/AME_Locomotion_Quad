"""Go2 traversal baseline with an isolated ongoing-air-time penalty.

Both stages share reward settings. The overlong-swing penalty is enabled only
on continuous terrain; set its weight to zero for the baseline control group.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp
from . import rewards as r
from . import stage_cfg


@configclass
class Go2RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=r.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=r.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    lin_vel_z_l2 = RewTerm(func=r.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=r.ang_vel_xy_l2, weight=-0.05)
    # Retained for isolated ablations; disabled in both stages after rollback.
    feet_slippage = RewTerm(
        func=r.feet_slippage,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=stage_cfg.FOOT_NAMES.copy(), preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=stage_cfg.FOOT_NAMES.copy(), preserve_order=True),
            "threshold": 1.0,
        },
    )
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 220.0},
    )
    standing_joint_positions = RewTerm(
        func=r.standing_joint_position_error,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=stage_cfg.JOINT_NAMES.copy(), preserve_order=True),
            "command_threshold": 0.1,
        },
    )
    standing_joint_velocities = RewTerm(
        func=r.standing_joint_velocity_error,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=stage_cfg.JOINT_NAMES.copy(), preserve_order=True),
            "command_threshold": 0.1,
        },
    )
    flat_orientation_l2 = RewTerm(func=r.flat_orientation_l2, weight=-0.2)
    base_height_l2 = RewTerm(
        func=r.base_height_l2,
        weight=0.0,
        params={
            "target_height": 0.33,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "terrain_mask": "continuous",
        },
    )
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_deviation_hip_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    joint_power = RewTerm(func=r.joint_power, weight=-2e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    smoothness = RewTerm(func=r.ActionSmoothnessPenalty, weight=-0.01)
    joint_mirror = RewTerm(
        func=r.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "terrain_mask": "continuous",
            "mirror_joints": [
                ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
                ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ],
        },
    )
    feet_slide = RewTerm(
        func=r.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_height_body = RewTerm(
        func=r.feet_height_body,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "target_height": -0.2,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "terrain_mask": "all",
        },
    )
    feet_gait = RewTerm(
        func=r.GaitReward,
        weight=0.1,
        params={
            "std": math.sqrt(0.1),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "terrain_mask": "continuous",
        },
    )
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-10.0)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )
    # Retained for later ablations; isolate current air time in this experiment.
    feet_air_time_variance_penalty = RewTerm(
        func=r.feet_air_time_variance_penalty,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=stage_cfg.FOOT_NAMES.copy(), preserve_order=True),
            "terrain_mask": "continuous",
        },
    )
    feet_air_time_excess_penalty = RewTerm(
        func=r.FeetAirTimeExcessPenalty,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=stage_cfg.FOOT_NAMES.copy(), preserve_order=True),
            "command_name": "base_velocity",
            "max_air_time": 0.6,
            "ramp_time": 0.4,
            "min_support_feet": 2,
            "terrain_mask": "continuous",
            "linear_command_threshold": 0.1,
            "angular_command_threshold": 0.2,
        },
    )
