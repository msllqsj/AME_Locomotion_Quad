"""Go2 rewards grouped into task, regulation and gait style.

Go2AMEEnvCfg.__post_init__ configures the final weights for both stages.
External terrain, level and iteration rules are configured in Go2AMEEnvCfg.reward_gates.
Weights adapt the repository's G1 stage-1 settings and AME Table 2 (ANYmal-D);
different robot morphology and reward formulas prevent a direct reproduction.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp
from . import rewards as r
from . import stage_cfg


@configclass
class Go2RewardsCfg:
    # -- task: command tracking
    # G1 uses 2.0 after stage overrides; retain Go2's existing tracking kernel.
    track_lin_vel_xy_exp = RewTerm(
        func=r.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=r.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # -- regulation: safety, energy and action smoothness
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-10.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )
    # Both G1 and the paper's quadruped configuration penalize soft-limit excess.
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # Mechanical power is not the paper's squared-torque term; keep its baseline.
    joint_power = RewTerm(func=r.joint_power, weight=-2e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    smoothness = RewTerm(func=r.ActionSmoothnessPenalty, weight=-0.01)
    lin_vel_z_l2 = RewTerm(func=r.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=r.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=r.flat_orientation_l2, weight=-0.2)

    # -- style: posture, foot contacts and coordination
    # AME Table 2, ANYmal-D: vertical velocity and roll/pitch angular velocity.

    base_height_l2 = RewTerm(
        func=r.base_height_l2,
        weight=0.0,
        params={
            "target_height": 0.33,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
        },
    )

    joint_deviation_hip_l1 = RewTerm(
        func=r.joint_deviation_hip_l1,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    joint_mirror = RewTerm(
        func=r.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
                ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ],
        },
    )
    feet_slide = RewTerm(
        func=r.feet_slide,
        # G1 stage-1 default; final weight is configured in Go2AMEEnvCfg.
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    # Retained for isolated ablations; disabled in both stages after rollback.
    feet_slippage_3d = RewTerm(
        func=r.feet_slippage_3d,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=stage_cfg.FOOT_NAMES.copy(), preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=stage_cfg.FOOT_NAMES.copy(), preserve_order=True),
            "threshold": 1.0,
        },
    )
    feet_contact_forces = RewTerm(
        func=r.feet_contact_forces,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 220.0},
    )
    feet_height_body = RewTerm(
        func=r.feet_height_body,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "target_height": -0.2,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_gait = RewTerm(
        func=r.GaitReward,
        weight=0.2,
        params={
            # GaitReward divides squared timing errors by std directly, not std**2.
            "std": 0.1,
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    # Reward completed swings, while the excess term penalizes ongoing overlong swings.
    feet_air_time = RewTerm(
        func=r.feet_air_time,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=stage_cfg.FOOT_NAMES.copy(), preserve_order=True),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    feet_air_time_variance_penalty = RewTerm(
        func=r.feet_air_time_variance_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=stage_cfg.FOOT_NAMES.copy(), preserve_order=True),
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
            "linear_command_threshold": 0.1,
            "angular_command_threshold": 0.2,
        },
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
