"""Go2 AME: original mixed/sparse terrains in both training stages."""

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    MySceneCfg,
    TerminationsCfg,
)

from .. import mdp
from ..assets.robots.unitree import UNITREE_GO2_CFG
from ..terrains.finetune_terrain_cfg import FINETUNE_ROUGH_TERRAINS_CFG
from ..terrains.terrain_cfg import ROUGH_TERRAINS_CFG
from . import stage_cfg
from .reward_gates import RewardGatesCfg, initialize_reward_gates
from .rewards_cfg import Go2RewardsCfg


def joint_entity():
    return SceneEntityCfg("robot", joint_names=stage_cfg.JOINT_NAMES.copy(), preserve_order=True)


@configclass
class Go2SceneCfg(MySceneCfg):
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0], ordering="xy"),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class Go2ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": joint_entity()}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={"asset_cfg": joint_entity()},
            noise=Unoise(n_min=-2.0, n_max=2.0),
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.elevation_map, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False}
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": joint_entity()})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, params={"asset_cfg": joint_entity()})
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.elevation_map, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False}
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2EventCfg(EventCfg):
    reward_gates = EventTerm(func=initialize_reward_gates, mode="startup")


@configclass
class Go2AMEEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=1024, env_spacing=2.5)
    observations: Go2ObservationsCfg = Go2ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: Go2RewardsCfg = Go2RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: Go2EventCfg = Go2EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    reward_gates: RewardGatesCfg = RewardGatesCfg()
    # Saved in env.yaml; source stage_cfg.FINETUNE is the supported selector.
    finetune: bool = False

    def __post_init__(self):
        self.finetune = stage_cfg.FINETUNE
        self.reward_gates.groups["gait_guidance"].scale = 0.0 if self.finetune else 1.0
        terrain = FINETUNE_ROUGH_TERRAINS_CFG if self.finetune else ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator = terrain.copy()
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 5
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.height_scanner_base.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.actions.joint_pos.joint_names = stage_cfg.JOINT_NAMES.copy()
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = 0.25
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.debug_vis = False

        # Reward weights: override Go2RewardsCfg defaults here for both stages.
        # -- task: command tracking
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        # -- regulation: safety, energy and action smoothness
        self.rewards.is_terminated.weight = -10.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_power.weight = -2e-5
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.smoothness.weight = -0.01
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.2

        # -- gait style: posture, foot contacts and coordination
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.joint_deviation_hip_l1.weight = -0.0
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slippage_3d.weight = 0.0
        self.rewards.feet_contact_forces.weight = 0.0
        self.rewards.feet_height_body.weight = -0.5
        self.rewards.feet_gait.weight = 0.2
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time_variance_penalty.weight = -0.1
        self.rewards.feet_air_time_excess_penalty.weight = -0.1
        self.rewards.standing_joint_positions.weight = 0.0
        self.rewards.standing_joint_velocities.weight = 0.0

        self.events.physics_material.params.update(
            static_friction_range=(0.3, 1.0), dynamic_friction_range=(0.3, 1.0), restitution_range=(0.0, 0.1)
        )
        self.events.reset_base.params["velocity_range"] = {
            k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params.update(position_range=(1.0, 1.0), velocity_range=(-1.0, 1.0))
        self.events.base_com = None
        self.events.base_external_force_torque = None
        if self.finetune:
            self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
            self.events.push_robot.interval_range_s = (5.0, 10.0)
            self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
            self.observations.policy.enable_corruption = True
            self.observations.policy.height_scan.params["noise"] = True
        else:
            self.events.add_base_mass = None
            self.events.push_robot = None


@configclass
class Go2AMEEnvCfg_PLAY(Go2AMEEnvCfg):
    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        env_index=0,
        eye=(2.0, 2.0, 1.5),
        lookat=(0.0, 0.0, -0.1),
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.params["noise"] = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {key: (0.0, 0.0) for key in ("x", "y", "yaw")}
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # Select one terrain for playback.
        terrain_name = "hf_gaps"
        generator = self.scene.terrain.terrain_generator
        generator.sub_terrains = {terrain_name: generator.sub_terrains[terrain_name].copy()}
        generator.sub_terrains[terrain_name].proportion = 1.0

        generator.num_rows = 1
        generator.num_cols = 1
        generator.curriculum = False
        generator.difficulty_range = (1.0, 1.0)
