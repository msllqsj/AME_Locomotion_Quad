# AME Locomotion: Unitree Go2 + G1

[中文说明](README_ZH.md) | English

This repository implements AME terrain-aware locomotion with Isaac Lab and a local
custom RSL-RL package. **Unitree Go2 (12 DoF)** is the default; G1 tasks/checkpoints
remain available. Use the existing `env_ame` environment on this Windows host,
or install both local packages in a compatible Isaac Lab environment:

```bash
python -m pip install -e source/ame_locomotion
python -m pip install -e rsl_rl
```

Robot assets and adapted rewards are
included; sibling reference repositories are not runtime dependencies.

Set the single `FINETUNE` constant in `go2/stage_cfg.py`, then restart train/play:

- `False`: original mixed base terrains, including 20% stepping stones and 20%
  gaps, with ideal perception; logs in `logs/rsl_rl/go2_ame`.
- `True`: original finetuning terrains with sparse supports and perturbations;
  logs in `logs/rsl_rl/go2_ame_finetune`.

Both stages use the same rewards, commands and PPO defaults. The current experiment
adds only a continuous-terrain ongoing-air-time penalty to the traversal reward
baseline. There is no rough-only pretraining stage.

```bash
# Stage 1: FINETUNE=False
python scripts/rsl_rl/train.py --num_envs 64 --headless --logger tensorboard
# Stage 2: edit FINETUNE=True and initialize from stage 1
python scripts/rsl_rl/train.py --num_envs 64 --headless --finetune_from <stage1_checkpoint>
# Resume a run, retaining optimizer and iteration
python scripts/rsl_rl/train.py --num_envs 64 --headless --resume --checkpoint <checkpoint>
# Play the stage selected by FINETUNE
python scripts/rsl_rl/play.py --num_envs 1 --checkpoint <go2_checkpoint>
```

`--finetune_from` loads weights with a fresh optimizer and iteration counter;
`--resume` restores training state. They are mutually exclusive. The task IDs are
`AME-Go2-v0` and `AME-Go2-Play-v0` in both stages. `--max_iterations` is the number
of additional PPO updates for this invocation. Use `agent.policy.attach_global=true`
consistently in train/play for AME2 checkpoints.

Add `--headless --video --video_length 100 --save_attention_weights --vis_attention`
for bounded playback with video and attention. Visualization and saving are
independent switches. `--max_steps` bounds playback without video. Videos and
`attention_weights.npy` are stored beside the checkpoint. Plot saved attention
with `python scripts/plot_attention.py <path/to/attention_weights.npy>`.

The Go2 action/joint observation order is FR, FL, RR, RL, with hip, thigh, calf
within each leg. The policy/critic observations have 2124/2127 entries, including
an xyz map of size 33x21; the output has 12 actions. Control runs at 50 Hz.

Local validation files are organized under the Git-ignored `test/` directory.
Run `test/test_go2.py` for configuration/network/reward checks and
`test/smoke_go2.py --terrain hf_steppingstones` (or `hf_gaps`) for
bounded physical checks. See [GO2_VALIDATION.md](test/GO2_VALIDATION.md) for results.
The shipped `pretrained/ame1.pt` and `ame2.pt` are G1 checkpoints:

```bash
python scripts/rsl_rl/train.py --task AME-G1-29DOF-v0 --num_envs 64 --headless
python scripts/rsl_rl/play.py --task AME-G1-29DOF-Play-v0 --checkpoint pretrained/ame1.pt --num_envs 1
python scripts/rsl_rl/play.py --task AME-G1-29DOF-Play-v0 --checkpoint pretrained/ame2.pt --num_envs 1 agent.policy.attach_global=true
```

Go2 uses its own stage selector; the G1 configuration retains its existing
`FINETUNE` constant. See [ENV_AME.md](ENV_AME.md) for Windows setup and
[the AME paper](https://arxiv.org/abs/2506.09588) for the method.

## Go2 rewards and shared terrain gates

Both stages share the same pure-RL base weights and formulas. External gates
control which terrain levels and training iterations receive style constraints.
These are experimental defaults; improved foot participation is not yet established.

For quick tuning, edit the grouped `self.rewards.<term>.weight` assignments in
`Go2AMEEnvCfg.__post_init__` (`go2/env_cfg.py`). These set the final weights for
both stages; `go2/rewards_cfg.py` defines the functions, parameters and defaults.
Reward gates still apply, and a zero weight disables a term.

| Term | Continuous terrain | Discrete terrain |
|---|---:|---:|
| Linear / yaw velocity tracking | 2 / 2 | 2 / 2 |
| Vertical velocity squared | -1 | -0.25 |
| Action rate squared | -0.01 | -0.01 |
| Joint position soft-limit excess | -1 | -1 |
| `base_height_l2`, `joint_mirror` | 0 | 0 |
| Historical air/contact-time variance | -0.1 | -0.1 |
| `feet_air_time` (touchdown, threshold 0.5 s) | 0.25 | 0.25 |
| `feet_slide`: world horizontal foot speed while in contact | -0.1 | -0.1 |
| `feet_height_body` | 0 | -0.5 |
| `feet_gait` (stage 1, level <=5 and iteration <3000) | 0.2 | 0.2 |
| Non-timeout termination | -10 | -10 |
| `feet_air_time_excess_penalty` | -0.1 | -0.1 |

The weights draw on the **effective G1 stage-1 configuration** (including its
post-init overrides) and [AME Table 2, ANYmal-D](https://arxiv.org/html/2506.09588v1).
Linear tracking and horizontal slip use G1 stage 1's 2.0 and -0.1; vertical velocity
and joint limits use the paper's coefficient magnitudes. This is
an adapted experiment, not a reproduction: G1 tracking uses `std=0.25` versus
Go2's 0.5, and the paper's action-rate term uses desired joint angles while
this implementation uses actions before the 0.25 scale and retains a -0.01
action-rate weight. Mechanical power is also distinct from squared torque.
Yaw tracking stays at 2 and termination at -10; the paper's -200 termination
and stronger slip penalty are not transferred. Gait, airborne-foot constraints
and terrain gates retain their current settings. Improvement must be checked
against both per-foot participation and sparse-terrain traversal.

`feet_slide` wraps G1's unchanged `mdp.feet_slide`: world XY foot speed without
subtracting base velocity, contact-history threshold 1 N, and no upright gate.
The TensorBoard reward tag is `Episode_Reward/feet_slide`; earlier runs using
`feet_slippage` keep their old tag. The separate
world-3D `feet_slippage_3d` term stays disabled. Gait pairs are FL-RR and FR-RL;
its `std` directly divides squared timing errors, so use `0.1`, not `sqrt(0.1)`.
Foot clearance retains its existing formula and target on discrete terrain.

`feet_air_time` follows robot_lab's quadruped touchdown formula:
`sum((last_air_time - 0.5) * first_contact)`. It uses all terrain categories,
command norm above 0.1 (including yaw), and upright orientation. Longer completed
swings contribute positively, shorter ones negatively; 0.5 s is a threshold,
not a target duration. An airborne foot earns nothing until touchdown. This
does not enforce diagonal timing or guarantee that all four feet participate.

The ongoing cost is `sum(clamp((current_air_time - 0.6) / 0.4, 0, 1)**2)`:
it starts after 0.6 s and saturates at 1 s. It requires a translation command
above 0.1 m/s or yaw command above 0.2 rad/s, at least two supporting feet, and
upright orientation. It reuses contact-sensor timers and their resets.
Terrain, commands, network and PPO are unchanged; no AMP or phase input is added.

`Go2AMEEnvCfg.reward_gates` replaces the old `terrain_mask` arguments and module.
The final reward is raw reward × base weight × scale × terrain × level × time × dt;
existing command, upright, touchdown and support conditions remain inside the formula.
Groups provide defaults and `overrides` merge individual fields per reward:

- `gait_guidance`: gait, mirror and hip deviation; all terrains at level <=5 and
  iteration <3000 in stage 1, disabled in stage 2.
- `air_contact`: touchdown air-time, variance and ongoing excess; all terrains in both stages.
- `default_style`: remaining style terms, vertical/angular velocity and orientation;
  default factors are 1. Base height is continuous-only, foot clearance discrete-only,
  and vertical velocity uses 0.25 on discrete terrain (1 on unknown terrain).

Each rule supports `scale`, `terrain_scales`, `max_level`, `above_level_scale`,
`end_iteration` and `after_iteration_scale`. Missing override fields inherit;
explicit `None` removes a level/time limit. Terrain dictionaries merge by key,
and a named terrain takes precedence over its category. For example:

```python
self.reward_gates.groups["gait_guidance"].max_level = 5
self.reward_gates.overrides["joint_mirror"] = {"end_iteration": 1500}
```

Hydra accepts `env.reward_gates.enabled=false` to set **all external factors to 1**;
zero base weights remain off. This does not restore the previous terrain masks.
The final configuration is saved in `env.yaml` and validated at environment startup,
including inactive rewards. Old `terrain_mask`/`terrain_masks` settings are unsupported.

Continuous names are configured in `reward_gates.continuous_terrain_names` (flat,
stairs, boxes, rough and slopes). Curriculum classification follows generator column
allocation, while levels are read dynamically. The spawn platform in a gap environment
is still discrete. Random mixed types are unknown; homogeneous categories and single
terrain names can be identified, but random row IDs are not difficulty levels.
Plane terrain has level 0. Unknown levels disable level-limited terms; unknown
categories disable the two height terms. Named terrain overrides still work when
the exact name is known. Restart after changing terrain layouts or gate configuration.

The Runner supplies its logged PPO iteration before each rollout. Resume uses the
existing saved iteration convention (the saved index is reused); weights-only
initialization resets to 0, and play uses checkpoint progress. No gate state enters
model weights, and G1 tasks retain their behavior. TensorBoard `RewardGates/<term>/`
records category/level/time/total factors, enabled fraction, base/effective weights,
and weighted reward per second from the **last reward control step**, not an episode
average; `RewardGates/last_control_step` identifies the sample. Logging is independent
of AME diagnostics. Attention and gate values alone do not establish locomotion quality.

For this stage-1 experiment, start a new run with `FINETUNE=False`:

```powershell
.\run_train.ps1 --seed 42 --run_name pure_rl_g1_paper
```

Keep older runs as controls. Evaluate fixed flat commands at 0.3, 0.5 and 0.8 m/s,
per-foot load participation, longest air phases, diagonal timing, and fixed
steppingstone/gap traversal. A higher gait reward alone does not establish success.
This is a combined experiment, not a single-term causal ablation.
New defaults require restarting the process; loading a checkpoint does not restore
its old environment/reward settings. Stage 2 still requires initialization or resume.

## AME TensorBoard diagnostics

Both Go2 stages enable diagnostics by default; G1 and other tasks leave them off.
Start training normally with `--logger tensorboard`, then run
`tensorboard --logdir logs/rsl_rl/go2_ame` (use `go2_ame_finetune` for stage 2).
Existing running processes need restarting to pick up this code. To disable or change sampling:

```bash
python scripts/rsl_rl/train.py --num_envs 64 --headless --logger tensorboard agent.diagnostics.enabled=false
python scripts/rsl_rl/train.py --num_envs 64 --headless --logger tensorboard agent.diagnostics.scalar_interval=10 agent.diagnostics.attribution_interval=50 agent.diagnostics.image_interval=200 agent.diagnostics.max_images=4
```

Intervals count PPO updates, including update 0; resumed runs use the restored iteration.
CNN and MHA have no independent loss. Their attribution uses gradients of
`surrogate_loss - entropy_coef * entropy` and `value_loss_coef * value_loss`.
The ordinary backward pass supplies total gradients, including any enabled auxiliary policy loss.

| TensorBoard group | What to examine |
|---|---|
| `AME/PPO` | Weighted objectives, total policy-network loss, pre-update/update KL, clip fraction, explained variance and gradient clipping |
| `AME/Modules/<module>` | Pre-clip gradient L2/RMS, parameter L2 and actual relative parameter update |
| `AME/Attribution/<module>` | Policy/value gradient norms, cosine similarity, value/policy ratio and validity/connectivity flags |
| `AME/Features` | Separate actor/critic CNN, Query and MHA statistics, nonfinite fractions, ReLU zeros and BN running variances |
| `AME/Attention` | Per-head normalized entropy/peak, head cosine similarity, actor/critic Jensen–Shannon divergence and rollout images |
| `AME/Terrain/<type>` | Linear velocity error, velocity along the command, stall fraction, non-timeout termination fraction, terrain level and sample counts |

Module/feature statistics and attribution use the first minibatch of each sampled update.
PPO statistics aggregate its minibatches; pre-update KL uses the first one and includes any
rollout/minibatch discrepancy already present before the optimizer step (including training-mode BN).
Explained variance uses returns and current predictions over those minibatches.
The clipping fraction is the fraction of minibatches exceeding `max_grad_norm`; clipping scale
is the mean applied scale. Missing/zero gradients have flags; undefined cosine or ratios are omitted.
AME2 also reports the global encoder and Query fusion projection.

Terrain statistics accumulate every rollout step and flush on scalar updates. Stall means a
translation command above 0.2 m/s and directed velocity below 0.1 m/s for at least 1 second.
Its denominator is moving-command samples; failure rates use classified completed episodes,
excluding timeouts from the numerator. No episodes means counts only. Random non-curriculum
terrain is labeled `unknown`; curriculum names follow the generator's actual column allocation.

Images prefer steppingstones/gaps and samples with height variation. Each image contains the
input height, mean-attention overlay, then all heads. x points right, y up; height colors span
−1.2 to 0 m and attention colors span 0 to four times uniform. Images reuse actual rollout
forwards; scalar feature/attention statistics come from the update minibatch. Per-head capture
uses [PyTorch's MHA interface](https://docs.pytorch.org/docs/2.7/generated/torch.nn.MultiheadAttention.html),
while playback still returns the original head-averaged attention shape.

For sparse-terrain stalls, first compare terrain stall/error curves and sample counts, then
look for small CNN/MHA gradients **and** updates, persistent value-dominated conflicting gradients,
or collapsed feature variance. High pre-update KL points to an existing rollout/update mismatch;
high update KL and frequent clipping suggest optimization instability. Attention concentration
and gradient conflict are diagnostic evidence, not proof of the cause of training failure.

Diagnostics add no forward pass, mode switch or random sampling, and do not change rewards,
BN behavior, optimizer updates or checkpoint keys. They are disabled with a warning in distributed
training. Local tests and measured overhead are in the ignored
`test/test_ame_diagnostics.py`, `test/diagnostics/` and
[validation report](test/AME_DIAGNOSTICS_VALIDATION.md). The overhead target is about 5% on average;
attribution/image updates can have larger individual spikes.
