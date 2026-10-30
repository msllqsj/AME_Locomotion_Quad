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

Both stages default to experiment B: the traversal reward baseline from
`2026-10-26_18-22-16_without_base-height-peneal`, plus one ongoing-air-time penalty.
The earlier combined posture/gait experiment is disabled for this comparison.

| Term | Continuous terrain | Discrete terrain |
|---|---:|---:|
| Linear / yaw velocity tracking | 3 / 2 | 3 / 2 |
| `base_height_l2`, `joint_mirror`, `feet_gait`, historical timing variance | 0 | 0 |
| Original robot_lab `feet_slide` | −0.05 | −0.05 |
| `feet_height_body` (`terrain_mask=all`) | −0.5 | −0.5 |
| Non-timeout termination | −10 | −10 |
| `feet_air_time_excess_penalty` | −0.2 | 0 |

The ongoing cost is `sum(clamp((current_air_time - 0.6) / 0.4, 0, 1)**2)`:
it starts after 0.6 s and saturates at 1 s. It requires a translation command
above 0.1 m/s or yaw command above 0.2 rad/s, at least two supporting feet, and
upright orientation. It reuses contact-sensor timers and their resets.
Historical timing variance and experimental style functions remain available,
with zero default weight. Tracking, foot-slide math, terrain, network and PPO
are unchanged in this isolated reward comparison.

The shared mask implementation remains available. Names are defined once in
`env.terrain_masks.continuous_terrain_names`: flat, stairs, boxes, rough and slopes.
Steppingstones, gaps, stakes, bridges and rails are excluded from the ongoing cost.
The optional selector supports `continuous`, `discrete`, `known`, and `all`;
`None` also disables gating. A nonzero weight is needed for a term to contribute.
Masks use assigned curriculum columns and are cached across resets/level changes.
They do not classify local footholds: a gap environment's spawn platform is discrete.
Random continuous/discrete mixtures are unknown and excluded from both groups;
homogeneous generators are classifiable. After manual runtime layout/list changes,
call `clear_terrain_mask_cache(env)`.

Use the same checkpoint and seed for both branches, with fresh optimizer state:

```powershell
# B: default experimental reward, initially check after 500 PPO updates.
.\run_train.ps1 --finetune_from logs/rsl_rl/go2_ame/2026-10-26_18-22-16_without_base-height-peneal/model_9999.pt --seed 42 --run_name air_excess_B --max_iterations 500
# A: same command with a distinct name and the only reward difference below.
.\run_train.ps1 --finetune_from logs/rsl_rl/go2_ame/2026-10-26_18-22-16_without_base-height-peneal/model_9999.pt --seed 42 --run_name air_excess_A --max_iterations 500 env.rewards.feet_air_time_excess_penalty.weight=0
```

Run branches sequentially on this single-GPU setup. Check per-foot contact and
load participation, longest air phases, tracking and fixed-terrain traversal at
0/100/300/500 updates. This is an inspection budget, not a convergence promise.
Loading a checkpoint does not restore its old environment/reward configuration.
Restart training for new defaults; `--finetune_from` initializes weights and
resets optimizer/iteration, while `--resume` restores full training state.
Decision evidence: `test/GO2_NEXT_STEP_20261029.md`; implementation validation:
`test/AIR_EXCESS_ISOLATED_VALIDATION_20261029.md`. These bounded checks establish
integration, not a proven improvement in foot participation or traversal.

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
