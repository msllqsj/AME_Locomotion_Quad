# AME Windows environment

## Go2 AME (2026-10-25)

Use `conda activate env_ame` from the repository root. The existing environment
and shared Isaac Lab source installation are used without package upgrades.
The local Isaac Lab checkout reports version 2.3.2 (package `isaaclab==0.54.2`),
with Isaac Sim 5.1.0, PyTorch 2.7.0+cu128 and local `rsl-rl-lib==3.0.1`.

```powershell
# Stage 1: go2/stage_cfg.py FINETUNE=False (the delivered default).
.\run_train.ps1 --logger tensorboard

# Stage 2: edit the same file to FINETUNE=True, then restart.
.\run_train.ps1 --logger tensorboard --finetune_from <stage1_checkpoint>

# Full-state resume; the stage selector must match the intended environment.
.\run_train.ps1 --resume --checkpoint <checkpoint>

# Play a Go2 model on the stage selected by the same source constant.
.\run_play.ps1 --checkpoint <go2_checkpoint>
.\run_play.ps1 --checkpoint <go2_checkpoint> --headless --video --video_length 100 --save_attention_weights --vis_attention

# Local validation in Git-ignored test/ (sparse checks require FINETUNE=False).
python -s test/test_go2.py
python -s test/smoke_go2.py --terrain hf_steppingstones
python -s test/smoke_go2.py --terrain hf_gaps
```

The phase selector is
`source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/go2/stage_cfg.py`.
Stage 1 already contains the original stepping stones and gaps. See
[README_ZH.md](README_ZH.md) for usage and [GO2_VALIDATION.md](test/GO2_VALIDATION.md)
for validation results and artifacts. Existing `pretrained/ame1.pt` and `ame2.pt`
remain G1 checkpoints.

## Original G1 environment record (2026-10-21)

Configured on 2026-10-21.

- `env_isaaclab`: restored official `rsl-rl-lib==3.1.2`.
- `env_ame`: cloned from `env_isaaclab`, with this repository's editable
  `rsl-rl-lib==3.0.1` and `ame_locomotion==0.1.0`.
- Python 3.11, PyTorch 2.7.0+cu128; CUDA available.
- `PYTHONNOUSERSITE=1` is configured through Conda for `env_ame`.
- Isaac Lab remains an editable installation pointing to the existing local
  `MyLocoInfra/IsaacLab` checkout. Editing that shared checkout affects both
  environments. AME's RSL-RL installation is separate from the original environment.

Run these commands from the repository root in PowerShell:

```powershell
conda activate env_ame

# Train (increase num_envs to suit available GPU memory).
python scripts/rsl_rl/train.py --task AME-G1-29DOF-v0 --num_envs 64 --headless --logger tensorboard

# Play the supplied checkpoint with a visible window.
python scripts/rsl_rl/play.py --task AME-G1-29DOF-Play-v0 --num_envs 10 --checkpoint pretrained/ame1.pt

# AME2 uses the global-context encoder (the field is declared in AMEActorCriticCfg).
python scripts/rsl_rl/play.py --task AME-G1-29DOF-Play-v0 --num_envs 10 --checkpoint pretrained/ame2.pt agent.policy.attach_global=true

# Record a bounded headless playback.
python scripts/rsl_rl/play.py --task AME-G1-29DOF-Play-v0 --num_envs 1 --checkpoint pretrained/ame1.pt --headless --video --video_length 300
```

Validation performed:

- `python -s -m pip check`: no broken requirements.
- Training: 4 environments, one PPO iteration, 96 timesteps, checkpoint saved.
  Smoke testing used one terrain row/column and initial terrain level zero.
- Playback: `pretrained/ame1.pt`, one environment, ten steps; generated and
  decoded a 10-frame 1280x720 MP4 at `pretrained/videos/play/rl-video-step-0.mp4`.
- Training logs: `ame_train_final_smoke.log`; playback log: `ame_play_smoke.log`.

These are short functional checks, not a full training or convergence run.
The visible-window playback has not been separately tested.

Compatibility pins installed in the new environment include
`botocore==1.39.11`, `wrapt==1.17.3`, `psutil==5.9.8`, `ipython==8.39.0`,
`ipywidgets==8.1.5`, `packaging==23.2`, `wheel==0.45.1`, `pytest==8.4.2`,
and OpenTelemetry API/SDK 1.44.0. Missing declared dependencies were also
installed. The original environment's other dependency issues were left as they were.
