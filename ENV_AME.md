# AME Windows environment

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
