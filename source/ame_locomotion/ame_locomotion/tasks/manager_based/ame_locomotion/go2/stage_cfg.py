"""Single source of truth for the Go2 AME training stage.

Change FINETUNE and restart train/play. Stage 1 already includes sparse terrain.
"""

FINETUNE = False

JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
FOOT_NAMES = [f"{leg}_foot" for leg in ("FR", "FL", "RR", "RL")]
MAP_SCAN_DIM = (33, 21, 3)


def experiment_name():
    return "go2_ame_finetune" if FINETUNE else "go2_ame"
