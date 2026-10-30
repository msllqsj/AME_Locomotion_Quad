#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python scripts/rsl_rl/train.py --num_envs 64 --headless "$@"
