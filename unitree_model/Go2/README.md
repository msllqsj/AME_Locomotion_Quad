# Go2 USD asset

Copied from the local `quad/unitree_rl_lab/unitree_model/Go2` asset on 2026-10-25.
Upstream: https://github.com/unitreerobotics/unitree_model/tree/main/Go2
Robot model reference: https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go2_description

The complete USD layer hierarchy is preserved under `usd/`; keep the root and
`configuration/` files together. Training uses this repository's existing
`UNITREE_GO2_CFG` and Go2HV actuator implementation. No sibling repository is
needed at runtime. These are upstream robot assets, not newly authored geometry.
