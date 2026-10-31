# AME Locomotion：Go2 四足机器人训练

[English](README.md)

项目在 Isaac Lab 中实现 AME 地形感知运动控制，默认机器人为 Unitree Go2（12 自由度），保留 G1 29 自由度任务和模型。
AME 使用 xyz 地图 CNN、本体状态条件化的注意力和 Actor/Critic；`agent.policy.attach_global=true` 启用 AME2 全局上下文分支。

## 环境

本机使用 `conda activate env_ame`，配置说明见 [ENV_AME.md](ENV_AME.md)。其他兼容 Isaac Lab 环境需要安装两个本地包：

```bash
python -m pip install -e source/ame_locomotion
python -m pip install -e rsl_rl
```

项目已包含 Go2 USD 模型，运行时不依赖参考用的 DreamWaQ 或 Unitree 仓库。

## 两阶段：一个源码开关

修改 [stage_cfg.py](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/go2/stage_cfg.py) 中的 `FINETUNE`，然后重新启动训练或回放：

| 开关 | 地形与感知 | 默认日志目录 |
|---|---|---|
| `False` | 原 AME 基础混合地形，含 20% 踏石、20% 间隙；理想感知 | `logs/rsl_rl/go2_ame` |
| `True` | 原 AME 微调地形，含桩阵、间隙、石桥和横栏；加入噪声、质量扰动和推扰 | `logs/rsl_rl/go2_ame_finetune` |

两阶段直接复制原仓库地形配置，保留类型、比例、尺寸及难度课程。第一阶段已经包含离散地形；当前不增加仅粗糙地形的阶段。
两阶段奖励、命令范围与 PPO 默认参数一致。当前纯 RL 实验使用世界水平足滑、连续地形步态和离散地形足高约束。来源和许可证随奖励模块保存。

第一阶段保持 `FINETUNE=False`：

```powershell
conda activate env_ame
python scripts/rsl_rl/train.py --num_envs 64 --headless --logger tensorboard
# 或 .\run_train.ps1；Linux 使用 bash run_train.sh
```

第二阶段改为 `FINETUNE=True`，指定第一阶段模型：

```powershell
python scripts/rsl_rl/train.py --num_envs 64 --headless --logger tensorboard --finetune_from logs/rsl_rl/go2_ame/<run>/model_9999.pt
```

`--finetune_from` 加载权重并重置优化器与迭代计数；`--resume` 恢复完整训练状态，二者互斥。第二阶段须提供其中一种加载方式。
训练任务始终为 `AME-Go2-v0`。源码默认 1024 个环境、10000 次更新；快速启动脚本使用 64 个环境并透传参数。
`--max_iterations` 表示本次启动追加执行的 PPO 更新次数。模型文件名使用实际保存的迭代编号。

```powershell
python scripts/rsl_rl/train.py --resume --checkpoint <断点模型路径> --num_envs 64 --headless
```

## 回放和注意力

回放任务为 `AME-Go2-Play-v0`，读取同一个阶段开关，保留对应地形并关闭噪声、动力学随机化及推扰。
省略模型参数时查找当前阶段日志目录的最新模型；也可显式指定：

```powershell
python scripts/rsl_rl/play.py --num_envs 1 --checkpoint <Go2模型路径>
python scripts/rsl_rl/play.py --num_envs 1 --checkpoint <Go2模型路径> --headless --video --video_length 100 --save_attention_weights --vis_attention
# 或 .\run_play.ps1 --checkpoint <Go2模型路径>
```

视频写入模型目录的 `videos/play/`，注意力写入同目录的 `attention_weights.npy`。
`--vis_attention` 与 `--save_attention_weights` 可以独立启用；`--max_steps` 可以限制无视频回放步数。

```powershell
python scripts/plot_attention.py <模型目录>/attention_weights.npy --max_frames 10
```

训练 AME2 时使用 `agent.policy.attach_global=true`，回放同一模型时使用相同设置。

## 控制和观测

- 关节和动作顺序：FR、FL、RR、RL，每腿依次 hip、thigh、calf；动作缩放 0.25。
- 控制频率 50 Hz，物理步长 0.005 s；训练回合 20 秒。
- 地图范围 1.6×1.0 m，分辨率 5 cm，尺寸 33×21×3；地图位于观测末尾。
- 策略和价值观测分别为 2124、2127 维，动作 12 维；CNN 下采样后有 187 个注意力位置。

## 验证

本地测试脚本、验收记录及短测产物统一放在 `test/`，该目录已加入 `.gitignore`。

```powershell
python test/test_go2.py
python test/smoke_go2.py --terrain hf_steppingstones
python test/smoke_go2.py --terrain hf_gaps
```

后两项在 `FINETUNE=False` 下运行，检查真实资产、离散地形扫描和有界环境步进。实际结果见 [GO2_VALIDATION.md](test/GO2_VALIDATION.md)。

## G1 兼容入口

`pretrained/ame1.pt` 和 `pretrained/ame2.pt` 是 G1 模型。Go2 需要重新训练。

```powershell
python scripts/rsl_rl/train.py --task AME-G1-29DOF-v0 --num_envs 64 --headless
python scripts/rsl_rl/play.py --task AME-G1-29DOF-Play-v0 --checkpoint pretrained/ame1.pt --num_envs 1
python scripts/rsl_rl/play.py --task AME-G1-29DOF-Play-v0 --checkpoint pretrained/ame2.pt --num_envs 1 agent.policy.attach_global=true
```

Go2 的阶段开关独立于 G1；G1 继续使用自身配置文件中的 `FINETUNE`。

## Go2 奖励与通用地形门控

两阶段共用纯 RL 奖励公式与基础权重，通过外部门控控制风格约束的地形、等级和训练时间。
这是实验配置，尚未证明能够改善单腿悬空。

快速调权重时，直接修改 `go2/env_cfg.py` 中 `Go2AMEEnvCfg.__post_init__` 的
`self.rewards.<奖励名>.weight` 分组配置，两个阶段均读取这里的最终值。
`go2/rewards_cfg.py` 保留函数、参数和默认值；地形门控仍会生效，权重为零表示关闭该项。

| 奖励项 | 连续地形 | 离散地形 |
|---|---:|---:|
| 线速度 / 转向跟踪 | 2 / 2 | 2 / 2 |
| 垂直速度平方 | −1 | −0.25 |
| 一阶动作变化平方 | −0.01 | −0.01 |
| 关节位置软限位越限 | −1 | −1 |
| 高度、关节镜像 | 0 | 0 |
| 历史腾空/接触时长方差 | −0.1 | −0.1 |
| `feet_air_time`（落地触发，阈值 0.5 s） | 0.25 | 0.25 |
| `feet_slide`：接触脚的世界水平速度 | −0.1 | −0.1 |
| `feet_height_body` | 0 | −0.5 |
| `feet_gait`，第一阶段等级≤5且迭代<3000 | 0.2 | 0.2 |
| 非超时终止 | −10 | −10 |
| `feet_air_time_excess_penalty` | −0.1 | −0.1 |

权重参考 **G1 第一阶段最终生效配置**（包含 `__post_init__` 覆盖）及
[AME 论文表 2 的 ANYmal-D 配置](https://arxiv.org/html/2506.09588v1)：
线速度跟踪和水平足滑采用 G1 第一阶段的 `2.0`、`−0.1`，垂直速度和关节限位参考论文系数。
这是适配实验，不是论文复现：G1 跟踪使用 `std=0.25`，Go2 保留 `0.5`；
论文动作变化使用目标关节角，本项目使用缩放前动作，动作缩放为 `0.25`，
当前动作变化权重保留 `−0.01`。机械功率也不等同于力矩平方。
转向跟踪保留 `2`、终止保留 `−10`，不直接引入论文的 `−200` 终止或更强的足滑惩罚。
gait、持续腾空约束和地形门控保持当前设置；需同时验证逐足参与和离散地形通过性。

`feet_slide` 通过薄封装调用 G1 原有 `mdp.feet_slide`：世界水平足速、历史接触判定（1 N），
不减去机身速度，没有直立门控。TensorBoard 奖励名称为 `Episode_Reward/feet_slide`；
之前使用 `feet_slippage` 的日志保留原名称。独立的世界三维 `feet_slippage_3d` 项仍关闭。
gait 配对为 FL–RR、FR–RL；实现直接用 `std` 除平方时长误差，因此填 `0.1`，不是 `sqrt(0.1)`。
离散地形足高公式和目标保持原值。

`feet_air_time` 使用 robot_lab 的四足落地公式：
`sum((last_air_time - 0.5) * first_contact)`，作用于所有地形，要求指令模长大于 0.1
（含转向）且机身直立时生效。摆动超过 0.5 秒后落地为正，短于该时间落地为负；
0.5 秒是奖励阈值，不是目标摆动时长。一直悬空但不落地的脚不得分。
该项不指定对角步态，也不能单独保证四足都参与行走。

持续腾空项使用 `sum(clamp((current_air_time - 0.6) / 0.4, 0, 1)**2)`：
0.6 秒后开始计罚，1 秒达到上限；要求平移指令大于 0.1 m/s 或转向指令大于 0.2 rad/s，
至少两脚支撑且机身直立。直接复用接触传感器计时和部分重置。
地形、命令、网络与 PPO 不变，不引入 AMP 或相位输入。

`Go2AMEEnvCfg.reward_gates` 统一替代旧 `terrain_mask` 参数及模块。
最终奖励为“原始奖励 × 基础权重 × 总倍率 × 类别倍率 × 等级倍率 × 时间倍率 × dt”；
原有指令、直立、落地及支撑条件保留在奖励公式内。

- `gait_guidance`：gait、mirror、髋关节偏差；第一阶段所有地形等级≤5且迭代<3000生效，第二阶段关闭。
- `air_contact`：落地腾空奖励、时长方差、持续腾空惩罚；两阶段所有地形持续启用。
- `default_style`：其余风格项，以及垂直速度、角速度和姿态约束；默认倍率1。
  单项覆盖：机身高度仅连续地形、足端高度仅离散地形、垂直速度在离散地形乘0.25（未知类别乘1）。

配置集中在 `reward_gates.groups`、`term_groups`、`overrides`。每组支持 `scale`、
`terrain_scales`、`max_level`、`above_level_scale`、`end_iteration`、`after_iteration_scale`。
单项未填写字段继承组值，显式 `None` 解除等级/时间限制；类别字典按键合并，具体地形名称优先：

```python
self.reward_gates.groups["gait_guidance"].max_level = 5
self.reward_gates.overrides["joint_mirror"] = {"end_iteration": 1500}
```

Hydra 可用 `env.reward_gates.enabled=false` 将所有外部门控倍率设为1；基础权重为零的项仍关闭，
不会恢复旧掩码。最终配置写入 `env.yaml`，在环境启动时验证，包括零权重项的配置。
旧 `terrain_mask`/`terrain_masks` 参数不再支持。

连续地形名单位于 `reward_gates.continuous_terrain_names`，默认包含平地、台阶、障碍块、粗糙和坡面。
分类遵循课程列分配，等级动态读取；gap 起始平台仍属离散地形。随机混合类别标为未知；
同类混合和单一地形可可靠分类，但随机行号不代表难度等级。纯平面等级为0，未知等级关闭带等级限制的项；
未知类别关闭两项高度约束。精确地形名称已知时仍可使用名称覆盖。修改布局或配置后重新启动环境。

Runner 在 rollout 前传递日志中的 PPO 迭代编号；恢复训练沿用现有编号语义（保存编号会再次使用），
权重初始化从0开始，play读取 checkpoint 进度。门控状态不进入模型权重，G1 行为保持原样。
TensorBoard `RewardGates/<奖励名>/` 记录类别/等级/时间/最终倍率、启用比例、基础/有效权重及
每秒加权奖励；统计来自最近一次奖励控制步，不是回合平均，`RewardGates/last_control_step` 标明采样步。
该日志独立于 AME diagnostics 开关；门控和注意力数值本身不能证明运动效果改善。

第一阶段保持 `FINETUNE=False`，本轮建议从头启动：

```powershell
.\run_train.ps1 --seed 42 --run_name pure_rl_g1_paper
```

保留旧模型作为对照；固定平地指令 0.3、0.5、0.8 m/s，检查逐足载荷参与、最长腾空、
对角腿时序和实际速度，并单独评估踏石/gap 通过性。gait 分数提高不代表四足有效承重。
这是组合实验，不能单独归因到某一个奖励项。
需要重启进程才能读取新配置；checkpoint 不会自动恢复旧的环境/奖励设置。
第二阶段仍须提供初始化模型或断点恢复。

## AME TensorBoard 分模块诊断

Go2 两阶段默认开启，G1 和其他任务默认关闭。正常训练时使用 `--logger tensorboard`，
再运行 `tensorboard --logdir logs/rsl_rl/go2_ame`；第二阶段目录换成 `go2_ame_finetune`。
已经运行的训练进程需要重新启动才能使用新代码。关闭或调整采样周期：

```powershell
python scripts/rsl_rl/train.py --num_envs 64 --headless --logger tensorboard agent.diagnostics.enabled=false
python scripts/rsl_rl/train.py --num_envs 64 --headless --logger tensorboard agent.diagnostics.scalar_interval=10 agent.diagnostics.attribution_interval=50 agent.diagnostics.image_interval=200 agent.diagnostics.max_images=4
```

周期按 PPO 更新次数计算，第 0 次也记录；恢复训练沿用恢复的迭代编号。
CNN、MHA 没有独立损失。梯度归因分别使用实际加权的策略目标
`surrogate_loss - entropy_coef * entropy` 和价值目标 `value_loss_coef * value_loss`。
总梯度来自原有反向传播，包含实际启用的其他策略网络损失。

| TensorBoard 分类 | 记录内容 |
|---|---|
| `AME/PPO` | 加权策略/价值目标、策略网络总损失、更新前/更新期间 KL、clip fraction、解释方差、梯度裁剪比例 |
| `AME/Modules/<模块>` | 裁剪前梯度 L2/RMS、参数 L2、实际参数更新相对幅度 |
| `AME/Attribution/<模块>` | 策略/价值梯度范数、余弦相似度、价值/策略范数比、有效性与计算图连通标志 |
| `AME/Features` | 分别记录 Actor/Critic 的 CNN、Query、MHA 均值/标准差/RMS/非有限值比例，以及 ReLU 零值比例和 BN running variance |
| `AME/Attention` | 各 head 归一化熵/最大权重、head 间余弦相似度、Actor/Critic JS 散度、rollout 高度图和注意力图 |
| `AME/Terrain/<地形>` | 线速度误差、沿指令方向速度、停滞比例、非超时终止比例、地形等级及样本数 |

模块/特征统计和梯度归因取记录轮次的第一个 minibatch。PPO 指标汇总该轮 minibatch，
其中更新前 KL 取首个 minibatch，反映优化器更新前已有的 rollout/更新前向差异，包含训练模式 BN 的影响。
解释方差根据这些 minibatch 的 returns 和当前预测计算。梯度裁剪比例表示超出 `max_grad_norm`
的 minibatch 比例，`gradient_clip_scale` 是平均裁剪系数。没有梯度、零范数会记录标志，
未定义的余弦和范数比不写为零。AME2 还记录全局编码器和 Query 融合投影。

分地形指标持续累计，每次标量日志后清空窗口。停滞定义为平移指令大于 0.2 m/s、
沿指令方向速度低于 0.1 m/s 且持续至少 1 秒，比例分母是有移动指令的采样数。
终止率分母是已知超时状态的结束回合数，分子排除超时；没有结束回合时只记录计数。
课程地形严格按生成器列分配命名，随机生成且无法确定类型时记作 `unknown`。

图像优先选择踏石、间隙及高度变化明显的样本，每张图依次排列输入高度、平均注意力叠加图、
各 head 热图。x 向右、y 向上；高度配色范围为 −1.2～0 m，注意力配色为均匀权重的 0～4 倍。
图像复用实际 rollout 前向，特征/注意力标量来自更新 minibatch，二者不是同一批样本。
逐 head 读取采用 [PyTorch MHA 接口](https://docs.pytorch.org/docs/2.7/generated/torch.nn.MultiheadAttention.html)，
回放返回的平均注意力形状保持不变。

排查离散地形停滞时，先看该地形停滞/速度误差和样本数，再检查 CNN/MHA 是否同时缺少梯度和参数更新，
共享模块是否长期由价值梯度主导且方向冲突，以及特征方差是否退化。更新前 KL 偏高提示已有的
rollout/更新计算差异；更新期间 KL 和裁剪比例一起升高则提示优化不稳定。
注意力集中、head 趋同或梯度冲突都只是诊断证据，不能单独证明训练失败的原因。

诊断不额外前向、不切换模式、不使用额外随机采样，不修改奖励、BN 行为、优化器更新及 checkpoint 键。
多卡训练会明确警告并关闭新增诊断。测试脚本与产物位于忽略的 `test/`：
`test/test_ame_diagnostics.py`、`test/diagnostics/`，实际验证与性能结果见
[AME_DIAGNOSTICS_VALIDATION.md](test/AME_DIAGNOSTICS_VALIDATION.md)。
平均额外开销目标约为 5%，记录归因和图像的单次更新会有较明显峰值。
