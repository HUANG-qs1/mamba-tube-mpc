# 轮式移动机器人在线扰动自适应管 MPC（Mamba-Tube-MPC）

> **English version follows the Chinese section / 英文版见下文**

本仓库为以下论文的官方代码与结果存档：

> 黄清苏、朱飞、朱晓飞，《轮式移动机器人在线扰动自适应管模型预测控制：预测器无关整定律兼顾经验标定误差裕量与跟踪精度》，投稿 *Mechatronics*，2026。

## 方法概览

经典管 MPC 离线固定管宽：窄管被反复突破，保守管则退化为标准 MPC。本项目根据扰动诱发误差演化的预报**在线**设定管宽，隐藏在**预测器无关接口**之后：

- 一个紧凑 **Mamba** 网络把最近 100 个跟踪误差样本映射为未来 10 步误差预报。扰动本身不可测量，误差序列是其穿过闭环的可观测痕迹。Mamba 在开环精度、跨轨迹泛化与序列长度扩展上是推荐选项，但它只是可插拔选项之一，并非闭环收益的来源。
- 整定律 `w = max(w_min, w_base + κ · max_i ‖ê_i‖)` 在每个控制周期把预报转换为管半宽（`w_base = 0.02 m`，`w_min = 0.08 m`，`κ = 1.0`）。经验标定的误差包络是这条律的性质，而非预报器表达能力的性质。
- MPC 优化保持标称运动学模型（CasADi + IPOPT，10 Hz 下时域 N = 10）；学习组件只重塑鲁棒裕量。

## 关键结果（仿真）

| 结果 | 数值 |
|---|---|
| 开环预测误差（对骨架匹配 LSTM） | 测试 MSE **低 29.3%**，参数少 11.7%（236,436 vs 267,668），held-out 螺旋轨迹族 |
| 标定边界覆盖（κ = 1.0） | **100%**，平均管宽 0.090 m 全场最紧；κ < 1 欠标定时仅 85.5% |
| 预测器无关性（iid / AR(1) / 双通道扰动） | 免训练持续性/滑动平均与 Mamba 闭环统计等价（配对 TOST，±1 mm）；双通道下不重训仍成立（+0.26 / +1.13 mm） |
| 预测器域内重训（fine-tune / 从头重训 / 数据规模律 5→40 条滚动） | 开环误差降至约一半，**闭环 RMSE 钉在 0.0679–0.0681 m 不变** |
| split-conformal 交叉验证（α=0.05，20 校准种子） | 同等 100% 覆盖下，自适应管 0.090 m 比 conformal 管 0.148 m **紧 39%** |
| 训练-部署控制器失配（DAgger 式对照） | 标准 MPC 语料 vs 管闭环语料训练，部署闭环配对差 **+0.06 mm**（TOST p < 10⁻¹⁵） |
| 执行器饱和/加速度限制应力（0.50 m/s，饱和触发约 21.6% 步） | 自适应管自身覆盖率 **100%**，固定 0.15 m 管降至 **97.4%**（最差种子 93.6%） |
| 零样本未见轨迹 | 方形 0.1958 m（−15.4% vs 标准 MPC）；利萨茹 0.0902 m（−9.8%），均 p < 0.001 |
| 计时（双口径） | 组件基准：MPC 持久求解器 **4.5 ms**、Mamba GPU 1.0 ms、EKF 0.026 ms；保守在环（每次重建求解器）**19–24 ms**；10 Hz 下裕量约 4–5 倍 |
| 长时域 | **900 s**（名义时长 9 倍）三个时间段覆盖率全 100%，无可测漂移 |
| 非理想条件 | 延迟/丢包/噪声等八种注入条件下平缓、可测退化，延迟最敏感（相位机制） |

闭环结果为 10 种子均值（每运行预热后 800 步）；成对比较用 Welch t 检验与 Cohen's d；"无差异"主张一律用 ±1 mm 界值的配对 TOST（预注册附录 A 确认集与事后探索集严格分离）；两条核心朴素预测器比较扩展至 30 个配对种子。

## 仓库结构

```
├── code/            # 闭环仿真库：sim_core_np.py（机器人模型、CasADi/IPOPT 管 MPC、EKF、整定律、指标；
│                    #   run_sim 模式：mamba / lstm / fixed / ekf / standard / model）
│                    #   mamba_predictor.py、lstm_predictor.py、pub_style.py
├── training/        # 语料采集与预测器训练：collect_data_v3.py、train_mamba_v3.py、train_lstm.py、exp23_train_all.py
├── experiments/     # 实验驱动脚本（主协议 + 事后研究 + 评审实验 exp21–23；exp24–31 协议见 EXPERIMENTS.md）
├── figures/         # 论文全部图的生成脚本
├── models/          # 训练权重与归一化统计（论文全部闭环结果用 best_mamba_v3.pt / best_lstm_model.pt / norm_params_v3.npz）
├── results/         # 论文每个数字背后的 .npz 存档
├── REPRODUCE.md     # 论文条目 → 脚本 → 存档对照表
├── EXPERIMENTS.md   # 评审驱动实验索引（exp21–exp31）
└── README.md
```

脚本 `import sim_core_np` 并期望权重/归一化文件在工作目录，运行前把 `code/` 与 `models/` 内容放到同一目录（或 PYTHONPATH）：

```bash
mkdir run && cp code/* models/* run/ && cd run
python ../experiments/rerun_no_penalty.py     # 主消融（表 3）
python ../experiments/exp21_dual_channel.py   # 双通道预测器无关（表 12）
```

## 环境

- Python 3.10
- PyTorch 2.4.1+cu121，mamba-ssm 2.3.0
- CasADi 3.7.2（含 IPOPT）
- NumPy 2.2.6，SciPy 1.15.3

参考平台：Lenovo Legion Y7000P IRH8（i7-13700H，RTX 4060 Laptop 8 GB），WSL2 Ubuntu。论文全部延迟数字指该机。

## 数据说明

训练语料（training_data_*.npz，40–260 MB）因 GitHub 单文件限制不入库，用 `training/collect_data_v3.py` 重新生成。论文报告的每个数字均有 `results/` 下存档或固定种子脚本支撑。

## 引用与许可

论文审稿中，引用条目将在发表后补充。代码以 [MIT License](LICENSE) 发布。

---

# Online Disturbance-Adaptive Tube MPC of WMR: A Predictor-Agnostic Tuning Law Trading Calibrated Error Envelope Against Tracking Accuracy

Official code and result archive for the paper:

> Q. Huang, F. Zhu, X. Zhu, "Online disturbance-adaptive tube model predictive control for wheeled mobile robots: a predictor-agnostic tuning law trading calibrated error envelope against tracking accuracy," submitted to *Mechatronics*, 2026.

## Overview

Classical tube MPC fixes the tube width offline: a narrow tube is violated repeatedly, while a conservative one degenerates toward standard MPC. This project sizes the tube **online** from a forecast of the disturbance-induced error evolution, behind a **predictor-agnostic interface**:

- A compact **Mamba** network maps the 100 most recent tracking-error samples to a 10-step forecast of the error evolution. The disturbance itself is unmeasurable; the error sequence is its observable trace through the closed loop. Mamba is the recommended predictor on open-loop accuracy, cross-trajectory generalization, and sequence-length scaling — but it is one pluggable option, not the source of the closed-loop benefit.
- A sizing law `w = max(w_min, w_base + κ · max_i ‖ê_i‖)` converts the forecast into the tube half-width at every control cycle (`w_base = 0.02 m`, `w_min = 0.08 m`, `κ = 1.0`). The calibrated envelope is a property of this law, not of the forecaster's expressive power.
- The MPC optimization keeps the nominal kinematic model (CasADi + IPOPT, horizon N = 10 at 10 Hz); the learned component reshapes the robustness margin only.

## Key results (simulation)

| Result | Value |
|---|---|
| Open-loop prediction error vs. matched LSTM | **−29.3%** test MSE, **11.7% fewer parameters** (236,436 vs. 267,668), held-out spiral family |
| Calibrated boundary coverage (κ = 1.0) | **100%** at the tightest mean tube width (0.090 m); under-calibrated κ < 1 retains only 85.5% |
| Predictor-agnostic (iid / AR(1) / dual-channel) | training-free persistence / moving-average match Mamba in closed loop (paired TOST, ±1 mm margin); holds without retraining under two-channel disturbances (+0.26 / +1.13 mm) |
| In-domain retraining (fine-tune / from-scratch / data scaling 5→40 rollouts) | open-loop error roughly halves; **closed-loop RMSE pinned at 0.0679–0.0681 m** |
| Split-conformal cross-check (α = 0.05, 20 calibration seeds) | at equal 100% coverage, adaptive tube 0.090 m vs. conformal 0.148 m — **39% tighter** |
| Train/deploy controller mismatch (DAgger-style control) | standard-MPC vs. tube-closed-loop corpora: deployed paired RMSE diff **+0.06 mm** (TOST p < 10⁻¹⁵) |
| Actuator saturation / rate-limit stress (0.50 m/s; saturation binds ~21.6% of steps) | adaptive tube self-coverage **100%** vs. fixed 0.15 m tube **97.4%** (worst seed 93.6%) |
| Zero-shot RMSE on unseen paths | square 0.1958 m (−15.4% vs. standard MPC); Lissajous 0.0902 m (−9.8%); both p < 0.001 |
| Timing (two protocols) | component benchmark: MPC 4.5 ms persistent solver, Mamba 1.0 ms GPU, EKF 0.026 ms; conservative in-loop (solver rebuilt per call) 19–24 ms; ~4–5× margin at 10 Hz |
| Long horizon | **900 s** (9× nominal): 100% coverage in every time third, no measurable drift |
| Non-ideal conditions | graceful, measurable degradation under eight injected conditions; latency hurts most (phase mechanism) |

All closed-loop results are averaged over 10 seeds (800 evaluated steps per run after warm-up); pairwise comparisons use Welch's t-test with Cohen's d; every "no-difference" claim uses a paired TOST at a pre-stated ±1 mm margin (the pre-specified confirmatory set of Appendix A is never mixed with post-hoc exploratory analyses). The two central naive-predictor comparisons are extended to 30 paired seeds.

## Repository structure

See the Chinese section above (identical layout). Scripts import `sim_core_np` and expect the weight/normalization files in the working directory:

```bash
mkdir run && cp code/* models/* run/ && cd run
python ../experiments/rerun_no_penalty.py     # main ablation (Table 3)
python ../experiments/exp21_dual_channel.py   # dual-channel predictor-agnostic (Table 12)
```

## Environment

- Python 3.10; PyTorch 2.4.1+cu121; mamba-ssm 2.3.0; CasADi 3.7.2 with IPOPT; NumPy 2.2.6; SciPy 1.15.3
- Reference platform: Lenovo Legion Y7000P IRH8 (i7-13700H, RTX 4060 Laptop 8 GB), Ubuntu under WSL2. All latency figures refer to this machine.

## Data policy

Training corpora (`training_data_*.npz`, 40–260 MB) are NOT included (GitHub file-size limit); regenerate with `training/collect_data_v3.py`. Every reported number is backed by an archive under `results/` or by fixed-seed scripts.

## Citation & License

The paper is under review; citation details will be added upon publication. Released under the [MIT License](LICENSE).
