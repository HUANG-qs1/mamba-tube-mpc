# Mamba-Based Disturbance Prediction for Adaptive Tube MPC of Wheeled Mobile Robots

Official code repository for the paper:

> Q. Huang, F. Zhu, X. Zhu, "Mamba-Based Disturbance Prediction for Adaptive Tube Model Predictive Control of Wheeled Mobile Robots," submitted to *Mechatronics*, 2026.

## Overview

Classical tube MPC fixes the tube width offline: a narrow tube is violated repeatedly, while a conservative one degenerates toward standard MPC. This project sizes the tube **online** from a learned forecast of the disturbance-induced error evolution:

- A compact **Mamba** network maps the 100 most recent tracking-error samples to a 10-step forecast of the error evolution. The disturbance itself is unmeasurable; the error sequence is its observable trace through the closed loop.
- A sizing law `w = max(w_min, w_base + κ · max_i ‖ê_i‖)` converts the forecast into the tube half-width at every control cycle (`w_base = 0.02 m`, `w_min = 0.08 m`, `κ = 1.0`).
- The MPC optimization keeps the nominal kinematic model (CasADi + IPOPT, horizon N = 10 at 10 Hz); the learned component reshapes the robustness margin only.

## Key results (simulation)

| Result | Value |
|---|---|
| Open-loop prediction error vs. matched LSTM | **−29.3%** test MSE, with **11.7% fewer parameters** (236,436 vs. 267,668), on a held-out spiral trajectory family |
| Calibrated boundary coverage (κ = 1.0) | **100%** at the tightest mean tube width (0.090 m); under-calibrated κ < 1 retains only 85.5% |
| Zero-shot RMSE on unseen square path | **0.1958 m** (−15.4% vs. standard MPC, p < 0.001) |
| Zero-shot RMSE on unseen Lissajous path | **0.0902 m** (−9.8% vs. standard MPC, p < 0.001) |
| Full pipeline cycle time | **3.7 ms** — a 27× real-time margin at 10 Hz |
| Non-ideal conditions | graceful degradation under nine injected conditions (delay, packet loss, velocity scaling, measurement noise) |

All closed-loop results are averaged over 10 seeds (800 evaluated steps per run after warm-up); pairwise comparisons use Welch's t-test with Cohen's d effect sizes.

## Repository structure

Code and data-generation scripts are being uploaded. Planned layout:

```
├── collect_data_v3.py        # v3 training corpus collection (3 path families × 5 disturbance profiles × 20 episodes)
├── collect_data_v4*.py       # v4 corpus (30-step prediction target)
├── generalization_spiral.py  # held-out spiral test-set generation (open-loop evaluation)
├── train_mamba_v3.py         # main Mamba predictor training recipe
├── exp23_train_all.py        # prediction-horizon & data-scale scans (v4, fixed seed 42)
├── sim_core_np.py            # closed-loop simulation core (modes: mamba / lstm / fixed / ekf / standard / model)
└── exp*.py                   # closed-loop experiments (scenarios, ablation, κ sweep, generalization, non-ideal conditions)
```

## Environment

- Python 3.10
- PyTorch 2.4.1+cu121, mamba-ssm 2.3.0
- CasADi 3.7.2 with IPOPT
- NumPy 2.2.6, SciPy 1.15.3

Reference platform: Lenovo Legion Y7000P IRH8 (i7-13700H, RTX 4060 Laptop 8 GB), Ubuntu under WSL2. All latency figures in the paper refer to this machine.

## Reproducing the results

1. Install the dependencies above (mamba-ssm requires a CUDA-capable PyTorch build).
2. Generate the training corpora with the data-collection scripts.
3. Train the predictor (`train_mamba_v3.py`; design scans in `exp23_train_all.py`).
4. Run the closed-loop experiments through `sim_core_np.py` and the `exp*.py` drivers; seeds 0–9 reproduce the statistics reported in the paper.

## Citation

If you use this code, please cite our paper (citation details will be added upon publication).

## License

This project is released under the [MIT License](LICENSE).
