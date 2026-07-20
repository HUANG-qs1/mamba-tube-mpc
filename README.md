# Online Disturbance-Adaptive Tube MPC of WMR: A Predictor-Agnostic Framework Validated with a Mamba State-Space Predictor

Official code repository for the paper:

> Q. Huang, F. Zhu, X. Zhu, "Online Disturbance-Adaptive Tube Model Predictive Control of Wheeled Mobile Robots: A Predictor-Agnostic Framework Validated with a Mamba State-Space Predictor," submitted to *Mechatronics*, 2026.

## Overview

Classical tube MPC fixes the tube width offline: a narrow tube is violated repeatedly, while a conservative one degenerates toward standard MPC. This project sizes the tube **online** from a forecast of the disturbance-induced error evolution, behind a **predictor-agnostic interface**:

- A compact **Mamba** network maps the 100 most recent tracking-error samples to a 10-step forecast of the error evolution. The disturbance itself is unmeasurable; the error sequence is its observable trace through the closed loop. Mamba is the recommended predictor on open-loop accuracy, cross-trajectory generalization, and sequence-length scaling — but it is one pluggable option, not the source of the closed-loop benefit.
- A sizing law `w = max(w_min, w_base + κ · max_i ‖ê_i‖)` converts the forecast into the tube half-width at every control cycle (`w_base = 0.02 m`, `w_min = 0.08 m`, `κ = 1.0`). The calibrated envelope is a property of this law, not of the forecaster's expressive power.
- The MPC optimization keeps the nominal kinematic model (CasADi + IPOPT, horizon N = 10 at 10 Hz); the learned component reshapes the robustness margin only.

## Key results (simulation)

| Result | Value |
|---|---|
| Open-loop prediction error vs. matched LSTM | **−29.3%** test MSE, with **11.7% fewer parameters** (236,436 vs. 267,668), on a held-out spiral trajectory family |
| Calibrated boundary coverage (κ = 1.0) | **100%** at the tightest mean tube width (0.090 m); under-calibrated κ < 1 retains only 85.5% |
| Predictor-agnostic property (iid disturbance) | training-free persistence / moving-average forecasts match Mamba in closed loop across a three-fold amplitude range |
| Predictor-agnostic property (AR(1), φ = 0.95) | all four predictors statistically indistinguishable in closed loop; mechanism traced to the path-induced deterministic error component |
| Zero-shot RMSE on unseen square path | **0.1958 m** (−15.4% vs. standard MPC, p < 0.001) |
| Zero-shot RMSE on unseen Lissajous path | **0.0902 m** (−9.8% vs. standard MPC, p < 0.001) |
| Full pipeline cycle time | **3.7 ms** — a 27× real-time margin at 10 Hz |
| Non-ideal conditions | graceful degradation under nine injected conditions (delay, packet loss, velocity scaling, measurement noise) |

All closed-loop results are averaged over 10 seeds (800 evaluated steps per run after warm-up); pairwise comparisons use Welch's t-test with Cohen's d effect sizes. Post-hoc analyses (configurations F/G, the φ-sensitivity scan, the zero-shot extension) are reported as exploratory and are never mixed into the pre-specified confirmatory set of Appendix A. The two central naive-predictor comparisons are extended to 30 paired seeds with TOST equivalence testing at a pre-stated ±1 mm margin.

## Repository structure

```
├── code/                     # closed-loop simulation library
│   ├── sim_core_np.py        #   simulation core: WMR model, CasADi/IPOPT tube MPC,
│   │                         #   EKF, sizing law, metrics; run_sim modes:
│   │                         #   mamba / lstm / fixed / ekf / standard / model
│   ├── mamba_predictor.py    #   Mamba disturbance-predictor definition
│   ├── lstm_predictor.py     #   matched-skeleton LSTM baseline
│   └── pub_style.py          #   publication figure style helpers
├── experiments/              # post-hoc experiment drivers (Sections 5.3, 5.6, 5.9)
│   ├── exp11_naive_baseline.py   # F/G: persistence & moving-average through the same interface
│   ├── exp11b_naive_higcamp.py   # naive baselines at 2x/3x disturbance amplitude
│   ├── exp12_ar1_correlated.py   # AR(1) correlated disturbance, phi = 0.95, 4 predictors
│   ├── exp12b_whitening_check.py # mechanism diagnostic: disturbance vs. error autocorrelation
│   ├── exp13_ar1_phi_scan.py     # phi = 0.5 / 0.7 sensitivity check
│   ├── exp14_zeroshot_naive.py   # zero-shot generalization of training-free forecasters
│   ├── exp17_seeds30.py          # seeds 10-29 top-up (pooled to 30 paired seeds for TOST)
│   ├── exp18_longhorizon.py      # 300 s envelope-stability check
│   ├── exp19_pool30_verify.py    # pooled 30-seed Welch + TOST verification utility
│   └── exp20_tost_verify.py      # TOST reproduction utility for the AR(1) experiments
├── models/                   # trained weights and normalization statistics
│   ├── best_mamba_v3.pt      #   main Mamba predictor (all closed-loop results)
│   ├── best_lstm_model.pt    #   main LSTM baseline
│   ├── best_mamba_seq100.pt, best_mamba_seq200.pt,
│   ├── best_lstm_seq100.pt,  best_lstm_seq200.pt   # sequence-length study (Section 5.7)
│   ├── best_mamba_v4_pl5.pt ... best_mamba_v4_pl30.pt  # prediction-horizon scan (Section 5.2)
│   └── norm_params_v3.npz, norm_params_v4.npz,
│       norm_params_seq200.npz                       # input/output normalization statistics
├── LICENSE                   # MIT
└── README.md
```

The scripts import `sim_core_np` and expect the weight/normalization files in the working directory, so place the contents of `code/` and `models/` in one folder (or on `PYTHONPATH`) before running, e.g.:

```bash
mkdir run && cp code/* models/* run/ && cd run
python exp11_naive_baseline.py     # reproduces configurations F/G of Table III
```

## Environment

- Python 3.10
- PyTorch 2.4.1+cu121, mamba-ssm 2.3.0
- CasADi 3.7.2 with IPOPT
- NumPy 2.2.6, SciPy 1.15.3

Reference platform: Lenovo Legion Y7000P IRH8 (i7-13700H, RTX 4060 Laptop 8 GB), Ubuntu under WSL2. All latency figures in the paper refer to this machine.

## Reproducing the results

The repository reproduces the closed-loop experiments end-to-end: simulation core, trained weights, normalization statistics, and the experiment drivers are all included. Seeds 0–9 (and 10–29 for the TOST extension) reproduce the statistics reported in the paper for the post-hoc studies.

The training-data generation and predictor-training scripts are not included in this release; they will be added upon publication. The main closed-loop results (Tables I–VIII) are produced by the same `sim_core_np.py` core included here, driven by the scenario scripts of the main experimental campaign.

## Citation

If you use this code, please cite our paper (citation details will be added upon publication).

## License

This project is released under the [MIT License](LICENSE).
