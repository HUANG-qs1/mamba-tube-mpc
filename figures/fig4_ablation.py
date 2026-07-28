# -*- coding: utf-8 -*-
"""fig4_ablation.py - Fig.4 v2: ablation A-E, RMSE (dot-whisker) + violations (bar)"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pub_style as ps

ps.apply_style()

r = np.load("rerun_no_penalty_results.npz", allow_pickle=True)["results"].tolist()
while isinstance(r, list):
    r = r[0]

COLS = [
    ("A (ours)\nMamba+Adap", "ablation_A_mamba_adaptive",  ps.C["blue"],   True),
    ("B/C\nFixed 0.15",      "ablation_B_fixed_015",       ps.C["red"],    False),
    ("D\nLSTM+Adap",         "ablation_D_lstm_adaptive",   ps.C["orange"], False),
    ("E\nEKF+Fixed",         "ablation_E_ekf_fixed_015",   ps.C["green"],  False),
]
XS = np.arange(len(COLS))
RMSE_A = r["ablation_A_mamba_adaptive"]["rmse"]

fig, (axa, axb) = plt.subplots(2, 1, figsize=ps.figsize(1.0, 1.15), sharex=True)

# ---- (a) RMSE: dot + 1std whisker (10 seeds) ----
axa.axhline(RMSE_A, ls=":", lw=0.6, color="0.55", zorder=1)
axa.text(3.55, RMSE_A, "A (ours)", fontsize=6, color="0.45", ha="right", va="bottom")
for i, (_, key, col, filled) in enumerate(COLS):
    m, s = r[key]["rmse"], r[key]["rmse_std"]
    axa.errorbar(i, m, yerr=s, fmt="o", ms=4.5, mew=1.0, mec=col,
                 mfc=col if filled else "white", ecolor=col, elinewidth=0.9,
                 capsize=2.5, zorder=3)
    axa.annotate(f"{m:.4f}", (i, m - s), textcoords="offset points",
                 xytext=(0, -11), ha="center", fontsize=6.5, color=col)
for i, mark in [(1, "**"), (2, "n.s."), (3, r"$\dagger\dagger$")]:
    m, s = r[COLS[i][1]]["rmse"], r[COLS[i][1]]["rmse_std"]
    axa.annotate(mark, (i, m + s), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=7, color="0.25")
axa.text(0.02, 0.04, "B/C identical: with a fixed tube the predictor is inactive.\n"
                     "vs A (Welch, n=10): ** p<0.01; n.s. = not significant;\n"
                     r"$\dagger\dagger$ p<0.01: E lower RMSE, but most violations (b).",
         transform=axb.transAxes if False else axa.transAxes, fontsize=6, color="0.35", va="bottom")
axa.set_ylabel("RMSE (m)")
axa.set_ylim(0.0650, 0.0688)
axa.set_yticks([0.065, 0.066, 0.067, 0.068])
axa.text(-0.14, 1.03, "(a)", transform=axa.transAxes, fontweight="bold")

# ---- (b) tube violations: bar ----
viol = [r[key]["viol"] for _, key, _, _ in COLS]
cols = [c[2] for c in COLS]
axb.bar(XS, viol, width=0.55, color=cols, edgecolor="white", lw=0.5, zorder=3)
for i, v in enumerate(viol):
    axb.annotate(f"{v:g}", (i, v), textcoords="offset points",
                 xytext=(0, 3), ha="center", fontsize=6.5, color=cols[i])
axb.text(0.02, 0.95, "mean over 10 seeds (800 steps per run)",
         transform=axb.transAxes, fontsize=6, color="0.35", va="top")
axb.set_ylabel("Tube violations (count)")
axb.set_ylim(0, 4.8)
axb.set_xticks(XS)
axb.set_xticklabels([c[0] for c in COLS], fontsize=6.5)
axb.set_xlim(-0.6, 3.6)
axb.text(-0.14, 1.03, "(b)", transform=axb.transAxes, fontweight="bold")

fig.subplots_adjust(hspace=0.16)
os.makedirs("figures", exist_ok=True)
ps.save_fig(fig, "figures/fig4_ablation")
