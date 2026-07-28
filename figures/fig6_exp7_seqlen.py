# -*- coding: utf-8 -*-
"""fig6_exp7_seqlen.py - Fig.6: exp7 latency & Test MSE vs seq_len (v3)"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pub_style as ps

ps.apply_style()

def load_results(path):
    r = np.load(path, allow_pickle=True)["results"].tolist()
    while isinstance(r, list):
        r = r[0]
    return r

tr = load_results("exp7_train_results.npz")
ev = load_results("exp7_eval_results.npz")

SEQS = [20, 100, 200]
lat_m = [ev[f"mamba_seq{s}"]["infer_ms"] for s in SEQS]
lat_l = [ev[f"lstm_seq{s}"]["infer_ms"] for s in SEQS]
mse_m = [tr[f"mamba_seq{s}"] for s in SEQS]
mse_l = [tr[f"lstm_seq{s}"] for s in SEQS]

fig, (axa, axb) = plt.subplots(2, 1, figsize=ps.figsize(1.0, 1.15), sharex=True)

# ---- (a) inference latency ----
axa.plot(SEQS, lat_m, **ps.style_of("mamba"))
axa.plot(SEQS, lat_l, **ps.style_of("lstm"))
OFF_M = [(0, 7), (0, -12), (0, -12)]
OFF_L = [(0, -12), (0, 7), (0, 7)]
for x, y, (dx, dy) in zip(SEQS, lat_m, OFF_M):
    axa.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                 xytext=(dx, dy), ha="center", fontsize=6.5, color=ps.C["blue"])
for x, y, (dx, dy) in zip(SEQS, lat_l, OFF_L):
    axa.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                 xytext=(dx, dy), ha="center", fontsize=6.5, color=ps.C["orange"])
axa.set_ylabel("Inference latency (ms)")
axa.set_ylim(0.70, 2.20)
axa.legend(loc="lower right")
axa.text(-0.14, 1.03, "(a)", transform=axa.transAxes, fontweight="bold")

# ---- (b) Test MSE ----
axb.plot(SEQS, mse_m, **ps.style_of("mamba"))
axb.plot(SEQS, mse_l, **ps.style_of("lstm"))
OFF_B = [("left", (7, 0)), ("center", (0, -13)), ("right", (-7, 0))]
for x, ym, yl, (ha, (dx, dy)) in zip(SEQS, mse_m, mse_l, OFF_B):
    gap = (ym - yl) / yl * 100.0
    ytxt = min(ym, yl) if ha == "center" else (ym + yl) / 2
    axb.annotate(f"{gap:+.1f}%", (x, ytxt), textcoords="offset points",
                 xytext=(dx, dy), ha=ha, va="center", fontsize=6.5, color="0.25")
axb.text(0.03, 0.05, r"$\Delta$% = (Mamba $-$ LSTM) / LSTM",
         transform=axb.transAxes, fontsize=6, color="0.35", ha="left", va="bottom")
axb.set_ylabel("Test MSE (normalized)")
axb.set_xlabel("Sequence length $L$")
axb.set_xticks(SEQS)
axb.set_xlim(5, 215)
axb.set_ylim(0.0050, 0.0063)
axb.text(-0.14, 1.03, "(b)", transform=axb.transAxes, fontweight="bold")

fig.subplots_adjust(hspace=0.16)
os.makedirs("figures", exist_ok=True)
ps.save_fig(fig, "figures/fig6_exp7_seqlen")
