# -*- coding: utf-8 -*-
"""fig8_exp6_traj.py - Fig.8 v2: exp6 generalization trajectories (2x2 composite)"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pub_style as ps

ps.apply_style()

d = np.load("fig8_trajectories.npz", allow_pickle=True)

def load(tr):
    ref = d[f"{tr}_ref"]
    sd = int(d[f"{tr}_seed"])
    errs = {"ours": d[f"{tr}_ours_err"], "standard": d[f"{tr}_std_err"], "ekf": d[f"{tr}_ekf_err"]}
    return ref, sd, errs

def style(key, **kw):
    s = ps.style_of(key)
    s["ms"] = 2.5
    s["markevery"] = 150
    s.update(kw)
    return s

fig, axes = plt.subplots(2, 2, figsize=ps.figsize(1.5, 1.02))
(axa, axb), (axc, axd) = axes

for ax, tr, tag in [(axa, "square", "(a)"), (axb, "spiral", "(b)"), (axc, "lissajous", "(c)")]:
    ref, sd, errs = load(tr)
    ax.plot(ref[:, 0], ref[:, 1], **style("ref", zorder=1))
    ax.plot(ref[:, 0] + errs["standard"][:, 0], ref[:, 1] + errs["standard"][:, 1], **style("standard", zorder=2))
    ax.plot(ref[:, 0] + errs["ekf"][:, 0], ref[:, 1] + errs["ekf"][:, 1], **style("ekf", zorder=3))
    ax.plot(ref[:, 0] + errs["ours"][:, 0], ref[:, 1] + errs["ours"][:, 1], **style("ours", zorder=4, lw=1.3))
    rr = {}
    for m in ["ours", "standard", "ekf"]:
        ea = errs[m][200:]
        rr[m] = float(np.sqrt(np.mean(ea[:, 0] ** 2 + ea[:, 1] ** 2)))
    ax.text(0.02, 0.03, f"RMSE (seed {sd}): ours {rr['ours']:.3f} / std {rr['standard']:.3f} / ekf {rr['ekf']:.3f}",
            transform=ax.transAxes, fontsize=6, color="0.30", va="bottom",
            bbox=dict(fc="white", alpha=0.85, ec="none", pad=1.5))
    ax.set_title(f"{tag} {tr} (unseen)", fontsize=8, loc="left", pad=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6.5)

# ---- (d) position error vs time: square (post-warmup) ----
ref, sd, errs = load("square")
t = np.arange(len(ref)) * 0.1
for m, key in [("standard", "standard"), ("ekf", "ekf"), ("ours", "ours")]:
    en = np.hypot(errs[m][:, 0], errs[m][:, 1])
    axd.plot(t[200:], en[200:], **style(key))
eo = np.hypot(errs["ours"][200:, 0], errs["ours"][200:, 1])
r_ours = float(np.sqrt(np.mean(eo ** 2)))
axd.axhline(r_ours, ls="--", lw=0.7, color=ps.C["blue"], alpha=0.6, zorder=2)
axd.text(99.5, r_ours, "ours RMSE", fontsize=6, color=ps.C["blue"], ha="right", va="bottom",
         bbox=dict(fc="white", alpha=0.85, ec="none", pad=1))
axd.set_title("(d) position error - square (post-warmup)", fontsize=8, loc="left", pad=3)
axd.set_xlabel("Time (s)")
axd.set_ylabel("Position error (m)")
axd.set_xlim(20, 100)
axd.set_ylim(bottom=0)
axd.tick_params(labelsize=6.5)

h, l = axa.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=7)
fig.subplots_adjust(hspace=0.30, wspace=0.26, top=0.93)
os.makedirs("figures", exist_ok=True)
ps.save_fig(fig, "figures/fig8_exp6_traj")
