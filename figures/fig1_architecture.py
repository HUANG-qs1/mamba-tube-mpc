# -*- coding: utf-8 -*-
"""fig1_architecture.py - Fig.1: system architecture (adaptation layer + control loop)
Top-journal style: layered block diagram, Okabe-Ito palette, sans-serif, vector PDF.
锚点：L=100、N=10、dt=0.1 s、κ=1.0/w_base=0.02/w_min=0.08（Table 1-15）、
Mamba 1.5 ms/step（exp2）、full pipeline 3.7 ms / 100 ms（Table 1-12）。
代价公式与 rerun_no_penalty.py / sim_core_np.py 的 mpc_with_tube 逐字一致
（含 2·max(0,||e||-w)^2 软管束项；0.01||u||^2 正则项图面从简，见正文方程）。"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C = {"blue": "#0072B2", "orange": "#E69F00", "red": "#D55E00",
     "gray": "#7F7F7F", "black": "#000000"}

CM = 1.0 / 2.54
fig, ax = plt.subplots(figsize=(14.0 * CM, 14.0 * CM * 0.60))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")

def box(x0, y0, w, h, title, sub, ec, fc, title_fs=6.9, sub_fs=5.4, sub2=None, sub2_fs=5.2):
    p = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.4,rounding_size=0.9",
                       fc=fc, ec=ec, lw=1.1, zorder=3)
    ax.add_patch(p)
    cy = y0 + h / 2
    if sub2:
        ax.text(x0 + w / 2, cy + 2.7, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color="0.12", zorder=4)
        ax.text(x0 + w / 2, cy - 0.5, sub, ha="center", va="center",
                fontsize=sub_fs, color="0.28", zorder=4)
        ax.text(x0 + w / 2, cy - 3.1, sub2, ha="center", va="center",
                fontsize=sub2_fs, color="0.28", zorder=4)
    elif sub:
        ax.text(x0 + w / 2, cy + 1.7, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color="0.12", zorder=4)
        ax.text(x0 + w / 2, cy - 1.8, sub, ha="center", va="center",
                fontsize=sub_fs, color="0.28", zorder=4)
    else:
        ax.text(x0 + w / 2, cy, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color="0.12", zorder=4)

def arr(p1, p2, color="0.15", lw=1.1):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=8, lw=lw,
                        color=color, zorder=2, connectionstyle="arc3,rad=0")
    ax.add_patch(a)

def seg(pts, color="0.15", lw=1.1):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, zorder=2, solid_capstyle="round")

# ---------------- 分组底框 ----------------
ax.add_patch(FancyBboxPatch((0.6, 35.0), 98.8, 24.2, boxstyle="round,pad=0.3,rounding_size=1.2",
                            fc="#FDF8EF", ec="0.55", lw=0.9, ls=(0, (4, 3)), zorder=1))
ax.add_patch(FancyBboxPatch((0.6, 0.6), 98.8, 32.6, boxstyle="round,pad=0.3,rounding_size=1.2",
                            fc="#F2F7FC", ec="0.55", lw=0.9, ls=(0, (4, 3)), zorder=1))
ax.text(2.2, 57.2, "ADAPTATION LAYER", fontsize=6.2, color="0.40",
        fontweight="bold", ha="left", va="center")
ax.text(2.2, 3.0, "CONTROL LOOP  ·  10 Hz", fontsize=6.2, color="0.40",
        fontweight="bold", ha="left", va="center")
ax.text(98.0, 3.0, r"full pipeline 3.7 ms $\ll$ 100 ms budget", fontsize=5.6,
        color="0.40", ha="right", va="center")

# ---------------- 上层：自适应层 ----------------
box(3, 39, 21, 11, "Error history", "sliding window $L=100$", C["gray"], "white")
box(27.5, 39, 13, 11, "Normalize", r"$(\mu,\sigma)$", C["gray"], "white")
box(44, 39, 23, 11, "Mamba predictor", "selective SSM", C["orange"], "#FDF3E0",
    sub2="offline trained · 1.5 ms/step", sub2_fs=5.0)
box(70, 39, 27.5, 11, "Adaptive tube law", r"$w=\max(w_{min},\,w_{base}+\kappa\max\|\hat{d}\|)$",
    C["blue"], "#E5F0F7", sub_fs=5.2, sub2=r"$\kappa=1.0,\ w_{base}=0.02,\ w_{min}=0.08$", sub2_fs=5.0)

arr((24.2, 44.5), (27.5, 44.5))
arr((40.7, 44.5), (44, 44.5))
arr((67.2, 44.5), (70, 44.5))

# ---------------- 下层：控制回路 ----------------
box(2.5, 11.5, 11.5, 12, "Reference", "trajectory", C["black"], "white", title_fs=6.4, sub_fs=5.2)
sum_c = Circle((19.8, 17.5), 2.2, fc="white", ec="0.15", lw=1.1, zorder=3)
ax.add_patch(sum_c)
ax.text(17.2, 19.6, "+", fontsize=8.5, ha="center", va="center", color="0.10", zorder=4)
ax.text(21.6, 14.7, r"$-$", fontsize=8.5, ha="center", va="center", color="0.10", zorder=4)

box(24.5, 11.5, 30, 12, "Tube MPC", r"$\min\ \Sigma\|e\|^2 + 2\max(0,\,\|e\|-w)^2$",
    C["blue"], "#E5F0F7", sub_fs=4.7, sub2=r"IPOPT · $N=10$ · $|u|\leq1$", sub2_fs=5.0)
box(63.5, 11.5, 25, 12, "Mobile robot", r"unicycle dynamics · $\Delta t=0.1$ s",
    C["gray"], "#EFEFEF", sub_fs=5.0)

# 外生扰动（右移，为 w 走线让出通道）
box(83.0, 26.5, 16.4, 4.6, None, None, C["red"], "white")
ax.text(91.2, 28.8, "true disturbance $d$", fontsize=5.1, ha="center", va="center",
        color=C["red"], zorder=4)
arr((87.5, 26.3), (87.5, 23.9), color=C["red"], lw=1.0)

# 主回路箭头
arr((14.2, 17.5), (17.4, 17.5))
arr((22.0, 17.5), (24.5, 17.5))
arr((54.7, 17.5), (63.5, 17.5))
seg([(88.5, 17.5), (93.5, 17.5), (93.5, 5.8), (19.8, 5.8)])
arr((19.8, 5.8), (19.8, 15.1))
ax.plot(93.5, 17.5, "o", ms=2.4, color="0.15", zorder=3)

# 信号标签
ax.text(15.8, 15.6, r"$r_k$", fontsize=6.0, ha="center", color="0.20")
ax.text(59.3, 19.2, r"$u_k=(v_k,\omega_k)$", fontsize=6.2, ha="center", color="0.20",
        bbox=dict(fc="white", alpha=0.85, ec="none", pad=1), zorder=4)
ax.text(56, 7.2, r"$x_k$", fontsize=6.2, ha="center", color="0.20")

# 误差分支 → 历史窗
ax.plot(23.2, 17.5, "o", ms=2.4, color="0.15", zorder=3)
seg([(23.2, 17.5), (23.2, 36.8)])
arr((23.2, 36.8), (23.2, 39.2))
ax.text(24.4, 30.5, r"$e_k$", fontsize=5.8, ha="left", color="0.35")

# Tube 宽度 w：tube law → MPC（下移走线，与层界虚线分离）
seg([(80, 39.0), (80, 29.8), (39.5, 29.8)])
arr((39.5, 29.8), (39.5, 23.9))
ax.text(40.6, 27.0, r"tube width $w_k$", fontsize=5.8, ha="left", va="center", color=C["blue"],
        bbox=dict(fc="white", alpha=0.85, ec="none", pad=1), zorder=4)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig1_architecture.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("figures/fig1_architecture.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("saved: figures/fig1_architecture.pdf / .png")
