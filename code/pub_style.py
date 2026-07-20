# -*- coding: utf-8 -*-
"""
pub_style.py — 论文图表全局样式模板（Mechatronics / Elsevier 规范）
所有 Fig. 脚本统一 import 本文件，保证全文风格一致。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

CM = 1.0 / 2.54  # cm -> inch

# ---- 版心宽度（Elsevier: 单栏 9.0cm / 1.5栏 14.0cm / 双栏 19.0cm）----
W_SINGLE = 9.0 * CM
W_ONEHALF = 14.0 * CM
W_DOUBLE = 19.0 * CM

# ---- 色板：Okabe-Ito 色盲安全色 ----
C = {
    "blue":   "#0072B2",  # 我方方法主色
    "orange": "#E69F00",  # 学习类基线
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "gray":   "#7F7F7F",  # 传统基线
    "black":  "#000000",
}

# ---- 方法 -> 样式映射（全文唯一出处，任何图不得绕过）----
# (color, linestyle, linewidth, marker, 显示名)
METHOD = {
    "ours":     (C["blue"],   "-",  1.6, "o", "Mamba-Tube (ours)"),
    "mamba":    (C["blue"],   "-",  1.6, "o", "Mamba"),
    "lstm":     (C["orange"], "--", 1.4, "s", "LSTM"),
    "standard": (C["gray"],   "-",  1.3, "d", "Standard MPC"),
    "ekf":      (C["green"],  "-.", 1.3, "^", "EKF-MPC"),
    "gp":       (C["purple"], ":",  1.4, "v", "GP-Tube"),
    "fixed":    (C["red"],    "--", 1.3, "x", "Fixed-Tube"),
    "nominal":  (C["black"],  ":",  1.2, None, "Nominal"),
    "ref":      (C["black"],  "-",  1.1, None, "Reference"),
}


def apply_style():
    """设置全局 rcParams。每个 Fig 脚本开头调用一次。"""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "legend.borderpad": 0.35,
        "legend.handlelength": 1.6,
        "legend.labelspacing": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "errorbar.capsize": 2.0,
    })


def figsize(cols=1.0, ratio=0.72):
    """cols: 1.0 单栏 / 1.5 / 2.0；ratio: 高/宽。"""
    w = {1.0: W_SINGLE, 1.5: W_ONEHALF, 2.0: W_DOUBLE}[cols]
    return (w, w * ratio)


def style_of(key):
    """取方法样式，返回 dict，直接 **解包进 plot()。"""
    c, ls, lw, mk, lab = METHOD[key]
    return dict(color=c, ls=ls, lw=lw, marker=mk, ms=3.2,
                mew=0.7, mec=c, mfc="white", label=lab)


def band(ax, x, mean, std, key, z=2):
    """均值线 + ±1σ 阴影带（10 种子图的标准画法）。"""
    st = style_of(key)
    lab = st.pop("label")
    ax.plot(x, mean, **st, label=lab, zorder=z + 1)
    ax.fill_between(x, mean - std, mean + std, color=st["color"],
                    alpha=0.15, lw=0, zorder=z)


def save_fig(fig, path_base):
    """同时输出 PDF（矢量，投稿用）与 PNG（600dpi，预览用）。"""
    fig.savefig(path_base + ".pdf")
    fig.savefig(path_base + ".png")
    plt.close(fig)
    print("saved:", path_base + ".pdf / .png")
