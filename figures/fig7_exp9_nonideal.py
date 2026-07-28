# -*- coding: utf-8 -*-
"""fig7_exp9_nonideal.py - Fig.7: exp9 non-ideal factors, RMSE + MaxErr grouped bars"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pub_style as ps

ps.apply_style()

r = np.load("exp9_results.npz", allow_pickle=True)["results"].tolist()
while isinstance(r, list):
    r = r[0]

CONDS = [
    ("baseline",        "baseline"),
    ("delay\n100 ms",   "delay1"),
    ("delay\n200 ms",   "delay2"),
    ("loss\n5%",        "loss5"),
    ("loss\n10%",       "loss10"),
    ("model\n+10%",     "vscale_p10"),
    ("model\n-10%",     "vscale_m10"),
    ("noise\nσ=0.01",   "meas001"),
    ("noise\nσ=0.02",   "meas002"),
]
MARKS = ["***", "***", "n.s.", "***", "***", "***", "***", "***", "***"]
X = np.arange(len(CONDS))
W = 0.38

om  = [r[f"{k}_mamba_tube"]["rmse"] for _, k in CONDS]
osd = [r[f"{k}_mamba_tube"]["rmse_std"] for _, k in CONDS]
sm  = [r[f"{k}_standard"]["rmse"] for _, k in CONDS]
ssd = [r[f"{k}_standard"]["rmse_std"] for _, k in CONDS]
omx = [r[f"{k}_mamba_tube"]["maxerr"] for _, k in CONDS]
smx = [r[f"{k}_standard"]["maxerr"] for _, k in CONDS]

fig, (axa, axb) = plt.subplots(2, 1, figsize=ps.figsize(1.5, 1.0), sharex=True)

# ---- (a) RMSE ----
axa.bar(X - W/2, om, W, yerr=osd, capsize=2, color=ps.C["blue"], edgecolor="white",
        lw=0.5, error_kw=dict(elinewidth=0.7, ecolor=ps.C["blue"]), label="Mamba-Tube (ours)", zorder=3)
axa.bar(X + W/2, sm, W, yerr=ssd, capsize=2, color=ps.C["gray"], edgecolor="white",
        lw=0.5, error_kw=dict(elinewidth=0.7, ecolor=ps.C["gray"]), label="Standard MPC", zorder=3)
axa.plot([], [], ls=":", lw=0.8, color="0.55", label="Baseline RMSE (ours)")
for i, mk in enumerate(MARKS):
    top = max(om[i] + osd[i], sm[i] + ssd[i])
    axa.annotate(mk, (X[i], top), textcoords="offset points", xytext=(0, 4),
                 ha="center", fontsize=7, color="0.25")
axa.axhline(om[0], ls=":", lw=0.8, color="0.55", zorder=1)
axa.set_ylabel("RMSE (m)")
axa.set_ylim(0, 0.095)
axa.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
axa.text(-0.09, 1.04, "(a)", transform=axa.transAxes, fontweight="bold")

# ---- (b) MaxErr ----
axb.bar(X - W/2, omx, W, color=ps.C["blue"], edgecolor="white", lw=0.5, zorder=3)
axb.bar(X + W/2, smx, W, color=ps.C["gray"], edgecolor="white", lw=0.5, zorder=3)
axb.axhline(omx[0], ls=":", lw=0.8, color="0.55", zorder=1)
axb.text(0.005, 0.96, "bars and dotted line as in (a);  *** p<0.001, n.s. (Welch, n=10)",
         transform=axb.transAxes, fontsize=6, color="0.35", va="top")
axb.set_ylabel("MaxErr (m)")
axb.set_xlabel("Non-ideal condition")
axb.set_ylim(0, 0.18)
axb.set_xticks(X)
axb.set_xticklabels([c[0] for c in CONDS], fontsize=6.5)
axb.set_xlim(-0.6, 8.6)
axb.text(-0.09, 1.04, "(b)", transform=axb.transAxes, fontweight="bold")

fig.subplots_adjust(hspace=0.14)
os.makedirs("figures", exist_ok=True)
ps.save_fig(fig, "figures/fig7_exp9_nonideal")
