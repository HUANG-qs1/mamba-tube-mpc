# -*- coding: utf-8 -*-
"""fig5_exp8_coverage.py - Fig.5: exp8 coverage & tube width vs mismatch amplitude"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pub_style as ps

ps.apply_style()

r = np.load("exp8_results.npz", allow_pickle=True)["results"].tolist()
while isinstance(r, list):
    r = r[0]

AMPS = [0.5, 1.0, 1.5, 2.0, 3.0]
SRC = {"ours": "adaptive", "f010": "fixed_010", "f005": "fixed_005"}
cov  = {m: [r[f"amp{a}_{k}"]["coverage"] * 100.0 for a in AMPS] for m, k in SRC.items()}
tube = {m: [r[f"amp{a}_{k}"]["tube"] for a in AMPS] for m, k in SRC.items()}

ST = {"ours": ps.style_of("ours"), "f010": ps.style_of("fixed"), "f005": ps.style_of("fixed")}
ST["ours"]["label"] = "Adaptive tube (ours)"
ST["f010"]["label"] = "Fixed $w = 0.10$"
ST["f005"].update(ls=":", marker="s", label="Fixed $w = 0.05$")

fig, (axa, axb) = plt.subplots(2, 1, figsize=ps.figsize(1.0, 1.15), sharex=True)

# ---- (a) coverage ----
for m in ["ours", "f010", "f005"]:
    axa.plot(AMPS, cov[m], **ST[m])
for m, c in [("ours", ps.C["blue"]), ("f010", ps.C["red"]), ("f005", ps.C["red"])]:
    axa.annotate(f"{cov[m][-1]:.1f}%", (AMPS[-1], cov[m][-1]),
                 textcoords="offset points", xytext=(7, 0), va="center",
                 fontsize=6.5, color=c)
axa.set_ylabel("Coverage (%)")
axa.set_ylim(-6, 108)
axa.set_yticks([0, 20, 40, 60, 80, 100])
axa.legend(loc="center right")
axa.text(-0.14, 1.03, "(a)", transform=axa.transAxes, fontweight="bold")

# ---- (b) mean tube width ----
for m in ["ours", "f010", "f005"]:
    axb.plot(AMPS, tube[m], **ST[m])
for m, c in [("ours", ps.C["blue"]), ("f010", ps.C["red"]), ("f005", ps.C["red"])]:
    axb.annotate(f"{tube[m][-1]:.3f}", (AMPS[-1], tube[m][-1]),
                 textcoords="offset points", xytext=(7, 0), va="center",
                 fontsize=6.5, color=c)
axb.text(0.03, 0.92, "line styles as in (a)", transform=axb.transAxes,
         fontsize=6, color="0.35", va="top")
axb.set_ylabel("Mean tube width (m)")
axb.set_xlabel(r"Disturbance amplitude scale $\alpha$")
axb.set_xticks(AMPS)
axb.set_xlim(0.35, 3.45)
axb.set_ylim(0.0, 0.27)
axb.text(-0.14, 1.03, "(b)", transform=axb.transAxes, fontweight="bold")

fig.subplots_adjust(hspace=0.16)
os.makedirs("figures", exist_ok=True)
ps.save_fig(fig, "figures/fig5_exp8_coverage")
