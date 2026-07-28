# -*- coding: utf-8 -*-
"""fig9_disturbance.py - Fig.9 v2: 三场景扰动剖面（纯 RNG 重放，无仿真）
(a) 场景1 恒定 d=[-0.1,0]；(b) 场景2 高斯随机 -0.1+0.05N(0,1)；
(c) 场景3 复合 = 随机 + 斜坡 -0.0005t（与 (b) 同 seed=0，差值即纯漂移）。
v2：补画 d_u（角速度通道），三场景生成函数第二分量恒为 0，以 sky 平线明示。
caption 备忘：注明 seed 0 重放 + "replay of the locked disturbance generator, no MPC rerun"。"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pub_style as ps

ps.apply_style()

N, DT = 1000, 0.1
t = np.arange(N) * DT

d1 = np.full(N, -0.1)
np.random.seed(0)
noise = 0.05 * np.random.randn(N)
d2 = -0.1 + noise
d3 = -0.1 + noise - 0.0005 * np.arange(N)
trend3 = -0.1 - 0.0005 * np.arange(N)
du = np.zeros(N)  # 三场景角速度通道扰动恒为 0（生成函数第二分量）

np.savez('fig9_data.npz', t=t, scene1_const=d1, scene2_random=d2,
         scene3_composite=d3, scene3_trend=trend3, scene_du=du, seed=0)

fig, (axa, axb, axc) = plt.subplots(3, 1, figsize=ps.figsize(1.5, 0.85), sharex=True)

axa.plot(t, d1, color=ps.C['black'], ls='-', lw=1.2, label=r'$d_v$')
axa.plot(t, du, color=ps.C['sky'], ls='-', lw=0.9, label=r'$d_u$')
axa.set_ylim(-0.7, 0.2)
axa.set_title('(a) Scene 1: constant', fontsize=8, loc='left', pad=3)
axa.legend(fontsize=6.5, loc='lower left', ncol=2)
axa.text(99, -0.1, r'$-0.1$ m/s', fontsize=6.5, color='0.30', ha='right', va='bottom')

axb.plot(t, d2, color=ps.C['black'], ls='-', lw=0.8)
axb.plot(t, du, color=ps.C['sky'], ls='-', lw=0.9)
axb.axhline(-0.1, ls='--', lw=0.7, color='0.45', zorder=1)
axb.set_ylim(-0.7, 0.2)
axb.set_title(r'(b) Scene 2: random, $-0.1+0.05\mathcal{N}(0,1)$', fontsize=8, loc='left', pad=3)

axc.plot(t, d3, color=ps.C['black'], ls='-', lw=0.8)
axc.plot(t, du, color=ps.C['sky'], ls='-', lw=0.9)
axc.plot(t, trend3, ls='--', lw=0.9, color=ps.C['blue'], zorder=3)
axc.set_ylim(-0.7, 0.2)
axc.set_title('(c) Scene 3: composite = random + ramp', fontsize=8, loc='left', pad=3)
axc.text(99, trend3[-1], r'trend $-0.1-0.005t$', fontsize=6.5, color=ps.C['blue'],
         ha='right', va='top', bbox=dict(fc='white', alpha=0.85, ec='none', pad=1))

for ax in (axa, axb, axc):
    ax.set_ylabel('(m/s)')
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=6.5)
axc.set_xlabel('Time (s)')

fig.subplots_adjust(hspace=0.42)
os.makedirs('figures', exist_ok=True)
ps.save_fig(fig, 'figures/fig9_disturbance')
