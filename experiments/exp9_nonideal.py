"""实验9：非理想因素注入（回应"纯仿真"质疑，独立新实验，不动已有34组数据）
场景2协议（figure8 + 随机扰动），10种子
四类非理想因素（每类单独注入）：
  delay: 控制延迟 {1, 2} 步（100/200ms）
  loss: 通信丢包率 {5%, 10%}（丢包时保持上一指令）
  vscale: 模型失配 实际速度 = {1.1, 0.9} × 指令速度（轮径/轮距±10%偏差）
  meas_noise: 位置测量噪声 σ ∈ {0.01, 0.02} m
方法：Mamba-Tube（自适应） vs Standard MPC（无Tube无补偿）
指标：RMSE / MaxErr
输出：exp9_results.npz"""
import numpy as np
from sim_core_np import figure8_path, run_sim, metric_vs_tube

fig8 = figure8_path()
conds = [('baseline', {}),
         ('delay1', dict(delay=1)), ('delay2', dict(delay=2)),
         ('loss5', dict(loss=0.05)), ('loss10', dict(loss=0.10)),
         ('vscale_p10', dict(vscale=1.1)), ('vscale_m10', dict(vscale=0.9)),
         ('meas001', dict(meas_noise=0.01)), ('meas002', dict(meas_noise=0.02))]

ALL = {}
print("===== 实验9：非理想因素注入 =====", flush=True)
for cname, ckw in conds:
    for mname, mkw in [('mamba_tube', dict(mode='mamba')),
                       ('standard', dict(mode='standard'))]:
        rl, ml = [], []
        for seed in range(10):
            e, tu, _ = run_sim(fig8, seed, **mkw, **ckw)
            r, mx, vi = metric_vs_tube(e, tu, 200)
            rl.append(r); ml.append(mx)
        ALL[f'{cname}_{mname}'] = {'rmse': np.mean(rl), 'rmse_std': np.std(rl),
                                   'maxerr': np.mean(ml)}
        print(f"{cname} {mname}: RMSE={np.mean(rl):.4f}±{np.std(rl):.4f} "
              f"MaxErr={np.mean(ml):.4f}", flush=True)

np.savez('exp9_results.npz', results=np.array([ALL], dtype=object))
print("\n===== 实验9完成，已保存 exp9_results.npz =====", flush=True)
