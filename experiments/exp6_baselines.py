"""实验6基线补跑（无惩罚）：为t-test与论文Table VI提供干净的Standard/EKF数据（mean±std）
方法：Standard MPC / EKF补偿 × 轨迹{square, spiral, lissajous} × 10种子
协议与rerun/exp10完全一致（同run_sim、同warmup=200）
输出：exp6_baselines.npz"""
import numpy as np
from sim_core_np import square_path, spiral_path, lissajous_path, run_sim, metric_vs_tube

ALL = {}
print("===== 实验6基线补跑（standard / ekf，无惩罚） =====", flush=True)
for pname, path in [('square', square_path()), ('spiral', spiral_path()), ('lissajous', lissajous_path())]:
    for mode in ['standard', 'ekf']:
        rl, ml, vl = [], [], []
        for seed in range(10):
            e, tu, _ = run_sim(path, seed, mode)
            r, mx, vi = metric_vs_tube(e, tu, 200)
            rl.append(r); ml.append(mx); vl.append(vi)
        ALL[f'{pname}_{mode}'] = {'rmse': np.mean(rl), 'rmse_std': np.std(rl),
                                  'maxerr': np.mean(ml), 'viol': np.mean(vl)}
        print(f"{pname} {mode}: RMSE={np.mean(rl):.4f}±{np.std(rl):.4f} "
              f"MaxErr={np.mean(ml):.4f} Viol={np.mean(vl):.1f}", flush=True)

np.savez('exp6_baselines.npz', results=np.array([ALL], dtype=object))
print("\n===== 完成，已保存 exp6_baselines.npz =====", flush=True)
