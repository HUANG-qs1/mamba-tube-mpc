"""t-test显著性检验（汇总统计版Welch t-test + Cohen's d）
数据：rerun_no_penalty_results.npz / exp6_baselines.npz / exp7_eval_results.npz /
      exp8_results.npz / exp9_results.npz / exp10_results.npz（均为mean±std, n=10）
输出：ttest_results.npz + 打印表格（p<0.05标*，p<0.01标**，p<0.001标***）
口径：diff = ours - baseline，RMSE越低越好；diff<0且p<0.05 = 我们的方法显著更优"""
import numpy as np
try:
    from scipy import stats
except ImportError:
    raise SystemExit("缺少scipy，请先执行: pip install scipy 然后重新运行")

N = 10
def load(f):
    return np.load(f, allow_pickle=True)['results'][0]

R = {'rerun': load('rerun_no_penalty_results.npz'),
     'exp6b': load('exp6_baselines.npz'),
     'exp7': load('exp7_eval_results.npz'),
     'exp8': load('exp8_results.npz'),
     'exp9': load('exp9_results.npz'),
     'exp10': load('exp10_results.npz')}

def welch(m1, s1, m2, s2, n=N):
    v1, v2 = s1**2/n, s2**2/n
    t = (m1-m2)/np.sqrt(v1+v2)
    df = (v1+v2)**2/(v1**2/(n-1)+v2**2/(n-1))
    p = 2*stats.t.sf(abs(t), df)
    d = (m1-m2)/np.sqrt((s1**2+s2**2)/2)
    return t, df, p, d

def star(p):
    return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'

# (标签, ours来源.key, baseline来源.key)
COMPS = [
    ('场景2: Mamba适应 vs Standard',      ('rerun','ablation_A_mamba_adaptive'), ('exp9','baseline_standard')),
    ('场景2: Mamba适应 vs EKF',           ('rerun','ablation_A_mamba_adaptive'), ('rerun','ablation_E_ekf_fixed_015')),
    ('场景2: Mamba适应 vs 固定0.05',      ('rerun','ablation_A_mamba_adaptive'), ('rerun','scene2_fixed_0.05')),
    ('场景2: Mamba适应 vs 固定0.10',      ('rerun','ablation_A_mamba_adaptive'), ('rerun','scene2_fixed_0.1')),
    ('消融: A适应Tube vs C固定Tube',      ('rerun','ablation_A_mamba_adaptive'), ('rerun','ablation_C_mamba_fixed_015')),
    ('消融: A Mamba vs D LSTM',           ('rerun','ablation_A_mamba_adaptive'), ('rerun','ablation_D_lstm_adaptive')),
    ('消融: A vs B 固定0.15',             ('rerun','ablation_A_mamba_adaptive'), ('rerun','ablation_B_fixed_015')),
    ('exp6 square: Mamba vs Standard',    ('rerun','exp6_mamba_square'),  ('exp6b','square_standard')),
    ('exp6 square: Mamba vs EKF',         ('rerun','exp6_mamba_square'),  ('exp6b','square_ekf')),
    ('exp6 spiral: Mamba vs Standard',    ('rerun','exp6_mamba_spiral'),  ('exp6b','spiral_standard')),
    ('exp6 spiral: Mamba vs EKF',         ('rerun','exp6_mamba_spiral'),  ('exp6b','spiral_ekf')),
    ('exp6 lissajous: Mamba vs Standard', ('rerun','exp6_mamba_lissajous'),('exp6b','lissajous_standard')),
    ('exp6 lissajous: Mamba vs EKF',      ('rerun','exp6_mamba_lissajous'),('exp6b','lissajous_ekf')),
    ('exp7 seq20: Mamba vs LSTM',         ('exp7','mamba_seq20'),  ('exp7','lstm_seq20')),
    ('exp7 seq100: Mamba vs LSTM',        ('exp7','mamba_seq100'), ('exp7','lstm_seq100')),
    ('exp7 seq200: Mamba vs LSTM',        ('exp7','mamba_seq200'), ('exp7','lstm_seq200')),
]
for amp in ['0.5','1.0','1.5','2.0','3.0']:
    COMPS.append((f'exp8 amp{amp}: 适应 vs 固定0.05', ('exp8',f'amp{amp}_adaptive'), ('exp8',f'amp{amp}_fixed_005')))
    COMPS.append((f'exp8 amp{amp}: 适应 vs 固定0.10', ('exp8',f'amp{amp}_adaptive'), ('exp8',f'amp{amp}_fixed_010')))
COMPS.append(('exp8 drift0.5: 适应 vs 固定0.05', ('exp8','drift0.5_adaptive'), ('exp8','drift0.5_fixed_005')))
COMPS.append(('exp8 drift0.5: 适应 vs 固定0.10', ('exp8','drift0.5_adaptive'), ('exp8','drift0.5_fixed_010')))
for cond in ['baseline','delay1','delay2','loss5','loss10','vscale_p10','vscale_m10','meas001','meas002']:
    COMPS.append((f'exp9 {cond}: Mamba vs Standard', ('exp9',f'{cond}_mamba_tube'), ('exp9',f'{cond}_standard')))
COMPS.append(('exp10 square: Mamba vs GP',  ('exp10','square_mamba_tube'),  ('exp10','square_gp_tube')))
COMPS.append(('exp10 lissajous: Mamba vs GP', ('exp10','lissajous_mamba_tube'), ('exp10','lissajous_gp_tube')))

OUT = {}
print(f"{'对比':<38} {'ours':>16} {'baseline':>16} {'t':>7} {'p':>10} {'d':>7}  显著性", flush=True)
print('-'*110, flush=True)
for label, (s1,k1), (s2,k2) in COMPS:
    a, b = R[s1][k1], R[s2][k2]
    m1, sd1, m2, sd2 = float(a['rmse']), float(a['rmse_std']), float(b['rmse']), float(b['rmse_std'])
    t, df, p, d = welch(m1, sd1, m2, sd2)
    OUT[label] = {'m1':m1,'s1':sd1,'m2':m2,'s2':sd2,'t':t,'df':df,'p':p,'cohen_d':d}
    print(f"{label:<38} {m1:.4f}±{sd1:.4f}   {m2:.4f}±{sd2:.4f} {t:>7.2f} {p:>10.2e} {d:>7.2f}  {star(p)}", flush=True)

np.savez('ttest_results.npz', results=np.array([OUT], dtype=object))
print('-'*110, flush=True)
print(f"共{len(COMPS)}组对比，已保存 ttest_results.npz", flush=True)
print("解读：diff=ours-baseline<0 且 p<0.05 → 我们的方法显著更优；d>0.8大效应 / 0.5中 / 0.2小", flush=True)
