# -*- coding: utf-8 -*-
"""export_fig3_traj.py — Fig.3（三场景轨迹对比复合图）数据导出，同源验证版
场景1 circle+恒定扰动  → 锚点 day44_scene1_v3_fix.npz（确定性，每方法单次）
场景2 figure8+随机扰动 → 锚点 rerun ablation_A / exp9 baseline_standard / rerun ablation_E
场景3 sine+复合扰动    → 锚点 scene3_nopen_probe.npz / day44_scene3_v3_fix3.npz（Standard/EKF 行）
引擎：mamba/ekf 逐行复刻 rerun_no_penalty.py；场景1 复刻 day44_scene1_v3_fix.py（含休眠惩罚分支）；
standard 走 sim_core_np.run_sim（与 exp9 生成路径一致）。重算值与锁定 npz 逐对核对，
全部通过才保存 fig3_trajectories.npz。
"""
import numpy as np
import torch
import casadi as ca
import glob

device = 'cuda'
from mamba_predictor import MambaDisturbancePredictor
mamba_model = MambaDisturbancePredictor(2, 128, 2, 10).to(device)
mamba_model.load_state_dict(torch.load('best_mamba_v3.pt', map_location=device, weights_only=True))
mamba_model.eval()
import sim_core_np as sc
print("模型加载完成", flush=True)

norm = np.load('norm_params_v3.npz')
x_mean = norm['x_mean'].squeeze(); x_std = norm['x_std'].squeeze()
y_mean = norm['y_mean'].squeeze(); y_std = norm['y_std'].squeeze()

# ========== 三条场景轨迹 ==========
def circle_path(radius=3.0, n=1000):
    ang = np.linspace(0, 2*np.pi, n)
    return np.column_stack([radius*np.cos(ang), radius*np.sin(ang)])

def figure8_path(a=3.0, n=1000):
    t = np.linspace(0, 2*np.pi, n)
    return np.column_stack([a*np.sin(2*t), a*np.sin(t)])

def sine_path(a=3.0, n=1000, cyc=2):
    x = np.linspace(0, 10, n)
    return np.column_stack([x, a*np.sin(cyc*np.pi*x/10)])

# ========== 引擎（逐行复刻 rerun_no_penalty.py） ==========
class RobotFree:
    def __init__(self, x=0.0, y=0.0, theta=0.0, dt=0.1):
        self.state = np.array([x, y, theta], float)
        self.dt = dt
    def step(self, v, omega, d=None):
        x, y, theta = self.state; dt = self.dt
        v_act = v + (d[0] if d is not None else 0)
        xn = x + v_act*np.cos(theta)*dt
        yn = y + v_act*np.sin(theta)*dt
        tn = np.arctan2(np.sin(theta+omega*dt), np.cos(theta+omega*dt))
        self.state = np.array([xn, yn, tn])
        return self.state

class EKF:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.x_est = np.zeros(4); self.P = np.eye(4)*0.1
        self.Q = np.diag([0.01, 0.01, 0.001, 0.001]); self.R = np.diag([0.01, 0.01, 0.001])
    def update(self, z, u):
        x, y, theta, d_v = self.x_est; v, omega = u; dt = self.dt
        xp = x + (v+d_v)*np.cos(theta)*dt
        yp = y + (v+d_v)*np.sin(theta)*dt
        tp = theta + omega*dt
        xpv = np.array([xp, yp, tp, d_v])
        F = np.array([[1, 0, -(v+d_v)*np.sin(theta)*dt, np.cos(theta)*dt],
                      [0, 1, (v+d_v)*np.cos(theta)*dt, np.sin(theta)*dt],
                      [0, 0, 1, 0], [0, 0, 0, 1]])
        Pp = F@self.P@F.T + self.Q
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        yt = z - np.array([xp, yp, tp])
        S = H@Pp@H.T + self.R
        K = Pp@H.T@np.linalg.inv(S)
        self.x_est = xpv + K@yt
        self.P = (np.eye(4)-K@H)@Pp
        return self.x_est[3]

def mpc_no_tube(state, ref_segment, N=10, dt=0.1):
    x, y, theta = state
    U = ca.MX.sym('U', 2*N); v = U[0::2]; o = U[1::2]
    X = [x, y, theta]; obj = 0
    for k in range(N):
        xn = X[0]+v[k]*ca.cos(X[2])*dt; yn = X[1]+v[k]*ca.sin(X[2])*dt; tn = X[2]+o[k]*dt
        X = [xn, yn, tn]
        obj += (X[0]-ref_segment[k, 0])**2 + (X[1]-ref_segment[k, 1])**2 + 0.01*v[k]**2 + 0.01*o[k]**2
    g = [X[0]-x, X[1]-y, X[2]-theta]
    solver = ca.nlpsol('s', 'ipopt', {'x': U, 'f': obj, 'g': ca.vertcat(*g)}, {'print_time': False, 'ipopt': {'print_level': 0}})
    res = solver(x0=[0.5, 0.0]*N, lbx=[-1.0]*(2*N), ubx=[1.0]*(2*N), lbg=0, ubg=0)
    sol = np.array(res['x']).flatten()
    return sol[0], sol[1]

def mpc_with_tube(state, ref_segment, tube_size, N=10, dt=0.1):
    x, y, theta = state
    U = ca.MX.sym('U', 2*N); v = U[0::2]; o = U[1::2]
    X = [x, y, theta]; obj = 0
    for k in range(N):
        xn = X[0]+v[k]*ca.cos(X[2])*dt; yn = X[1]+v[k]*ca.sin(X[2])*dt; tn = X[2]+o[k]*dt
        X = [xn, yn, tn]
        ex = X[0]-ref_segment[k, 0]; ey = X[1]-ref_segment[k, 1]
        dist = ca.sqrt(ex**2+ey**2)
        tv = ca.fmax(0, dist-tube_size)
        obj += ex**2+ey**2 + 2.0*tv**2 + 0.01*v[k]**2 + 0.01*o[k]**2
    g = [X[0]-x, X[1]-y, X[2]-theta]
    solver = ca.nlpsol('s', 'ipopt', {'x': U, 'f': obj, 'g': ca.vertcat(*g)}, {'print_time': False, 'ipopt': {'print_level': 0}})
    res = solver(x0=[0.5, 0.0]*N, lbx=[-1.0]*(2*N), ubx=[1.0]*(2*N), lbg=0, ubg=0)
    sol = np.array(res['x']).flatten()
    return sol[0], sol[1]

def compute_tube(history_errors, model, w_base=0.02, w_min=0.08, kappa=1.0):
    x = (history_errors - x_mean) / (x_std + 1e-8)
    x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(x_tensor).cpu().numpy()[0]
    pred = pred_norm * y_std + y_mean
    pred_norms = np.sqrt(pred[:, 0]**2 + pred[:, 1]**2)
    return max(w_min, w_base + kappa*np.max(pred_norms))

def get_random_disturbance(t):
    return [-0.1 + 0.05*np.random.randn(), 0]

def get_composite_disturbance(t):
    return [-0.1 + 0.05*np.random.randn() - 0.0005*t, 0]

def metric_vs_tube(errors, tubes, warmup):
    ea = errors[warmup:]; ta = tubes[warmup:]
    en = np.hypot(ea[:, 0], ea[:, 1])
    return np.sqrt(np.mean(en**2)), en.max(), int(np.sum(en > ta))

# ========== run_sim（rerun 版 + standard 分支；scene2/3 用） ==========
def run_sim(ref_path, seed, mode, N=10, dt=0.1, disturb='random', init_theta=0.0,
            fixed_tube=0.15, w_base=0.02, w_min=0.08, kappa=1.0, total_time=100.0):
    np.random.seed(seed)
    robot = RobotFree(0, 0, init_theta, dt)
    ekf = EKF(dt) if mode == 'ekf' else None
    steps = int(total_time/dt)
    errors, tubes = [], []
    for t in range(steps):
        idx = min(int(t*dt*10), len(ref_path)-1)
        seg_ids = np.clip(idx + (np.arange(N)*dt*10).astype(int), 0, len(ref_path)-1)
        ref_seg = ref_path[seg_ids]
        errors.append([robot.state[0]-ref_path[idx, 0], robot.state[1]-ref_path[idx, 1]])
        if mode == 'mamba':
            tube = compute_tube(np.array(errors[-100:]), mamba_model, w_base, w_min, kappa) if len(errors) >= 100 else 0.15
        else:
            tube = fixed_tube
        tubes.append(tube)
        d = get_composite_disturbance(t) if disturb == 'composite' else get_random_disturbance(t)
        if mode == 'ekf':
            v_mpc, w_mpc = mpc_no_tube(robot.state, ref_seg, N, dt)
            d_est = ekf.update(robot.state.copy(), [v_mpc, w_mpc])
            v, w = v_mpc - d_est, w_mpc
        elif mode == 'standard':
            v, w = mpc_no_tube(robot.state, ref_seg, N, dt)
        else:
            v, w = mpc_with_tube(robot.state, ref_seg, tube, N, dt)
        robot.step(v, w, d)
    return np.array(errors), np.array(tubes)

# ========== 场景1专用（复刻 day44_scene1_v3_fix.py：init(3,0,pi/2)、恒定扰动、休眠惩罚分支） ==========
PENALTY_HITS = 0

def run_scene1(ref_path, mode, N=10, dt=0.1):
    global PENALTY_HITS
    np.random.seed(0)
    robot = RobotFree(3.0, 0.0, np.pi/2, dt)
    ekf = EKF(dt) if mode == 'ekf' else None
    errors, tubes = [], []
    for t in range(1000):
        idx = min(t, len(ref_path)-1)
        ref_seg = ref_path[idx:idx+N]
        if len(ref_seg) < N:
            ref_seg = np.vstack([ref_seg, np.tile(ref_path[-1], (N-len(ref_seg), 1))])
        errors.append([robot.state[0]-ref_path[idx, 0], robot.state[1]-ref_path[idx, 1]])
        d = [-0.1, 0]
        if mode == 'mamba':
            tube = compute_tube(np.array(errors[-100:]), mamba_model) if len(errors) >= 100 else 0.15
            tubes.append(tube)
            v, w = mpc_with_tube(robot.state, ref_seg, tube, N, dt)
            if np.hypot(errors[-1][0], errors[-1][1]) > tube:  # day44 惩罚分支（本场景预期不触发）
                v, w = v*0.5, w*0.5
                PENALTY_HITS += 1
            robot.step(v, w, d)
        elif mode == 'ekf':
            tubes.append(np.nan)
            v_mpc, w_mpc = mpc_no_tube(robot.state, ref_seg, N, dt)
            d_est = ekf.update(robot.state.copy(), [v_mpc, w_mpc])
            robot.step(v_mpc - d_est, w_mpc, d)
        else:
            tubes.append(np.nan)
            v, w = mpc_no_tube(robot.state, ref_seg, N, dt)
            robot.step(v, w, d)
    return np.array(errors), np.array(tubes)

# ========== 工具 ==========
def as_dict(x):
    while not isinstance(x, dict):
        x = x[0]
    return x

def rmse_200(errors):
    ea = errors[200:]
    en = np.hypot(ea[:, 0], ea[:, 1])
    return np.sqrt(np.mean(en**2))

REPORT = []
def check(name, got, want, tol=1e-12):
    diff = abs(float(got) - float(want))
    ok = diff <= tol
    REPORT.append(ok)
    print(f"[{'OK' if ok else 'BAD'}] {name}: 重算={float(got):.12f} 锁定={float(want):.12f} |diff|={diff:.2e}", flush=True)

def rep_seed(rl, target):
    return int(np.argmin(np.abs(np.array(rl) - target)))

OUT = {}

# ========== 场景1：circle + 恒定扰动（确定性） ==========
print("\n===== 场景1 circle+恒定扰动（确定性，单run） =====", flush=True)
s1_anchor = np.load('day44_scene1_v3_fix.npz', allow_pickle=True)['all_results'].item()
ref1 = circle_path()
for mode, aname, tag in [('standard', 'Standard MPC', 'std'), ('ekf', 'EKF-MPC', 'ekf'), ('mamba', 'Mamba-Tube(Ours)', 'ours')]:
    e, tu = run_scene1(ref1, mode)
    r = rmse_200(e)
    check(f"scene1 {mode}", r, s1_anchor[aname][0][0])
    OUT[f'scene1_{tag}_err'] = e
    OUT[f'scene1_{tag}_rmse'] = r
    OUT[f'scene1_{tag}_rmse_locked'] = float(s1_anchor[aname][0][0])
    if mode == 'mamba':
        OUT['scene1_ours_tube'] = tu
print(f"scene1 惩罚分支触发次数 = {PENALTY_HITS}（预期 0，即与无惩罚等价）", flush=True)
OUT['scene1_ref'] = ref1
OUT['scene1_seed'] = 0

# ========== 场景2：figure8 + 随机扰动（10 种子） ==========
print("\n===== 场景2 figure8+随机扰动（10种子） =====", flush=True)
rerun = as_dict(np.load('rerun_no_penalty_results.npz', allow_pickle=True)['results'])
exp9 = None
for f in sorted(glob.glob('exp9*.npz')):
    try:
        cand = as_dict(np.load(f, allow_pickle=True)['results'])
        if 'baseline_standard' in cand:
            exp9 = cand
            print(f"exp9 锚点文件: {f}", flush=True)
            break
    except Exception:
        continue
assert exp9 is not None, "未找到含 baseline_standard 的 exp9 npz"
ref2 = figure8_path()
rl, errs, tubs = [], {}, {}
for seed in range(10):
    e, tu = run_sim(ref2, seed, 'mamba')
    r, _, _ = metric_vs_tube(e, tu, 200)
    rl.append(r); errs[seed] = e; tubs[seed] = tu
    print(f"  scene2 ours seed{seed}: RMSE={r:.6f}", flush=True)
lock_a = float(rerun['ablation_A_mamba_adaptive']['rmse'])
check("scene2 ours(ablation_A)", np.mean(rl), lock_a)
s2 = rep_seed(rl, lock_a)
OUT.update(scene2_ours_err=errs[s2], scene2_ours_tube=tubs[s2],
           scene2_ours_rmse=rl[s2], scene2_ours_rmse_locked=lock_a, scene2_seed=s2)
rl2, errs2 = [], {}
for seed in range(10):
    e, tu, _ = sc.run_sim(ref2, seed, 'standard')
    r, _, _ = metric_vs_tube(e, tu, 200)
    rl2.append(r); errs2[seed] = e
    print(f"  scene2 standard seed{seed}: RMSE={r:.6f}", flush=True)
lock_s2 = float(exp9['baseline_standard']['rmse'])
check("scene2 standard(exp9 baseline)", np.mean(rl2), lock_s2)
OUT.update(scene2_std_err=errs2[s2], scene2_std_rmse=rl2[s2], scene2_std_rmse_locked=lock_s2)
rl3, errs3 = [], {}
for seed in range(10):
    e, tu = run_sim(ref2, seed, 'ekf')
    r, _, _ = metric_vs_tube(e, tu, 200)
    rl3.append(r); errs3[seed] = e
    print(f"  scene2 ekf seed{seed}: RMSE={r:.6f}", flush=True)
lock_e2 = float(rerun['ablation_E_ekf_fixed_015']['rmse'])
check("scene2 ekf(ablation_E)", np.mean(rl3), lock_e2)
OUT.update(scene2_ekf_err=errs3[s2], scene2_ekf_rmse=rl3[s2], scene2_ekf_rmse_locked=lock_e2)
OUT['scene2_ref'] = ref2

# ========== 场景3：sine + 复合扰动（10 种子） ==========
print("\n===== 场景3 sine+复合扰动（10种子） =====", flush=True)
probe = np.load('scene3_nopen_probe.npz')
fix3 = np.load('day44_scene3_v3_fix3.npz', allow_pickle=True)['all_results'].item()
ref3 = sine_path()
th3 = np.arctan(3.0*2*np.pi/10)
rl, vls, errs, tubs = [], [], {}, {}
for seed in range(10):
    e, tu = run_sim(ref3, seed, 'mamba', disturb='composite', init_theta=th3)
    r, _, vi = metric_vs_tube(e, tu, 200)
    rl.append(r); vls.append(vi); errs[seed] = e; tubs[seed] = tu
    print(f"  scene3 ours seed{seed}: RMSE={r:.6f} Viol={vi}", flush=True)
dr = float(np.max(np.abs(np.array(rl) - probe['rmse'])))
dv = int(np.max(np.abs(np.array(vls) - probe['viol_tube'])))
ok = (dr <= 1e-12) and (dv == 0)
REPORT.append(ok)
print(f"[{'OK' if ok else 'BAD'}] scene3 ours(probe): rmse逐种子max|diff|={dr:.2e} viol_tube逐种子maxdiff={dv}", flush=True)
s3 = rep_seed(rl, float(np.mean(probe['rmse'])))
OUT.update(scene3_ours_err=errs[s3], scene3_ours_tube=tubs[s3],
           scene3_ours_rmse=rl[s3], scene3_ours_rmse_locked=float(np.mean(probe['rmse'])), scene3_seed=s3)
for mode, aname, tag in [('standard', 'Standard MPC', 'std'), ('ekf', 'EKF-MPC', 'ekf')]:
    rlx, errx = [], {}
    for seed in range(10):
        e, tu, _ = sc.run_sim(ref3, seed, mode, disturb='composite', init_theta=th3)
        r, _, _ = metric_vs_tube(e, tu, 200)
        rlx.append(r); errx[seed] = e
        print(f"  scene3 {mode} seed{seed}: RMSE={r:.6f}", flush=True)
    want = np.array([row[0] for row in fix3[aname]], dtype=float)
    dx = float(np.max(np.abs(np.array(rlx) - want)))
    ok = dx <= 1e-12
    REPORT.append(ok)
    print(f"[{'OK' if ok else 'BAD'}] scene3 {mode}(fix3 {aname}): rmse逐种子max|diff|={dx:.2e}", flush=True)
    OUT[f'scene3_{tag}_err'] = errx[s3]
    OUT[f'scene3_{tag}_rmse'] = rlx[s3]
    OUT[f'scene3_{tag}_rmse_locked'] = float(np.mean(want))
OUT['scene3_ref'] = ref3

# ========== 路径一致性 & 保存 ==========
try:
    pc8 = float(np.max(np.abs(sc.figure8_path() - ref2)))
    pcs = float(np.max(np.abs(sc.sine_path() - ref3)))
    print(f"\npath check: figure8 max|diff|={pc8:.2e}  sine max|diff|={pcs:.2e}", flush=True)
except Exception as ex:
    print(f"\npath check 跳过（sim_core 无同名路径函数）: {ex}", flush=True)

n_bad = REPORT.count(False)
print(f"\n===== 同源验证汇总: {len(REPORT)-n_bad}/{len(REPORT)} 通过 =====", flush=True)
if n_bad == 0:
    np.savez('fig3_trajectories.npz', **OUT)
    print("已保存 fig3_trajectories.npz", flush=True)
else:
    print("存在不一致，未保存。请把日志发回。", flush=True)
