"""无惩罚重跑包：消融A-E / 场景2固定Tube / 场景3固定Tube / 实验1 / 实验4 Mamba行 / 实验6 Mamba行
实现逐行复刻原脚本，唯一改动：robot.step不施加半速惩罚"""
import numpy as np
import torch
import casadi as ca
import time

device = 'cuda'
from mamba_predictor import MambaDisturbancePredictor
from lstm_predictor import LSTMDisturbancePredictor
mamba_model = MambaDisturbancePredictor(2,128,2,10).to(device)
mamba_model.load_state_dict(torch.load('best_mamba_v3.pt', map_location=device, weights_only=True))
mamba_model.eval()
lstm_model = LSTMDisturbancePredictor(2,128,2,10).to(device)
lstm_model.load_state_dict(torch.load('best_lstm_model.pt', map_location=device, weights_only=True))
lstm_model.eval()
print("模型加载完成", flush=True)

norm = np.load('norm_params_v3.npz')
x_mean = norm['x_mean'].squeeze(); x_std = norm['x_std'].squeeze()
y_mean = norm['y_mean'].squeeze(); y_std = norm['y_std'].squeeze()

# ========== 轨迹 ==========
def figure8_path(a=3.0, n=1000):
    t = np.linspace(0, 2*np.pi, n)
    return np.column_stack([a*np.sin(2*t), a*np.sin(t)])

def sine_path(a=3.0, n=1000, cyc=2):
    x = np.linspace(0, 10, n)
    return np.column_stack([x, a*np.sin(cyc*np.pi*x/10)])

def square_path(side=2.0, speed=0.5, n=1000, dt=0.1):
    pts = []; period = 4*side/speed
    for i in range(n):
        s = ((i*dt) % period)*speed
        if s < side:      p = [s - side/2, -side/2]
        elif s < 2*side:  p = [side/2, s - 1.5*side]
        elif s < 3*side:  p = [2.5*side - s, side/2]
        else:             p = [-side/2, 3.5*side - s]
        pts.append(p)
    return np.array(pts)

def spiral_path(a=0.1, b=0.15, n=1000, dt=0.1):
    pts = []
    for i in range(n):
        theta = 0.2*(i*dt); r = a + b*theta
        pts.append([r*np.cos(theta), r*np.sin(theta)])
    return np.array(pts)

def lissajous_path(A=2.0, B=2.0, n=1000, dt=0.1):
    pts = []
    for i in range(n):
        t = i*dt
        pts.append([A*np.sin(0.3*t + np.pi/2), B*np.sin(0.2*t)])
    return np.array(pts)

# ========== 机器人（无惩罚） ==========
class RobotFree:
    def __init__(self, x=0.0, y=0.0, theta=0.0, dt=0.1):
        self.state = np.array([x,y,theta], float)
        self.dt = dt
    def step(self, v, omega, d=None):
        x,y,theta = self.state; dt = self.dt
        v_act = v + (d[0] if d is not None else 0)
        xn = x + v_act*np.cos(theta)*dt
        yn = y + v_act*np.sin(theta)*dt
        tn = np.arctan2(np.sin(theta+omega*dt), np.cos(theta+omega*dt))
        self.state = np.array([xn,yn,tn])
        return self.state

# ========== EKF（复刻原实现） ==========
class EKF:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.x_est = np.zeros(4); self.P = np.eye(4)*0.1
        self.Q = np.diag([0.01,0.01,0.001,0.001]); self.R = np.diag([0.01,0.01,0.001])
    def update(self, z, u):
        x,y,theta,d_v = self.x_est; v,omega = u; dt = self.dt
        xp = x + (v+d_v)*np.cos(theta)*dt
        yp = y + (v+d_v)*np.sin(theta)*dt
        tp = theta + omega*dt
        xpv = np.array([xp,yp,tp,d_v])
        F = np.array([[1,0,-(v+d_v)*np.sin(theta)*dt,np.cos(theta)*dt],
                      [0,1,(v+d_v)*np.cos(theta)*dt,np.sin(theta)*dt],
                      [0,0,1,0],[0,0,0,1]])
        Pp = F@self.P@F.T + self.Q
        H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        yt = z - np.array([xp,yp,tp])
        S = H@Pp@H.T + self.R
        K = Pp@H.T@np.linalg.inv(S)
        self.x_est = xpv + K@yt
        self.P = (np.eye(4)-K@H)@Pp
        return self.x_est[3]

# ========== MPC（复刻原实现） ==========
def mpc_no_tube(state, ref_segment, N=10, dt=0.1):
    x,y,theta = state
    U = ca.MX.sym('U',2*N); v = U[0::2]; o = U[1::2]
    X = [x,y,theta]; obj = 0
    for k in range(N):
        xn = X[0]+v[k]*ca.cos(X[2])*dt; yn = X[1]+v[k]*ca.sin(X[2])*dt; tn = X[2]+o[k]*dt
        X = [xn,yn,tn]
        obj += (X[0]-ref_segment[k,0])**2 + (X[1]-ref_segment[k,1])**2 + 0.01*v[k]**2 + 0.01*o[k]**2
    g = [X[0]-x, X[1]-y, X[2]-theta]
    solver = ca.nlpsol('s','ipopt',{'x':U,'f':obj,'g':ca.vertcat(*g)},{'print_time':False,'ipopt':{'print_level':0}})
    res = solver(x0=[0.5,0.0]*N, lbx=[-1.0]*(2*N), ubx=[1.0]*(2*N), lbg=0, ubg=0)
    sol = np.array(res['x']).flatten()
    return sol[0], sol[1]

def mpc_with_tube(state, ref_segment, tube_size, N=10, dt=0.1):
    x,y,theta = state
    U = ca.MX.sym('U',2*N); v = U[0::2]; o = U[1::2]
    X = [x,y,theta]; obj = 0
    for k in range(N):
        xn = X[0]+v[k]*ca.cos(X[2])*dt; yn = X[1]+v[k]*ca.sin(X[2])*dt; tn = X[2]+o[k]*dt
        X = [xn,yn,tn]
        ex = X[0]-ref_segment[k,0]; ey = X[1]-ref_segment[k,1]
        dist = ca.sqrt(ex**2+ey**2)
        tv = ca.fmax(0, dist-tube_size)
        obj += ex**2+ey**2 + 2.0*tv**2 + 0.01*v[k]**2 + 0.01*o[k]**2
    g = [X[0]-x, X[1]-y, X[2]-theta]
    solver = ca.nlpsol('s','ipopt',{'x':U,'f':obj,'g':ca.vertcat(*g)},{'print_time':False,'ipopt':{'print_level':0}})
    res = solver(x0=[0.5,0.0]*N, lbx=[-1.0]*(2*N), ubx=[1.0]*(2*N), lbg=0, ubg=0)
    sol = np.array(res['x']).flatten()
    return sol[0], sol[1]

def compute_tube(history_errors, model, w_base=0.02, w_min=0.08, kappa=1.0):
    x = (history_errors - x_mean) / (x_std + 1e-8)
    x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(x_tensor).cpu().numpy()[0]
    pred = pred_norm * y_std + y_mean
    pred_norms = np.sqrt(pred[:,0]**2 + pred[:,1]**2)
    max_d = np.max(pred_norms)
    tube_size = w_base + kappa * max_d
    return max(w_min, tube_size)

def get_random_disturbance(t):
    return [-0.1 + 0.05*np.random.randn(), 0]

def get_composite_disturbance(t):
    return [-0.1 + 0.05*np.random.randn() - 0.0005*t, 0]

# ========== 通用闭环仿真 ==========
def run_sim(ref_path, seed, mode, N=10, dt=0.1, disturb='random', init_theta=0.0,
            fixed_tube=0.15, w_base=0.02, w_min=0.08, kappa=1.0, total_time=100.0):
    """mode: 'mamba'/'lstm'/'fixed'/'ekf'。返回(errors, tubes, solve_times)"""
    np.random.seed(seed)
    robot = RobotFree(0, 0, init_theta, dt)
    ekf = EKF(dt) if mode == 'ekf' else None
    steps = int(total_time/dt)
    errors, tubes, solve_times = [], [], []
    for t in range(steps):
        idx = min(int(t*dt*10), len(ref_path)-1)
        seg_ids = np.clip(idx + (np.arange(N)*dt*10).astype(int), 0, len(ref_path)-1)
        ref_seg = ref_path[seg_ids]
        ex = robot.state[0]-ref_path[idx,0]; ey = robot.state[1]-ref_path[idx,1]
        errors.append([ex,ey])
        if mode == 'mamba':
            tube = compute_tube(np.array(errors[-100:]), mamba_model, w_base, w_min, kappa) if len(errors)>=100 else 0.15
        elif mode == 'lstm':
            tube = compute_tube(np.array(errors[-100:]), lstm_model, w_base, w_min, kappa) if len(errors)>=100 else 0.15
        else:
            tube = fixed_tube
        tubes.append(tube)
        d = get_composite_disturbance(t) if disturb=='composite' else get_random_disturbance(t)
        t0 = time.perf_counter()
        if mode == 'ekf':
            v_mpc, w_mpc = mpc_no_tube(robot.state, ref_seg, N, dt)
            d_est = ekf.update(robot.state.copy(), [v_mpc, w_mpc])
            v, w = v_mpc - d_est, w_mpc
        else:
            v, w = mpc_with_tube(robot.state, ref_seg, tube, N, dt)
        solve_times.append(time.perf_counter()-t0)
        robot.step(v, w, d)
    return np.array(errors), np.array(tubes), np.array(solve_times)

def metric_vs_tube(errors, tubes, warmup):
    ea = errors[warmup:]; ta = tubes[warmup:]
    en = np.hypot(ea[:,0], ea[:,1])
    return np.sqrt(np.mean(en**2)), en.max(), int(np.sum(en > ta))

def metric_vs_015(errors, warmup):
    ea = errors[warmup:]
    en = np.hypot(ea[:,0], ea[:,1])
    return np.sqrt(np.mean(en**2)), en.max(), int(np.sum(en > 0.15))

ALL = {}
fig8 = figure8_path()
sine = sine_path()
init_theta_s3 = np.arctan(3.0*2*np.pi/10)

# ========== PART 1: 消融A-E（场景2协议，Viol按tube） ==========
print("\n===== PART1 消融A-E =====", flush=True)
ablation_defs = [('A_mamba_adaptive','mamba',{}), ('B_fixed_015','fixed',{'fixed_tube':0.15}),
                 ('C_mamba_fixed_015','fixed',{'fixed_tube':0.15}), ('D_lstm_adaptive','lstm',{}),
                 ('E_ekf_fixed_015','ekf',{'fixed_tube':0.15})]
for name, mode, kw in ablation_defs:
    rl, ml, vl = [], [], []
    for seed in range(10):
        e, tu, _ = run_sim(fig8, seed, mode, **kw)
        r, m, vi = metric_vs_tube(e, tu, 200)
        rl.append(r); ml.append(m); vl.append(vi)
    ALL[f'ablation_{name}'] = {'rmse': np.mean(rl), 'rmse_std': np.std(rl), 'maxerr': np.mean(ml), 'viol': np.mean(vl)}
    print(f"{name}: RMSE={np.mean(rl):.4f}+-{np.std(rl):.4f} MaxErr={np.mean(ml):.4f} Viol={np.mean(vl):.1f}", flush=True)

# ========== PART 2: 场景2固定Tube 0.05/0.10 ==========
print("\n===== PART2 场景2固定Tube =====", flush=True)
for tt in [0.05, 0.10]:
    rl, ml, vl = [], [], []
    for seed in range(10):
        e, tu, _ = run_sim(fig8, seed, 'fixed', fixed_tube=tt)
        r, m, vi = metric_vs_tube(e, tu, 200)
        rl.append(r); ml.append(m); vl.append(vi)
    ALL[f'scene2_fixed_{tt}'] = {'rmse': np.mean(rl), 'rmse_std': np.std(rl), 'maxerr': np.mean(ml), 'viol': np.mean(vl)}
    print(f"scene2 fixed={tt}: RMSE={np.mean(rl):.4f}+-{np.std(rl):.4f} MaxErr={np.mean(ml):.4f} Viol={np.mean(vl):.1f}", flush=True)

# ========== PART 3: 场景3固定Tube 0.05/0.10（正弦+复合扰动） ==========
print("\n===== PART3 场景3固定Tube =====", flush=True)
for tt in [0.05, 0.10]:
    rl, ml, vl = [], [], []
    for seed in range(10):
        e, tu, _ = run_sim(sine, seed, 'fixed', fixed_tube=tt, disturb='composite', init_theta=init_theta_s3)
        r, m, vi = metric_vs_tube(e, tu, 200)
        rl.append(r); ml.append(m); vl.append(vi)
    ALL[f'scene3_fixed_{tt}'] = {'rmse': np.mean(rl), 'rmse_std': np.std(rl), 'maxerr': np.mean(ml), 'viol': np.mean(vl)}
    print(f"scene3 fixed={tt}: RMSE={np.mean(rl):.4f}+-{np.std(rl):.4f} MaxErr={np.mean(ml):.4f} Viol={np.mean(vl):.1f}", flush=True)

# ========== PART 4: 实验1（15配置，Viol按0.15） ==========
print("\n===== PART4 实验1 Tube敏感性 =====", flush=True)
exp1_defs = []
for wb in [0.00,0.01,0.02,0.05,0.10]: exp1_defs.append((f'wbase_{wb}', dict(w_base=wb, w_min=0.08, kappa=1.0)))
for wm in [0.02,0.05,0.08,0.12,0.15]: exp1_defs.append((f'wmin_{wm}', dict(w_base=0.02, w_min=wm, kappa=1.0)))
for kp in [0.0,0.5,1.0,1.5,2.0]:      exp1_defs.append((f'kappa_{kp}', dict(w_base=0.02, w_min=0.08, kappa=kp)))
for name, kw in exp1_defs:
    rl, ml, vl = [], [], []
    for seed in range(10):
        e, tu, _ = run_sim(fig8, seed, 'mamba', **kw)
        r, m, vi = metric_vs_015(e, 200)
        rl.append(r); ml.append(m); vl.append(vi)
    ALL[f'exp1_{name}'] = {'rmse': np.mean(rl), 'rmse_std': np.std(rl), 'maxerr': np.mean(ml), 'viol': np.mean(vl)}
    print(f"{name}: RMSE={np.mean(rl):.4f}+-{np.std(rl):.4f} MaxErr={np.mean(ml):.4f} Viol={np.mean(vl):.1f}", flush=True)

# ========== PART 5: 实验4 Mamba行（Viol按0.15） ==========
print("\n===== PART5 实验4 Mamba行 =====", flush=True)
for N in [5,10,20]:
    for dt in [0.05,0.1,0.2]:
        rl, ml, vl, sl = [], [], [], []
        for seed in range(10):
            e, tu, st = run_sim(fig8, seed, 'mamba', N=N, dt=dt)
            wu = int(20/dt)
            r, m, vi = metric_vs_015(e, wu)
            rl.append(r); ml.append(m); vl.append(vi); sl.append(np.mean(st)*1000)
        ALL[f'exp4_mamba_N{N}_dt{dt}'] = {'rmse': np.mean(rl), 'rmse_std': np.std(rl), 'maxerr': np.mean(ml), 'viol': np.mean(vl), 'solve_ms': np.mean(sl)}
        print(f"N={N} dt={dt}: RMSE={np.mean(rl):.4f}+-{np.std(rl):.4f} MaxErr={np.mean(ml):.4f} Viol={np.mean(vl):.1f} Solve={np.mean(sl):.2f}ms", flush=True)

# ========== PART 6: 实验6 Mamba行（三轨迹，Viol按0.15） ==========
print("\n===== PART6 实验6 Mamba行 =====", flush=True)
for tname, tpath in [('square',square_path()), ('spiral',spiral_path()), ('lissajous',lissajous_path())]:
    rl, ml, vl = [], [], []
    for seed in range(10):
        e, tu, _ = run_sim(tpath, seed, 'mamba')
        r, m, vi = metric_vs_015(e, 200)
        rl.append(r); ml.append(m); vl.append(vi)
    ALL[f'exp6_mamba_{tname}'] = {'rmse': np.mean(rl), 'rmse_std': np.std(rl), 'maxerr': np.mean(ml), 'viol': np.mean(vl)}
    print(f"{tname}: RMSE={np.mean(rl):.4f}+-{np.std(rl):.4f} MaxErr={np.mean(ml):.4f} Viol={np.mean(vl):.1f}", flush=True)

np.savez('rerun_no_penalty_results.npz', results=np.array([ALL], dtype=object))
print("\n全部重跑完成！保存: rerun_no_penalty_results.npz", flush=True)
