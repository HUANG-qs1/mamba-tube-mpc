# -*- coding: utf-8 -*-
"""export_fig8_traj.py - Fig.8 轨迹导出：同源验证(90 run) + 代表种子轨迹落盘"""
import numpy as np
import torch
import casadi as ca

device = 'cuda'
from mamba_predictor import MambaDisturbancePredictor
mamba_model = MambaDisturbancePredictor(2,128,2,10).to(device)
mamba_model.load_state_dict(torch.load('best_mamba_v3.pt', map_location=device, weights_only=True))
mamba_model.eval()
print("mamba loaded", flush=True)

norm = np.load('norm_params_v3.npz')
x_mean = norm['x_mean'].squeeze(); x_std = norm['x_std'].squeeze()
y_mean = norm['y_mean'].squeeze(); y_std = norm['y_std'].squeeze()

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

def mpc_with_tube(state, ref_segment, tube_size, N=10, dt=0.1):
    x,y,theta = state
    U = ca.MX.sym('U',2*N); v = U[0::2]; o = U[1::2]
    X = [x,y,theta]; obj = 0
    for k in range(N):
        xn = X[0]+v[k]*ca.cos(X[2])*dt; yn = X[1]+v[k]*ca.sin(X[2])*dt; tn = X[2]+o[k]*dt
        X = [xn, yn, tn]
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

def run_sim_mamba(ref_path, seed, N=10, dt=0.1, total_time=100.0):
    """逐行复刻 rerun_no_penalty.py run_sim 的 mamba 路径"""
    np.random.seed(seed)
    robot = RobotFree(0, 0, 0.0, dt)
    steps = int(total_time/dt)
    errors, tubes = [], []
    for t in range(steps):
        idx = min(int(t*dt*10), len(ref_path)-1)
        seg_ids = np.clip(idx + (np.arange(N)*dt*10).astype(int), 0, len(ref_path)-1)
        ref_seg = ref_path[seg_ids]
        ex = robot.state[0]-ref_path[idx,0]; ey = robot.state[1]-ref_path[idx,1]
        errors.append([ex,ey])
        tube = compute_tube(np.array(errors[-100:]), mamba_model) if len(errors)>=100 else 0.15
        tubes.append(tube)
        d = get_random_disturbance(t)
        v, w = mpc_with_tube(robot.state, ref_seg, tube, N, dt)
        robot.step(v, w, d)
    return np.array(errors), np.array(tubes)

def metric_vs_015(errors, warmup):
    ea = errors[warmup:]
    en = np.hypot(ea[:,0], ea[:,1])
    return np.sqrt(np.mean(en**2)), en.max(), int(np.sum(en > 0.15))

# ---- standard / ekf 走 sim_core_np（与 exp6_baselines.py 相同入口） ----
from sim_core_np import run_sim as run_sim_core, metric_vs_tube
from sim_core_np import square_path as sq_c, spiral_path as sp_c, lissajous_path as li_c

PATHS = {'square': square_path(), 'spiral': spiral_path(), 'lissajous': lissajous_path()}
for k, p2 in [('square', sq_c()), ('spiral', sp_c()), ('lissajous', li_c())]:
    print(f"path check {k}: max diff = {np.abs(PATHS[k]-p2).max():.2e}", flush=True)

rerun = np.load('rerun_no_penalty_results.npz', allow_pickle=True)['results'].tolist()
while isinstance(rerun, list): rerun = rerun[0]
base = np.load('exp6_baselines.npz', allow_pickle=True)['results'].tolist()
while isinstance(base, list): base = base[0]

OUT = {}
for tname, tpath in PATHS.items():
    rm_m, rm_s, rm_e, cache = [], [], [], {}
    for seed in range(10):
        em, tm = run_sim_mamba(tpath, seed)
        es, ts, _ = run_sim_core(tpath, seed, 'standard')
        ee, te, _ = run_sim_core(tpath, seed, 'ekf')
        cache[seed] = (em, tm, es, ee)
        rm_m.append(metric_vs_015(em, 200)[0])
        rm_s.append(metric_vs_tube(es, ts, 200)[0])
        rm_e.append(metric_vs_tube(ee, te, 200)[0])
    lm = rerun[f'exp6_mamba_{tname}']['rmse']
    ls = base[f'{tname}_standard']['rmse']
    le = base[f'{tname}_ekf']['rmse']
    print(f"[{tname}] mamba {np.mean(rm_m):.6f} vs locked {lm:.6f}", flush=True)
    print(f"[{tname}] standard {np.mean(rm_s):.6f} vs locked {ls:.6f}", flush=True)
    print(f"[{tname}] ekf {np.mean(rm_e):.6f} vs locked {le:.6f}", flush=True)
    sd = int(np.argmin([abs(x - lm) for x in rm_m]))
    print(f"[{tname}] display seed = {sd} (mamba rmse {rm_m[sd]:.4f} vs mean {lm:.4f})", flush=True)
    em, tm, es, ee = cache[sd]
    OUT[f'{tname}_ref'] = tpath
    OUT[f'{tname}_ours_err'] = em
    OUT[f'{tname}_std_err'] = es
    OUT[f'{tname}_ekf_err'] = ee
    OUT[f'{tname}_tube'] = tm
    OUT[f'{tname}_seed'] = np.int64(sd)

np.savez('fig8_trajectories.npz', **OUT)
print("saved: fig8_trajectories.npz", flush=True)
