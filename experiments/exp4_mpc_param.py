"""
实验4：MPC参数鲁棒性 (N x dt) — Standard MPC vs Mamba-Tube
场景2（8字轨迹+随机扰动），10 seeds
"""
import numpy as np
import torch
import casadi as ca
import time

device = 'cuda'
from mamba_predictor import MambaDisturbancePredictor
mamba_model = MambaDisturbancePredictor(2,128,2,10).to(device)
mamba_model.load_state_dict(torch.load('best_mamba_v3.pt', map_location=device, weights_only=True))
mamba_model.eval()
print("Mamba模型加载成功", flush=True)

norm = np.load('norm_params_v3.npz')
x_mean = norm['x_mean'].squeeze(); x_std = norm['x_std'].squeeze()
y_mean = norm['y_mean'].squeeze(); y_std = norm['y_std'].squeeze()

def figure8_path(a=3.0, num_points=1000):
    t = np.linspace(0, 2*np.pi, num_points)
    return np.column_stack([a*np.sin(2*t), a*np.sin(t)])

ref_path = figure8_path()

class RobotWithPenalty:
    def __init__(self, x=0.0, y=0.0, theta=0.0, dt=0.1):
        self.state = np.array([x,y,theta], float)
        self.dt = dt
        self.history = [self.state.copy()]
    def step(self, v, omega, d=None, tube_size=None):
        x,y,theta = self.state; dt = self.dt
        if tube_size is not None:
            step_count = len(self.history)-1
            ref_idx = min(int(step_count*self.dt*10), len(ref_path)-1)
            rp = ref_path[ref_idx]
            err = np.hypot(x-rp[0], y-rp[1])
            if err > tube_size:
                v, omega = v*0.5, omega*0.5
        v_act = v + (d[0] if d is not None else 0)
        xn = x + v_act*np.cos(theta)*dt
        yn = y + v_act*np.sin(theta)*dt
        tn = np.arctan2(np.sin(theta+omega*dt), np.cos(theta+omega*dt))
        self.state = np.array([xn,yn,tn])
        self.history.append(self.state.copy())
        return self.state

def solve_mpc(state, ref_seg, N, dt, tube_size=None):
    x,y,theta = state
    U = ca.MX.sym('U', 2*N)
    v = U[0::2]; o = U[1::2]
    X = [x,y,theta]; obj = 0
    for k in range(N):
        xn = X[0]+v[k]*ca.cos(X[2])*dt
        yn = X[1]+v[k]*ca.sin(X[2])*dt
        tn = X[2]+o[k]*dt
        X = [xn,yn,tn]
        ex = X[0]-ref_seg[k,0]; ey = X[1]-ref_seg[k,1]
        obj += ex**2 + ey**2 + 0.01*v[k]**2 + 0.01*o[k]**2
        if tube_size is not None:
            tv = ca.fmax(0, ca.sqrt(ex**2+ey**2)-tube_size)
            obj += 2.0*tv**2
    g = [X[0]-x, X[1]-y, X[2]-theta]
    nlp = {'x':U,'f':obj,'g':ca.vertcat(*g)}
    solver = ca.nlpsol('s','ipopt',nlp,{'print_time':False,'ipopt':{'print_level':0}})
    res = solver(x0=[0.5,0.0]*N, lbx=[-1.0]*(2*N), ubx=[1.0]*(2*N), lbg=0, ubg=0)
    sol = np.array(res['x']).flatten()
    return sol[0], sol[1]

def compute_tube(history_errors):
    x = (history_errors - x_mean) / (x_std + 1e-8)
    x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = mamba_model(x_tensor).cpu().numpy()[0]
    pred = pred_norm * y_std + y_mean
    pred_norms = np.sqrt(pred[:,0]**2 + pred[:,1]**2)
    max_d = np.max(pred_norms)
    tube_size = 0.02 + 1.0 * max_d
    return max(0.08, tube_size)

def run_trial(N, dt, method, seed):
    np.random.seed(seed)
    robot = RobotWithPenalty(0,0,0,dt=dt)
    steps = int(100/dt)
    warmup = int(20/dt)
    errors = []; solve_times = []
    for t in range(steps):
        idx = min(int(t*dt*10), len(ref_path)-1)
        seg_ids = np.clip(idx + (np.arange(N)*dt*10).astype(int), 0, len(ref_path)-1)
        ref_seg = ref_path[seg_ids]
        ex = robot.state[0]-ref_path[idx,0]
        ey = robot.state[1]-ref_path[idx,1]
        errors.append([ex,ey])
        if method == 'mamba':
            tube = compute_tube(np.array(errors[-100:])) if len(errors) >= 100 else 0.15
        else:
            tube = None
        t0 = time.perf_counter()
        v,w = solve_mpc(robot.state, ref_seg, N, dt, tube_size=tube)
        solve_times.append(time.perf_counter()-t0)
        d = [-0.1 + 0.05*np.random.randn(), 0]
        robot.step(v,w,d,tube_size=tube)
    ea = np.array(errors[warmup:])
    en = np.hypot(ea[:,0], ea[:,1])
    rmse = np.sqrt(np.mean(ea[:,0]**2+ea[:,1]**2))
    maxerr = np.max(en)
    viol = int(np.sum(en > 0.15))
    return rmse, maxerr, viol, float(np.mean(solve_times)*1000)

print("="*60, flush=True)
print("实验4：MPC参数鲁棒性扫描 (N x dt)", flush=True)
print("="*60, flush=True)

results = []
for N in [5, 10, 20]:
    for dt in [0.05, 0.1, 0.2]:
        for method in ['standard', 'mamba']:
            rmse_l, max_l, viol_l, t_l = [], [], [], []
            for seed in range(10):
                r, m, vi, ts = run_trial(N, dt, method, seed)
                rmse_l.append(r); max_l.append(m); viol_l.append(vi); t_l.append(ts)
            rec = {'N': N, 'dt': dt, 'method': method,
                   'rmse_mean': np.mean(rmse_l), 'rmse_std': np.std(rmse_l),
                   'maxerr_mean': np.mean(max_l), 'viol_mean': np.mean(viol_l),
                   'solve_ms': np.mean(t_l)}
            results.append(rec)
            print(f"N={N:2d} dt={dt:.2f} {method:8s}: RMSE={rec['rmse_mean']:.4f}+-{rec['rmse_std']:.4f} "
                  f"MaxErr={rec['maxerr_mean']:.4f} Viol={rec['viol_mean']:.1f} Solve={rec['solve_ms']:.2f}ms", flush=True)

np.savez('exp4_mpc_param.npz', results=np.array(results, dtype=object))
print("实验4完成！保存: exp4_mpc_param.npz", flush=True)
