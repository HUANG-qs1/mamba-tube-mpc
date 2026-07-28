"""实验1补测：各参数下的Tube覆盖率（err>tube的次数）+ 平均Tube宽度"""
import numpy as np
import torch
import casadi as ca
import time

device = 'cuda'
from mamba_predictor import MambaDisturbancePredictor
mamba_model = MambaDisturbancePredictor(2,128,2,10).to(device)
mamba_model.load_state_dict(torch.load('best_mamba_v3.pt', map_location=device, weights_only=True))
mamba_model.eval()

norm = np.load('norm_params_v3.npz')
x_mean = norm['x_mean'].squeeze(); x_std = norm['x_std'].squeeze()
y_mean = norm['y_mean'].squeeze(); y_std = norm['y_std'].squeeze()

def figure8_path(a=3.0, n=1000):
    t = np.linspace(0, 2*np.pi, n)
    return np.column_stack([a*np.sin(2*t), a*np.sin(t)])
ref_path = figure8_path()

class RobotFree:
    def __init__(self, x=0.0, y=0.0, theta=0.0, dt=0.1):
        self.state = np.array([x,y,theta], float); self.dt = dt
    def step(self, v, omega, d=None):
        x,y,theta = self.state; dt = self.dt
        v_act = v + (d[0] if d is not None else 0)
        xn = x + v_act*np.cos(theta)*dt; yn = y + v_act*np.sin(theta)*dt
        tn = np.arctan2(np.sin(theta+omega*dt), np.cos(theta+omega*dt))
        self.state = np.array([xn,yn,tn])
        return self.state

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

def compute_tube(hist, w_base, w_min, kappa):
    x = (hist - x_mean)/(x_std+1e-8)
    xt = torch.FloatTensor(x).unsqueeze(0).to(device)
    with torch.no_grad():
        pn = mamba_model(xt).cpu().numpy()[0]
    pred = pn*y_std + y_mean
    max_d = np.max(np.hypot(pred[:,0], pred[:,1]))
    return max(w_min, w_base + kappa*max_d)

print("实验1补测：Tube覆盖率", flush=True)
results = []
defs = []
for wb in [0.00,0.01,0.02,0.05,0.10]: defs.append((f'wbase_{wb}', wb, 0.08, 1.0))
for wm in [0.02,0.05,0.08,0.12,0.15]: defs.append((f'wmin_{wm}', 0.02, wm, 1.0))
for kp in [0.0,0.5,1.0,1.5,2.0]:      defs.append((f'kappa_{kp}', 0.02, 0.08, kp))

for name, wb, wm, kp in defs:
    rl, cl, tl = [], [], []
    for seed in range(10):
        np.random.seed(seed)
        robot = RobotFree(0,0,0,0.1)
        errors, tubes = [], []
        for t in range(1000):
            idx = min(t, len(ref_path)-1)
            ref_seg = ref_path[idx:idx+10]
            if len(ref_seg) < 10:
                ref_seg = np.vstack([ref_seg, np.tile(ref_path[-1], (10-len(ref_seg),1))])
            ex = robot.state[0]-ref_path[idx,0]; ey = robot.state[1]-ref_path[idx,1]
            errors.append([ex,ey])
            tube = compute_tube(np.array(errors[-100:]), wb, wm, kp) if len(errors)>=100 else 0.15
            tubes.append(tube)
            v, w = mpc_with_tube(robot.state, ref_seg, tube)
            robot.step(v, w, [-0.1+0.05*np.random.randn(), 0])
        ea = np.array(errors[200:]); ta = np.array(tubes[200:])
        en = np.hypot(ea[:,0], ea[:,1])
        cov = 100.0 * (1 - np.sum(en > ta)/len(en))
        rl.append(np.sqrt(np.mean(en**2))); cl.append(cov); tl.append(np.mean(ta))
    results.append({'name':name,'rmse':np.mean(rl),'coverage':np.mean(cl),'tube_mean':np.mean(tl)})
    print(f"{name}: RMSE={np.mean(rl):.4f} 覆盖率={np.mean(cl):.1f}% 平均Tube={np.mean(tl):.3f}", flush=True)

np.savez('exp1_coverage.npz', results=np.array(results, dtype=object))
print("完成，保存: exp1_coverage.npz", flush=True)
