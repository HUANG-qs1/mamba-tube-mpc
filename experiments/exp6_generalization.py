"""
实验6：训练未见轨迹的泛化测试
轨迹：方形/螺旋/Lissajous（训练仅用圆/8字/正弦）
方法：Standard MPC / EKF-MPC / Mamba-Tube（实现严格复刻 day44_scene2_v3_fix2.py）
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

# ========== 三条未见轨迹（1000个参考点，dt=0.1） ==========
def square_path(side=2.0, speed=0.5, n=1000, dt=0.1):
    pts = []
    period = 4*side/speed
    for i in range(n):
        s = ((i*dt) % period)*speed
        if s < side:        p = [s - side/2, -side/2]
        elif s < 2*side:    p = [side/2, s - 1.5*side]
        elif s < 3*side:    p = [2.5*side - s, side/2]
        else:               p = [-side/2, 3.5*side - s]
        pts.append(p)
    return np.array(pts)

def spiral_path(a=0.1, b=0.15, n=1000, dt=0.1):
    pts = []
    for i in range(n):
        theta = 0.2*(i*dt)
        r = a + b*theta
        pts.append([r*np.cos(theta), r*np.sin(theta)])
    return np.array(pts)

def lissajous_path(A=2.0, B=2.0, n=1000, dt=0.1):
    pts = []
    for i in range(n):
        t = i*dt
        pts.append([A*np.sin(0.3*t + np.pi/2), B*np.sin(0.2*t)])
    return np.array(pts)

# ========== 机器人（与主实验一致） ==========
class RobotWithPenalty:
    def __init__(self, x=0.0, y=0.0, theta=0.0, dt=0.1):
        self.state = np.array([x,y,theta], float)
        self.dt = dt
        self.history = [self.state.copy()]
    def step(self, v, omega, d=None, tube_size=None):
        x,y,theta = self.state; dt = self.dt
        if tube_size is not None:
            ref_idx = min(len(self.history)-1, len(ref_path)-1)
            rp = ref_path[ref_idx % len(ref_path)]
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

# ========== EKF估计器（与主实验一致） ==========
class EKFDisturbanceEstimator:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.x_est = np.array([0.0, 0.0, 0.0, 0.0])
        self.P = np.eye(4) * 0.1
        self.Q = np.diag([0.01, 0.01, 0.001, 0.001])
        self.R = np.diag([0.01, 0.01, 0.001])
    def update(self, z, u):
        x, y, theta, d_v = self.x_est
        v, omega = u
        dt = self.dt
        x_pred = x + (v + d_v) * np.cos(theta) * dt
        y_pred = y + (v + d_v) * np.sin(theta) * dt
        theta_pred = theta + omega * dt
        d_v_pred = d_v
        x_pred_vec = np.array([x_pred, y_pred, theta_pred, d_v_pred])
        F = np.array([
            [1, 0, -(v + d_v) * np.sin(theta) * dt, np.cos(theta) * dt],
            [0, 1,  (v + d_v) * np.cos(theta) * dt, np.sin(theta) * dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        P_pred = F @ self.P @ F.T + self.Q
        H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        z_pred = np.array([x_pred, y_pred, theta_pred])
        y_tilde = z - z_pred
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x_est = x_pred_vec + K @ y_tilde
        self.P = (np.eye(4) - K @ H) @ P_pred
        return self.x_est[3]

# ========== MPC（与主实验一致） ==========
def mpc_no_tube(state, ref_segment, N=10, dt=0.1):
    x, y, theta = state
    U = ca.MX.sym('U', 2*N)
    v = U[0::2]; o = U[1::2]
    X = [x, y, theta]; obj = 0
    for k in range(N):
        xn = X[0] + v[k]*ca.cos(X[2])*dt
        yn = X[1] + v[k]*ca.sin(X[2])*dt
        tn = X[2] + o[k]*dt
        X = [xn, yn, tn]
        obj += (X[0]-ref_segment[k,0])**2 + (X[1]-ref_segment[k,1])**2 + 0.01*v[k]**2 + 0.01*o[k]**2
    g = [X[0]-x, X[1]-y, X[2]-theta]
    nlp = {'x':U, 'f':obj, 'g':ca.vertcat(*g)}
    solver = ca.nlpsol('s','ipopt',nlp,{'print_time':False,'ipopt':{'print_level':0}})
    res = solver(x0=[0.5,0.0]*N, lbx=[-1.0]*N+[-1.0]*N, ubx=[1.0]*N+[1.0]*N, lbg=0, ubg=0)
    sol = np.array(res['x']).flatten()
    return sol[0], sol[1]

def mpc_with_tube(state, ref_segment, tube_size, N=10, dt=0.1):
    x, y, theta = state
    U = ca.MX.sym('U', 2*N)
    v = U[0::2]; o = U[1::2]
    X = [x, y, theta]; obj = 0
    for k in range(N):
        xn = X[0] + v[k]*ca.cos(X[2])*dt
        yn = X[1] + v[k]*ca.sin(X[2])*dt
        tn = X[2] + o[k]*dt
        X = [xn, yn, tn]
        err_x = X[0] - ref_segment[k,0]
        err_y = X[1] - ref_segment[k,1]
        dist = ca.sqrt(err_x**2 + err_y**2)
        tube_viol = ca.fmax(0, dist - tube_size)
        obj += err_x**2 + err_y**2 + 2.0*tube_viol**2 + 0.01*v[k]**2 + 0.01*o[k]**2
    g = [X[0]-x, X[1]-y, X[2]-theta]
    nlp = {'x':U, 'f':obj, 'g':ca.vertcat(*g)}
    solver = ca.nlpsol('s','ipopt',nlp,{'print_time':False,'ipopt':{'print_level':0}})
    res = solver(x0=[0.5,0.0]*N, lbx=[-1.0]*N+[-1.0]*N, ubx=[1.0]*N+[1.0]*N, lbg=0, ubg=0)
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

def get_random_disturbance(t):
    return [-0.1 + 0.05*np.random.randn(), 0]

# ========== 三种方法的单次运行 ==========
def run_once(method, ref_path_in, seed):
    global ref_path
    ref_path = ref_path_in
    np.random.seed(seed)
    robot = RobotWithPenalty(x=0, y=0, theta=0, dt=0.1)
    ekf = EKFDisturbanceEstimator(dt=0.1)
    dt, N = 0.1, 10
    errors = []
    for t in range(1000):
        idx = min(t, len(ref_path)-1)
        ref_seg = ref_path[idx:idx+N]
        if len(ref_seg) < N:
            ref_seg = np.vstack([ref_seg, np.tile(ref_path[-1], (N-len(ref_seg), 1))])
        ex = robot.state[0] - ref_path[idx,0]
        ey = robot.state[1] - ref_path[idx,1]
        errors.append([ex, ey])
        if method == 'standard':
            v, w = mpc_no_tube(robot.state, ref_seg, N, dt)
            d = get_random_disturbance(t)
            robot.step(v, w, d)
        elif method == 'ekf':
            v_mpc, w_mpc = mpc_no_tube(robot.state, ref_seg, N, dt)
            d_est = ekf.update(robot.state.copy(), [v_mpc, w_mpc])
            v_cmd = v_mpc - d_est
            d = get_random_disturbance(t)
            robot.step(v_cmd, w_mpc, d)
        elif method == 'mamba':
            tube = compute_tube(np.array(errors[-100:])) if len(errors) >= 100 else 0.15
            v, w = mpc_with_tube(robot.state, ref_seg, tube, N, dt)
            d = get_random_disturbance(t)
            robot.step(v, w, d, tube_size=tube)
    ea = np.array(errors[200:])
    en = np.hypot(ea[:,0], ea[:,1])
    rmse = np.sqrt(np.mean(ea[:,0]**2 + ea[:,1]**2))
    return rmse, float(np.max(en)), int(np.sum(en > 0.15))

# ========== 主实验 ==========
print("="*60, flush=True)
print("实验6：未见轨迹泛化测试", flush=True)
print("="*60, flush=True)

trajectories = {
    'square': square_path(),
    'spiral': spiral_path(),
    'lissajous': lissajous_path()
}
methods = ['standard', 'ekf', 'mamba']

results = []
for tname, tpath in trajectories.items():
    for mname in methods:
        rmse_l, max_l, viol_l = [], [], []
        for seed in range(10):
            r, m, vi = run_once(mname, tpath, seed)
            rmse_l.append(r); max_l.append(m); viol_l.append(vi)
        rec = {'traj': tname, 'method': mname,
               'rmse_mean': np.mean(rmse_l), 'rmse_std': np.std(rmse_l),
               'maxerr_mean': np.mean(max_l), 'viol_mean': np.mean(viol_l)}
        results.append(rec)
        print(f"{tname:10s} {mname:8s}: RMSE={rec['rmse_mean']:.4f}+-{rec['rmse_std']:.4f} "
              f"MaxErr={rec['maxerr_mean']:.4f} Viol={rec['viol_mean']:.1f}", flush=True)

np.savez('exp6_generalization.npz', results=np.array(results, dtype=object))
print("实验6完成！保存: exp6_generalization.npz", flush=True)
