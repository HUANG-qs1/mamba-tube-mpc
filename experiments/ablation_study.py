import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from mamba_predictor import MambaDisturbancePredictor
import casadi as ca

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== LSTM模型定义 ====================
class LSTMDisturbancePredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, pred_len=10):
        super().__init__()
        self.pred_len = pred_len
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim * pred_len)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm1(x)
        x = self.dropout(self.norm1(x))
        x, _ = self.lstm2(x)
        x = self.dropout(self.norm2(x))
        last = x[:, -1, :]
        out = self.output_proj(last)
        out = out.view(-1, self.pred_len, 2)
        return out

# ==================== 加载模型 ====================
print("[1/3] 加载 Mamba 模型...")
mamba_model = MambaDisturbancePredictor(input_dim=2, hidden_dim=128, output_dim=2, pred_len=10).to(device)
mamba_model.load_state_dict(torch.load('best_mamba_v3.pt', map_location=device))
mamba_model.eval()

print("[2/3] 加载 LSTM 模型...")
lstm_model = LSTMDisturbancePredictor(input_dim=2, hidden_dim=128, output_dim=2, pred_len=10).to(device)
try:
    lstm_model.load_state_dict(torch.load('best_lstm_model.pt', map_location=device))
    lstm_model.eval()
    lstm_available = True
    print("    -> LSTM 加载成功")
except Exception as e:
    lstm_available = False
    print(f"    -> LSTM 加载失败: {e}")
    print("    -> 配置D (LSTM-Tube) 将跳过")

norm = np.load('norm_params_v3.npz')
x_mean = norm['x_mean'].squeeze()
x_std = norm['x_std'].squeeze()
y_mean = norm['y_mean'].squeeze()
y_std = norm['y_std'].squeeze()

# ==================== 场景设置 ====================
def figure8_path(a=3.0, num_points=1000):
    t = np.linspace(0, 2*np.pi, num_points)
    x = a * np.sin(2*t)
    y = a * np.sin(t)
    return np.column_stack([x, y])

ref_path = figure8_path()

class RobotWithPenalty:
    def __init__(self, x=0.0, y=0.0, theta=0.0, dt=0.1):
        self.state = np.array([x, y, theta], dtype=float)
        self.dt = dt
        self.history = [self.state.copy()]
    def step(self, v, omega, d=None, tube_size=None):
        x, y, theta = self.state
        dt = self.dt
        if tube_size is not None:
            ref_idx = min(len(self.history)-1, len(ref_path)-1)
            ref_point = ref_path[ref_idx % len(ref_path)]
            err = np.sqrt((x - ref_point[0])**2 + (y - ref_point[1])**2)
            if err > tube_size:
                v, omega = v * 0.5, omega * 0.5
        v_act = v + (d[0] if d is not None else 0)
        x_new = x + v_act * np.cos(theta) * dt
        y_new = y + v_act * np.sin(theta) * dt
        theta_new = theta + omega * dt
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        self.state = np.array([x_new, y_new, theta_new])
        self.history.append(self.state.copy())
        return self.state

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
            [0, 0, 0, 1]
        ])
        P_pred = F @ self.P @ F.T + self.Q
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        z_pred = np.array([x_pred, y_pred, theta_pred])
        y_tilde = z - z_pred
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x_est = x_pred_vec + K @ y_tilde
        self.P = (np.eye(4) - K @ H) @ P_pred
        return self.x_est[3]

def mpc_no_tube(state, ref_segment, N=10, dt=0.1):
    x, y, theta = state
    U = ca.MX.sym('U', 2*N)
    v = U[0::2]
    o = U[1::2]
    X = [x, y, theta]
    obj = 0
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
    v = U[0::2]
    o = U[1::2]
    X = [x, y, theta]
    obj = 0
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

def get_random_disturbance(t):
    return [-0.1 + 0.05*np.random.randn(), 0]

# ==================== Tube计算函数 ====================
def compute_tube_mamba(history_errors):
    x = (history_errors - x_mean) / (x_std + 1e-8)
    x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = mamba_model(x_tensor).cpu().numpy()[0]
    pred = pred_norm * y_std + y_mean
    pred_norms = np.sqrt(pred[:,0]**2 + pred[:,1]**2)
    max_d = np.max(pred_norms)
    tube_size = 0.02 + 1.0 * max_d
    return max(0.08, tube_size)

def compute_tube_lstm(history_errors):
    x = (history_errors - x_mean) / (x_std + 1e-8)
    x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = lstm_model(x_tensor).cpu().numpy()[0]
    pred = pred_norm * y_std + y_mean
    pred_norms = np.sqrt(pred[:,0]**2 + pred[:,1]**2)
    max_d = np.max(pred_norms)
    tube_size = 0.02 + 1.0 * max_d
    return max(0.08, tube_size)

# ==================== 5组消融配置 ====================
def run_config_A(seed=42):  # Mamba + Adaptive Tube
    np.random.seed(seed)
    robot = RobotWithPenalty(x=0, y=0, theta=0, dt=0.1)
    dt, N = 0.1, 10
    errors, tubes = [], []
    for t in range(1000):
        idx = min(t, len(ref_path)-1)
        ref_seg = ref_path[idx:idx+N]
        if len(ref_seg) < N:
            ref_seg = np.vstack([ref_seg, np.tile(ref_path[-1], (N-len(ref_seg), 1))])
        ex = robot.state[0] - ref_path[idx,0]
        ey = robot.state[1] - ref_path[idx,1]
        errors.append([ex, ey])
        if len(errors) >= 100:
            tube = compute_tube_mamba(np.array(errors[-100:]))
        else:
            tube = 0.15
        tubes.append(tube)
        v, w = mpc_with_tube(robot.state, ref_seg, tube, N, dt)
        d = get_random_disturbance(t)
        robot.step(v, w, d, tube_size=tube)
    return np.array(errors), np.array(tubes)

def run_config_B(seed=42):  # Fixed Tube=0.15
    np.random.seed(seed)
    robot = RobotWithPenalty(x=0, y=0, theta=0, dt=0.1)
    dt, N = 0.1, 10
    errors, tubes = [], []
    for t in range(1000):
        idx = min(t, len(ref_path)-1)
        ref_seg = ref_path[idx:idx+N]
        if len(ref_seg) < N:
            ref_seg = np.vstack([ref_seg, np.tile(ref_path[-1], (N-len(ref_seg), 1))])
        ex = robot.state[0] - ref_path[idx,0]
        ey = robot.state[1] - ref_path[idx,1]
        errors.append([ex, ey])
        tube = 0.15
        tubes.append(tube)
        v, w = mpc_with_tube(robot.state, ref_seg, tube, N, dt)
        d = get_random_disturbance(t)
        robot.step(v, w, d, tube_size=tube)
    return np.array(errors), np.array(tubes)

def run_config_C(seed=42):  # Mamba预测但固定Tube=0.15
    np.random.seed(seed)
    robot = RobotWithPenalty(x=0, y=0, theta=0, dt=0.1)
    dt, N = 0.1, 10
    errors, tubes = [], []
    for t in range(1000):
        idx = min(t, len(ref_path)-1)
        ref_seg = ref_path[idx:idx+N]
        if len(ref_seg) < N:
            ref_seg = np.vstack([ref_seg, np.tile(ref_path[-1], (N-len(ref_seg), 1))])
        ex = robot.state[0] - ref_path[idx,0]
        ey = robot.state[1] - ref_path[idx,1]
        errors.append([ex, ey])
        # 计算但不使用（模拟"有预测但不调整"）
        if len(errors) >= 100:
            _ = compute_tube_mamba(np.array(errors[-100:]))
        tube = 0.15  # 强制固定
        tubes.append(tube)
        v, w = mpc_with_tube(robot.state, ref_seg, tube, N, dt)
        d = get_random_disturbance(t)
        robot.step(v, w, d, tube_size=tube)
    return np.array(errors), np.array(tubes)

def run_config_D(seed=42):  # LSTM + Adaptive Tube
    if not lstm_available:
        return None, None
    np.random.seed(seed)
    robot = RobotWithPenalty(x=0, y=0, theta=0, dt=0.1)
    dt, N = 0.1, 10
    errors, tubes = [], []
    for t in range(1000):
        idx = min(t, len(ref_path)-1)
        ref_seg = ref_path[idx:idx+N]
        if len(ref_seg) < N:
            ref_seg = np.vstack([ref_seg, np.tile(ref_path[-1], (N-len(ref_seg), 1))])
        ex = robot.state[0] - ref_path[idx,0]
        ey = robot.state[1] - ref_path[idx,1]
        errors.append([ex, ey])
        if len(errors) >= 100:
            tube = compute_tube_lstm(np.array(errors[-100:]))
        else:
            tube = 0.15
        tubes.append(tube)
        v, w = mpc_with_tube(robot.state, ref_seg, tube, N, dt)
        d = get_random_disturbance(t)
        robot.step(v, w, d, tube_size=tube)
    return np.array(errors), np.array(tubes)

def run_config_E(seed=42):  # EKF + Fixed Tube=0.15
    np.random.seed(seed)
    robot = RobotWithPenalty(x=0, y=0, theta=0, dt=0.1)
    ekf = EKFDisturbanceEstimator(dt=0.1)
    dt, N = 0.1, 10
    errors, tubes = [], []
    for t in range(1000):
        idx = min(t, len(ref_path)-1)
        ref_seg = ref_path[idx:idx+N]
        if len(ref_seg) < N:
            ref_seg = np.vstack([ref_seg, np.tile(ref_path[-1], (N-len(ref_seg), 1))])
        ex = robot.state[0] - ref_path[idx,0]
        ey = robot.state[1] - ref_path[idx,1]
        errors.append([ex, ey])
        tube = 0.15
        tubes.append(tube)
        v_mpc, w_mpc = mpc_no_tube(robot.state, ref_seg, N, dt)
        d_est = ekf.update(robot.state.copy(), [v_mpc, w_mpc])
        v_cmd = v_mpc - d_est
        d = get_random_disturbance(t)
        robot.step(v_cmd, w_mpc, d, tube_size=tube)
    return np.array(errors), np.array(tubes)

def compute_metrics(errors, tubes, start_idx=200):
    main_errors = errors[start_idx:]
    err_norm = np.sqrt(main_errors[:,0]**2 + main_errors[:,1]**2)
    rmse = np.sqrt(np.mean(err_norm**2))
    max_err = np.max(err_norm)
    violations = 0
    if tubes is not None:
        main_tubes = tubes[start_idx:]
        violations = np.sum(err_norm > main_tubes)
    return rmse, max_err, violations

configs = {
    'A: Mamba+Adaptive (Ours)': run_config_A,
    'B: Fixed Tube=0.15': run_config_B,
    'C: Mamba+Fixed=0.15': run_config_C,
    'D: LSTM+Adaptive': run_config_D,
    'E: EKF+Fixed=0.15': run_config_E,
}

n_runs = 10
all_results = {}

print(f"\n🚀 消融实验：5组配置 × {n_runs}次（场景2：8字形+随机扰动）")
print("预计50-70分钟，请耐心等待...\n")

for config_name, config_func in configs.items():
    print(f"🧪 {config_name}...")
    if 'D: LSTM' in config_name and not lstm_available:
        print("  ⚠️ 跳过（LSTM模型不可用）")
        continue
    results = []
    for run in range(n_runs):
        errors, tubes = config_func(run)
        if errors is None:
            break
        metrics = compute_metrics(errors, tubes)
        results.append(metrics)
        if run == 0:
            print(f"  Run 1: RMSE={metrics[0]:.6f}, Viol={metrics[2]}")
    if results:
        all_results[config_name] = results
        print(f"  ✅ {config_name} 完成")

print("\n" + "="*80)
print("📊 消融实验结果（Ablation Study）")
print("场景：8字形 + 随机扰动（10次运行，mean ± std）")
print("="*80)
print(f"{'配置':<25} {'RMSE[m]':<20} {'MaxErr[m]':<20} {'ViolCount':<15}")
print("-"*80)

for config_name, results in all_results.items():
    rmses = [r[0] for r in results]
    maxerrs = [r[1] for r in results]
    viols = [r[2] for r in results]
    print(f"{config_name:<25} {np.mean(rmses):.6f}±{np.std(rmses):.6f}  {np.mean(maxerrs):.6f}±{np.std(maxerrs):.6f}  {np.mean(viols):.1f}±{np.std(viols):.1f}")

print("="*80)

np.savez('ablation_study.npz', all_results=all_results)
print("\n✅ 数据保存: ablation_study.npz")

# 画图
plt.figure(figsize=(10, 5))
config_names = list(all_results.keys())
rmse_means = [np.mean([r[0] for r in all_results[c]]) for c in config_names]
rmse_stds = [np.std([r[0] for r in all_results[c]]) for c in config_names]
colors = ['green', 'red', 'orange', 'purple', 'blue']
plt.bar(range(len(config_names)), rmse_means, yerr=rmse_stds, color=colors[:len(config_names)], alpha=0.7, capsize=5)
plt.xticks(range(len(config_names)), [c.split(':')[0] for c in config_names], rotation=0)
plt.ylabel('RMSE [m]')
plt.title('Ablation Study: RMSE (mean ± std, n=10)')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('ablation_study.png', dpi=300)
print("✅ 图表保存: ablation_study.png")
