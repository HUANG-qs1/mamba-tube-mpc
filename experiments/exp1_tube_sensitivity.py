"""
实验1：Tube参数 w_base / w_min / kappa 的敏感性分析
目的：回答审稿人Q3，证明Tube参数选择的合理性
"""
import numpy as np
import torch
import casadi as ca
import time

# ========== 加载模型和归一化参数 ==========
device = 'cuda'
from mamba_predictor import MambaDisturbancePredictor
mamba_model = MambaDisturbancePredictor(2,128,2,10).to(device)
mamba_model.load_state_dict(torch.load('best_mamba_v3.pt', map_location=device, weights_only=True))
mamba_model.eval()

norm = np.load('norm_params_v3.npz')
x_mean = norm['x_mean'].squeeze()
x_std = norm['x_std'].squeeze()
y_mean = norm['y_mean'].squeeze()
y_std = norm['y_std'].squeeze()

# ========== 复制ablation_study.py核心逻辑 ==========
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

# ========== 修改版：通用Tube计算函数 ==========
def compute_tube(history_errors, w_base, w_min, kappa):
    """通用Tube计算，接受参数"""
    x = (history_errors - x_mean) / (x_std + 1e-8)
    x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = mamba_model(x_tensor).cpu().numpy()[0]
    pred = pred_norm * y_std + y_mean
    pred_norms = np.sqrt(pred[:,0]**2 + pred[:,1]**2)
    max_d = np.max(pred_norms)
    tube_size = w_base + kappa * max_d
    return max(w_min, tube_size)

# ========== 单次实验运行 ==========
def run_trial(w_base, w_min, kappa, seed=42):
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
            tube = compute_tube(np.array(errors[-100:]), w_base, w_min, kappa)
        else:
            tube = 0.15
        
        tubes.append(tube)
        v, w = mpc_with_tube(robot.state, ref_seg, tube, N, dt)
        d = get_random_disturbance(t)
        robot.step(v, w, d, tube_size=tube)
    
    # 计算指标（跳过前200步预热）
    errors_arr = np.array(errors[200:])
    rmse = np.sqrt(np.mean(errors_arr[:,0]**2 + errors_arr[:,1]**2))
    max_err = np.max(np.sqrt(errors_arr[:,0]**2 + errors_arr[:,1]**2))
    
    # 约束违反次数
    viol_count = 0
    for i in range(200, len(errors)):
        err = np.sqrt(errors[i][0]**2 + errors[i][1]**2)
        if err > 0.15:  # 假设约束边界
            viol_count += 1
    
    return rmse, max_err, viol_count

# ========== 主实验：参数扫描 ==========
print("="*60)
print("🧪 实验1：Tube参数敏感性分析")
print("="*60)

# 扫描范围
w_base_list = [0.00, 0.01, 0.02, 0.05, 0.10]
w_min_list = [0.02, 0.05, 0.08, 0.12, 0.15]
kappa_list = [0.0, 0.5, 1.0, 1.5, 2.0]

results = {'w_base': {}, 'w_min': {}, 'kappa': {}}

# 1. 扫描 w_base（固定 w_min=0.08, kappa=1.0）
print("\n📊 扫描 w_base...")
for wb in w_base_list:
    print(f"  w_base={wb:.2f}...", end=" ")
    rmse_list, maxerr_list, viol_list = [], [], []
    for seed in range(10):
        rmse, max_err, viol = run_trial(wb, 0.08, 1.0, seed)
        rmse_list.append(rmse)
        maxerr_list.append(max_err)
        viol_list.append(viol)
    results['w_base'][wb] = {
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'maxerr_mean': np.mean(maxerr_list),
        'viol_mean': np.mean(viol_list)
    }
    print(f"RMSE={np.mean(rmse_list):.4f}±{np.std(rmse_list):.4f}")

# 2. 扫描 w_min（固定 w_base=0.02, kappa=1.0）
print("\n📊 扫描 w_min...")
for wm in w_min_list:
    print(f"  w_min={wm:.2f}...", end=" ")
    rmse_list, maxerr_list, viol_list = [], [], []
    for seed in range(10):
        rmse, max_err, viol = run_trial(0.02, wm, 1.0, seed)
        rmse_list.append(rmse)
        maxerr_list.append(max_err)
        viol_list.append(viol)
    results['w_min'][wm] = {
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'maxerr_mean': np.mean(maxerr_list),
        'viol_mean': np.mean(viol_list)
    }
    print(f"RMSE={np.mean(rmse_list):.4f}±{np.std(rmse_list):.4f}")

# 3. 扫描 kappa（固定 w_base=0.02, w_min=0.08）
print("\n📊 扫描 kappa...")
for k in kappa_list:
    print(f"  kappa={k:.1f}...", end=" ")
    rmse_list, maxerr_list, viol_list = [], [], []
    for seed in range(10):
        rmse, max_err, viol = run_trial(0.02, 0.08, k, seed)
        rmse_list.append(rmse)
        maxerr_list.append(max_err)
        viol_list.append(viol)
    results['kappa'][k] = {
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'maxerr_mean': np.mean(maxerr_list),
        'viol_mean': np.mean(viol_list)
    }
    print(f"RMSE={np.mean(rmse_list):.4f}±{np.std(rmse_list):.4f}")

# ========== 保存结果 ==========
np.savez('exp1_tube_sensitivity.npz', **results)
print("\n" + "="*60)
print("✅ 实验1完成！结果保存: exp1_tube_sensitivity.npz")
print("="*60)

# 打印汇总
print("\n📈 结果汇总:")
print("\n【w_base 敏感性】")
for wb, vals in results['w_base'].items():
    print(f"  w_base={wb:.2f}: RMSE={vals['rmse_mean']:.4f}±{vals['rmse_std']:.4f}, MaxErr={vals['maxerr_mean']:.4f}, Viol={vals['viol_mean']:.1f}")

print("\n【w_min 敏感性】")
for wm, vals in results['w_min'].items():
    print(f"  w_min={wm:.2f}: RMSE={vals['rmse_mean']:.4f}±{vals['rmse_std']:.4f}, MaxErr={vals['maxerr_mean']:.4f}, Viol={vals['viol_mean']:.1f}")

print("\n【kappa 敏感性】")
for k, vals in results['kappa'].items():
    print(f"  kappa={k:.1f}: RMSE={vals['rmse_mean']:.4f}±{vals['rmse_std']:.4f}, MaxErr={vals['maxerr_mean']:.4f}, Viol={vals['viol_mean']:.1f}")
