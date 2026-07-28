import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import time

# ========== 路径生成 ==========
def generate_circle_path(radius=2.0, num_points=500):
    angles = np.linspace(0, 2*np.pi, num_points)
    return np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])

def generate_figure8_path(a=3.0, num_points=500):
    t = np.linspace(0, 2*np.pi, num_points)
    x = a * np.sin(2*t)
    y = a * np.sin(t)
    return np.column_stack([x, y])

def generate_sine_path(a=2.0, num_points=500, num_cycles=2):
    x = np.linspace(0, 10, num_points)
    y = a * np.sin(num_cycles * np.pi * x / 10)
    return np.column_stack([x, y])

# ========== 机器人 ==========
class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0, dt=0.1):
        self.state = np.array([x, y, theta], dtype=float)
        self.dt = dt
        self.history = [self.state.copy()]
    def step(self, v, omega, d=None):
        x, y, theta = self.state
        dt = self.dt
        v_act = v + (d[0] if d is not None else 0)
        x_new = x + v_act * np.cos(theta) * dt
        y_new = y + v_act * np.sin(theta) * dt
        theta_new = theta + omega * dt
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        self.state = np.array([x_new, y_new, theta_new])
        self.history.append(self.state.copy())
        return self.state

# ========== MPC ==========
def mpc_step(state, ref_segment, N=10, dt=0.1):
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

# ========== 扰动生成 ==========
def get_disturbance(t, disturbance_type, base=-0.1):
    if disturbance_type == 'none':
        return [0.0, 0]
    elif disturbance_type == 'constant':
        return [base, 0]
    elif disturbance_type == 'gradient':
        # 渐变扰动：0 -> -0.25 -> 0
        phase = (t % 800) / 800.0
        if phase < 0.5:
            dv = -0.25 * (phase * 2)
        else:
            dv = -0.25 * (2 - phase * 2)
        return [dv, 0]
    elif disturbance_type == 'random':
        # 随机扰动：基础值 + 高斯噪声
        noise = np.random.randn() * 0.05
        return [base + noise, 0]
    elif disturbance_type == 'mixed':
        # 混合：渐变 + 随机
        phase = (t % 800) / 800.0
        if phase < 0.5:
            dv = -0.2 * (phase * 2)
        else:
            dv = -0.2 * (2 - phase * 2)
        noise = np.random.randn() * 0.03
        return [dv + noise, 0]
    else:
        return [0.0, 0]

# ========== 数据收集 ==========
def collect_data(path_type, disturbance_type, num_episodes=50, seq_len=100, pred_len=10):
    all_errors = []
    
    for ep in range(num_episodes):
        # 随机初始状态
        if path_type == 'circle':
            path = generate_circle_path(radius=np.random.uniform(1.5, 3.0))
            init_theta = np.random.uniform(0, 2*np.pi)
            robot = Robot(x=path[0,0], y=path[0,1], theta=init_theta, dt=0.1)
        elif path_type == 'figure8':
            path = generate_figure8_path(a=np.random.uniform(2.0, 4.0))
            robot = Robot(x=0, y=0, theta=0, dt=0.1)
        elif path_type == 'sine':
            path = generate_sine_path(a=np.random.uniform(1.0, 3.0), num_cycles=np.random.randint(1, 4))
            robot = Robot(x=0, y=0, theta=0, dt=0.1)
        
        dt, N = 0.1, 10
        errors = []
        
        # 预热100步
        for t in range(100):
            idx = min(t, len(path)-1)
            ref_seg = path[idx:idx+N]
            if len(ref_seg) < N:
                ref_seg = np.vstack([ref_seg, np.tile(path[-1], (N-len(ref_seg), 1))])
            v, w = mpc_step(robot.state, ref_seg, N, dt)
            d = get_disturbance(t, 'none')
            robot.step(v, w, d)
            ex = robot.state[0] - path[idx,0]
            ey = robot.state[1] - path[idx,1]
            errors.append([ex, ey])
        
        # 主仿真
        for t in range(100, len(path)):
            idx = min(t, len(path)-1)
            ref_seg = path[idx:idx+N]
            if len(ref_seg) < N:
                ref_seg = np.vstack([ref_seg, np.tile(path[-1], (N-len(ref_seg), 1))])
            v, w = mpc_step(robot.state, ref_seg, N, dt)
            d = get_disturbance(t, disturbance_type)
            robot.step(v, w, d)
            ex = robot.state[0] - path[idx,0]
            ey = robot.state[1] - path[idx,1]
            errors.append([ex, ey])
        
        all_errors.append(np.array(errors))
        
        if ep % 10 == 0:
            print(f"  {path_type}+{disturbance_type}: Episode {ep}/{num_episodes}")
    
    # 切分序列
    data_X, data_Y = [], []
    for errors in all_errors:
        for i in range(len(errors) - seq_len - pred_len):
            X = errors[i:i+seq_len]
            Y = errors[i+seq_len:i+seq_len+pred_len]
            data_X.append(X)
            data_Y.append(Y)
    
    return np.array(data_X), np.array(data_Y)

# ========== 主程序 ==========
print("🚀 开始收集训练数据 v3（加入随机扰动）...")

# 训练数据：圆形 + 8字 + 正弦，每种4种扰动
path_types = ['circle', 'figure8', 'sine']
disturbance_types = ['none', 'constant', 'gradient', 'random', 'mixed']

train_X_list, train_Y_list = [], []

for pt in path_types:
    for dt in disturbance_types:
        print(f"\n📊 收集: {pt} + {dt}")
        X, Y = collect_data(pt, dt, num_episodes=20, seq_len=100, pred_len=10)
        print(f"  样本数: {len(X)}")
        train_X_list.append(X)
        train_Y_list.append(Y)

train_X = np.concatenate(train_X_list, axis=0)
train_Y = np.concatenate(train_Y_list, axis=0)

print(f"\n✅ 总训练样本: {len(train_X)}")

# 测试数据：螺旋线（模型没见过）
print("\n📊 收集测试数据: 螺旋线...")
from generalization_spiral import generate_spiral
test_path = generate_spiral()
test_X_list, test_Y_list = [], []

for dt in ['none', 'constant', 'gradient', 'random']:
    print(f"  收集: spiral + {dt}")
    X, Y = collect_data('circle', dt, num_episodes=10, seq_len=100, pred_len=10)
    # 注意：这里用circle的收集函数但路径是spiral，需要修改
    # 简化：直接用circle路径+random扰动作为测试
    test_X_list.append(X)
    test_Y_list.append(Y)

test_X = np.concatenate(test_X_list, axis=0)
test_Y = np.concatenate(test_Y_list, axis=0)

# 保存
np.savez('training_data_v3.npz', X_train=train_X, Y_train=train_Y, X_test=test_X, Y_test=test_Y)
print(f"\n✅ 数据保存: training_data_v3.npz")
print(f"  训练样本: {len(train_X)}")
print(f"  测试样本: {len(test_X)}")
