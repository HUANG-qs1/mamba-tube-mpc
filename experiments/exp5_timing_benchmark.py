"""
实验5：各方法推理时间与求解时间对比
"""
import torch
import time
import numpy as np
import casadi as ca

from mamba_predictor import MambaDisturbancePredictor
device = 'cuda'
mamba_model = MambaDisturbancePredictor(2,128,2,10).to(device)
mamba_model.load_state_dict(torch.load('best_mamba_v3.pt', map_location=device, weights_only=True))
mamba_model.eval()
print("✅ Mamba模型加载成功")

from lstm_predictor import LSTMDisturbancePredictor
lstm_model = LSTMDisturbancePredictor(2,128,2,10).to(device)
lstm_model.load_state_dict(torch.load('best_lstm_model.pt', map_location=device, weights_only=True))
lstm_model.eval()
print("✅ LSTM模型加载成功")

norm = np.load('norm_params_v3.npz')
x_mean = torch.from_numpy(norm['x_mean'].astype(np.float32)).to(device)
x_std = torch.from_numpy(norm['x_std'].astype(np.float32)).to(device)
x_test = torch.randn(1,100,2).to(device)
x_test_norm = (x_test - x_mean) / x_std

def benchmark(model, x, n_warmup=50, n_iter=1000):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup): _ = model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter): _ = model(x)
        torch.cuda.synchronize()
        return (time.time()-t0)/n_iter*1000

t_mamba = benchmark(mamba_model, x_test_norm)
print(f"\n🚀 Mamba GPU: {t_mamba:.3f} ms/次")
t_lstm = benchmark(lstm_model, x_test_norm)
print(f"🚀 LSTM GPU:  {t_lstm:.3f} ms/次")

class EKF:
    def __init__(self, dt=0.1):
        self.dt=dt; self.x=np.zeros(5); self.P=np.eye(5)*0.1
        self.Q=np.diag([0.01,0.01,0.001,0.001,0.001]); self.R=np.diag([0.01,0.01,0.001])
    def update(self,z,u):
        x,y,th,dv,dw=self.x; v,omega=u; va=v+dv; oa=omega+dw
        xp=x+va*np.cos(th)*self.dt; yp=y+va*np.sin(th)*self.dt; tp=th+oa*self.dt
        F=np.array([[1,0,-va*self.dt*np.sin(th),self.dt*np.cos(th),0],[0,1,va*self.dt*np.cos(th),self.dt*np.sin(th),0],[0,0,1,0,self.dt],[0,0,0,1,0],[0,0,0,0,1]])
        H=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]])
        Pp=F@self.P@F.T+self.Q; yr=z-H@np.array([xp,yp,tp,dv,dw])
        S=H@Pp@H.T+self.R; K=Pp@H.T@np.linalg.inv(S)
        self.x=np.array([xp,yp,tp,dv,dw])+K@yr; self.P=(np.eye(5)-K@H)@Pp
        return self.x[3:5]

ekf=EKF()
for _ in range(100): ekf.update(np.array([0.1,0.1,0.0]),np.array([0.5,0.1]))
t0=time.time()
for _ in range(1000): ekf.update(np.array([0.1,0.1,0.0]),np.array([0.5,0.1]))
t_ekf=(time.time()-t0)/1000*1000
print(f"🚀 EKF更新:   {t_ekf:.3f} ms/次")

def mpc_time():
    state=np.zeros(3); ref=np.array([[0.1*i,0.0] for i in range(10)]); tube=0.15
    U=ca.MX.sym('U',20); v=U[0::2]; o=U[1::2]; X=[state[0],state[1],state[2]]; obj=0
    for k in range(10):
        xn=X[0]+v[k]*ca.cos(X[2])*0.1; yn=X[1]+v[k]*ca.sin(X[2])*0.1; tn=X[2]+o[k]*0.1
        X=[xn,yn,tn]; ex=X[0]-ref[k,0]; ey=X[1]-ref[k,1]
        tv=ca.fmax(0,ca.sqrt(ex**2+ey**2)-tube); obj+=ex**2+ey**2+2.0*tv**2+0.01*v[k]**2+0.01*o[k]**2
    solver=ca.nlpsol('s','ipopt',{'x':U,'f':obj},{'print_time':False,'ipopt':{'print_level':0}})
    for _ in range(10): solver(x0=[0.0]*20,lbx=[-1.0]*20,ubx=[1.0]*20)
    t0=time.time()
    for _ in range(100): solver(x0=[0.0]*20,lbx=[-1.0]*20,ubx=[1.0]*20)
    return (time.time()-t0)/100*1000

t_mpc=mpc_time()
print(f"🚀 MPC求解:   {t_mpc:.3f} ms/次")

print(f"\n{'='*50}")
print("📊 计算时间对比汇总")
print(f"{'='*50}")
print(f"Mamba GPU:      {t_mamba:.3f} ms")
print(f"LSTM GPU:       {t_lstm:.3f} ms")
print(f"EKF更新:        {t_ekf:.3f} ms")
print(f"MPC求解:        {t_mpc:.3f} ms")
print(f"{'='*50}")
print(f"MPC控制周期:    100 ms (10Hz)")
print(f"Mamba占比:      {t_mamba/100*100:.1f}%")
print(f"LSTM占比:       {t_lstm/100*100:.1f}%")
print(f"完整循环(Mamba):{t_mamba+t_mpc:.3f} ms ({(t_mamba+t_mpc)/100*100:.1f}%)")
print(f"完整循环(LSTM): {t_lstm+t_mpc:.3f} ms ({(t_lstm+t_mpc)/100*100:.1f}%)")
print(f"{'='*50}")

np.savez('exp5_timing.npz',mamba_gpu_ms=t_mamba,lstm_gpu_ms=t_lstm,ekf_update_ms=t_ekf,mpc_solve_ms=t_mpc)
print("\n✅ 结果已保存: exp5_timing.npz")
