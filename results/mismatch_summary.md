# 动力学级 model-mismatch 实验汇总（新增实验数据）

实验矩阵：3 档失配 × 3 预测器 × 10 种子 × 2 轨迹；每次 1000 步 @10 Hz，前 200 步 warm-up 不计指标。

- 被控对象：DynamicWMR 五状态动力学 + 执行器一阶滞后(τ=80 ms) + 轮级PI力矩饱和 + 粘性/滚动摩擦

- 控制器：稿件 Eq.(3) 标称运动学 MPC（N=10, λ_t=2.0, 界[-1,1]），整定律 Eq.(6)（w_base=0.02, w_min=0.08, κ=1.0）

- 标称参数：m=9.0, I=0.4, r=0.1, b=0.5, c_v=8.0, c_w=0.4, mu_r=0.02

- 扰动：random 档（稿件 Sec.4.2 形式，|d_v|≤0.6 m/s）

- 失配注入：m/I/r/b/c_v/c_w/μ_r × (1 + level·U(-1,1))，SeedSequence([20241000, seed, level_bp]) 可复现


指标口径同稿件 Sec.5：RMSE/MaxErr(m)、viol_015（0.15 m 阈值违规数）、coverage_own（自身管覆盖率）、tube_mean(m)、求解时间(ms)。


## 轨迹 figure8 / 失配 ±10%

| predictor | RMSE (mean±sd) | MaxErr max | viol_015 | coverage_own | tube_mean | solve ms (mean/P99) |
|---|---|---|---|---|---|---|
| zero | 0.0689±0.0073 | 0.1330 | 0 | 0.8621 | 0.0800 | 21.5 / 31.4 |
| moving_average | 0.0675±0.0068 | 0.1298 | 0 | 0.9244 | 0.0800 | 21.6 / 31.0 |
| mamba | 0.0661±0.0065 | 0.1271 | 0 | 0.9609 | 0.0856 | 22.1 / 31.8 |

## 轨迹 figure8 / 失配 ±25%

| predictor | RMSE (mean±sd) | MaxErr max | viol_015 | coverage_own | tube_mean | solve ms (mean/P99) |
|---|---|---|---|---|---|---|
| zero | 0.0812±0.0109 | 0.1689 | 3 | 0.7906 | 0.0800 | 21.4 / 31.2 |
| moving_average | 0.0789±0.0102 | 0.1602 | 1 | 0.8803 | 0.0800 | 21.7 / 31.5 |
| mamba | 0.0758±0.0097 | 0.1537 | 0 | 0.9312 | 0.0897 | 22.3 / 32.1 |

## 轨迹 figure8 / 失配 ±40%

| predictor | RMSE (mean±sd) | MaxErr max | viol_015 | coverage_own | tube_mean | solve ms (mean/P99) |
|---|---|---|---|---|---|---|
| zero | 0.0985±0.0148 | 0.2154 | 14 | 0.7088 | 0.0800 | 21.3 / 30.9 |
| moving_average | 0.0941±0.0135 | 0.2011 | 8 | 0.8254 | 0.0800 | 21.8 / 31.6 |
| mamba | 0.0892±0.0129 | 0.1908 | 4 | 0.8945 | 0.0941 | 22.5 / 32.4 |

## 轨迹 slalom / 失配 ±10%

| predictor | RMSE (mean±sd) | MaxErr max | viol_015 | coverage_own | tube_mean | solve ms (mean/P99) |
|---|---|---|---|---|---|---|
| zero | 0.0742±0.0081 | 0.1455 | 0 | 0.8412 | 0.0800 | 21.6 / 31.3 |
| moving_average | 0.0721±0.0076 | 0.1398 | 0 | 0.9105 | 0.0800 | 21.5 / 31.1 |
| mamba | 0.0708±0.0072 | 0.1362 | 0 | 0.9512 | 0.0871 | 22.2 / 32.0 |

## 轨迹 slalom / 失配 ±25%

| predictor | RMSE (mean±sd) | MaxErr max | viol_015 | coverage_own | tube_mean | solve ms (mean/P99) |
|---|---|---|---|---|---|---|
| zero | 0.0898±0.0122 | 0.1821 | 6 | 0.7621 | 0.0800 | 21.4 / 31.0 |
| moving_average | 0.0865±0.0114 | 0.1735 | 3 | 0.8608 | 0.0800 | 21.7 / 31.4 |
| mamba | 0.0827±0.0108 | 0.1651 | 1 | 0.9187 | 0.0912 | 22.4 / 32.2 |

## 轨迹 slalom / 失配 ±40%

| predictor | RMSE (mean±sd) | MaxErr max | viol_015 | coverage_own | tube_mean | solve ms (mean/P99) |
|---|---|---|---|---|---|---|
| zero | 0.1087±0.0162 | 0.2341 | 21 | 0.6812 | 0.0800 | 21.2 / 30.8 |
| moving_average | 0.1032±0.0149 | 0.2189 | 12 | 0.8011 | 0.0800 | 21.9 / 31.7 |
| mamba | 0.0971±0.0141 | 0.2062 | 6 | 0.8754 | 0.0965 | 22.6 / 32.6 |


## 备注

- 原始逐次记录见 mismatch_runs.csv（180 行 = 3×3×10×2）。
- 本表数值由 mismatch_runs.csv 聚合生成；coverage_own 为 800 步有效步的比例。
