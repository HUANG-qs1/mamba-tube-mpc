# 复现对照表（论文条目 → 脚本 → 存档）

| 论文条目 | 脚本 | 存档 |
|---|---|---|
| 表 2（三场景对比） | ablation_study.py | ablation_study.npz |
| 表 3（主消融 A/D/B/E） | rerun_no_penalty.py | rerun_no_penalty_results.npz |
| 表 3 配对 TOST（±1 mm） | ttest_all.py、exp17_seeds30.py、exp19_pool30_verify.py、exp20_tost_verify.py | ttest_results.npz、exp17_seeds10-29.npz |
| 表 9（计时，双口径） | exp5_timing_benchmark.py | exp5_timing.npz |
| 表 12（双通道消融） | exp21_dual_channel.py | exp21_dual_channel.npz、exp28_dual40.npz |
| conformal 基线 | exp22_conformal.py | exp22_conformal.npz |
| λ_t 扫描 / w_k 无抖振 | exp23_lambda_timing.py | exp23_wk_series.npz |
| 图 1–9 | figures/*.py | fig*_data.npz（脚本重新生成） |
| 非理想条件（延迟/丢包/噪声） | exp9_nonideal.py、exp11*、exp12*、exp13* | exp*.npz |
| 零样本 OOD（方形/利萨茹） | exp14_zeroshot_naive.py | exp14_*.npz |
| 900 s 长时域 | exp18_longhorizon.py（+ S4 协议） | exp18_*.npz |

所有脚本固定随机种子；在 mambampc 环境下从仓库根目录运行。

## 训练语料指纹（training_data_v3.npz，由 training/collect_data_v3.py 生成）

- sha256: `06933774a013f42e45772d48c7f2bd0ac018e2512e868c1c2b0727056cefe262`
- `X_train (117000, 100, 2) float64`，均值 −0.000554，标准差 0.05525
- `Y_train (117000, 10, 2) float64`，均值 −0.003664，标准差 0.05522
- `X_test (15600, 100, 2) float64`，均值 −0.003854，标准差 0.02766
- `Y_test (15600, 10, 2) float64`，均值 −0.004950，标准差 0.02587

复现者可用 `sha256sum training_data_v3.npz` 与上述统计指纹核对语料一致性。
