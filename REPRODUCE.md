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
