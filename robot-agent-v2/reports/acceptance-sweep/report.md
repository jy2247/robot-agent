# 三票接受参数离线扫参

使用已保存的逐票输出，固定录像中的 expected_phase；没有新增模型推理或测量 Hz。
这是同一条成功源录像上的参数敏感性分析，不是独立测试集，也不是闭环 FC 重放。
接受阶段数不代表准确率；完成时刻没有独立标注，不能据此判断提前误接受或真实检测延迟。

## 参数含义

- confidence threshold：每个成功/失败投票进入严格多数决的置信门槛。
- success confirmations：成功多数需要连续出现的窗口数；失败仍固定两次确认，安全结果原样保留。
- score threshold：历史成功证据乘以 duration prior 后的接受门槛。
- duration center：logistic 时间先验的中心，不是硬性等待时长；scale 固定 0.8 秒，最后阶段豁免。
- decay：每个原录像采样周期保留的证据比例；数值越大，记忆越长。

leaky 模式使用 `score=max(0, decay**steps*score + success_mass - 0.05*in_progress_mass - 0.20*fail_mass)`。
mass 用配置票数作分母，错误票弃权；还必须有当前窗口的高置信严格多数和 success_mass≥0.45。
进入新 expected_phase 时重置历史，先验时钟从该阶段首个被观察窗口开始，不读取录像边界或标注。
已接受完成会保留，直到后续接受失败矛盾；两次确认与 2 秒 duration center 是独立参数。

连续确认模式扫描 confidence=0.50/0.60/0.70/0.80/0.90，确认窗口数=1/2/3。
历史证据模式固定 confidence=0.70，扫描成功确认窗口数=1/2；其他参数范围见 CSV 和 JSON。
旧五链实验采用 0.4 秒采样周期，本次采用 1 秒；相同 decay 数值对应的实际记忆时长不同。

## rtx5080-graphs3

50 个窗口，3 票，录像采样间隔 1 秒；共比较 315 个参数配置。基线逐窗口与归档状态一致。

| 配置 | 成功连续窗口 | score 阈值 | prior 中心/s | decay | 接受阶段数 | 阶段1/s | 阶段3/s | 阶段5/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2 | — | — | — | 2/5 | 9.75 | — | 50.75 |
| confidence_0.50 | 2 | — | — | — | 2/5 | 9.75 | — | 50.75 |
| one_window | 1 | — | — | — | 3/5 | 8.75 | 20.75 | 49.75 |
| three_windows | 3 | — | — | — | 0/5 | — | — | — |
| leaky_reference | 1 | 0.45 | 4.0 | 0.65 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_duration_2s | 1 | 0.45 | 2 | 0.65 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_decay_0.85 | 1 | 0.45 | 4.0 | 0.85 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_threshold_0.35 | 1 | 0.35 | 4.0 | 0.65 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_combined | 1 | 0.35 | 2 | 0.85 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_two_windows | 2 | 0.35 | 2 | 0.85 | 2/5 | 9.75 | — | 50.75 |
| leaky_no_prior | 1 | 0.45 | 关闭 | 0.65 | 3/5 | 8.75 | 20.75 | 49.75 |

扫参最多接受 3/5 阶段。单窗口确认配置最多 3/5；保留至少两个连续窗口时最多 2/5。

| 阶段 | 成功票 / 总票槽 | 错误弃权 |
|---|---:|---:|
| step1_pipette_pick | 5/27 | 0 |
| step2_tube_pick | 0/27 | 0 |
| step3_pipette_aim_at_tube | 6/24 | 0 |
| step4_tube_return | 0/27 | 0 |
| step5_pipette_return | 6/45 | 0 |

归档安全首次接受停止：11.75 秒。表中继续分析该时刻之后的录像，仅用于诊断；这些接受不能当作实际机器人会继续执行的结果。逐次事件中的安全冲突见 JSON。

来源：`robot-agent-v2/outputs/rtx5080-graphs3/predictions.jsonl`，SHA-256 `90d97f76491b033f30bf17874c8125fda16f310dee913308f665491a4607ccc2`。

## rtx5080-concurrent3

50 个窗口，3 票，录像采样间隔 1 秒；共比较 315 个参数配置。基线逐窗口与归档状态一致。

| 配置 | 成功连续窗口 | score 阈值 | prior 中心/s | decay | 接受阶段数 | 阶段1/s | 阶段3/s | 阶段5/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2 | — | — | — | 2/5 | 9.75 | — | 50.75 |
| confidence_0.50 | 2 | — | — | — | 2/5 | 9.75 | — | 50.75 |
| one_window | 1 | — | — | — | 3/5 | 8.75 | 20.75 | 49.75 |
| three_windows | 3 | — | — | — | 0/5 | — | — | — |
| leaky_reference | 1 | 0.45 | 4.0 | 0.65 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_duration_2s | 1 | 0.45 | 2 | 0.65 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_decay_0.85 | 1 | 0.45 | 4.0 | 0.85 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_threshold_0.35 | 1 | 0.35 | 4.0 | 0.65 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_combined | 1 | 0.35 | 2 | 0.85 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_two_windows | 2 | 0.35 | 2 | 0.85 | 2/5 | 9.75 | — | 50.75 |
| leaky_no_prior | 1 | 0.45 | 关闭 | 0.65 | 3/5 | 8.75 | 20.75 | 49.75 |

扫参最多接受 3/5 阶段。单窗口确认配置最多 3/5；保留至少两个连续窗口时最多 2/5。

| 阶段 | 成功票 / 总票槽 | 错误弃权 |
|---|---:|---:|
| step1_pipette_pick | 5/27 | 0 |
| step2_tube_pick | 0/27 | 0 |
| step3_pipette_aim_at_tube | 6/24 | 0 |
| step4_tube_return | 0/27 | 0 |
| step5_pipette_return | 6/45 | 0 |

归档安全首次接受停止：11.75 秒。表中继续分析该时刻之后的录像，仅用于诊断；这些接受不能当作实际机器人会继续执行的结果。逐次事件中的安全冲突见 JSON。

来源：`robot-agent-v2/outputs/rtx5080-concurrent3/predictions.jsonl`，SHA-256 `1855a0571567fed4fa1a354b1d2d1fab78edb4aeb9e97385925164f7cd9297f9`。

## rtx5080-vote3-kv

50 个窗口，3 票，录像采样间隔 1 秒；共比较 315 个参数配置。基线逐窗口与归档状态一致。

| 配置 | 成功连续窗口 | score 阈值 | prior 中心/s | decay | 接受阶段数 | 阶段1/s | 阶段3/s | 阶段5/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2 | — | — | — | 2/5 | 9.75 | — | 50.75 |
| confidence_0.50 | 2 | — | — | — | 2/5 | 9.75 | — | 50.75 |
| one_window | 1 | — | — | — | 3/5 | 8.75 | 20.75 | 49.75 |
| three_windows | 3 | — | — | — | 0/5 | — | — | — |
| leaky_reference | 1 | 0.45 | 4.0 | 0.65 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_duration_2s | 1 | 0.45 | 2 | 0.65 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_decay_0.85 | 1 | 0.45 | 4.0 | 0.85 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_threshold_0.35 | 1 | 0.35 | 4.0 | 0.65 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_combined | 1 | 0.35 | 2 | 0.85 | 3/5 | 8.75 | 26.75 | 49.75 |
| leaky_two_windows | 2 | 0.35 | 2 | 0.85 | 2/5 | 9.75 | — | 50.75 |
| leaky_no_prior | 1 | 0.45 | 关闭 | 0.65 | 3/5 | 8.75 | 20.75 | 49.75 |

扫参最多接受 3/5 阶段。单窗口确认配置最多 3/5；保留至少两个连续窗口时最多 2/5。

| 阶段 | 成功票 / 总票槽 | 错误弃权 |
|---|---:|---:|
| step1_pipette_pick | 5/27 | 0 |
| step2_tube_pick | 0/27 | 0 |
| step3_pipette_aim_at_tube | 6/24 | 0 |
| step4_tube_return | 0/27 | 0 |
| step5_pipette_return | 6/45 | 0 |

归档安全首次接受停止：11.75 秒。表中继续分析该时刻之后的录像，仅用于诊断；这些接受不能当作实际机器人会继续执行的结果。逐次事件中的安全冲突见 JSON。

来源：`robot-agent-v2/reports/rtx5080-vote3-kv/predictions.jsonl`，SHA-256 `0c208c8cdb2ea92f2a2b6645e7a3404d58eb221ee6321d1fbbde403d8f3b92a9`。

## 使用边界

这些参数没有写入服务或生产配置。若接受数增加来自取消连续确认，需要独立完成时刻标注与失败录像来判断是否更好。
阶段 confidence 是模型对其输出状态的自报置信度；in_progress 的高置信度不能转换为 success。
原始投票、推理延迟和安全判定均未改写。此分析只衡量源视频时间上的接受变化，不报告新的推理频率。

## 复现

在仓库根目录执行：

```bash
python3 robot-agent-v2/evaluate_acceptance.py robot-agent-v2/outputs/rtx5080-graphs3 robot-agent-v2/outputs/rtx5080-concurrent3 robot-agent-v2/reports/rtx5080-vote3-kv --output robot-agent-v2/reports/acceptance-sweep
```

可通过 `--confidence-thresholds`、`--score-thresholds`、`--duration-centers`、`--decays` 指定扫描值。
