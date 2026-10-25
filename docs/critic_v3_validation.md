# Universal critic V3 验证记录

2026-10-21，根据 AGENTS.md 的五阶段物理判据修改。原有工作区修改保留。

实现：三份 pipette TOML 统一功能识别与 Success/Failure 判据；第三步要求对准、接触/插入、
可见按压释放；第四、五步要求松手后支撑稳定。保留原任务 ID、phase 名与 JSON 视觉 verdict。
`uncertain` 对应建议中的 UNKNOWN。内部检查点目前写入证据文本，尚未增加结构化检查点输出。

程序独立记忆各阶段完成，不因前一步未确认而丢弃后续阶段证据，也不以切换阶段替代完成确认。
高置信度失败会撤销对应阶段的完成；重新确认可解除该阶段失败。明确终止后由程序汇总结果，
未全部完成则返回 INCOMPLETE_AT_END；未解决的已确认失败返回 PHASE_FAILED。视觉 status 与
整体结论分开，后续帧要求 reset。新增一张初始全景参考帧，辅助比较支架位置；这不是专门的
目标跟踪器，也不能保证物体身份正确。

验证：27 项 unittest 全部通过，包含 HTTP、本地假模型、完成记忆、失败撤销、恢复、终止别名、
后端报错终止、初始图像记忆及 reset 隔离。Mock replay 通过。没有执行真实机器人控制。

真实 Qwen/Qwen3.8-27B / vLLM 回放使用已有 manifest 中的 success_02 和 failure_05，
两个独立 episode 并发，各 episode 内按时间顺序推理，未发送数据集阶段或成败标签。
两轮均为 0.5 Hz 源抽帧、历史 1 个观测，共各 46 次推理。

| 回放 | success_02 | failure_05 |
|---|---|---|
| 第一轮：新判据，336px，无固定参考帧 | 阶段准确率 42.86%；确认 1、2；最终 INCOMPLETE_AT_END | 末帧报告 object_dropped，整体 PHASE_FAILED；归因第 2 步而参考标签是 S3 |
| 最终版本：修订阶段识别，448px，加初始参考帧 | 阶段准确率 67.86%；确认 1、2、5；最终 INCOMPLETE_AT_END | 未检出物理掉落；最终 INCOMPLETE_AT_END，不再继续 CONTINUE |

最终版本平均调用耗时 5192.2 ms，约 0.193 次/秒，不满足 3–5 Hz。
对比已有 optimized-tp4 结果，success_02 的阶段准确率是 82.14%，但没有接受任何阶段完成。
这些指标不是整体准确率提升的证明；修改了判据，且第二轮同时调整 prompt、参考图和分辨率，
不能将差异单独归因于某一项。失败数据的标签也不是人工逐帧验证的物理原因。

目前明确剩余的问题：成功样本第 3 步未确认可见按压；第 4 步被模型持续标成第 3 步，未形成
完成 verdict。最终第 5 步证据文字甚至提到管子已在架上，但单阶段输出没有独立确认第 4 步。
不能用后续阶段自动补齐这两个缺口。下一步应验证更密集采样及独立的逐阶段物理检查点输出，
尤其要区分「动作已发生但漏采/遮挡」和「根本没有发生」。2 Hz 已设为视频入口的默认抽帧率，
本次没有做 2 Hz 全量回放，不能将其写成已验证的提升。

最终回放命令：

```bash
PYTHONPATH=src:. .venv-model/bin/python scripts/infer_continuous_review.py \
  --sample-id success_02 --sample-id failure_05 --sample-hz 0.5 \
  --image-side 448 --workers 2 --out-dir outputs/universal-critic-v3-reference-review
```

再次运行必须换一个新输出目录，防止把旧预测当作新 prompt 的结果。
最终产物位于 `outputs/universal-critic-v3-reference-review/`：batch_summary.json、两个样本的
result.mp4 / predictions.json / summary.json，以及本次 observer.txt 和 TOML 快照。
第一轮产物保留在 `outputs/universal-critic-v3-review/`。
