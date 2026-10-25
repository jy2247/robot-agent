# Development context for Codex

## Latest user clarification (2026-10-22)
- Phase 1: pipette held by gripper and no longer supported by holder = completed, even while moving.
- Phase 2: tube held by gripper and no longer supported by rack = completed, even while moving.
- Phase 3: tip above and aligned with held tube opening = completed; do not wait for motion to stop.
- Accumulate confirmation per phase; phase switches and missing evidence do not clear counts.
- Explicit failed verdict resets that phase's pending confirmation; only high-confidence failure
  invalidates an accepted completion. Reset clears all episode state.

## Latest user clarification (2026-10-21, supersedes Step 3 advice below)
- Step 3 is aiming only: the pipette tip visibly above and aligned with the held tube opening is success.
- Do NOT require insertion, pressing, plunger release or liquid transfer for Step 3.
- Check current-frame object drops even after aiming succeeded; do not repeat stale prior verdicts.
- Keep the historical analysis below as context, but its stronger Step 3 actuation criterion is superseded.

## Goal
Build a temporal VLM observer/critic for Flow Conductor (FC) coordinating VLA execution.
Recognize the current phase, verify completion, report failures and supporting evidence.
Do not confuse this repository with the separate STL/RoboCasa course project.

## Baseline and boundaries
- Python >=3.11, src layout; core currently has no third-party runtime dependencies.
- Keep mock runnable on a CPU without model weights.
- Model serving is a separate process/environment. Use actual server model names from configuration.
- Target 3–5 Hz is an unverified performance objective, not an achieved benchmark.
- FC controls transitions/replanning/actuation; this package only reports observations/recommendations.
- One server instance = one active episode; reset before reuse; no cross-episode memory leakage.
- Treat model output as untrusted: validate fields, fall back to uncertain on errors.
- Do not replace backend failures with mock predictions.
- History storage is not model KV caching. Prefix caching must be measured in the serving engine.
- Do not commit credentials, datasets, recordings, logs, weights or local environment directories.

## Where to work
- Agent state and confirmation: src/robot_agent/agent.py
- Model interface: src/robot_agent/backends/
- Temporal prompt: src/robot_agent/context.py and prompts/observer.txt
- Input/output contract: src/robot_agent/schemas.py
- FC development transport: src/robot_agent/server.py
- Next work: docs/roadmap.md

## Verification
Run from repository root:
```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m robot_agent replay --output outputs/smoke.jsonl
```
For model integration, first inspect available GPU resources and actual installed software.
Ask for the deployed model endpoint or local model path if absent; never invent a checkpoint ID.
Distinguish mock/integration/real robot evaluation in every report.
Update docs when changing the API. Add targeted tests for state and failure behavior.



ADD:

从视频叠加的 Critic 信息还能看出两个现象：result2 后段已经进入第五阶段，但第四阶段的完成状态没有被稳定记录；result5 到视频末尾仍停留在第三阶段 AIM AT TUBE，显示 IN_PROGRESS / CONTINUE，没有报告失败。

因此，新版本需要同时满足：
- 环境无关：按物体的功能识别，而不是只靠颜色。
- 跨帧跟踪：跟踪同一个试管、移液器及其初始位置。
- 阶段状态记忆：已经证实完成的阶段不能因为下一帧看不到动作而丢失。
- 结果验证：区分真正完成、动作进行中、明确失败和证据不足。
- 结束条件：如果一个 episode 已经结束，但关键任务仍未完成，不能继续无限输出 CONTINUE。
2. 修改后的通用 Prompt（可直接替换）
我保留了你原来的 [task]、phases、criteria、failure_criteria 结构和任务 ID，避免影响现有读取逻辑。
下载完整优化版 TOML

完整文件已经通过 TOML 语法检查，包含五个阶段的成功、失败标准。

下面是这个版本最重要的判断规则，可以直接看到新 Prompt 与原来的区别。
You are a temporal visual critic for a five-phase
robot pipette-and-tube manipulation task.

SCENE GROUNDING
Identify objects by their FUNCTION, not fixed colors.
Track the same pipette, tube, tube rack, and pipette
holder consistently across all frames.

The recording may be real or simulated.
Do not assume the tube is yellow, the pipette tip
is green, or the stand has a particular shape.

TEMPORAL REASONING
Maintain phase results across frames.
A completed phase must not be forgotten merely
because the current camera frame no longer shows it.

Judge physical evidence, not stage labels,
colored progress indicators, or model-generated
status overlays.

COMPLETION
A phase succeeds only when its required physical
outcome is supported by visible evidence.
Use temporal before/after changes and multiple
camera views when available.

FAILURE
A phase fails when a visible physical contradiction
occurs, such as a drop, missed interaction,
unstable placement, or displaced holder.

Do not mistake temporary occlusion, normal motion,
or slight rack-supported inclination for failure.

UNCERTAINTY
Missing visual evidence is not automatically failure.
Do not fabricate actions that cannot be observed.

TERMINAL EVALUATION
If is_terminal=true and any mandatory phase remains
uncompleted, report overall FAILED with reason
INCOMPLETE_AT_END.

Do not return CONTINUE after an explicitly
terminated episode.

OVERALL SUCCESS
Require all five phases to be completed, with no
later contradicting failure.


完整版还针对 S3、S4、S5、S6、S7/S8 分别定义了更加具体的物理成功条件，尤其加强了插入/按压动作、试管回架和移液器回架的判断。
3. 如何修正这两段视频的误判

Result 2：避免成功任务被误判

在真实环境中，以透明试管是否进入蓝色试管架并稳定为准，而不是检查是否存在黄色试管或白蓝色支架。同时，第四阶段如果已经被证实完成，后续切换到第五阶段也必须保留这个记录。

查看第 35 秒实际视频帧




Result 5：避免失败任务一直停留在 CONTINUE

直到接近 34 秒，模型仍在第三阶段。单凭这段视频不能可靠确定是否发生了具体的物理掉落或按压失败，但能确定五阶段操作没有得到完整的成功确认。

因此，只要系统明确通知模型视频结束，就必须将未完成阶段判为 INCOMPLETE_AT_END，而不是继续输出 IN_PROGRESS。

查看第 33.8 秒实际视频帧



上方第一部分的两张场景图片仅为示意；这里链接的是上传视频的实际截帧。
4. 仅改 Prompt 还不够：必须增加一个终止判断
对于你目前每 2 秒推理一次的 Temporal Critic，我建议由外部状态机负责最后的强制判定：
if is_terminal:    if any(        state[p] == "FAILED"        for p in phases    ):        overall = "FAILED"    elif all(        state[p] == "COMPLETED"        for p in phases    ):        overall = "SUCCESS"    else:        overall = "FAILED"        reason = "INCOMPLETE_AT_END"



这里的 state[p] 应该是跨帧累计、经证据确认的阶段状态，而不是只看最后一帧的输出。

另一个重要问题是采样频率。 你目前每 2 秒抽取一帧，可能恰好错过移液器接触试管、按压和松手这些短暂动作。建议在阶段 3、4、5 的关键交互区间临时提高采样频率，或传入短视频片段，而不是只依赖单张截图。

最终，我建议将这份 TOML 作为 Universal Visual Critic Prompt V2，先用同一份 Prompt 对 Result 2 和 Result 5 做离线回放，逐阶段检查状态和可见证据。这样既能降低对某个环境的过拟合，也能判断误判究竟来自 Prompt、视觉模型还是状态机逻辑。


我觉得需要完善，而且重点应该放在五个阶段的 Success / Failure 定义上，而不只是优化整体 Prompt。

你目前的五步划分是合理的，不需要增加阶段，但每一步的完成条件还有一些容易让 VLM 误判的地方。
1. 五个步骤具体应该怎么改
阶段	原定义的问题	我建议的改进
Step 1：取移液器	只要看起来离开支架就可能判成功	必须确认机器人稳定抓持，移液器已脱离支架
Step 2：取试管	没有充分考虑遮挡、移动过程	识别同一个试管，并确认已经被稳定拿起
Step 3：移液器对准试管	问题最大：对准、接触、按压混在一起	拆成内部三个判断：对准 → 接触/插入 → 按压释放
Step 4：放回试管	对 upright 的要求容易过于严格	重点判断是否进入正确孔位、松手后是否稳定
Step 5：放回移液器	容易把接近支架当作放置成功	必须确认支架实际承重、机械手释放、支架没有被推走

最重要的两个修改
第一，Step 3 必须区分“对准”和“执行成功”。
对准

Aligned

接触 / 插入

Contact

按压 / 释放

Actuation



这三个是 Step 3 的内部检查点，不需要把整个任务改成七个阶段。

如果只有对准，没有明确接触证据，就不能判 Step 3 成功；如果接触已经完成，但按钮动作被手指遮挡，就应输出 UNKNOWN 或 IN_PROGRESS，不能擅自判成功或失败。

第二，Step 4/5 要用松手后的物理状态判断。

真正的成功不是机器人把东西送到了正确位置，而是机器人松手后，物体仍然被目标支架稳定支撑。

比如试管轻微倾斜但仍插在正确的孔位里，并不等于失败；反之，即使试管竖直站在桌面上，也不能认为完成了回架。
2. 我建议直接替换成下面的定义
保留你现有的五个 phase 名字，避免破坏代码兼容性。
[task]
id = "vega_sim_pipette_five_step_workflow"

instruction = """
You are a visual critic evaluating a five-phase
robot manipulation task.

Identify objects by their functional roles and
track the same pipette, tube, rack, and stand
throughout the episode.

Do not depend on fixed object colors, exact
geometry, hand side, or camera viewpoint.

S1/R1, S2, S3/S4/S5, S6, and S7/S8 are
nominal phase mappings, not proof of success.

A phase is COMPLETED only when its required
physical outcome is supported by visual evidence.

Distinguish:
- COMPLETED: required outcome visibly achieved.
- IN_PROGRESS: action is visibly progressing.
- UNKNOWN: evidence is insufficient or occluded.
- FAILED: a definitive failure is visible.

Temporary misalignment, motion, or occlusion
does not by itself establish failure.

Preserve previously verified phase completions
unless subsequent evidence contradicts them.

If the episode explicitly ends before all five
phases are confirmed, the overall result is
FAILED (INCOMPLETE_AT_END).

Never infer an unseen interaction solely from
the subsequent phase or intended action.
"""

phases = [
  "step1_pipette_pick",
  "step2_tube_pick",
  "step3_pipette_aim_at_tube",
  "step4_tube_return",
  "step5_pipette_return"
]

criteria = [
  # STEP 1
  "The robot securely grasps the target pipette and lifts it clear of its initial stand. The pipette is visibly supported by the gripper rather than the stand, without uncontrolled motion.",

  # STEP 2
  "The robot securely grasps the target tube and lifts the same tube clear of its initial support. The tube remains controlled by the gripper and is available for the pipetting interaction.",

  # STEP 3
  "The pipette tip is aligned with the opening of the tracked tube, reaches the intended contact or insertion position, and the required press-and-release interaction visibly occurs. Alignment alone or proximity alone is insufficient. Do not infer hidden actuation without evidence.",

  # STEP 4
  "The same tracked tube is returned to its designated rack pocket or support location. After the gripper releases it, the tube remains seated and stable, approximately upright as required by the holder. Minor inclination within a supporting pocket is acceptable.",

  # STEP 5
  "The pipette is returned to its designated stand or holder. The holder visibly supports the pipette after gripper release, the pipette remains stable, and the holder retains its original position relative to the workspace."
]

failure_criteria = [
  # STEP 1
  "The pipette is visibly dropped, becomes uncontrolled or lost, or the stand is permanently displaced. A temporary failed grasp attempt that is successfully corrected is not automatically terminal failure.",

  # STEP 2
  "The tracked tube is visibly dropped, lost, or left unsupported outside its intended handling path. Temporary occlusion or a recoverable grasp adjustment alone is insufficient to declare failure.",

  # STEP 3
  "The pipette visibly contacts the wrong location, causes an unintended collision, or either object is dropped. A visibly aborted or stalled interaction that remains incomplete at episode termination is failure. Mere temporary misalignment is not failure.",

  # STEP 4
  "The tube is released outside its designated support, falls onto the table, remains unsupported or unstable, or the wrong tube is returned. A tube properly seated in a rack pocket must not be rejected solely for minor inclination.",

  # STEP 5
  "The pipette is released outside its holder, falls after release, remains unsupported, or the holder is visibly displaced from its intended fixed position. Approaching the stand without a verified stable release is not completion."
]


3. 还有一个比 Prompt 更关键的问题
不要让 VLM 同时负责“看到了什么”和“整个任务是否最终失败”。

更稳定的实现是：
- VLM：负责识别每个阶段的物理证据，并输出 COMPLETED / IN_PROGRESS / UNKNOWN / FAILED。
- Finite-State Controller（FC）：负责记录五个阶段是否完成、什么时候切换阶段，以及视频结束时是否判整体失败。
特别是你的 result5，如果没有看到明确掉落，也没有完成第三阶段，VLM 不应该提前编造一个物理失败原因。但视频终止后，FC 必须输出 INCOMPLETE_AT_END。

而对于 result2，不能仅凭当前帧没有看到第四阶段的回架动作，就把之前已经验证的第四阶段完成状态清空。
4. 推荐的最终策略
我建议保留五阶段结构，同时采用两级证据：

阶段内证据：接近、抓稳、接触、释放、稳定等，用于判断当前动作进展。

阶段完成证据：满足当前任务真正要求的最终物理状态，用于决定是否允许推进下一阶段。

尤其 Step 3，可以额外返回：
{
  "phase": "step3_pipette_aim_at_tube",
  "evidence": {
    "tip_aligned": true,
    "tip_contact": true,
    "actuation_observed": false
  },
  "status": "UNKNOWN",
  "advance": false
}


这意味着模型已观察到对准和接触，但按压动作没有足够证据，因此不能把该阶段判断为完成。

我的最终建议：五个阶段不用增加，但 Step 3 应细化内部检查点，Step 4/5 必须强调松手后的稳定状态，再由 FC 负责跨帧记忆和终止判定。 这会比单纯把 Prompt 写得更长，更有希望同时修复 Result 2 的假失败和 Result 5 的漏判失败。
