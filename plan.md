# Long-Horizon Agentic GRPO — 项目计划与状态

> 这份文档是交接记录。下次回来先读这个,就知道做了什么、卡在哪、怎么接。
> 最后更新:腿1/腿2 算法核心 + advantage 消费端 + patch 已完成并本地验证;
> 剩余产出端(rollout→DataProto)待服务器接入。

## 一句话定位

**outcome-only GRPO + 腿1(hard constraint)+ 腿2(divergence credit)。**

这不是 credit assignment 的根本解法(critic/PRM 被 anti-hack 约束排除了),
而是在 outcome-GRPO 框架内、不引入可 hack 的 learned signal 的前提下,能
做到的最大化改善。本质还是 GRPO(group-relative advantage + clipped ratio
+ outcome reward)。

## 核心原则(不要违背)

reward = **terminal outcome only**(任务成功 0/1)。任何更细的过程信号都
**不进 reward**(进了就被优化、被 hack)。过程信息只走两个非 hack 通道:
- **腿1 = hard constraint**(机械检查 → 违反的 rollout reject / score=0)
- **腿2 = 固定结构权重**(divergence turn 拿更多 credit,不学习、不进 reward)

## 方法

### 腿1 — 过程正确性 hard constraint
机械可检查的错误 → 违反的 rollout 即使环境判成功也强制失败:
- placeholder 参数 / schema 违反 / loop / max-turn stall / **ungrounded write**
  (schema 驱动:entity 参数值没在前序 observation 出现)
- reward 层:`constrained_reward = success AND NOT violated`(score masking)
- advantage 层:violated rollout 从 group baseline 排除 + advantage=0(patch)
- curriculum:`warmup_steps` 早期放宽(烂 policy 能学)

### 腿2 — divergence-based per-turn credit
同 group G 条 rollout,逐 turn 对齐 action,action 不一致的 turn = divergence
turn(真正决策点)→ 拿更高 credit;shared turn 拿低。权重读自轨迹结构,总
credit 守恒(只改分布,不改来源/粒度)。

## 完成状态

### ✅ 本地完成 + 已验证
- **算法核心**:`constraint_gate`、`grounded_write_verifier`(schema 驱动)、
  `divergence`、`advantage` 的 per-turn credit helper —— `test_legs.py` 17/17、
  `run_tests.py` pass、py_compile 全 OK、顶层 import 22 exports。
- **reward 接入**:`reward_state.compute_long_horizon_components` 算 grounding +
  constraint + curriculum(score masking)。
- **advantage 消费端**(patch 注入 veRL):`compute_grpo_long_horizon_advantage`
  消费 `turn_token_weights`(腿2)+ `constraint_violated` reject(腿1);
  `patch_ray_trainer` 传这两个字段进 adv_kwargs。
- **patch 已 apply** 到本项目 `verl/`(0.6.1,锚点匹配):core_algos 有
  `constraint_violated`/`turn_token_weights`,ray_trainer 有两个 marker。
- **清理**:删了 Delta-Critic/Evidence Ledger/mock/calibrated 等旧方向;
  `docs/method.md`、`README.md` 已重写为腿1+腿2;死代码/死脚本/无引用配置已删。

### ❌ 未完成(服务器侧,本地跑不了真实 rollout 无法验证)

| 项 | 描述 | 本地能写 | 本地能验 | 难度 |
|---|---|---|---|---|
| ② token→turn 在线对齐 | rollout 产出每个 assistant turn 的 token 范围 | 能 | **不能** | 🔴 最硬 |
| ③ turn_token_weights 产出 | 对齐 response_mask 存 DataProto.batch | 能 | 不能 | 中(卡 ②) |
| ④ constraint_violated 存 batch | reward function 把 `rejected` 写进 batch | 能 | 不能 | 🟢 最简单 |
| ⑤ divergence 在 ray_trainer 算 | 同 group action_history → divergence → tensor | 能 | 不能 | 中 |
| ① true reject-resample | trainer 违反 rollout 丢弃+重生成补采 | 能 | 不能 | 增量(可不做) |

**依赖关系**:腿1 只要 ④ 就在线生效;腿2 要 ②③⑤ 三个一起(② 是瓶颈);
① 是增量(advantage reject 已覆盖"违反的不学习/不污染 baseline"这个主要效果)。

## 协作流程(上服务器接产出端)

1. **确认服务器 veRL = 0.6.1**(和本项目 `verl/` 同版本,patch 锚点才匹配)。
2. **跑一次 dump**(只要一次):在 `verl/verl/trainer/ppo/ray_trainer.py` 的
   `generate_sequences` 之后(约 1042 行后)临时插入下面的代码,设
   `DCL_DUMP_ROLLOUT` 跑一个 step,Ctrl-C,把 dump 文件内容贴回来:

   ```python
   import os as _os
   if _os.environ.get("DCL_DUMP_ROLLOUT"):
       import json as _json
       _b = gen_batch_output
       _dump = {
           "batch_keys": list(_b.batch.keys()),
           "non_tensor_keys": list(_b.non_tensor_batch.keys()) if hasattr(_b, "non_tensor_batch") else [],
           "response_mask_shape": list(_b.batch["response_mask"].shape),
           "response_mask_row0": _b.batch["response_mask"][0].tolist()[:600],
       }
       try:
           _dump["responses_row0_decoded"] = self.tokenizer.decode(_b.batch["responses"][0].tolist())[:1000]
       except Exception as _e:
           _dump["responses_decode_error"] = repr(_e)
       for _k in ("uid", "action_history", "__num_turns__", "total_tool_calls"):
           if hasattr(_b, "non_tensor_batch") and _k in _b.non_tensor_batch:
               _dump[f"non_tensor_{_k}_0"] = str(_b.non_tensor_batch[_k][0])[:400]
       with open(_os.environ["DCL_DUMP_ROLLOUT"], "w") as _f:
           _f.write(_json.dumps(_dump, indent=2, default=str))
       print("[DCL_DUMP] wrote", _os.environ["DCL_DUMP_ROLLOUT"])
   ```

3. **我基于 dump 写 ②③④⑤**(② 看 `response_mask_row0` 的分段确定 turn 边界;
   ⑤ 看 `non_tensor_keys` 有没有 `action_history`)。
4. **你服务器跑,贴报错/输出,我改,迭代** 到在线生效。

## 关键决策日志(为什么这么做,别改回去)

1. **弃 Delta-Critic / Evidence Ledger / calibrated process model**:本质都是
   learned per-step signal,进 reward 就被 hack。已全删。
2. **弃 critic / GAE**:agentic 长 horizon critic 难训、不稳。这导致 credit
   assignment 在 outcome-GRPO 里只能"筛选 + 重分配",做不到每步独立信号。
3. **腿1 用 reject(hard constraint)不用 reward shaping**:约束是二值、不可
   trade-off,agent 没有"部分满足"的梯度,所以不可 hack。
4. **腿2 用 divergence(结构)不用 learned scorer**:权重读自轨迹分叉结构,
   固定、不学习、不在 reward 里,不可 hack。
5. **grounded verifier schema 驱动**(`entity_keys_from_schema`):从工具 schema
   描述判断 entity 字段,泛化到任意 tool agent(tau-bench airline/retail/telecom),
   不靠 airline 命名。"such as" 太宽(origin/destination 也用),用
   stored-in/identifier/id/number 信号。
6. **curriculum warmup**:`set_training_step` + `warmup_steps`,早期放宽。
   reward 阶段没有全局 step,靠 trainer 调 `set_training_step` 注入。
7. **四个保留设计**(用户明确):loop guard / balanced aggregation / dynamic
   clipping / group variance soft filtering —— 不删。balanced agg 默认关(opt-in)。
8. **patch 比 veRL 源码新**:每次 `patch_verl_long_horizon_grpo.py` 重新 apply
   才生效;服务器 veRL 要 re-apply。
9. **`observation_preview` 不能截**(SFT 真实数据验证发现):`tools.py` 的
   observation 必须存**完整**,不能截 500/3000。`get_user_details` 返回几 KB,
   截断会把后面的 `payment_id`/`reservation_id` 截掉 → grounding 把干净 write
   误判 ungrounded → 腿1 误杀。SFT 干净 teacher 数据实测:截 500→17 FP、
   截 3000→7 FP、完整→0 FP。已改成完整(`obs`)。
10. **divergence 在策略发散的任务退化**(SFT 验证):只在"有 shared 前缀 +
    少数决策分叉"的任务有效(task 0/24 准);策略高度发散时(first_divergence_turn=0,
    几乎全 turn 分叉)退化成无信号(task 33/37/38);完全相同的 rollout 无信号
    (task 34)。GRPO 同 prompt 同温度采样 shared 前缀通常更多,可能比 SFT 多温度
    sample 好,但必须服务器用 GRPO 数据验。
11. **GRPO config 已对齐参考项目**(agentic-grpo-longhorizon):`rollout.n=8`
    (G=4 零方差组太多,成功率 0.7 时 25% 零方差 vs G=8 的 6%)、
    `max_user/assistant_turns=15`、`max_response_length=12288`(6144 不够 15 turn
    累计会被截断 → 任务没跑完)、`max_model_len=24576`(prompt 10240+response 12288
    =22528)、`max_num_batched_tokens=24576`。`max_tool_response_length=256`+
    `truncate_side=middle`(veRL 默认,保留头尾各 128 字符,entity ID 在头尾够)。
    **别改回 n=4 / response 6144**。保留本项目特色:`adv_estimator=grpo_long_horizon`、
    `loss_mode=long_horizon_balanced`(参考用 grpo_lata / vanilla)。
    显存注意:max_model_len 24576 + n=8,GPU0 同时训练+rollout+logprob,`gpu_memory_utilization`
    0.32 可能要调 0.4+,OOM 则 `max_model_len` 回 20480(prompt 10240+response 8192)。
12. **SFT 收集配置**(对齐参考项目,它出了 67 条=train 45+holdout 22):
    `collect_sft_teacher_72b_user_32b_2xa800.sh` 用 **best-of-n=8**(temperatures
    `0,0,0.5,0.5,0.8,0.8,1.0,1.0`,成对;sh 原本是 best-of-4 已改 8)、50 airline task
    (test split)、72B teacher + 32B user sim、`--holdout-size 10`、
    `--contamination-char-limit 35000`。预计 ~80-150 条成功 trajectory。
    **contamination 35000 够**(实测参考 trajectory 最长 21233、中位 10116,0 条超 35000;
    之前担心"长 task 被丢"是错判,已收回)。**别改回 best-of-4**(数据量不够)。
    体检指标看 `summary.json`:`total_contaminated_trajectories` 占比、`task_coverage_rate`
    (低说明 teacher 弱,加大 best-of-n 或换 teacher)。

## 模块索引

- `src/delta_critic_ledger/long_horizon/`:
  - 腿1:`constraint_gate.py`、`grounded_write_verifier.py`
  - 腿2:`divergence.py`、`advantage.py`(per-turn credit + dynamic clip)
  - 护栏:`loop_guard.py`、`reward_envelope.py`
  - GRPO 改进:`aggregation.py`(balanced,默认关)、`group_filter.py`(软过滤)
  - 信号源:`process_features.py`
- `src/delta_critic_ledger/verl_integration/`:
  - `interaction.py`(LongHorizonTauBenchInteraction)、`agent_loop.py`、
    `tools.py`、`reward_state.py`(腿1 接入)、`reward.py`(compute_score)、`context.py`
- 顶层:`adaptive_control.py`、`sft_dataset.py`、`evaluation.py`、`tau_compat.py`、
  `prompts.py`、`schemas.py`、`config.py`
- `scripts/setup/patch_verl_long_horizon_grpo.py`:注入 veRL 的 patch(advantage +
  ray_trainer 传参)。**服务器每次重 apply。**
- `scripts/test/test_legs.py`:腿1/腿2 单测(17/17)。

## 已清理(不要再找)

Delta-Critic / Evidence Ledger / oracle delta / mock_airline / calibrated
process model —— 全弃用已删。`docs/method.md`、`README.md` 已重写。死代码
(`failure_taxonomy` 等)、死脚本(`eval_delta_grpo_*`)、旧 ablation
(`airline_mini`)、无引用配置(`terminal_only` 等)已删。

## 下次回来怎么接

1. 读本文件。
2. 跑 `PYTHONPATH=src python3 scripts/test/test_legs.py` 确认算法核心在(应 17/17)。
3. 跑 `grep -c LONG_HORIZON_GRPO_PATCH verl/verl/trainer/ppo/core_algos.py` 确认 patch 在。
4. 上服务器:按"协作流程"跑 dump → 我写 ②③④⑤ → 迭代。
