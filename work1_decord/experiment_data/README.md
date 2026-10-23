# 实验结果

目录是 `experiment_data/<model>/<method>/<dataset>.jsonl`。`runs/` 被忽略，只留在当次机器上，换环境就会丢。

每一行至少有 `question`、`answer`、`pred`。`answer` 是标准答案，`pred` 是模型给出的字母。新跑的记录另外保留模型原文 `response`，以及接口给出的 `finish_reason`。9B 的三份是 2026-10-20 的旧日志，当时没有留下原文。文件按题目顺序保存，中断后用同一条命令可以接着写。

当前格子的四个模型是 `qwen3.5-9b`、`qwen3.5-27b`、`deepseek-v3`、`gpt-4`。

## qwen3.5-9b

2026-10-20 重新跑的 Direct 和 AQUA CoT。温度 0.4，`top_p` 1，`enable_thinking` 关闭。Direct 的 `max_tokens` 是 512。CoT 是一次补全，最多 2048 token，字母只取回复里的答案句。10 月 17 日那次的逐题文件没有进 git，已经不在磁盘上，这里不是那次的原件。

| 文件 | 数据 | 结果 |
| --- | --- | --- |
| `qwen3.5-9b/direct/aqua.jsonl` | AQUA-RAT test，254 | 94/254 |
| `qwen3.5-9b/direct/mmlu.jsonl` | MMLU 每科 40 题，2280 | 1792/2280 |
| `qwen3.5-9b/cot/aqua.jsonl` | AQUA-RAT test，254。一次 CoT，最多 2048 token | 211/254 |

## qwen3.5-27b

2026-10-23 用 `qwen3.5-27b` 跑的 AQUA Direct，请求发到 `https://www.dmxapi.cn/v1`。温度 0.4，`top_p` 1，`max_tokens` 512，`enable_thinking` 关闭。254 题都有返回，`finish_reason` 都是 `stop`。每一行的 `response` 是模型原文。

| 文件 | 数据 | 结果 |
| --- | --- | --- |
| `qwen3.5-27b/direct/aqua.jsonl` | AQUA-RAT test，254 | 141/254 |
