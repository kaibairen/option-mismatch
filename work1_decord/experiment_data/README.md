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

2026-10-23 用 `qwen3.5-27b` 跑的 AQUA Direct、AQUA CoT、AQUA Self-Refine，以及 AQUA DecoRD 的前 3 题，请求发到 `https://www.dmxapi.cn/v1`。温度 0.4，`top_p` 1，`enable_thinking` 关闭。Direct 的 `max_tokens` 是 512。CoT 是一次补全，最多 2048 token，字母只取回复里的答案句。Self-Refine 的初稿、反馈和重写也是 2048 token。每一行的 `response` 是用来计分的模型原文；Self-Refine 另有 `trace`，按调用顺序留下每一次回复。

Direct 的 254 题都在 512 token 内结束。CoT 有 28 题在 2048 token 处被截断（`finish_reason` 为 `length`），其中 15 题没有可计的答案句。另有 14 题正常结束，但收尾写成了 `Answer Choice: X` 这一类，同一套答案句规则不把它们计成字母。这 29 题的 `pred` 为空，仍留在分母里，原文在 `response`。

| 文件 | 数据 | 结果 |
| --- | --- | --- |
| `qwen3.5-27b/direct/aqua.jsonl` | AQUA-RAT test，254 | 141/254 |
| `qwen3.5-27b/cot/aqua.jsonl` | AQUA-RAT test，254。一次 CoT，最多 2048 token | 210/254 |
| `qwen3.5-27b/self_refine/aqua.jsonl` | AQUA-RAT test，254。Self-Refine，最多 4 轮 | 217/254 |
| `qwen3.5-27b/decord_standard/aqua.jsonl` | AQUA-RAT test 前 3 题。DecoRD，题干推理再匹配选项 | 3/3 |

DecoRD 这份目前只有前 3 题，按题目顺序写着，后面用同一条命令可以继续。三题的决策字母和结论都对上了标准答案，关系分别是 `≡`、`≈`、`≡`。`trace` 里是推理和决策两次原文。没有跑置信度反思。

Self-Refine 从 14:35 跑到 16:15（Asia/Shanghai）。254 题都有返回，没有请求失败，一共 590 次调用。232 题在第一次反馈后停下。22 题至少重写了一次，其中 7 题改对、1 题改错，另外 14 题对错没有变。18 题没有抽出字母，仍留在分母里；多数是初稿以 `Answer Choice: X` 收尾，反馈认定它正确，所以没有重写成答案句。
