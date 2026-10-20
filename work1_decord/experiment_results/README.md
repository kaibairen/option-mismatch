# 实验结果

这里放会提交进 git 的逐题记录。`runs/` 被忽略，只留在当次机器上，换环境就会丢。

每一行是 `{"question","answer","pred"}`。`answer` 是标准答案，`pred` 是模型给出的字母。文件按题目顺序保存，中断后用同一条命令可以接着写。

## qwen3.5-9b

2026-10-20 重新跑的 Direct。温度 0.4，`top_p` 1，`max_tokens` 512，`enable_thinking` 关闭。10 月 17 日那次的逐题文件没有进 git，已经不在磁盘上，这里不是那次的原件。

| 文件 | 数据 | 结果 |
| --- | --- | --- |
| `qwen3.5-9b/aqua_direct.jsonl` | AQUA-RAT test，254 | 94/254 |
| `qwen3.5-9b/mmlu_direct.jsonl` | MMLU 每科 40 题，2280 | 1792/2280 |
