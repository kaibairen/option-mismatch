# DecoRD

工作 1 的代码按 AAAI 稿 *We Get the Answer, but Where is the Answer?* 重写。这一轮只用 DecoRD 框架，两步：

1. 推理智能体只看题干，输出 `reasoning` 和 `conclusion`。
2. 决策智能体用 `≈`、`≡`、`→` 把结论配到选项，输出一个字母。

稿子里的置信度反思还留在代码里，这一轮不跑。

耦合的 CoT 基线仍把题干和选项放在同一次调用里。

生成设置来自稿子：温度 0.4，`top_p` 1，`max_tokens` 512。CoT 用 Kojima 等人的零样本触发句，题干和选项在同一条 `Q:` 里，模型从 `A: Let's think step by step.` 往后写。这一模型在 512 token 时还停在推导中途，所以 CoT 和 DecoRD 的题干推理放到 2048，让结论能写完。选项匹配仍是 512。字母只取这次回复里最后一处明确选择：`Answer: X`、`answer is X`、`the correct option is X`，或 `matches option X`。推理里顺手提到的选项字母不算，截断后也不再补一次调用。采样次数稿子只写成 Sample N，包内默认 5。单次请求超时 180 秒，这是这次重写的工程选择，用来替换旧 `evaluate.py` 里的 20 秒超时。失败会记成 `error`，不再当成空答案从分母里拿掉。

## 这一轮要跑的格子

`configs/experiment.json` 锁定四个模型、五个零样本系统，题目是 `dataset/manifest.json` 里的 `experiment_files`：AQUA-RAT 全部 254 题，加上 MMLU 每科 40 题（seed 42，共 2280 题）。LogiQA 和高考仍留在 `dataset/`，不进这一轮。

| 模型 | 请求里的模型名 | 思考开关 |
| --- | --- | --- |
| Qwen3.5-9B | `qwen3.5-9b` | `enable_thinking: false` |
| Qwen3.5-27B | `qwen3.5-27b` | `enable_thinking: false` |
| DeepSeek-V3 | `deepseek-v3` | 不带 |
| GPT-4 | `gpt-4` | 不带 |

五个系统都看同一批题。Direct、CoT、Plan-and-Solve、Self-Refine 每次调用都看题干和选项。DecoRD(Standard) 是题干推理一次，再用 `≈`、`≡`、`→` 匹配选项一次。逐题记录除了决策字母，还留下 `trace`、`conclusion`、`reasoning_correct` 和 `relation`，事后可以重算 DCDI。Plan-and-Solve 用 Wang 等人的一句触发语，计划和求解写在同一次回复里，字母规则与 CoT 相同。Self-Refine 用 Madaan 等人 NeurIPS 2023 的算法：初稿和 CoT 用同一句提示，然后同一个模型写反馈；反馈里出现 `It is correct` 就停，否则带着以往的解答和反馈重写。最多 4 轮。标准答案只用来计分。论文的数学实验是带示例的 Python 程序，附录里还用标签决定是否继续；这一轮保持零样本，答案是选项字母。置信度反思的函数还在包里，这一轮不使用。

MMLU 的大类分数和 57 科总分都是学科准确率的不加权平均。AQUA 单独计分，并在 DecoRD 上计算 DCDI。两套数据不合成一个分数。

```bash
cd work1_decord
PYTHONPATH=src python -m decord run --dry-run
# 真正调用前设置 BASE_URL（OpenAI 兼容根路径，含 /v1）和 API_KEY
PYTHONPATH=src python -m decord run --model qwen3.5-9b --method direct --limit 2
PYTHONPATH=src python -m decord score --runs runs
```

`--dry-run` 不访问网络。正式运行把每题一行写到 `runs/<model>/<method>.jsonl`，已经写过的题会跳过。`runs/` 不进 git。

AQUA 和 MMLU 的逐题记录用 `record`。每答完一题就追加一行并 `fsync`。一行至少有 `question`、`answer`、`pred`；这次调用留下的模型原文写在 `response`，接口给出的结束原因写在 `finish_reason`。文件按题目顺序保存，已有前缀会跳过。请求失败时停在该题，不把后面的题写进去，下次从这题继续。文件末尾如果是写到一半的半行，恢复时会丢掉这一行再重跑该题。MMLU 有重复题干，所以不能只按题干去重。要进 git 的记录放在 `experiment_data/<模型>/<方法>/<数据集>.jsonl`。

```bash
BASE_URL=http://127.0.0.1:16006/v1 API_KEY=local \
PYTHONPATH=src python -m decord record --model qwen3.5-9b --method direct \
  --dataset aqua_rat --output experiment_data/qwen3.5-9b/direct/aqua.jsonl
PYTHONPATH=src python -m decord record --model qwen3.5-9b --method direct \
  --dataset mmlu --output experiment_data/qwen3.5-9b/direct/mmlu.jsonl
```

## 已有结果

`reports/` 里是三套已经打印出来的数，导入包不会重算它们。

| 文件 | 来源 |
| --- | --- |
| `aaai_reported.json` | AAAI 稿 Table 1–3 |
| `nota_sheet.json` | 上传的 `data.xlsx`，含 NOTA 两列 |
| `aqua_logiqa_valid_only.json` | 后来的 AQUA / LogiQA 汇总。分母是抽出了字母的题，温度 0.7，超时 20 秒 |

Deepseek-v3 在 Table 1 印出的 DCDI 是 15.93%。按公式 `(0.6181 - 0.5629) / 0.6181` 不是这个数。`dcdi()` 保留公式。

## 用法

```bash
cd work1_decord
python -m pytest
PYTHONPATH=src python -m decord
PYTHONPATH=src python -m decord --log runs/decord_standard.jsonl --n-full 250
```

新跑出来的 JSONL 每一行要有 `status`（`ok`、`empty`、`no_letter`、`error`）和 `correct`。`summarize_predictions` 同时给全量准确率、有效准确率和失败计数。

工作一的代码和归档表都在这个目录。
