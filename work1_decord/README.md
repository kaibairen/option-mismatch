# DecoRD

工作 1 的代码按 AAAI 稿 *We Get the Answer, but Where is the Answer?* 重写。一次调用的路径是：

1. 推理智能体只看题干，输出 `reasoning` 和 `conclusion`。
2. 决策智能体用 `≈`、`≡`、`→` 把结论配到选项，输出一个字母。
3. 置信度反思在决策步采样。置信度是 `1 - H(p) / log(M)`。低于 0.5 时再反思一次。

耦合的 CoT 基线仍把题干和选项放在同一次调用里。

生成设置来自稿子：温度 0.4，`top_p` 1，`max_tokens` 512。采样次数稿子只写成 Sample N，包内默认 5。单次请求超时 180 秒，这是这次重写的工程选择，用来替换旧 `evaluate.py` 里的 20 秒超时。失败会记成 `error`，不再当成空答案从分母里拿掉。

## 这一轮要跑的格子

`configs/experiment.json` 锁定三个模型、五个零样本系统，题目是 `dataset/manifest.json` 里的 `experiment_files`：AQUA-RAT 全部 254 题，加上 MMLU 每科 40 题（seed 42，共 2280 题）。

| 模型 | 角色 |
| --- | --- |
| `qwen3-8b` | 开源 8B。请求带 `enable_thinking: false` |
| `gpt-4o-mini` | 当前闭源轻量模型 |
| `deepseek-v4-pro-guan` | 当前较强的官方模型。请求带 `thinking.type = disabled` |

五个系统都看同一批题。Direct、CoT、Plan-and-Solve、Reflexion 每次调用都看题干和选项。DecoRD 是题干推理一次，再匹配选项一次。Plan-and-Solve 拆成两次调用，第二次读第一次写出的计划。Reflexion 是一次 CoT 再加一次修改，不再重试。置信度反思的函数还在包里，这一轮的 `methods` 不包含它。

MMLU 的大类分数和 57 科总分都是学科准确率的不加权平均。AQUA 单独计分，并在 DecoRD 上计算 DCDI。两套数据不合成一个分数。

```bash
cd work1_decord
PYTHONPATH=src python -m decord run --dry-run
# 真正调用前设置 BASE_URL（OpenAI 兼容根路径，含 /v1）和 API_KEY
PYTHONPATH=src python -m decord run --model qwen3-8b --method direct --limit 2
PYTHONPATH=src python -m decord score --runs runs
```

`--dry-run` 不访问网络。正式运行把每题一行写到 `runs/<model>/<method>.jsonl`，已经写过的题会跳过。`runs/` 不进 git。

AQUA 的逐题记录用 `record`。每答完一题就追加一行 `{"question","answer","pred"}` 并 `fsync`。请求失败的题不写入，下次从断点继续。文件末尾如果是写到一半的半行，恢复时会丢掉这一行再重跑该题。

```bash
BASE_URL=http://127.0.0.1:16006/v1 API_KEY=local \
PYTHONPATH=src python -m decord record --model qwen3.5-9b --method direct \
  --output runs/qwen3.5-9b/aqua_direct.jsonl
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
