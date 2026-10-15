# DecoRD

工作 1 的代码按 AAAI 稿 *We Get the Answer, but Where is the Answer?* 重写。一次调用的路径是：

1. 推理智能体只看题干，输出 `reasoning` 和 `conclusion`。
2. 决策智能体用 `≈`、`≡`、`→` 把结论配到选项，输出一个字母。
3. 置信度反思在决策步采样。置信度是 `1 - H(p) / log(M)`。低于 0.5 时再反思一次。

耦合的 CoT 基线仍把题干和选项放在同一次调用里。

生成设置来自稿子：温度 0.4，`top_p` 1，`max_tokens` 512。采样次数稿子只写成 Sample N，包内默认 5。单次请求超时 180 秒，这是这次重写的工程选择，用来替换旧 `evaluate.py` 里的 20 秒超时。失败会记成 `error`，不再当成空答案从分母里拿掉。

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
python -m decord
python -m decord --log runs/decord_standard.jsonl --n-full 250
```

新跑出来的 JSONL 每一行要有 `status`（`ok`、`empty`、`no_letter`、`error`）和 `correct`。`summarize_predictions` 同时给全量准确率、有效准确率和失败计数。

工作一的代码和归档表都在这个目录。
