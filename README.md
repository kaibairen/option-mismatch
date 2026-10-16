# 硕士工作

两项工作放在同一个仓库里，目录互不调用。

## work1_decord

多步智能体：题干推理、选项匹配、置信度反思。AAAI 稿的表在 `work1_decord/reports/`。

```bash
cd work1_decord
python -m pytest
python -m decord
```

## work2_option_mismatch

Phase III 选项错配的表征测量，以及 SFT / DPO / Rep-DPO / DECORD-SFT。冻结结果在 `work2_option_mismatch/reports/`，其中 DECORD 的 AQUA 门禁在 `reports/decord/`。

```bash
cd work2_option_mismatch
bash scripts/ci/cloud_agent_install.sh
```

脚本里的根目录按脚本自身位置计算，所以从仓库根目录直接执行 `work2_option_mismatch/scripts/` 下的脚本也可以。

## dataset

工作一锁定的评测集，和上面两个目录平级。来源、题量和 MMLU 抽样都写在 `dataset/manifest.json`。

| 文件 | 内容 |
| --- | --- |
| `dataset/aqua_rat/test.jsonl` | AQUA-RAT test，254 |
| `dataset/logiqa/test.jsonl` | LogiQA 英文 test，651 |
| `dataset/mmlu/eval/` | STEM、Social Sciences、Humanities、Other 各 100，种子 42 |
| `dataset/mmlu/pools/` | 上面 400 题所在的官方 test 全量，14042 |
| `dataset/gaokao_bench/` | 理科数学 214、文科数学 218、英语单选 105、生物 150、化学 124、历史 287 |

每行一条记录：`id`、`dataset`、`split`、`subset`、`question`、`options`、`answer`、`meta`。`meta` 里的解析和原题不进入提示词。`dataset/prepare.py` 可以从原始文件重新生成这批 jsonl。
