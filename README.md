# 硕士工作

两项工作放在同一个仓库里，目录互不调用。

## work1_decord

多步智能体 DecoRD。这一轮四个模型是 Qwen3.5-9B、Qwen3.5-27B、DeepSeek-V3、GPT-4，对照是 Direct、CoT、Plan-and-Solve、Reflexion。题目是 AQUA-RAT test 和 MMLU 每科 40 题。已提交的逐题记录在 `work1_decord/experiment_data/<模型>/<方法>/`，稿子里的旧表在 `work1_decord/reports/`。

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
| `dataset/mmlu/subject40/` | 论文用的 MMLU：57 个学科各 40 题，种子 42，共 2280。抽中的题号在 `draw.json` |
| `dataset/mmlu/eval/` | 早先的连通性样本：四个大类各 100，种子 42 |
| `dataset/mmlu/pools/` | 官方 test 全量，14042 |
| `dataset/gaokao_bench/` | 理科数学 213、文科数学 218、英语单选 105、生物 148、化学 103、历史 287 |
| `dataset/excluded/gaokao_bench.jsonl` | 从高考评测集剔除的 24 题：选项只是字母，或题干/选项是图片 |

每行一条记录：`id`、`dataset`、`split`、`subset`、`question`、`options`、`answer`、`meta`。`meta` 里的解析和原题不进入提示词。`dataset/prepare.py` 可以从原始文件重新生成这批 jsonl。
