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

Phase III 选项错配的表征测量，以及 SFT / DPO / Rep-DPO 修复。冻结结果在 `work2_option_mismatch/reports/`。

```bash
cd work2_option_mismatch
bash scripts/ci/cloud_agent_install.sh
```

脚本里的根目录按脚本自身位置计算，所以从仓库根目录直接执行 `work2_option_mismatch/scripts/` 下的脚本也可以。
