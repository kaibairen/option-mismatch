"""Build the locked Work 1 evaluation sets into dataset/.

Reads the raw files listed in RAW_FILES (cached outside the repo) and writes
jsonl plus manifest.json next to this script. The committed jsonl is the
locked copy; manifest.json records sources, counts, and the MMLU draw.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

# EleutherAI lm-evaluation-harness, lm_eval/tasks/mmlu/_generate_configs.py
# Insertion order is the MMLU pool order within each supercategory.
MMLU_SUBJECTS = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

MMLU_GROUPS = ("stem", "social_sciences", "humanities", "other")
MMLU_SAMPLE_SEED = 42
MMLU_SAMPLE_SIZE = 100

GAOKAO_FILES = (
    ("sci_math", "math_i.json", "2010-2022_Math_I_MCQs.json", 214),
    ("hum_math", "2010-2022_Math_II_MCQs.json", "2010-2022_Math_II_MCQs.json", 218),
    ("english", "2010-2013_English_MCQs.json", "2010-2013_English_MCQs.json", 105),
    ("biology", "2010-2022_Biology_MCQs.json", "2010-2022_Biology_MCQs.json", 150),
    ("chemistry", "2010-2022_Chemistry_MCQs.json", "2010-2022_Chemistry_MCQs.json", 124),
    ("history", "2010-2022_History_MCQs.json", "2010-2022_History_MCQs.json", 287),
)

GAOKAO_BASE = (
    "https://raw.githubusercontent.com/OpenLMLab/GAOKAO-Bench/main/"
    "Data/Objective_Questions/"
)
OPTION_MARKER = re.compile(r"([A-D])\s*([\.．])")
AQUA_OPTION = re.compile(r"^([A-E])\)\s*(.*)$", re.S)
LOGIQA_LABEL = re.compile(r"^([A-D])[\.\．、\?\)\:\s]\s*(.*)$", re.S)
LETTERISH = re.compile(
    r"^(?:\$+)?(?:\\mathrm\{)?([A-D])(?:\})?(?:\$+)?$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_line(row: dict) -> str:
    # U+0085 / U+2028 / U+2029 are line breaks that json.dumps leaves raw
    # when ensure_ascii is false. Official MMLU contains U+0085.
    line = json.dumps(row, ensure_ascii=False, sort_keys=False)
    return (
        line.replace("\u0085", "\\u0085")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(jsonl_line(row) + "\n" for row in rows)
    for line in payload.splitlines():
        json.loads(line)
    path.write_text(payload, encoding="utf-8")
    return {
        "path": path.as_posix(),
        "count": len(rows),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "bytes": len(payload.encode("utf-8")),
    }


def normalize_option(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # A line break inside a Chinese word is an extraction artifact.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def letterish(text: str) -> bool:
    return LETTERISH.match(text.strip()) is not None


def record(
    *,
    item_id: str,
    dataset: str,
    subset: str | None,
    question: str,
    options: list[dict],
    answer: str,
    meta: dict,
) -> dict:
    letters = [opt["letter"] for opt in options]
    if len(letters) < 2 or len(set(letters)) != len(letters):
        raise ValueError(f"{item_id} has option letters {letters}")
    if answer not in letters:
        raise ValueError(f"{item_id} answer {answer} not in {letters}")
    if not question.strip():
        raise ValueError(f"{item_id} has an empty question")
    for opt in options:
        if not opt["text"]:
            raise ValueError(f"{item_id} option {opt['letter']} is empty")
    row = {
        "id": item_id,
        "dataset": dataset,
        "split": "test",
        "subset": subset,
        "question": question.strip(),
        "options": options,
        "answer": answer,
        "meta": meta,
    }
    return row


def split_embedded_options(question: str) -> tuple[str, list[dict], str]:
    matches = list(OPTION_MARKER.finditer(question))
    letters = [match.group(1) for match in matches]
    start = None
    for index in range(len(letters) - 3):
        if letters[index : index + 4] == ["A", "B", "C", "D"]:
            start = index
    if start is None:
        raise ValueError("no A. B. C. D. option block")
    chosen = matches[start : start + 4]
    stem = question[: chosen[0].start()]
    stem = re.sub(r"([）\)])\d{1,3}\s*$", r"\1", stem).strip()
    options = []
    for index, match in enumerate(chosen):
        end = chosen[index + 1].start() if index + 1 < len(chosen) else len(question)
        options.append(
            {
                "letter": match.group(1),
                "text": normalize_option(question[match.end() : end]),
            }
        )
    layout = "embedded"
    if all(letterish(opt["text"]) for opt in options):
        layout = "table_row_letters"
    return stem, options, layout


def build_aqua(raw: Path) -> list[dict]:
    rows = []
    lines = raw.read_text(encoding="utf-8").splitlines()
    if len(lines) != 254:
        raise ValueError(f"AQUA test has {len(lines)} lines, expected 254")
    for index, line in enumerate(lines):
        item = json.loads(line)
        options = []
        for raw_opt in item["options"]:
            match = AQUA_OPTION.match(raw_opt.strip())
            if not match:
                raise ValueError(f"AQUA option {raw_opt!r} at {index}")
            options.append({"letter": match.group(1), "text": match.group(2).strip()})
        answer = str(item["correct"]).strip().upper()
        rows.append(
            record(
                item_id=f"aqua_rat-test-{index:04d}",
                dataset="aqua_rat",
                subset=None,
                question=item["question"].strip(),
                options=options,
                answer=answer,
                meta={"rationale": item.get("rationale", "").strip()},
            )
        )
    return rows


def build_logiqa(raw: Path) -> list[dict]:
    lines = raw.read_text(encoding="utf-8").splitlines()
    rows = []
    cursor = 0
    while cursor < len(lines):
        if lines[cursor].strip() == "":
            cursor += 1
            continue
        answer = lines[cursor].strip()
        cursor += 1
        chunk = []
        while cursor < len(lines) and lines[cursor].strip() != "":
            chunk.append(lines[cursor])
            cursor += 1
        if len(answer) != 1 or answer not in "abcd" or len(chunk) != 6:
            raise ValueError(f"LogiQA block {len(rows)} is malformed")
        context, query = chunk[0].strip(), chunk[1].strip()
        options = []
        for expected, line in zip("ABCD", chunk[2:]):
            match = LOGIQA_LABEL.match(line.strip())
            if match and match.group(1).upper() == expected:
                text = match.group(2).strip()
            else:
                text = line.strip()
            options.append({"letter": expected, "text": normalize_option(text)})
        rows.append(
            record(
                item_id=f"logiqa-test-{len(rows):04d}",
                dataset="logiqa",
                subset=None,
                question=f"{context}\n{query}",
                options=options,
                answer=answer.upper(),
                meta={"context": context, "query": query},
            )
        )
    if len(rows) != 651:
        raise ValueError(f"LogiQA test has {len(rows)} items, expected 651")
    return rows


def exclude_reason(row: dict) -> str | None:
    """Why a parsed item cannot be used as text stem plus text options."""
    reasons = []
    if row["meta"].get("option_layout") == "table_row_letters":
        reasons.append("table_row_letters")
    visible = "\n".join([row["question"], *(opt["text"] for opt in row["options"])])
    if "![](" in visible or "mathpix.com" in visible:
        reasons.append("image")
    if row["meta"].get("missing_option_text"):
        reasons.append("missing_option_text")
    return "+".join(reasons) if reasons else None


def build_gaokao(raw_dir: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for subset, local_name, source_name, expected in GAOKAO_FILES:
        payload = json.loads((raw_dir / local_name).read_text(encoding="utf-8"))
        if payload.get("keywords") != source_name.replace(".json", ""):
            raise ValueError(f"{local_name} keywords {payload.get('keywords')}")
        items = payload["example"]
        if len(items) != expected:
            raise ValueError(f"{subset} has {len(items)} items, expected {expected}")
        rows = []
        for item in items:
            answers = item["answer"]
            if not isinstance(answers, list) or len(answers) != 1:
                raise ValueError(f"{subset} {item.get('index')} answer {answers}")
            stem, options, layout = split_embedded_options(item["question"])
            year = int(item["year"])
            index = int(item["index"])
            missing = [opt["letter"] for opt in options if not opt["text"]]
            for opt in options:
                if not opt["text"]:
                    opt["text"] = "[source has no text]"
            meta = {
                "year": year,
                "category": item.get("category", ""),
                "index": index,
                "score": item.get("score"),
                "analysis": (item.get("analysis") or "").strip(),
                "question_raw": item["question"],
                "source_file": source_name,
                "option_layout": layout,
            }
            if missing:
                meta["missing_option_text"] = missing
            if "mathpix.com" in item["question"] or "![](" in item["question"]:
                meta["has_image"] = True
            rows.append(
                record(
                    item_id=f"gaokao_bench-{subset}-{year}-{index}",
                    dataset="gaokao_bench",
                    subset=subset,
                    question=stem,
                    options=options,
                    answer=str(answers[0]).strip().upper(),
                    meta=meta,
                )
            )
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate ids in {subset}")
        grouped[subset] = rows
    return grouped


def build_mmlu(test_dir: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict]:
    csv_paths = sorted(test_dir.glob("*.csv"))
    found = {path.name[: -len("_test.csv")] for path in csv_paths}
    expected = set(MMLU_SUBJECTS)
    if found != expected:
        raise ValueError(
            f"MMLU subjects mismatch missing={sorted(expected - found)} extra={sorted(found - expected)}"
        )
    pools: dict[str, list[dict]] = {group: [] for group in MMLU_GROUPS}
    subject_counts: dict[str, dict[str, int]] = {group: {} for group in MMLU_GROUPS}
    for subject, group in MMLU_SUBJECTS.items():
        path = test_dir / f"{subject}_test.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            parsed = list(csv.reader(handle))
        subject_counts[group][subject] = len(parsed)
        for row_index, cols in enumerate(parsed):
            if len(cols) != 6:
                raise ValueError(f"{path.name} row {row_index} has {len(cols)} columns")
            question, *choices, answer = cols
            answer = answer.strip().upper()
            options = [
                {"letter": letter, "text": choice.strip()}
                for letter, choice in zip("ABCD", choices)
            ]
            pools[group].append(
                record(
                    item_id=f"mmlu-{subject}-{row_index:04d}",
                    dataset="mmlu",
                    subset=group,
                    question=question.strip(),
                    options=options,
                    answer=answer,
                    meta={"subject": subject, "row": row_index},
                )
            )
    total = sum(len(pool) for pool in pools.values())
    if total != 14042:
        raise ValueError(f"MMLU test has {total} rows, expected 14042")
    eval_sets: dict[str, list[dict]] = {}
    draw: dict[str, dict] = {}
    for group in MMLU_GROUPS:
        pool = pools[group]
        rng = random.Random(MMLU_SAMPLE_SEED)
        picked = rng.sample(range(len(pool)), MMLU_SAMPLE_SIZE)
        eval_sets[group] = [pool[index] for index in sorted(picked)]
        draw[group] = {
            "pool_size": len(pool),
            "seed": MMLU_SAMPLE_SEED,
            "sample_size": MMLU_SAMPLE_SIZE,
            "subjects": subject_counts[group],
            "sample_ids_draw_order": [pool[index]["id"] for index in picked],
            "eval_ids_file_order": [row["id"] for row in eval_sets[group]],
        }
    return pools, eval_sets, draw


def rel_file(out: Path, info: dict) -> dict:
    path = Path(info["path"])
    info = dict(info)
    info["path"] = path.relative_to(out.parent).as_posix()
    return info


def build_manifest(
    out: Path,
    files: list[dict],
    draw: dict,
    raw_dir: Path,
    excluded: list[dict],
) -> dict:
    gaokao_layouts: dict[str, int] = defaultdict(int)
    kept_counts: dict[str, int] = {}
    for subset, _, _, _ in GAOKAO_FILES:
        rows = [
            json.loads(line)
            for line in (out / "gaokao_bench" / f"{subset}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        kept_counts[subset] = len(rows)
        for row in rows:
            meta = row["meta"]
            if exclude_reason(row):
                raise ValueError(f"{row['id']} stayed in the eval file")
            gaokao_layouts[f"{subset}:{meta['option_layout']}"] += 1
    sources = {
        "aqua_rat": {
            "name": "AQUA-RAT",
            "split": "test",
            "count": 254,
            "url": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json",
            "hf": "deepmind/aqua_rat test",
            "note": "全部 254 道 test。meta.rationale 是官方解析，不能放进提示词。",
        },
        "logiqa": {
            "name": "LogiQA",
            "split": "test",
            "language": "en",
            "count": 651,
            "url": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt",
            "note": "英文 Test.txt，全部 651 道。题干是段落加问题。",
        },
        "mmlu": {
            "name": "MMLU",
            "split": "test",
            "url": "https://people.eecs.berkeley.edu/~hendrycks/data.tar",
            "category_map": (
                "EleutherAI lm-evaluation-harness "
                "lm_eval/tasks/mmlu/_generate_configs.py SUBJECTS"
            ),
            "category_map_url": (
                "https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/"
                "main/lm_eval/tasks/mmlu/_generate_configs.py"
            ),
            "official_test_count": 14042,
            "eval_per_group": MMLU_SAMPLE_SIZE,
            "sampling": (
                "每个大类单独用 random.Random(42).sample 从该大类官方 test 池抽 100 题。"
                "池内顺序是 SUBJECTS 的插入顺序，再按 CSV 行号。"
                "评测文件按池内位置排序。sample_ids_draw_order 才是抽样函数的返回顺序。"
                "Python 3.11 起 random.sample 的算法与更早版本不同，已提交的 jsonl 是锁定结果。"
            ),
            "python": "3.12.3",
            "prompting": "zero-shot。不要使用 MMLU 的 5-shot dev。评测只读 dataset/mmlu/eval，pools 是抽这 100 题的官方 test 全量。",
            "text": "官方 CSV 中少量 Windows-1252 标点以 U+0080–U+009F 原样保留。U+0085 在 jsonl 里写成 \\u0085，读回后仍是原字符。",
            "draw": draw,
        },
        "gaokao_bench": {
            "name": "GAOKAO-Bench",
            "paper": "Zhang et al. 2023, arXiv:2305.12474",
            "url_base": GAOKAO_BASE,
            "scope": "全国卷单选题。生物和化学在结果里合成一列「生化」，文件仍分开保存。",
            "subsets": {
                "sci_math": {
                    "file": "2010-2022_Math_I_MCQs.json",
                    "years": "2010-2022",
                    "official_count": 214,
                    "count": kept_counts["sci_math"],
                },
                "hum_math": {
                    "file": "2010-2022_Math_II_MCQs.json",
                    "years": "2010-2022",
                    "official_count": 218,
                    "count": kept_counts["hum_math"],
                },
                "english": {
                    "file": "2010-2013_English_MCQs.json",
                    "years": "2010-2013",
                    "official_count": 105,
                    "count": kept_counts["english"],
                    "note": "GAOKAO-Bench 的英语单选文件只收录 2010–2013。完形、阅读、短文改错不在这个文件里。",
                },
                "biology": {
                    "file": "2010-2022_Biology_MCQs.json",
                    "years": "2010-2022",
                    "official_count": 150,
                    "count": kept_counts["biology"],
                },
                "chemistry": {
                    "file": "2010-2022_Chemistry_MCQs.json",
                    "years": "2010-2022",
                    "official_count": 124,
                    "count": kept_counts["chemistry"],
                },
                "history": {
                    "file": "2010-2022_History_MCQs.json",
                    "years": "2010-2022",
                    "official_count": 287,
                    "count": kept_counts["history"],
                },
            },
            "option_layout_counts": dict(gaokao_layouts),
            "excluded": {
                "count": len(excluded),
                "file": "dataset/excluded/gaokao_bench.jsonl",
                "rule": (
                    "评测只保留题干和选项都是文字、并且选项不是单纯字母的题。"
                    "table_row_letters：卷面选项只是 A/B/C/D，表格留在题干里，推理步会看到选项。"
                    "image：题干或选项是 mathpix 图片，文字里没有对应内容。"
                    "missing_option_text：源数据里该选项没有文字。"
                ),
                "items": [
                    {"id": row["id"], "reason": row["meta"]["exclude_reason"]}
                    for row in excluded
                ],
            },
            "note": "meta.analysis 是官方解析，meta.question_raw 是未拆分的原题，都不能放进提示词。",
        },
    }
    return {
        "role": "工作一锁定的评测集。目录与 work1_decord、work2_option_mismatch 平级。",
        "built_on": "2026-10-16",
        "fields": {
            "id": "稳定题目编号",
            "dataset": "aqua_rat | logiqa | mmlu | gaokao_bench",
            "split": "test",
            "subset": "MMLU 大类或高考科目；AQUA 与 LogiQA 为 null",
            "question": "题干。DecoRD 的推理步只看这一字段。",
            "options": "[{letter, text}, ...]。决策步才看。",
            "answer": "大写字母",
            "meta": "来源字段。rationale、analysis、question_raw 不进入提示词。",
        },
        "eval_files": [
            item["path"]
            for item in files
            if item["path"].endswith(".jsonl")
            and "/pools/" not in item["path"]
            and "/excluded/" not in item["path"]
        ],
        "sources": sources,
        "raw_sha256": {
            "aqua_test.json": sha256_file(raw_dir / "aqua_test.json"),
            "logiqa_test.txt": sha256_file(raw_dir / "logiqa_test.txt"),
            **{
                source_name: sha256_file(raw_dir / local_name)
                for _, local_name, source_name, _ in GAOKAO_FILES
            },
        },
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("/tmp/dsraw"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.out
    file_infos = []

    aqua = build_aqua(args.raw / "aqua_test.json")
    file_infos.append(rel_file(out, write_jsonl(out / "aqua_rat" / "test.jsonl", aqua)))

    logiqa = build_logiqa(args.raw / "logiqa_test.txt")
    file_infos.append(rel_file(out, write_jsonl(out / "logiqa" / "test.jsonl", logiqa)))

    kept_expect = {
        "sci_math": 213,
        "hum_math": 218,
        "english": 105,
        "biology": 148,
        "chemistry": 103,
        "history": 287,
    }
    gaokao = build_gaokao(args.raw)
    excluded: list[dict] = []
    for subset, _, _, _ in GAOKAO_FILES:
        kept = []
        for row in gaokao[subset]:
            reason = exclude_reason(row)
            if reason is None:
                kept.append(row)
                continue
            dropped = dict(row)
            dropped["meta"] = dict(row["meta"])
            dropped["meta"]["exclude_reason"] = reason
            excluded.append(dropped)
        info = write_jsonl(out / "gaokao_bench" / f"{subset}.jsonl", kept)
        if info["count"] != kept_expect[subset]:
            raise ValueError(f"{subset} kept {info['count']}, expected {kept_expect[subset]}")
        file_infos.append(rel_file(out, info))
    if len(excluded) != 24:
        raise ValueError(f"excluded {len(excluded)}, expected 24")
    file_infos.append(
        rel_file(out, write_jsonl(out / "excluded" / "gaokao_bench.jsonl", excluded))
    )

    pools, eval_sets, draw = build_mmlu(args.raw / "mmlu" / "data" / "test")
    for group in MMLU_GROUPS:
        file_infos.append(
            rel_file(out, write_jsonl(out / "mmlu" / "pools" / f"{group}.jsonl", pools[group]))
        )
        info = write_jsonl(out / "mmlu" / "eval" / f"{group}.jsonl", eval_sets[group])
        if info["count"] != MMLU_SAMPLE_SIZE:
            raise ValueError(group)
        file_infos.append(rel_file(out, info))

    manifest = build_manifest(out, file_infos, draw, args.raw, excluded)
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")
    for info in file_infos:
        print(f"{info['count']:6d}  {info['path']}")


if __name__ == "__main__":
    main()
