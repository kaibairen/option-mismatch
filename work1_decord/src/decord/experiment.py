"""Run and score the locked Work 1 grid.

Four models, five zero-shot systems, AQUA-RAT test plus MMLU subject-40.
DecoRD is one reasoning call and one decision call. Confidence reflection
stays implemented and is not part of this grid. A real run appends JSONL
under ``work1_decord/runs/`` and skips item ids already present.
``--dry-run`` only counts calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from decord.client import OpenAIChatClient
from decord.data import Item, load_experiment_items
from decord.metrics import score_experiment
from decord.pipeline import RUNNERS, GenerateConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "experiment.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG
    return json.loads(config_path.read_text(encoding="utf-8"))


def calls_per_item(sample_k: int, *, reflect_every_item: bool) -> dict[str, int]:
    confidence = 1 + sample_k + (1 if reflect_every_item else 0)
    return {
        "direct": 1,
        "cot": 1,
        "plan_and_solve": 1,
        "reflexion": 2,
        "decord_standard": 2,
        "decord_confidence_reflection": confidence,
    }


def dry_run_report(config: dict[str, Any] | None = None, items: list[Item] | None = None) -> dict[str, Any]:
    settings = config or load_config()
    question_list = items if items is not None else load_experiment_items()
    sample_k = int(settings["sample_k"])
    by_dataset: dict[str, int] = {}
    subjects: dict[str, int] = {}
    for item in question_list:
        by_dataset[item.dataset] = by_dataset.get(item.dataset, 0) + 1
        if item.subject:
            subjects[item.subject] = subjects.get(item.subject, 0) + 1
    catalog = calls_per_item(sample_k, reflect_every_item=False)
    selected = [method for method in settings["methods"]]
    per_item = {method: catalog[method] for method in selected}
    n_items = len(question_list)
    return {
        "models": [model["id"] for model in settings["models"]],
        "methods": selected,
        "n_items": n_items,
        "n_by_dataset": by_dataset,
        "n_subjects": len(subjects),
        "subject_sizes": subjects,
        "calls_per_item": per_item,
        "calls_per_model": n_items * sum(per_item.values()),
    }


def _log_path(runs: Path, model: str, method: str) -> Path:
    return runs / model / f"{method}.jsonl"


def _completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not JSON") from exc
        done.add(str(row["id"]))
    return done


def _public_row(model: str, item: Item, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.id,
        "dataset": item.dataset,
        "subset": item.subset,
        "subject": item.subject,
        "model": model,
        "method": result["method"],
        "gold": item.answer,
        "status": result["status"],
        "pred": result["pred"],
        "correct": result["correct"],
        "reasoning_correct": result.get("reasoning_correct"),
        "confidence": result.get("confidence"),
        "reflected": result.get("reflected"),
        "error": result.get("error") or "",
        "n_calls": result.get("n_calls"),
        "raw": result.get("raw") or {},
    }


def run_grid(
    *,
    config: dict[str, Any] | None = None,
    items: list[Item] | None = None,
    runs: Path | None = None,
    models: list[str] | None = None,
    methods: list[str] | None = None,
    limit: int = 0,
    api_key: str | None = None,
    base_url: str | None = None,
    client_factory: Any = None,
) -> dict[str, int]:
    """Append one JSON object per newly run item. Existing ids are skipped."""

    settings = config or load_config()
    question_list = list(items if items is not None else load_experiment_items())
    if limit > 0:
        question_list = question_list[:limit]
    selected_models = models or [model["id"] for model in settings["models"]]
    selected_methods = methods or list(settings["methods"])
    unknown = [method for method in selected_methods if method not in RUNNERS]
    if unknown:
        raise ValueError(f"unknown method: {', '.join(unknown)}")
    model_body = {model["id"]: dict(model.get("extra_body") or {}) for model in settings["models"]}
    for model in selected_models:
        if model not in model_body:
            raise ValueError(f"model {model} is not in the experiment config")
    generate = GenerateConfig(
        temperature=float(settings["temperature"]),
        top_p=float(settings["top_p"]),
        max_tokens=int(settings["max_tokens"]),
        sample_k=int(settings["sample_k"]),
        confidence_threshold=float(settings["confidence_threshold"]),
    )
    run_dir = runs or (PACKAGE_ROOT / "runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol.json").write_text(
        json.dumps(
            {
                "models": selected_models,
                "methods": selected_methods,
                "temperature": generate.temperature,
                "top_p": generate.top_p,
                "max_tokens": generate.max_tokens,
                "sample_k": generate.sample_k,
                "confidence_threshold": generate.confidence_threshold,
                "request_timeout_s": settings["request_timeout_s"],
                "n_items": len(question_list),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written = 0
    skipped = 0
    for model in selected_models:
        if client_factory is not None:
            client = client_factory(model, model_body[model])
        else:
            client = OpenAIChatClient(
                model,
                base_url=base_url,
                api_key=api_key,
                timeout_s=float(settings["request_timeout_s"]),
                extra_body=model_body[model],
            )
        for method in selected_methods:
            path = _log_path(run_dir, model, method)
            path.parent.mkdir(parents=True, exist_ok=True)
            done = _completed_ids(path)
            runner = RUNNERS[method]
            with path.open("a", encoding="utf-8") as handle:
                for index, item in enumerate(question_list, start=1):
                    if item.id in done:
                        skipped += 1
                        continue
                    result = runner(
                        client,
                        question=item.question,
                        options=item.options,
                        gold=item.answer,
                        config=generate,
                    )
                    handle.write(json.dumps(_public_row(model, item, result), ensure_ascii=False) + "\n")
                    handle.flush()
                    written += 1
                    if written % 25 == 0:
                        print(f"{model} {method} {index}/{len(question_list)}", file=sys.stderr)
    return {"written": written, "skipped": skipped}


def score_runs(
    runs: Path,
    *,
    items: list[Item] | None = None,
    model: str = "",
    method: str = "",
) -> dict[str, Any]:
    question_list = items if items is not None else load_experiment_items()
    catalog = [item.as_dict() for item in question_list]
    tables: dict[str, Any] = {}
    paths = sorted(runs.glob("*/*.jsonl"))
    for path in paths:
        model_id = path.parent.name
        method_id = path.stem
        if model and model_id != model:
            continue
        if method and method_id != method:
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        tables.setdefault(model_id, {})[method_id] = score_experiment(rows, catalog)
    return tables


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run or score the Work 1 grid")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--model", action="append", default=[])
    run_parser.add_argument("--method", action="append", default=[])
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--runs", default=str(PACKAGE_ROOT / "runs"))
    score_parser.add_argument("--model", default="")
    score_parser.add_argument("--method", default="")
    args = parser.parse_args(argv)
    if args.command == "run":
        config = load_config(Path(args.config))
        if args.dry_run:
            print(json.dumps(dry_run_report(config), ensure_ascii=False, indent=2))
            return
        counts = run_grid(
            config=config,
            models=args.model or None,
            methods=args.method or None,
            limit=args.limit,
        )
        print(json.dumps(counts, ensure_ascii=False))
        return
    root = repo_root()
    del root
    tables = score_runs(Path(args.runs), model=args.model, method=args.method)
    print(json.dumps(tables, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
