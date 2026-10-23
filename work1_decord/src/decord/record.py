"""Append one prediction per finished item and resume after an interruption.

Each line is ``{"question", "answer", "pred"}``. ``answer`` is the gold letter
and ``pred`` is the letter returned by the method. When the runner captured the
model text, the same line also keeps ``response`` and, when known,
``finish_reason``. A multi-step method also keeps ``trace``, one entry per
call. A call that raises, or a result whose status is ``error``,
is left unwritten so the next run retries it. A trailing partial line from a
crash is dropped before new rows are appended. Extra evidence fields are kept.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from decord.client import OpenAIChatClient
from decord.data import Item, load_experiment_items
from decord.pipeline import RUNNERS, GenerateConfig

REQUIRED_KEYS = ("question", "answer", "pred")


def prediction_row(
    question: str,
    answer: str,
    pred: str,
    response: str | None = None,
    finish_reason: str | None = None,
    trace: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"question": question, "answer": answer, "pred": pred}
    if response is not None:
        row["response"] = response
    if finish_reason:
        row["finish_reason"] = finish_reason
    if trace:
        row["trace"] = trace
    return row


def _response_text(result: dict[str, Any]) -> str | None:
    raw = result.get("raw") or {}
    if not isinstance(raw, dict):
        return None
    for key in ("solution", "direct", "cot", "plan_solve", "revision", "decision", "reasoning"):
        value = raw.get(key)
        if isinstance(value, str):
            return value
    return None


def load_saved_rows(path: Path) -> list[dict[str, str]]:
    """Return stored rows in order. A corrupt final line is truncated."""

    if not path.is_file():
        return []
    data = path.read_bytes()
    if not data:
        return []
    parts = data.split(b"\n")
    rows: list[dict[str, str]] = []
    keep: list[bytes] = []
    for index, part in enumerate(parts):
        if part == b"":
            continue
        try:
            row = json.loads(part.decode("utf-8"))
        except json.JSONDecodeError:
            if index != len(parts) - 1:
                raise ValueError(f"{path}:{index + 1} is not JSON") from None
            break
        if not isinstance(row, dict) or any(key not in row for key in REQUIRED_KEYS):
            raise ValueError(f"{path}:{index + 1} is missing question, answer, or pred")
        kept = {key: str(row[key]) for key in REQUIRED_KEYS}
        for key, value in row.items():
            if key not in kept:
                kept[key] = value
        rows.append(kept)
        keep.append(json.dumps(kept, ensure_ascii=False).encode("utf-8"))
    rebuilt = b"\n".join(keep) + (b"\n" if keep else b"")
    if rebuilt != data:
        path.write_bytes(rebuilt)
    return rows


def append_prediction(
    path: Path,
    question: str,
    answer: str,
    pred: str,
    response: str | None = None,
    finish_reason: str | None = None,
    trace: list[dict[str, str]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        prediction_row(
            question,
            answer,
            pred,
            response=response,
            finish_reason=finish_reason,
            trace=trace,
        ),
        ensure_ascii=False,
    ) + "\n"
    with path.open("ab") as handle:
        handle.write(line.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def run_letter_log(
    *,
    output: Path,
    items: list[Item],
    method: str = "direct",
    client: Any = None,
    model: str = "",
    base_url: str | None = None,
    api_key: str | None = None,
    extra_body: dict | None = None,
    config: GenerateConfig | None = None,
    request_timeout_s: float = 180.0,
) -> dict[str, int]:
    """Run ``method`` and append one row after each item that does not error."""

    if method not in RUNNERS:
        raise ValueError(f"unknown method: {method}")
    settings = config or GenerateConfig()
    runner = RUNNERS[method]
    chat = client
    if chat is None:
        chat = OpenAIChatClient(
            model,
            base_url=base_url,
            api_key=api_key,
            timeout_s=request_timeout_s,
            extra_body=dict(extra_body or {}),
        )
    saved = load_saved_rows(output)
    if len(saved) > len(items):
        raise ValueError(f"{output} has {len(saved)} rows for {len(items)} items")
    for index, row in enumerate(saved):
        item = items[index]
        if row["question"] != item.question or row["answer"] != item.answer:
            raise ValueError(f"{output}:{index + 1} does not match {item.id}")
    written = 0
    errors = 0
    total = len(items)
    for index, item in enumerate(items[len(saved) :], start=len(saved) + 1):
        result = runner(
            chat,
            question=item.question,
            options=item.options,
            gold=item.answer,
            config=settings,
        )
        if result.get("status") == "error":
            errors += 1
            print(f"{index}/{total} error {item.id} {result.get('error')}", file=sys.stderr)
            break
        trace = result.get("trace")
        append_prediction(
            output,
            item.question,
            item.answer,
            str(result.get("pred") or ""),
            response=_response_text(result),
            finish_reason=str(result.get("finish_reason") or ""),
            trace=trace if isinstance(trace, list) else None,
        )
        written += 1
        print(f"{index}/{total} saved", file=sys.stderr)
    return {"written": written, "skipped": len(saved), "errors": errors}


def dataset_items(dataset: str) -> list[Item]:
    name = {"aqua": "aqua_rat", "aqua_rat": "aqua_rat", "mmlu": "mmlu"}.get(dataset)
    if name is None:
        raise ValueError(f"unknown dataset: {dataset}")
    return [item for item in load_experiment_items() if item.dataset == name]


def main(argv: list[str] | None = None) -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Save one prediction per finished item")
    parser.add_argument("command", nargs="?", default="record")
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", default="direct")
    parser.add_argument("--dataset", default="aqua_rat", choices=("aqua", "aqua_rat", "mmlu"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    items = dataset_items(args.dataset)
    if args.limit > 0:
        items = items[: args.limit]
    extra: dict[str, Any] = {}
    if "qwen" in args.model.lower():
        extra = {"enable_thinking": False}
    counts = run_letter_log(
        output=Path(args.output),
        items=items,
        method=args.method,
        model=args.model,
        base_url=os.environ.get("BASE_URL"),
        api_key=os.environ.get("API_KEY"),
        extra_body=extra,
    )
    print(json.dumps(counts, ensure_ascii=False))
