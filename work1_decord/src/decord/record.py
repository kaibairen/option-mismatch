"""Append one prediction per finished item and resume after an interruption.

Each line is ``{"question", "answer", "pred"}``. ``answer`` is the gold letter
and ``pred`` is the letter returned by the method. A call that raises, or a
result whose status is ``error``, is left unwritten so the next run retries it.
A trailing partial line from a crash is dropped before new rows are appended.
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


def prediction_row(question: str, answer: str, pred: str) -> dict[str, str]:
    return {"question": question, "answer": answer, "pred": pred}


def load_saved_questions(path: Path) -> set[str]:
    """Return questions already stored. A corrupt final line is truncated."""

    if not path.is_file():
        return set()
    data = path.read_bytes()
    if not data:
        return set()
    parts = data.split(b"\n")
    done: set[str] = set()
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
        done.add(str(row["question"]))
        keep.append(part)
    rebuilt = b"\n".join(keep) + (b"\n" if keep else b"")
    if rebuilt != data:
        path.write_bytes(rebuilt)
    return done


def append_prediction(path: Path, question: str, answer: str, pred: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(prediction_row(question, answer, pred), ensure_ascii=False) + "\n"
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
    done = load_saved_questions(output)
    written = 0
    skipped = 0
    errors = 0
    total = len(items)
    for index, item in enumerate(items, start=1):
        if item.question in done:
            skipped += 1
            continue
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
            continue
        append_prediction(output, item.question, item.answer, str(result.get("pred") or ""))
        done.add(item.question)
        written += 1
        print(f"{index}/{total} saved", file=sys.stderr)
    return {"written": written, "skipped": skipped, "errors": errors}


def aqua_items() -> list[Item]:
    return [item for item in load_experiment_items() if item.dataset == "aqua_rat"]


def main(argv: list[str] | None = None) -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Save one AQUA prediction per finished item")
    parser.add_argument("command", nargs="?", default="record")
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", default="direct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    items = aqua_items()
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
