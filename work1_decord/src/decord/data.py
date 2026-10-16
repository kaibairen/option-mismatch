"""Load the locked Work 1 question list.

The list is ``experiment_files`` in ``dataset/manifest.json``: the AQUA-RAT
test split and the MMLU subject-40 draw. ``meta`` stays out of the objects
that the runners see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Item:
    id: str
    dataset: str
    subset: str | None
    subject: str | None
    question: str
    options: list[str]
    answer: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "subset": self.subset,
            "subject": self.subject,
            "question": self.question,
            "options": list(self.options),
            "answer": self.answer,
        }


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "dataset" / "manifest.json").is_file():
            return parent
    raise FileNotFoundError("dataset/manifest.json not found above the package")


def format_option_lines(options: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for option in options:
        letter = str(option["letter"]).strip().upper()
        text = str(option["text"]).strip()
        if letter not in "ABCDE" or not text:
            raise ValueError(f"option is missing a letter or text: {option!r}")
        lines.append(f"{letter}) {text}")
    if len(lines) < 2:
        raise ValueError("an item needs at least two options")
    return lines


def _item_from_row(row: dict[str, Any], source: str) -> Item:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    answer = str(row.get("answer") or "").strip().upper()[:1]
    if answer not in "ABCDE":
        raise ValueError(f"{source}: {row.get('id')} has no A–E answer")
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError(f"{source}: {row.get('id')} has an empty stem")
    subject = meta.get("subject")
    return Item(
        id=str(row["id"]),
        dataset=str(row.get("dataset") or ""),
        subset=row.get("subset"),
        subject=str(subject) if subject else None,
        question=question,
        options=format_option_lines(list(row.get("options") or [])),
        answer=answer,
    )


def load_experiment_items(root: Path | None = None) -> list[Item]:
    base = root or repo_root()
    manifest = json.loads((base / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    files = list(manifest["experiment_files"])
    items: list[Item] = []
    seen: set[str] = set()
    for relative in files:
        path = base / relative
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            item = _item_from_row(row, f"{relative}:{line_number}")
            if item.id in seen:
                raise ValueError(f"duplicate item id {item.id}")
            seen.add(item.id)
            items.append(item)
    return items
