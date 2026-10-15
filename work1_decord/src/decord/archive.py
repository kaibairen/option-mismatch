"""Load the three result sets already on disk. None of them is recomputed here."""

from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def load_report(name: str) -> dict:
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8"))


def available_reports() -> list[str]:
    return sorted(path.name for path in REPORTS.glob("*.json"))
