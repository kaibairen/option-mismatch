"""Print archived results, or score a JSONL log produced by this package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decord.archive import available_reports, load_report
from decord.metrics import summarize_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="DecoRD archived tables and log scoring")
    parser.add_argument("--report", default="", help="JSON file name under reports/")
    parser.add_argument("--log", default="", help="JSONL of per-item records from this package")
    parser.add_argument("--n-full", type=int, default=0)
    args = parser.parse_args()
    if args.log:
        rows = [json.loads(line) for line in Path(args.log).read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = summarize_predictions(rows, n_full=args.n_full or None)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    names = [args.report] if args.report else available_reports()
    for name in names:
        payload = load_report(name)
        print(f"# {name}")
        print(payload.get("provenance", ""))
        print()


if __name__ == "__main__":
    main()
