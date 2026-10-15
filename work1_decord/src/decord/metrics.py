"""Accuracy on a fixed item set, entropy confidence, and DCDI."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")
_LETTER = re.compile(r"\b([A-E])\b")


def _entropy(probabilities: Sequence[float]) -> float:
    total = 0.0
    for probability in probabilities:
        if probability > 0.0:
            total -= probability * math.log(probability)
    return total


def normalized_confidence(counts: Mapping[str, int], letters: Sequence[str]) -> float | None:
    """Confidence in ``[0, 1]`` from the paper's normalized entropy.

    ``1 - H(p) / log(M)``. One option taking every sample scores 1. A uniform
    distribution scores 0. ``M`` is the number of candidate letters, including
    letters that received zero samples. Returns ``None`` when there are no
    samples or fewer than two letters.
    """

    labels = [str(letter).upper() for letter in letters]
    if len(labels) < 2:
        return None
    total = sum(int(counts.get(letter, 0)) for letter in labels)
    if total <= 0:
        return None
    probabilities = [int(counts.get(letter, 0)) / total for letter in labels]
    maximum = math.log(len(labels))
    if maximum == 0.0:
        return None
    score = 1.0 - _entropy(probabilities) / maximum
    return max(0.0, min(1.0, score))


def dcdi(accuracy_reasoning: float, accuracy_decision: float) -> float | None:
    """Fractional drop from reasoning accuracy to decision accuracy.

    The manuscript prints this quantity as a percent:

    ``(Acc_reasoning - Acc_decision) / Acc_reasoning``.
    """

    if accuracy_reasoning <= 0.0:
        return None
    return (accuracy_reasoning - accuracy_decision) / accuracy_reasoning


def reasoning_matches(conclusion: str, options: Sequence[str], gold: str) -> bool | None:
    """Whether a stem-only conclusion names the gold option.

    Numeric options match by value. If the conclusion has no number, a trailing
    option letter is used. ``None`` means the conclusion is empty or the gold
    letter is not among the options.
    """

    text = (conclusion or "").strip()
    gold_letter = (gold or "").strip().upper()[:1]
    if not text or gold_letter not in "ABCDE":
        return None
    gold_option = ""
    for option in options:
        body = str(option).strip()
        if body.upper().startswith(gold_letter):
            gold_option = body
            break
    if not gold_option:
        return None
    targets = [float(raw) for raw in _NUMBER.findall(gold_option.replace(",", ""))]
    found = [float(raw) for raw in _NUMBER.findall(text.replace(",", ""))]
    if targets and found:
        return any(abs(left - right) <= max(1e-6, 1e-3 * max(abs(left), abs(right), 1.0)) for left in found for right in targets)
    letters = _LETTER.findall(text.upper())
    if letters:
        return letters[-1] == gold_letter
    return False


def majority_letter(letters: Sequence[str]) -> str:
    counts: Counter[str] = Counter(letter for letter in letters if letter)
    if not counts:
        return ""
    return sorted(counts, key=lambda letter: (-counts[letter], letter))[0]


def summarize_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_full: int | None = None,
) -> dict[str, Any]:
    """Score one method on one fixed question list.

    ``status`` is ``ok`` when a letter was parsed, ``empty`` when the call
    returned no text, ``no_letter`` when text contained no A–E choice, and
    ``error`` for a transport or API failure. Missing rows, when ``n_full``
    is larger than ``len(rows)``, count as ``error``.

    Full-set accuracy treats every non-correct row as wrong. Valid accuracy
    divides only by rows whose status is ``ok``.
    """

    recorded = list(rows)
    expected = len(recorded) if n_full is None else int(n_full)
    if expected < len(recorded):
        raise ValueError("n_full is smaller than the number of recorded rows")
    correct = 0
    valid = 0
    failures = {"empty": 0, "no_letter": 0, "error": 0}
    reasoning_correct = 0
    reasoning_scored = 0
    for row in recorded:
        status = str(row.get("status") or "")
        if status == "ok":
            valid += 1
            if bool(row.get("correct")):
                correct += 1
        elif status in failures:
            failures[status] += 1
        else:
            failures["error"] += 1
        if "reasoning_correct" in row and row.get("reasoning_correct") is not None:
            reasoning_scored += 1
            if bool(row.get("reasoning_correct")):
                reasoning_correct += 1
    missing = expected - len(recorded)
    failures["error"] += missing
    decision_rate = correct / expected if expected else 0.0
    reasoning_rate = (reasoning_correct / reasoning_scored) if reasoning_scored else None
    return {
        "n_full": expected,
        "n_recorded": len(recorded),
        "n_valid": valid,
        "n_correct": correct,
        "n_fail": expected - valid,
        "failures": failures,
        "accuracy_full": decision_rate,
        "accuracy_valid": (correct / valid) if valid else 0.0,
        "accuracy_reasoning": reasoning_rate,
        "dcdi": None if reasoning_rate is None else dcdi(reasoning_rate, decision_rate),
    }
