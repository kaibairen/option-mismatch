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


def _last_row_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        indexed[str(row["id"])] = row
    return indexed


def _slice_summary(
    indexed: Mapping[str, Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    recorded = [indexed[str(item["id"])] for item in items if str(item["id"]) in indexed]
    return summarize_predictions(recorded, n_full=len(items))


def score_experiment(
    rows: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score one model and one method on the locked item list.

    AQUA is one slice of 254. Each MMLU subject is scored on its own locked
    items, with missing rows kept in ``accuracy_full``. A category score is
    the unweighted mean of its subject accuracies. The MMLU overall score is
    the unweighted mean of every subject, not the mean of the four category
    scores and not a micro average over items. Category DCDI is filled only
    when every subject in that category has a reasoning accuracy.
    """

    indexed = _last_row_by_id(rows)
    aqua_items = [item for item in items if item.get("dataset") == "aqua_rat"]
    mmlu_items = [item for item in items if item.get("dataset") == "mmlu"]
    subjects: list[str] = []
    for item in mmlu_items:
        subject = str(item.get("subject") or "")
        if subject and subject not in subjects:
            subjects.append(subject)
    per_subject: dict[str, Any] = {}
    subject_group: dict[str, str] = {}
    for subject in subjects:
        owned = [item for item in mmlu_items if item.get("subject") == subject]
        per_subject[subject] = _slice_summary(indexed, owned)
        subject_group[subject] = str(owned[0].get("subset") or "")
    groups: dict[str, list[str]] = {}
    for subject, group in subject_group.items():
        groups.setdefault(group, []).append(subject)
    categories: dict[str, Any] = {}
    for group, members in groups.items():
        decision = [per_subject[subject]["accuracy_full"] for subject in members]
        valid = [per_subject[subject]["accuracy_valid"] for subject in members]
        reasoning_values = [
            per_subject[subject]["accuracy_reasoning"]
            for subject in members
            if per_subject[subject]["accuracy_reasoning"] is not None
        ]
        reasoning_rate = (
            sum(reasoning_values) / len(reasoning_values)
            if reasoning_values and len(reasoning_values) == len(members)
            else None
        )
        decision_rate = sum(decision) / len(decision) if decision else 0.0
        categories[group] = {
            "n_subjects": len(members),
            "n_full": sum(per_subject[subject]["n_full"] for subject in members),
            "accuracy_full": decision_rate,
            "accuracy_valid": (sum(valid) / len(valid)) if valid else 0.0,
            "accuracy_reasoning": reasoning_rate,
            "dcdi": None if reasoning_rate is None else dcdi(reasoning_rate, decision_rate),
        }
    subject_rates = [per_subject[subject]["accuracy_full"] for subject in subjects]
    return {
        "aqua": _slice_summary(indexed, aqua_items),
        "mmlu": {
            "n_subjects": len(subjects),
            "accuracy_full": (sum(subject_rates) / len(subject_rates)) if subject_rates else 0.0,
            "categories": categories,
            "subjects": per_subject,
        },
    }
