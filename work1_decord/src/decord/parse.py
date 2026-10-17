"""Pull a choice letter or a JSON object out of a model response."""

from __future__ import annotations

import json
import re
from typing import Any

LETTER = re.compile(r"\b([A-E])\b", re.IGNORECASE)
FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
BRACED = re.compile(r"\{.*\}", re.DOTALL)


COT_ANSWER = re.compile(
    r"(?:\*+)?(?:therefore,\s*among\s+[A-E]\s+through\s+[A-E],\s*)?"
    r"(?:the\s+)?(?:final\s+)?\banswer\b(?:\*+)?\s*(?:is\b\s*)?[:：]?\s*(?:\*+)?\s*[\(\[]?\s*([A-E])\b"
    r"|(?:the\s+)?correct\s+(?:option|choice)\s+is\s*(?:\*+)?\s*[\(\[]?\s*([A-E])\b"
    r"|(?:corresponds\s+to|matches)\s+option\s*(?:\*+)?\s*[\(\[]?\s*([A-E])\b",
    re.IGNORECASE,
)


def cot_answer_letter(text: str) -> str:
    """Last stated choice. A letter used only as a variable or an option label does not count."""

    found: list[str] = []
    for match in COT_ANSWER.finditer(text or ""):
        letter = next((group for group in match.groups() if group), "")
        if letter:
            found.append(letter.upper())
    return found[-1] if found else ""


def extract_letter(text: str) -> str:
    """Prefer an explicit answer field, then the last standalone A–E."""

    raw = text or ""
    payload = parse_json_object(raw)
    if isinstance(payload, dict):
        for key in ("answer", "choice", "option", "final_answer", "pred"):
            value = payload.get(key)
            if isinstance(value, str):
                match = LETTER.search(value.upper())
                if match:
                    return match.group(1).upper()
    for pattern in (
        r"<<\s*([A-E])\s*>>",
        r"Final answer:\s*([A-E])\b",
        r"(?:答案是|answer\s+is|所以选)\s*[:：]?\s*([A-E])\b",
    ):
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    letters = LETTER.findall(raw.upper())
    return letters[-1].upper() if letters else ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    fenced = FENCED.search(raw)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    braced = BRACED.search(raw)
    if braced:
        candidates.append(braced.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None
