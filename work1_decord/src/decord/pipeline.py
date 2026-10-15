"""Standard DecoRD, confidence reflection, and the coupled CoT baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from decord.metrics import majority_letter, normalized_confidence, reasoning_matches
from decord.parse import extract_letter, parse_json_object
from decord.prompts import cot_prompt, decision_prompt, reasoning_prompt, reflection_prompt

LETTERS = "ABCDE"


class ChatClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str: ...


@dataclass(frozen=True)
class GenerateConfig:
    temperature: float = 0.4
    top_p: float = 1.0
    max_tokens: int = 512
    sample_k: int = 5
    confidence_threshold: float = 0.5

    def as_kwargs(self) -> dict[str, float | int]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }


def option_letters(options: list[str]) -> list[str]:
    found: list[str] = []
    for option in options:
        text = str(option).strip().upper()
        if text and text[0] in LETTERS and text[0] not in found:
            found.append(text[0])
    return found or list(LETTERS)


def _complete(client: ChatClient, prompt: str, config: GenerateConfig) -> tuple[str, str]:
    try:
        text = client.complete(prompt, **config.as_kwargs())
    except Exception as exc:  # noqa: BLE001 — recorded per item, not hidden
        return "", f"{type(exc).__name__}: {exc}"
    return text or "", ""


def _status_for(text: str, error: str, letter: str) -> str:
    if error:
        return "error"
    if not text.strip():
        return "empty"
    if not letter:
        return "no_letter"
    return "ok"


def run_cot(
    client: ChatClient,
    *,
    question: str,
    options: list[str],
    gold: str,
    config: GenerateConfig | None = None,
) -> dict[str, Any]:
    settings = config or GenerateConfig()
    prompt = cot_prompt(question, options)
    text, error = _complete(client, prompt, settings)
    letter = extract_letter(text) if text else ""
    gold_letter = (gold or "").strip().upper()[:1]
    return {
        "method": "cot",
        "status": _status_for(text, error, letter),
        "pred": letter,
        "correct": bool(letter) and letter == gold_letter,
        "reasoning_correct": None,
        "error": error,
        "raw": {"cot": text},
    }


def _reason(client: ChatClient, question: str, config: GenerateConfig) -> tuple[str, str, str, str]:
    prompt = reasoning_prompt(question)
    text, error = _complete(client, prompt, config)
    payload = parse_json_object(text) or {}
    reasoning = str(payload.get("reasoning") or text)
    conclusion = str(payload.get("conclusion") or "")
    return text, reasoning, conclusion, error


def run_standard(
    client: ChatClient,
    *,
    question: str,
    options: list[str],
    gold: str,
    config: GenerateConfig | None = None,
) -> dict[str, Any]:
    """Reason on the stem, then map the conclusion onto the options once."""

    settings = config or GenerateConfig()
    reason_text, reasoning, conclusion, reason_error = _reason(client, question, settings)
    if reason_error or not reason_text.strip():
        return {
            "method": "decord_standard",
            "status": _status_for(reason_text, reason_error, ""),
            "pred": "",
            "correct": False,
            "reasoning_correct": None,
            "error": reason_error,
            "raw": {"reasoning": reason_text},
        }
    prompt = decision_prompt(question, reasoning, conclusion, options)
    decision_text, decision_error = _complete(client, prompt, settings)
    letter = extract_letter(decision_text) if decision_text else ""
    gold_letter = (gold or "").strip().upper()[:1]
    return {
        "method": "decord_standard",
        "status": _status_for(decision_text, decision_error, letter),
        "pred": letter,
        "correct": bool(letter) and letter == gold_letter,
        "reasoning_correct": reasoning_matches(conclusion or reasoning, options, gold),
        "conclusion": conclusion,
        "error": decision_error,
        "raw": {"reasoning": reason_text, "decision": decision_text},
    }


def run_confidence_reflection(
    client: ChatClient,
    *,
    question: str,
    options: list[str],
    gold: str,
    config: GenerateConfig | None = None,
) -> dict[str, Any]:
    """Sample the decision step and reflect when confidence is below 0.5."""

    settings = config or GenerateConfig()
    reason_text, reasoning, conclusion, reason_error = _reason(client, question, settings)
    if reason_error or not reason_text.strip():
        return {
            "method": "decord_confidence_reflection",
            "status": _status_for(reason_text, reason_error, ""),
            "pred": "",
            "correct": False,
            "reasoning_correct": None,
            "error": reason_error,
            "raw": {"reasoning": reason_text},
        }
    labels = option_letters(options)
    samples: list[str] = []
    sample_text: list[str] = []
    errors: list[str] = []
    prompt = decision_prompt(question, reasoning, conclusion, options)
    for _ in range(settings.sample_k):
        text, error = _complete(client, prompt, settings)
        sample_text.append(text)
        if error:
            errors.append(error)
            continue
        letter = extract_letter(text)
        if letter:
            samples.append(letter)
    counts = {letter: samples.count(letter) for letter in labels}
    confidence = normalized_confidence(counts, labels)
    reflected = False
    final_text = ""
    final_error = ""
    if confidence is None:
        letter = ""
        final_error = errors[0] if errors else "no decision sample produced a letter"
    elif confidence < settings.confidence_threshold:
        reflected = True
        final_text, final_error = _complete(
            client,
            reflection_prompt(question, reasoning, conclusion, options, counts, confidence),
            settings,
        )
        letter = extract_letter(final_text) if final_text else ""
    else:
        letter = majority_letter(samples)
    gold_letter = (gold or "").strip().upper()[:1]
    status = _status_for(letter or final_text, final_error, letter)
    return {
        "method": "decord_confidence_reflection",
        "status": status,
        "pred": letter,
        "correct": bool(letter) and letter == gold_letter,
        "reasoning_correct": reasoning_matches(conclusion or reasoning, options, gold),
        "conclusion": conclusion,
        "confidence": confidence,
        "reflected": reflected,
        "sample_counts": counts,
        "error": final_error,
        "raw": {"reasoning": reason_text, "samples": sample_text, "reflection": final_text},
    }
