"""Zero-shot systems on one multiple-choice item.

Coupled baselines see the stem and the options in every call: direct, CoT,
Plan-and-Solve, and one-revision Reflexion. DecoRD reasoning sees the stem
only; the decision and reflection calls see the options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from decord.metrics import majority_letter, normalized_confidence, reasoning_matches
from decord.parse import cot_answer_letter, extract_letter, parse_json_object
from decord.prompts import (
    cot_prompt,
    decision_prompt,
    direct_prompt,
    plan_prompt,
    reasoning_prompt,
    reflection_prompt,
    reflexion_prompt,
    solve_prompt,
)

COT_MAX_TOKENS = 2048

LETTERS = "ABCDE"

METHODS = (
    "direct",
    "cot",
    "plan_and_solve",
    "reflexion",
    "decord_standard",
)


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


def _with_max_tokens(config: GenerateConfig, max_tokens: int) -> GenerateConfig:
    return GenerateConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=max_tokens,
        sample_k=config.sample_k,
        confidence_threshold=config.confidence_threshold,
    )


def _complete_meta(client: ChatClient, prompt: str, config: GenerateConfig) -> tuple[str, str, str]:
    details = getattr(client, "complete_details", None)
    try:
        if details is not None:
            payload = details(prompt, **config.as_kwargs())
            return str(payload.get("content") or ""), "", str(payload.get("finish_reason") or "")
        text = client.complete(prompt, **config.as_kwargs())
    except Exception as exc:  # noqa: BLE001 — recorded per item, not hidden
        return "", f"{type(exc).__name__}: {exc}", ""
    return text or "", "", ""


def _complete(client: ChatClient, prompt: str, config: GenerateConfig) -> tuple[str, str]:
    text, error, _finish = _complete_meta(client, prompt, config)
    return text, error


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
    """One zero-shot CoT completion. No follow-up call fills in a missing letter.

    The letter is the last ``answer is X`` or ``Final answer: X`` in that completion.
    A trace that stops before such a sentence, including one cut off by the token
    limit, is left blank.
    """

    settings = config or GenerateConfig()
    prompt = cot_prompt(question, options)
    text, error = _complete(client, prompt, _with_max_tokens(settings, COT_MAX_TOKENS))
    letter = cot_answer_letter(text) if text else ""
    gold_letter = (gold or "").strip().upper()[:1]
    return {
        "method": "cot",
        "status": _status_for(text, error, letter),
        "pred": letter,
        "correct": bool(letter) and letter == gold_letter,
        "reasoning_correct": None,
        "error": error,
        "n_calls": 1,
        "raw": {"cot": text},
    }


def run_direct(
    client: ChatClient,
    *,
    question: str,
    options: list[str],
    gold: str,
    config: GenerateConfig | None = None,
) -> dict[str, Any]:
    settings = config or GenerateConfig()
    prompt = direct_prompt(question, options)
    text, error = _complete(client, prompt, settings)
    letter = extract_letter(text) if text else ""
    gold_letter = (gold or "").strip().upper()[:1]
    return {
        "method": "direct",
        "status": _status_for(text, error, letter),
        "pred": letter,
        "correct": bool(letter) and letter == gold_letter,
        "reasoning_correct": None,
        "error": error,
        "n_calls": 1,
        "raw": {"direct": text},
    }


def run_plan_and_solve(
    client: ChatClient,
    *,
    question: str,
    options: list[str],
    gold: str,
    config: GenerateConfig | None = None,
) -> dict[str, Any]:
    """Devise a plan, then carry that plan out. Both calls see the options."""

    settings = config or GenerateConfig()
    plan_text, plan_error = _complete(client, plan_prompt(question, options), settings)
    if plan_error or not plan_text.strip():
        return {
            "method": "plan_and_solve",
            "status": _status_for(plan_text, plan_error, ""),
            "pred": "",
            "correct": False,
            "reasoning_correct": None,
            "error": plan_error,
            "n_calls": 1,
            "raw": {"plan": plan_text},
        }
    payload = parse_json_object(plan_text) or {}
    plan = str(payload.get("plan") or plan_text)
    solve_text, solve_error = _complete(client, solve_prompt(question, options, plan), settings)
    letter = extract_letter(solve_text) if solve_text else ""
    gold_letter = (gold or "").strip().upper()[:1]
    return {
        "method": "plan_and_solve",
        "status": _status_for(solve_text, solve_error, letter),
        "pred": letter,
        "correct": bool(letter) and letter == gold_letter,
        "reasoning_correct": None,
        "error": solve_error,
        "n_calls": 2,
        "raw": {"plan": plan_text, "solve": solve_text},
    }


def run_reflexion(
    client: ChatClient,
    *,
    question: str,
    options: list[str],
    gold: str,
    config: GenerateConfig | None = None,
) -> dict[str, Any]:
    """One coupled CoT attempt, then one critique. No further retry."""

    settings = config or GenerateConfig()
    first_text, first_error = _complete(client, cot_prompt(question, options), settings)
    if first_error or not first_text.strip():
        return {
            "method": "reflexion",
            "status": _status_for(first_text, first_error, ""),
            "pred": "",
            "correct": False,
            "reasoning_correct": None,
            "error": first_error,
            "n_calls": 1,
            "raw": {"cot": first_text},
        }
    revised_text, revised_error = _complete(
        client,
        reflexion_prompt(question, options, first_text),
        settings,
    )
    letter = extract_letter(revised_text) if revised_text else ""
    gold_letter = (gold or "").strip().upper()[:1]
    return {
        "method": "reflexion",
        "status": _status_for(revised_text, revised_error, letter),
        "pred": letter,
        "correct": bool(letter) and letter == gold_letter,
        "reasoning_correct": None,
        "error": revised_error,
        "n_calls": 2,
        "raw": {"cot": first_text, "revision": revised_text},
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
            "n_calls": 1,
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
        "n_calls": 2,
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
            "n_calls": 1,
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
        "n_calls": 1 + len(sample_text) + (1 if reflected else 0),
        "raw": {"reasoning": reason_text, "samples": sample_text, "reflection": final_text},
    }


RUNNERS = {
    "direct": run_direct,
    "cot": run_cot,
    "plan_and_solve": run_plan_and_solve,
    "reflexion": run_reflexion,
    "decord_standard": run_standard,
    "decord_confidence_reflection": run_confidence_reflection,
}
