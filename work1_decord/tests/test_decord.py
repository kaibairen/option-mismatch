import json
from pathlib import Path

import pytest

from decord.metrics import dcdi, normalized_confidence, reasoning_matches, summarize_predictions
from decord.pipeline import (
    METHODS,
    RUNNERS,
    GenerateConfig,
    run_confidence_reflection,
    run_cot,
    run_direct,
    run_plan_and_solve,
    run_reflexion,
    run_standard,
)
from decord.prompts import decision_prompt, direct_prompt, plan_prompt, reasoning_prompt, reflexion_prompt


class ScriptedClient:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def complete(self, prompt: str, *, temperature: float, top_p: float, max_tokens: int) -> str:
        self.prompts.append(prompt)
        self.kwargs.append({"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens})
        if not self.replies:
            raise RuntimeError("no scripted reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def test_reasoning_prompt_hides_options():
    prompt = reasoning_prompt("Shweta rides 15 km. How long for 100 km?")
    assert "15 km" in prompt
    assert "12 hours" not in prompt
    assert "A)" not in prompt


def test_decision_prompt_keeps_relation_symbols():
    prompt = decision_prompt("How long?", "Total is 11 hours.", "11 hours", ["A) 10", "D) 12 hours"])
    assert "≈" in prompt and "≡" in prompt and "→" in prompt
    assert "11 hours" in prompt
    assert "D) 12 hours" in prompt


def test_confidence_ends():
    assert normalized_confidence({"A": 4, "B": 0, "C": 0, "D": 0}, "ABCD") == pytest.approx(1.0)
    assert normalized_confidence({"A": 1, "B": 1, "C": 1, "D": 1}, "ABCD") == pytest.approx(0.0)
    assert normalized_confidence({}, "ABCD") is None


def test_dcdi_matches_qwen_row_and_not_the_printed_deepseek_cell():
    qwen = dcdi(0.7913, 0.6732)
    assert qwen is not None
    assert qwen * 100 == pytest.approx(14.92, abs=0.02)
    gemini = dcdi(0.8111, 0.7244)
    assert gemini is not None
    assert gemini * 100 == pytest.approx(10.69, abs=0.02)
    deepseek = dcdi(0.6181, 0.5629)
    assert deepseek is not None
    assert abs(deepseek * 100 - 15.93) > 1


def test_reasoning_match_uses_the_gold_value():
    options = ["A) 10", "B) 11", "C) 12", "D) 13"]
    assert reasoning_matches("The trip takes 11 hours.", options, "B") is True
    assert reasoning_matches("The trip takes 12 hours.", options, "B") is False
    assert reasoning_matches("", options, "B") is None


def test_summary_keeps_failures_in_the_full_denominator():
    rows = [
        {"status": "ok", "correct": True, "reasoning_correct": True},
        {"status": "ok", "correct": False, "reasoning_correct": True},
        {"status": "empty", "correct": False, "reasoning_correct": None},
        {"status": "no_letter", "correct": False},
    ]
    summary = summarize_predictions(rows, n_full=5)
    assert summary["n_full"] == 5
    assert summary["n_valid"] == 2
    assert summary["n_correct"] == 1
    assert summary["n_fail"] == 3
    assert summary["failures"] == {"empty": 1, "no_letter": 1, "error": 1}
    assert summary["accuracy_full"] == pytest.approx(1 / 5)
    assert summary["accuracy_valid"] == pytest.approx(1 / 2)
    assert summary["dcdi"] == pytest.approx((1.0 - 0.2) / 1.0)


def test_standard_path_calls_reason_then_decide_once():
    client = ScriptedClient(
        [
            json.dumps({"reasoning": "15 km takes 1.5 h.", "conclusion": "11 hours"}),
            json.dumps({"matches": [{"letter": "B", "relation": "≈", "note": "11"}], "answer": "B"}),
        ]
    )
    result = run_standard(
        client,
        question="How long does the 100 km trip take?",
        options=["A) 10 hours", "B) 11 hours", "D) 12 hours"],
        gold="B",
        config=GenerateConfig(),
    )
    assert result["status"] == "ok"
    assert result["pred"] == "B"
    assert result["correct"] is True
    assert result["reasoning_correct"] is True
    assert len(client.prompts) == 2
    assert "11 hours" not in client.prompts[0]
    assert "B) 11 hours" in client.prompts[1]
    assert client.kwargs[0] == {"temperature": 0.4, "top_p": 1.0, "max_tokens": 512}


def test_low_confidence_triggers_one_reflection():
    client = ScriptedClient(
        [
            json.dumps({"reasoning": "11 hours", "conclusion": "11 hours"}),
            json.dumps({"answer": "A"}),
            json.dumps({"answer": "B"}),
            json.dumps({"answer": "C"}),
            json.dumps({"answer": "D"}),
            json.dumps({"answer": "B", "note": "11 matches B"}),
        ]
    )
    result = run_confidence_reflection(
        client,
        question="How long?",
        options=["A) 10", "B) 11", "C) 12", "D) 13"],
        gold="B",
        config=GenerateConfig(sample_k=4),
    )
    assert result["reflected"] is True
    assert result["pred"] == "B"
    assert result["confidence"] == pytest.approx(0.0)
    assert len(client.prompts) == 6
    assert "Sample counts" in client.prompts[-1]


def test_high_confidence_skips_reflection():
    client = ScriptedClient(
        [
            json.dumps({"reasoning": "11", "conclusion": "11"}),
            json.dumps({"answer": "B"}),
            json.dumps({"answer": "B"}),
        ]
    )
    result = run_confidence_reflection(
        client,
        question="How long?",
        options=["A) 10", "B) 11"],
        gold="B",
        config=GenerateConfig(sample_k=2),
    )
    assert result["reflected"] is False
    assert result["pred"] == "B"
    assert result["confidence"] == pytest.approx(1.0)
    assert len(client.prompts) == 3


def test_transport_error_is_not_scored_as_a_valid_miss():
    client = ScriptedClient([RuntimeError("timed out")])
    result = run_cot(client, question="Q", options=["A) 1", "B) 2"], gold="A")
    assert result["status"] == "error"
    summary = summarize_predictions([result], n_full=1)
    assert summary["n_valid"] == 0
    assert summary["accuracy_full"] == 0.0
    assert summary["failures"]["error"] == 1


def test_direct_sees_options_and_returns_only_the_letter():
    client = ScriptedClient([json.dumps({"answer": "C"})])
    result = run_direct(
        client,
        question="How long?",
        options=["A) 10", "B) 11", "C) 12"],
        gold="C",
    )
    assert result["pred"] == "C"
    assert result["n_calls"] == 1
    assert "A) 10" in client.prompts[0]
    assert "Reply with the option letter only." in client.prompts[0]


def test_plan_and_solve_is_two_calls_and_the_second_reads_the_plan():
    client = ScriptedClient(
        [
            json.dumps({"plan": "Compute the time from the speed."}),
            json.dumps({"reasoning": "100 / 10 = 10", "answer": "A"}),
        ]
    )
    result = run_plan_and_solve(
        client,
        question="How long?",
        options=["A) 10 hours", "B) 11 hours"],
        gold="A",
    )
    assert result["n_calls"] == 2
    assert result["pred"] == "A"
    assert "devise a plan" in client.prompts[0]
    assert "A) 10 hours" in client.prompts[0]
    assert "Compute the time from the speed." in client.prompts[1]
    assert "A) 10 hours" in client.prompts[1]


def test_reflexion_revises_once_and_does_not_sample():
    client = ScriptedClient(
        [
            json.dumps({"reasoning": "Maybe 10.", "answer": "A"}),
            json.dumps({"critique": "The conclusion was 11.", "answer": "B"}),
        ]
    )
    result = run_reflexion(
        client,
        question="How long?",
        options=["A) 10", "B) 11"],
        gold="B",
    )
    assert result["n_calls"] == 2
    assert result["pred"] == "B"
    assert len(client.prompts) == 2
    assert "Previous attempt" in client.prompts[1]
    assert "Maybe 10." in client.prompts[1]


def test_reasoning_prompt_still_hides_the_direct_and_plan_context():
    stem = "Shweta rides 15 km. How long for 100 km?"
    options = ["A) 10 hours", "B) 11 hours"]
    hidden = reasoning_prompt(stem)
    shown = direct_prompt(stem, options)
    planned = plan_prompt(stem, options)
    revised = reflexion_prompt(stem, options, "answer A")
    assert "A) 10 hours" not in hidden
    assert "A) 10 hours" in shown and "A) 10 hours" in planned and "A) 10 hours" in revised


def test_experiment_methods_omit_confidence_reflection():
    assert METHODS == (
        "direct",
        "cot",
        "plan_and_solve",
        "reflexion",
        "decord_standard",
    )
    assert set(METHODS) <= set(RUNNERS)
    assert "decord_confidence_reflection" in RUNNERS


def test_archived_tables_stay_separate():
    root = Path(__file__).resolve().parents[1] / "reports"
    names = [
        "aaai_reported.json",
        "nota_sheet.json",
        "aqua_logiqa_valid_only.json",
    ]
    provenances = []
    for name in names:
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        provenances.append(payload["provenance"])
        assert payload["recomputed"] is False
    assert len(set(provenances)) == 3
