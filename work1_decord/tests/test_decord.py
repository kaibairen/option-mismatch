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
    run_self_refine,
    run_standard,
)
from decord.prompts import (
    decision_prompt,
    direct_prompt,
    plan_solve_prompt,
    reasoning_prompt,
    self_refine_feedback_prompt,
    self_refine_refine_prompt,
)


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
    assert "smaller absolute difference" in prompt
    assert "recompute the quantity" in prompt
    assert "exactly one letter" in prompt
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
    assert "≈" in client.prompts[1]
    assert client.kwargs[0]["max_tokens"] == 2048
    assert client.kwargs[1]["max_tokens"] == 512
    assert result["relation"] == "≈"
    assert [step["step"] for step in result["trace"]] == ["reasoning", "decision"]


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
    assert client.kwargs[0]["max_tokens"] == 2048
    assert client.kwargs[1]["max_tokens"] == 512


def test_finished_cot_is_one_call_and_reads_the_answer_sentence():
    client = ScriptedClient(["36 is the multiple. The answer is (C)."])
    result = run_cot(client, question="Which multiple?", options=["A) 15", "C) 36"], gold="C")
    assert result["status"] == "ok"
    assert result["pred"] == "C"
    assert result["n_calls"] == 1
    assert client.kwargs[0]["max_tokens"] == 2048
    assert client.prompts[0].startswith("Q: Which multiple? Answer Choices: (A) 15 (C) 36")
    assert client.prompts[0].rstrip().endswith("A: Let's think step by step.")
    assert "reasoning" not in client.prompts[0]
    assert "Partial reply" not in client.prompts[0]


def test_truncated_cot_does_not_scrape_a_letter_or_ask_again():
    client = ScriptedClient(['distance $d = x_1 - x_2$ and option A looks long'])
    result = run_cot(
        client,
        question="How long?",
        options=["A) 5(√3 + 1)", "D) 8(√3 – 2)"],
        gold="A",
    )
    assert result["pred"] == ""
    assert result["status"] == "no_letter"
    assert result["n_calls"] == 1
    assert result["correct"] is False
    assert len(client.prompts) == 1


def test_cot_without_an_answer_sentence_stays_blank():
    client = ScriptedClient(["the distance is $d = 5$"])
    result = run_cot(client, question="Q", options=["A) 1", "B) 2"], gold="A")
    assert result["pred"] == ""
    assert result["status"] == "no_letter"
    assert result["n_calls"] == 1
    assert result["correct"] is False


def test_cot_keeps_the_last_answer_sentence():
    client = ScriptedClient(["The answer is A. Recalculating gives 11. The answer is B."])
    result = run_cot(client, question="Q", options=["A) 10", "B) 11"], gold="B")
    assert result["pred"] == "B"
    assert result["n_calls"] == 1


def test_cot_reads_a_markdown_answer_line_and_ignores_option_mentions():
    text = "The calculated value matches option (A).\n\n**Answer:** (A)"
    client = ScriptedClient([text])
    result = run_cot(client, question="Q", options=["A) 5", "B) 6"], gold="A")
    assert result["pred"] == "A"
    assert result["n_calls"] == 1


def test_cot_reads_a_correct_option_line():
    client = ScriptedClient(["The greatest value is 0.26.\n\nThe correct option is **(B)**."])
    result = run_cot(client, question="Q", options=["A) 0.22", "B) 0.26"], gold="B")
    assert result["pred"] == "B"
    assert result["n_calls"] == 1


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


def test_plan_and_solve_is_one_completion_and_reads_the_answer_line():
    client = ScriptedClient(["Plan: divide distance by speed.\n\nAnswer: (A)"])
    result = run_plan_and_solve(
        client,
        question="How long?",
        options=["A) 10 hours", "B) 11 hours"],
        gold="A",
    )
    assert result["n_calls"] == 1
    assert result["pred"] == "A"
    assert result["status"] == "ok"
    assert client.kwargs[0]["max_tokens"] == 2048
    prompt = client.prompts[0]
    assert prompt.startswith("Q: How long? Answer Choices: (A) 10 hours (B) 11 hours")
    assert "devise a plan to solve the problem" in prompt
    assert "carry out the plan" in prompt
    assert "reasoning" not in prompt


def test_plan_and_solve_ignores_a_letter_that_is_only_in_the_plan():
    client = ScriptedClient(["Step 1: compare option A with the distance. Step 2: still calculating"])
    result = run_plan_and_solve(
        client,
        question="How long?",
        options=["A) 10 hours", "B) 11 hours"],
        gold="A",
    )
    assert result["pred"] == ""
    assert result["status"] == "no_letter"
    assert result["n_calls"] == 1


def test_self_refine_stops_when_the_feedback_says_the_draft_is_correct():
    client = ScriptedClient(
        [
            "The distance is 11. The answer is B.",
            "It is correct.",
            "The answer is A.",
        ]
    )
    result = run_self_refine(
        client,
        question="How long?",
        options=["A) 10", "B) 11"],
        gold="B",
    )
    assert result["n_calls"] == 2
    assert result["pred"] == "B"
    assert result["status"] == "ok"
    assert len(client.prompts) == 2
    assert client.prompts[0].rstrip().endswith("A: Let's think step by step.")
    assert "The distance is 11." in client.prompts[1]
    assert "It is correct." in client.prompts[1]
    assert all(call["max_tokens"] == 2048 for call in client.kwargs)
    assert all("gold" not in prompt.lower() for prompt in client.prompts)
    assert [step["step"] for step in result["trace"]] == ["init", "feedback_0"]


def test_self_refine_rewrites_with_history_and_stops_at_four_iterations():
    replies = ["The answer is A."]
    for _ in range(4):
        replies.append("Step 2 subtracts the wrong quantity.")
        replies.append("The answer is B.")
    replies.append("this call must not be sent")
    client = ScriptedClient(replies)
    result = run_self_refine(
        client,
        question="How long?",
        options=["A) 10", "B) 11"],
        gold="B",
    )
    assert result["n_calls"] == 9
    assert result["pred"] == "B"
    assert client.replies == ["this call must not be sent"]
    assert "Okay! Here is the rewrite:" in client.prompts[2]
    assert "Step 2 subtracts the wrong quantity." in client.prompts[-1]
    assert client.prompts[-1].count("Feedback:") == 4
    assert "The answer is A." in client.prompts[-1]


def test_self_refine_keeps_the_draft_when_the_second_feedback_stops():
    client = ScriptedClient(
        [
            "The answer is A.",
            "Step 2 subtracts the wrong quantity.",
            "Recalculating gives 11. The answer is B.",
            "It is correct.",
        ]
    )
    result = run_self_refine(
        client,
        question="How long?",
        options=["A) 10", "B) 11"],
        gold="B",
    )
    assert result["n_calls"] == 4
    assert result["pred"] == "B"
    assert "The answer is A." in client.prompts[2]
    assert "Step 2 subtracts the wrong quantity." in client.prompts[2]
    assert "Recalculating gives 11." in client.prompts[3]
    assert "The answer is A." not in client.prompts[3]
    assert "Okay! Here is the rewrite:" not in client.prompts[3]


def test_reasoning_prompt_still_hides_the_direct_and_plan_context():
    stem = "Shweta rides 15 km. How long for 100 km?"
    options = ["A) 10 hours", "B) 11 hours"]
    hidden = reasoning_prompt(stem)
    shown = direct_prompt(stem, options)
    planned = plan_solve_prompt(stem, options)
    feedback = self_refine_feedback_prompt(stem, options, "The answer is A.")
    revised = self_refine_refine_prompt(stem, options, [("The answer is A.", "Step 2 is wrong.")])
    assert "A) 10 hours" not in hidden
    assert "A) 10 hours" in shown and "A) 10 hours" in planned
    assert "A) 10 hours" in feedback and "A) 10 hours" in revised


def test_experiment_methods_omit_confidence_reflection():
    assert METHODS == (
        "direct",
        "cot",
        "plan_and_solve",
        "self_refine",
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
