import json
from pathlib import Path

import pytest

from decord.client import OpenAIChatClient
from decord.data import load_experiment_items
from decord.experiment import dry_run_report, load_config, run_grid, score_runs
from decord.metrics import score_experiment
from decord.pipeline import METHODS
from decord.prompts import reasoning_prompt


class ScriptedClient:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, temperature: float, top_p: float, max_tokens: int) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise RuntimeError("no scripted reply left")
        return self.replies.pop(0)


def test_config_locks_the_four_models_and_disables_qwen_thinking():
    config = load_config()
    assert [model["id"] for model in config["models"]] == [
        "qwen3.5-9b",
        "qwen3.5-27b",
        "deepseek-v3",
        "gpt-4",
    ]
    bodies = {model["id"]: model["extra_body"] for model in config["models"]}
    assert bodies["qwen3.5-9b"] == {"enable_thinking": False}
    assert bodies["qwen3.5-27b"] == {"enable_thinking": False}
    assert bodies["deepseek-v3"] == {}
    assert bodies["gpt-4"] == {}
    assert config["methods"] == list(METHODS)
    assert config["temperature"] == 0.4
    assert config["top_p"] == 1.0
    assert config["max_tokens"] == 512
    assert config["sample_k"] == 5
    assert config["confidence_threshold"] == 0.5
    assert config["request_timeout_s"] == 180


def test_request_payload_carries_only_that_models_thinking_switch():
    qwen = OpenAIChatClient("qwen3.5-9b", api_key="test", extra_body={"enable_thinking": False})
    closed = OpenAIChatClient("gpt-4", api_key="test", extra_body={})
    deepseek = OpenAIChatClient("deepseek-v3", api_key="test", extra_body={})
    qwen_payload = qwen.request_payload("Q", temperature=0.4, top_p=1, max_tokens=512)
    closed_payload = closed.request_payload("Q", temperature=0.4, top_p=1, max_tokens=512)
    deepseek_payload = deepseek.request_payload("Q", temperature=0.4, top_p=1, max_tokens=512)
    assert qwen_payload["enable_thinking"] is False
    assert "thinking" not in qwen_payload
    assert "enable_thinking" not in closed_payload
    assert "thinking" not in closed_payload
    assert "enable_thinking" not in deepseek_payload
    assert "thinking" not in deepseek_payload


def test_locked_question_list_is_aqua_254_and_mmlu_40_per_subject():
    items = load_experiment_items()
    aqua = [item for item in items if item.dataset == "aqua_rat"]
    mmlu = [item for item in items if item.dataset == "mmlu"]
    sizes: dict[str, int] = {}
    groups: dict[str, set[str]] = {}
    for item in mmlu:
        assert item.subject
        sizes[item.subject] = sizes.get(item.subject, 0) + 1
        groups.setdefault(item.subset or "", set()).add(item.subject)
    assert len(aqua) == 254
    assert len(mmlu) == 2280
    assert len(sizes) == 57
    assert set(sizes.values()) == {40}
    assert {name: len(subjects) for name, subjects in groups.items()} == {
        "stem": 19,
        "social_sciences": 12,
        "humanities": 13,
        "other": 13,
    }
    prompt = reasoning_prompt(aqua[0].question)
    assert "Explanation" not in prompt
    assert aqua[0].answer == "A"
    assert all(not hasattr(item, "meta") for item in items[:3])


def test_dry_run_counts_the_maximum_self_refine_calls():
    report = dry_run_report()
    assert report["n_items"] == 2534
    assert report["n_by_dataset"] == {"aqua_rat": 254, "mmlu": 2280}
    assert report["n_subjects"] == 57
    assert report["methods"] == list(METHODS)
    assert "decord_confidence_reflection" not in report["methods"]
    assert report["calls_per_item"] == {
        "direct": 1,
        "cot": 1,
        "plan_and_solve": 1,
        "self_refine": 9,
        "decord_standard": 2,
    }
    assert report["calls_per_model"] == 2534 * 14


def test_mmlu_overall_is_the_unweighted_mean_of_subjects():
    items = [
        {"id": "s1", "dataset": "mmlu", "subset": "stem", "subject": "algebra"},
        {"id": "s2", "dataset": "mmlu", "subset": "stem", "subject": "physics"},
        {"id": "h1", "dataset": "mmlu", "subset": "humanities", "subject": "history"},
        {"id": "a1", "dataset": "aqua_rat", "subset": None, "subject": None},
    ]
    rows = [
        {"id": "s1", "status": "ok", "correct": True, "reasoning_correct": True},
        {"id": "s2", "status": "ok", "correct": True, "reasoning_correct": True},
        {"id": "h1", "status": "ok", "correct": False, "reasoning_correct": False},
        {"id": "a1", "status": "error", "correct": False, "reasoning_correct": None},
    ]
    scored = score_experiment(rows, items)
    assert scored["mmlu"]["n_subjects"] == 3
    assert scored["mmlu"]["accuracy_full"] == pytest.approx(2 / 3)
    assert scored["mmlu"]["categories"]["stem"]["accuracy_full"] == pytest.approx(1.0)
    assert scored["mmlu"]["categories"]["humanities"]["accuracy_full"] == pytest.approx(0.0)
    assert scored["aqua"]["n_full"] == 1
    assert scored["aqua"]["accuracy_full"] == 0.0
    assert scored["aqua"]["failures"]["error"] == 1


def test_run_appends_and_skips_completed_ids(tmp_path: Path):
    from decord.data import Item

    item = Item(
        id="aqua_rat-test-0000",
        dataset="aqua_rat",
        subset=None,
        subject=None,
        question="How long does the trip take?",
        options=["A) 5", "B) 6"],
        answer="A",
    )
    config = {
        "models": [{"id": "gpt-4", "extra_body": {}}],
        "methods": ["direct"],
        "temperature": 0.4,
        "top_p": 1.0,
        "max_tokens": 512,
        "sample_k": 5,
        "confidence_threshold": 0.5,
        "request_timeout_s": 180,
    }
    client = ScriptedClient([json.dumps({"answer": "A"})])

    def factory(model: str, body: dict) -> ScriptedClient:
        assert model == "gpt-4"
        assert body == {}
        return client

    first = run_grid(config=config, items=[item], runs=tmp_path, client_factory=factory)
    second = run_grid(config=config, items=[item], runs=tmp_path, client_factory=factory)
    assert first == {"written": 1, "skipped": 0}
    assert second == {"written": 0, "skipped": 1}
    log = (tmp_path / "gpt-4" / "direct.jsonl").read_text(encoding="utf-8").strip()
    row = json.loads(log)
    assert row["correct"] is True
    assert row["status"] == "ok"
    assert "Explanation" not in client.prompts[0]
    tables = score_runs(tmp_path, items=[item])
    assert tables["gpt-4"]["direct"]["aqua"]["accuracy_full"] == 1.0
