import json
from pathlib import Path

from decord.data import Item
from decord.pipeline import GenerateConfig
from decord.record import append_prediction, load_saved_questions, prediction_row, run_letter_log


class ScriptedClient:
    def __init__(self, replies: list[str | Exception]):
        self.replies = list(replies)

    def complete(self, prompt: str, *, temperature: float, top_p: float, max_tokens: int) -> str:
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _item(number: int, answer: str = "A") -> Item:
    return Item(
        id=f"aqua_rat-test-{number:04d}",
        dataset="aqua_rat",
        subset=None,
        subject=None,
        question=f"Question {number}",
        options=["A) one", "B) two"],
        answer=answer,
    )


def test_row_uses_question_answer_and_pred():
    assert prediction_row("How long?", "A", "D") == {
        "question": "How long?",
        "answer": "A",
        "pred": "D",
    }


def test_partial_tail_is_dropped_and_completed_questions_are_skipped(tmp_path: Path):
    path = tmp_path / "aqua_direct.jsonl"
    append_prediction(path, "Question 0", "A", "D")
    with path.open("ab") as handle:
        handle.write(b'{"question": "Question 1", "answer":')
    done = load_saved_questions(path)
    assert done == {"Question 0"}
    assert path.read_bytes().endswith(b"\n")
    assert b"Question 1" not in path.read_bytes()

    client = ScriptedClient([json.dumps({"answer": "B"}), json.dumps({"answer": "A"})])
    first = run_letter_log(
        output=path,
        items=[_item(0), _item(1, "E")],
        client=client,
        config=GenerateConfig(),
    )
    second = run_letter_log(
        output=path,
        items=[_item(0), _item(1, "E")],
        client=ScriptedClient([]),
        config=GenerateConfig(),
    )
    assert first == {"written": 1, "skipped": 1, "errors": 0}
    assert second == {"written": 0, "skipped": 2, "errors": 0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"question": "Question 0", "answer": "A", "pred": "D"},
        {"question": "Question 1", "answer": "E", "pred": "B"},
    ]
    assert set(rows[0]) == {"question", "answer", "pred"}


def test_failed_call_is_not_saved_so_a_rerun_can_retry(tmp_path: Path):
    path = tmp_path / "aqua_direct.jsonl"
    item = _item(3, "B")
    failed = run_letter_log(
        output=path,
        items=[item],
        client=ScriptedClient([RuntimeError("down")]),
        config=GenerateConfig(),
    )
    assert failed == {"written": 0, "skipped": 0, "errors": 1}
    assert not path.exists() or path.read_text(encoding="utf-8") == ""
    recovered = run_letter_log(
        output=path,
        items=[item],
        client=ScriptedClient([json.dumps({"answer": "B"})]),
        config=GenerateConfig(),
    )
    assert recovered == {"written": 1, "skipped": 0, "errors": 0}
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["pred"] == "B"
    assert row["answer"] == "B"
