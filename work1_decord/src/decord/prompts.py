"""Prompt templates for the three DecoRD agents and the CoT baseline.

The wrapper matches the manuscript's agent template: a task line, instructions,
and a JSON object that ``json.loads`` can read. The reasoning agent receives
the question stem only.
"""

from __future__ import annotations

import json
from typing import Any


def _options_block(options: list[str]) -> str:
    return "\n".join(str(option) for option in options)


def _wrap(task: str, instructions: str, schema: dict[str, Any], context: str) -> str:
    return (
        f"You are a {task} expert.\n"
        "### Instructions:\n"
        f"{instructions.strip()}\n"
        "### Output Format\n"
        "- Provide the output in valid JSON format, parsable by json.loads().\n"
        "- Strictly adhere to the following structure:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n"
        "---------------------------\n"
        f"{context.strip()}\n"
    )


def split_prompt(raw_question: str) -> str:
    schema = {"question": "", "options": [{"letter": "A", "text": ""}]}
    return _wrap(
        "question splitting",
        "Separate the question stem from the candidate options. Do not solve the question.",
        schema,
        raw_question,
    )


def reasoning_prompt(question: str) -> str:
    schema = {"reasoning": "", "conclusion": ""}
    return _wrap(
        "reasoning",
        (
            "Solve the question stem with a single chain of thought. "
            "Begin from the stem alone. Let's think step by step. "
            "Do not mention or choose an option letter."
        ),
        schema,
        f"Question:\n{question.strip()}",
    )


def decision_prompt(question: str, reasoning: str, conclusion: str, options: list[str]) -> str:
    schema = {
        "matches": [{"letter": "A", "relation": "≈", "note": ""}],
        "answer": "A",
    }
    instructions = (
        "Match the reasoning conclusion to exactly one option. "
        "Use ≈ for numerical approximation, ≡ for conceptual equivalence, "
        "and → for a range or semantic containment. "
        "Put the chosen letter in answer."
    )
    context = (
        f"Question:\n{question.strip()}\n\n"
        f"Reasoning:\n{reasoning.strip()}\n\n"
        f"Conclusion:\n{conclusion.strip()}\n\n"
        f"Options:\n{_options_block(options)}"
    )
    return _wrap("option decision", instructions, schema, context)


def reflection_prompt(
    question: str,
    reasoning: str,
    conclusion: str,
    options: list[str],
    counts: dict[str, int],
    confidence: float,
) -> str:
    schema = {"answer": "A", "note": ""}
    distribution = ", ".join(f"{letter}:{count}" for letter, count in sorted(counts.items()))
    instructions = (
        "The decision samples are diffident. Re-read the reasoning and the options, "
        "correct a mismatch between the conclusion and the letter, and choose one letter."
    )
    context = (
        f"Question:\n{question.strip()}\n\n"
        f"Reasoning:\n{reasoning.strip()}\n\n"
        f"Conclusion:\n{conclusion.strip()}\n\n"
        f"Options:\n{_options_block(options)}\n\n"
        f"Sample counts: {distribution}\n"
        f"Confidence: {confidence:.4f}"
    )
    return _wrap("confidence reflection", instructions, schema, context)


def cot_prompt(question: str, options: list[str]) -> str:
    schema = {"reasoning": "", "answer": "A"}
    context = f"Question:\n{question.strip()}\n\nOptions:\n{_options_block(options)}"
    return _wrap(
        "multiple-choice solving",
        "Let's think step by step. Then put the final option letter in answer.",
        schema,
        context,
    )
