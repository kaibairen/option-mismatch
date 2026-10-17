"""Prompt templates for DecoRD and the four coupled baselines.

Every template is zero-shot. None of them includes a worked example, an MMLU
dev item, or a field from ``meta`` (AQUA rationales, GAOKAO analysis).

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


def direct_prompt(question: str, options: list[str]) -> str:
    schema = {"answer": "A"}
    context = f"Question:\n{question.strip()}\n\nOptions:\n{_options_block(options)}"
    return _wrap(
        "multiple-choice solving",
        "Read the question and the options. Reply with the option letter only.",
        schema,
        context,
    )


def plan_prompt(question: str, options: list[str]) -> str:
    """First call of Plan-and-Solve (Wang et al., ACL 2023).

    The paper puts “devise a plan” and “carry out the plan” in one trigger
    sentence. This runner asks for the plan alone so the next call can read it.
    """

    schema = {"plan": ""}
    context = f"Question:\n{question.strip()}\n\nOptions:\n{_options_block(options)}"
    return _wrap(
        "problem planning",
        (
            "Let's first understand the problem and devise a plan to solve the problem. "
            "Do not choose an option letter."
        ),
        schema,
        context,
    )


def solve_prompt(question: str, options: list[str], plan: str) -> str:
    """Second call of Plan-and-Solve. The plan from the first call is an input."""

    schema = {"reasoning": "", "answer": "A"}
    context = (
        f"Question:\n{question.strip()}\n\n"
        f"Options:\n{_options_block(options)}\n\n"
        f"Plan:\n{plan.strip()}"
    )
    return _wrap(
        "plan execution",
        "Carry out the plan and solve the problem step by step. Then put the final option letter in answer.",
        schema,
        context,
    )


def reflexion_prompt(question: str, options: list[str], previous: str) -> str:
    """One critique-and-revise call after a single coupled CoT attempt.

    Shinn et al. (2023) retry until the attempt improves. This baseline stops
    after this one revision: no extra sample and no vote.
    """

    schema = {"critique": "", "answer": "A"}
    context = (
        f"Question:\n{question.strip()}\n\n"
        f"Options:\n{_options_block(options)}\n\n"
        f"Previous attempt:\n{previous.strip()}"
    )
    return _wrap(
        "answer reflection",
        (
            "Critique the previous attempt, correct a wrong option choice, "
            "and put the revised option letter in answer."
        ),
        schema,
        context,
    )


def cot_prompt(question: str, options: list[str]) -> str:
    schema = {"reasoning": "", "answer": "A"}
    context = f"Question:\n{question.strip()}\n\nOptions:\n{_options_block(options)}"
    return _wrap(
        "multiple-choice solving",
        "Let's think step by step. Then put the final option letter in answer.",
        schema,
        context,
    )


def cot_continue_prompt(question: str, options: list[str], partial: str) -> str:
    """Second call when the first CoT reply was cut off before ``answer``."""

    schema = {"answer": "A"}
    context = (
        f"Question:\n{question.strip()}\n\n"
        f"Options:\n{_options_block(options)}\n\n"
        f"Partial reply:\n{partial.strip()}"
    )
    return _wrap(
        "multiple-choice solving",
        "The partial reply was cut off before the final answer. Using that reasoning, put the option letter in answer.",
        schema,
        context,
    )
