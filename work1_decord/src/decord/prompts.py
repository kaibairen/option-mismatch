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


def _answer_choices(options: list[str]) -> str:
    choices: list[str] = []
    for option in options:
        text = str(option).strip()
        if len(text) >= 2 and text[0] in "ABCDE" and text[1] == ")":
            choices.append(f"({text[0]}) {text[2:].strip()}")
        else:
            choices.append(text)
    return "Answer Choices: " + " ".join(choices)


def plan_solve_prompt(question: str, options: list[str]) -> str:
    """Zero-shot Plan-and-Solve trigger from Wang et al., ACL 2023.

    The plan and the solution are one completion. The trigger replaces
    ``Let's think step by step`` in the same ``Q:`` / ``A:`` frame as CoT.
    """

    return (
        f"Q: {question.strip()} {_answer_choices(options)}\n"
        "A: Let's first understand the problem and devise a plan to solve the problem. "
        "Then, let's carry out the plan and solve the problem step by step."
    )


def self_refine_feedback_prompt(question: str, options: list[str], solution: str) -> str:
    """Feedback step from Madaan et al., NeurIPS 2023, Algorithm 1.

    The math-task wording asks for a specific error. The stop sentence is the
    one the released GSM runner treats as “no further refinement”.
    """

    return (
        f"Q: {question.strip()} {_answer_choices(options)}\n"
        f"Solution:\n{solution.strip()}\n\n"
        "There may be an error in the solution above because of a lack of understanding of the question. "
        "What is the error? To find the error, go through the solution step by step and check if everything looks good.\n"
        "If the solution is already correct, write exactly: It is correct.\n"
        "If it is not correct, name the specific wrong step and state how to correct it."
    )


def self_refine_refine_prompt(
    question: str,
    options: list[str],
    history: list[tuple[str, str]],
) -> str:
    """Refine step. Equation 4 keeps every earlier solution and its feedback."""

    blocks = [f"Q: {question.strip()} {_answer_choices(options)}"]
    for solution, feedback in history:
        blocks.append(f"Solution:\n{solution.strip()}\n\nFeedback:\n{feedback.strip()}")
    blocks.append(
        "Okay! Here is the rewrite:\n"
        "Write a corrected step-by-step solution to the question above. "
        "End with the line: The answer is X"
    )
    return "\n\n".join(blocks)


def cot_prompt(question: str, options: list[str]) -> str:
    """Zero-shot CoT trigger from Kojima et al. 2022, with the options in the question.

    The completion starts after ``Let's think step by step.`` There is no JSON
    schema and no second call that fills in a missing letter.
    """

    return (
        f"Q: {question.strip()} {_answer_choices(options)}\n"
        "A: Let's think step by step."
    )
