"""Load MCQ questions for pilot and main runs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

Question = dict[str, Any]


def load_handwritten() -> list[Question]:
    path = DATA_DIR / "handwritten_mcq.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_mmlu_subset(n: int = 60, seed: int = 42) -> list[Question]:
    """Load a fixed MMLU subset via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets: pip install datasets") from exc

    subjects = [
        "logical_fallacies",
        "moral_scenarios",
        "high_school_mathematics",
        "formal_logic",
    ]
    rng = random.Random(seed)
    rows: list[Question] = []

    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        per_subject = max(1, n // len(subjects))
        for idx in indices[:per_subject]:
            item = ds[idx]
            choices = {
                "A": item["choices"][0],
                "B": item["choices"][1],
                "C": item["choices"][2],
                "D": item["choices"][3],
            }
            letter = "ABCD"[item["answer"]]
            rows.append(
                {
                    "id": f"mmlu_{subject}_{idx}",
                    "source": "mmlu",
                    "subject": subject,
                    "question": item["question"],
                    "choices": choices,
                    "correct": letter,
                }
            )
        if len(rows) >= n:
            break

    return rows[:n]


def load_all_questions(
    mmlu_n: int = 60,
    include_handwritten: bool = True,
    seed: int = 42,
) -> list[Question]:
    questions: list[Question] = []
    if include_handwritten:
        questions.extend(load_handwritten())
    questions.extend(load_mmlu_subset(n=mmlu_n, seed=seed))
    return questions


def assign_conditions(
    questions: list[Question],
    conditions: list[str],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Split questions evenly across conditions for between-subject design."""
    rng = random.Random(seed)
    shuffled = questions[:]
    rng.shuffle(shuffled)

    assigned: list[dict[str, Any]] = []
    for i, q in enumerate(shuffled):
        condition = conditions[i % len(conditions)]
        assigned.append({**q, "condition": condition})
    return assigned


def load_pilot_questions(n: int = 10, seed: int = 42) -> list[Question]:
    all_q = load_all_questions(mmlu_n=max(n, 10), include_handwritten=True, seed=seed)
    rng = random.Random(seed)
    rng.shuffle(all_q)
    return all_q[:n]
