#!/usr/bin/env python3
"""Smoke-test prompts and answer parsing without a GPU."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from parsing import cot_mentions_hint_keywords, extract_final_answer, split_cot_and_answer
from prompts import CONDITIONS, build_prompt
from questions import load_handwritten


def test_prompts() -> None:
    q = load_handwritten()[0]
    for condition in CONDITIONS:
        prompt = build_prompt(q["question"], q["choices"], condition, q["correct"])
        assert q["question"] in prompt
        if "hint" in condition:
            assert "trusted advisor" in prompt.lower()
        if "monitored" in condition:
            assert "reviewed" in prompt.lower()
    print(f"prompts ok ({len(CONDITIONS)} conditions)")


def test_parser() -> None:
    raw = (
        "The advisor suggests B, but the premises imply A.\n"
        "Final answer: A"
    )
    cot, ans = split_cot_and_answer(raw)
    assert ans == "A"
    assert extract_final_answer(raw) == "A"
    assert cot_mentions_hint_keywords(cot) is True
    assert extract_final_answer("I pick C.") == "C"
    print("parser ok")


if __name__ == "__main__":
    test_prompts()
    test_parser()
    print("smoke test passed")
