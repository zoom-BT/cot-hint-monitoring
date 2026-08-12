#!/usr/bin/env python3
"""Run CoT faithfulness evaluation on MCQ benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from parsing import cot_mentions_hint_keywords, extract_final_answer, split_cot_and_answer
from prompts import CONDITIONS, PILOT_CONDITIONS, build_prompt, pick_wrong_letter
from questions import assign_conditions, load_all_questions, load_pilot_questions


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_model(model_name: str):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            done.add((row["question_id"], row["condition"]))
    return done


def run_job(
    *,
    model_name: str,
    output_path: Path,
    mode: str,
    max_questions: int | None,
    max_new_tokens: int,
    include_no_cot_baseline: bool,
    seed: int,
) -> None:
    model, tokenizer = load_model(model_name)
    done = load_done_keys(output_path)

    if mode == "pilot":
        base_questions = load_pilot_questions(n=max_questions or 10, seed=seed)
        jobs = []
        for q in base_questions:
            for condition in PILOT_CONDITIONS:
                jobs.append({**q, "condition": condition})
    else:
        questions = load_all_questions(mmlu_n=60, include_handwritten=True, seed=seed)
        if max_questions:
            questions = questions[:max_questions]
        jobs = assign_conditions(questions, list(CONDITIONS.keys()), seed=seed)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"Run {run_id} | jobs={len(jobs)} | resume skips={len(done)}")

    for job in tqdm(jobs, desc="eval"):
        qid = job["id"]
        condition = job["condition"]
        key = (qid, condition)
        if key in done:
            continue

        prompt = build_prompt(
            job["question"],
            job["choices"],
            condition,
            job["correct"],
            use_cot=True,
        )
        started = time.time()
        raw = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        cot, parsed = split_cot_and_answer(raw)
        if parsed is None:
            parsed = extract_final_answer(raw)

        hinted = None
        flags = CONDITIONS[condition]
        if "hint_correct" in flags:
            hinted = job["correct"]
        elif "hint_incorrect" in flags:
            hinted = pick_wrong_letter(job["correct"], job["choices"])

        row = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "question_id": qid,
            "source": job.get("source"),
            "subject": job.get("subject"),
            "condition": condition,
            "correct_answer": job["correct"],
            "hinted_letter": hinted,
            "prompt": prompt,
            "raw_output": raw,
            "cot": cot,
            "final_answer": parsed,
            "is_correct": parsed == job["correct"] if parsed else None,
            "cot_mentions_hint_kw": cot_mentions_hint_keywords(cot),
            "latency_sec": round(time.time() - started, 2),
            "faithfulness_code": None,
        }
        append_jsonl(output_path, row)
        done.add(key)

        if include_no_cot_baseline:
            no_cot_key = (qid, f"{condition}__no_cot")
            if no_cot_key not in done:
                no_cot_prompt = build_prompt(
                    job["question"],
                    job["choices"],
                    condition,
                    job["correct"],
                    use_cot=False,
                )
                raw_nc = generate(
                    model, tokenizer, no_cot_prompt, max_new_tokens=64
                )
                _, parsed_nc = split_cot_and_answer(raw_nc)
                append_jsonl(
                    output_path,
                    {
                        **row,
                        "condition": f"{condition}__no_cot",
                        "prompt": no_cot_prompt,
                        "raw_output": raw_nc,
                        "cot": "",
                        "final_answer": parsed_nc,
                        "is_correct": parsed_nc == job["correct"] if parsed_nc else None,
                        "cot_mentions_hint_kw": False,
                    },
                )
                done.add(no_cot_key)

    print(f"Saved to {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "generations.jsonl",
    )
    p.add_argument("--mode", choices=["pilot", "main"], default="pilot")
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--no-cot-baseline", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_job(
        model_name=args.model,
        output_path=args.output,
        mode=args.mode,
        max_questions=args.max_questions,
        max_new_tokens=args.max_new_tokens,
        include_no_cot_baseline=args.no_cot_baseline,
        seed=args.seed,
    )
