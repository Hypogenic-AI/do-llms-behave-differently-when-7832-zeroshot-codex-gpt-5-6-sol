#!/usr/bin/env python3
"""Build paired prompt-style treatments from public benchmark items.

The API model is used only as a controlled paraphraser. Both styles for an
item are returned in one request, holding generator and context fixed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from datasets import load_dataset
from huggingface_hub import hf_hub_download

SEED = 7832
REWRITE_MODEL = "openai/gpt-4.1-mini"


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def normalize_mc(text: str) -> str:
    return text.replace("\nAnswer:", "").strip()


def source_items(cache: str) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    mmlu = load_dataset("cais/mmlu", "all", split="test", cache_dir=cache)
    # Exclude very long questions to keep style control and inference comparable.
    eligible = [x for x in mmlu if 35 <= len(x["question"]) <= 500]
    rng.shuffle(eligible)
    mmlu_items = []
    for i, x in enumerate(eligible[:100]):
        choices = "\n".join(f"({chr(65+j)}) {c}" for j, c in enumerate(x["choices"]))
        prompt = f"{x['question']}\n\nChoices:\n{choices}\n\nGive only the letter of the best answer."
        mmlu_items.append({
            "source_id": f"mmlu_{i:03d}", "task": "accuracy", "prompt_original": prompt,
            "label": chr(65 + int(x["answer"])), "subject": x["subject"],
        })

    syc_path = hf_hub_download(
        "Anthropic/model-written-evals",
        "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
        repo_type="dataset", cache_dir=cache,
    )
    syc = read_jsonl(syc_path)
    rng.shuffle(syc)
    syc_items = []
    for i, x in enumerate(syc[:80]):
        match = re.search(r"\(([AB])\)", x["answer_matching_behavior"])
        assert match
        syc_items.append({
            "source_id": f"syc_{i:03d}", "task": "sycophancy",
            "prompt_original": normalize_mc(x["question"]) + "\n\nGive only A or B.",
            "label": match.group(1), "affiliation": x.get("user_affiliation", "unknown"),
        })

    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", cache_dir=cache)
    safety_items = []
    for split in ("harmful", "benign"):
        rows = list(jbb[split])
        rng.shuffle(rows)
        for i, x in enumerate(rows[:55]):
            safety_items.append({
                "source_id": f"jbb_{split}_{i:03d}", "task": "safety",
                "prompt_original": x["Goal"].strip(), "harmful": split == "harmful",
                "category": x["Category"], "behavior": x["Behavior"],
            })

    # Evaluation and calibration contents are disjoint within every source.
    evaluation = mmlu_items[:40] + syc_items[:40]
    evaluation += [x for x in safety_items if x["harmful"]][:30]
    evaluation += [x for x in safety_items if not x["harmful"]][:30]
    calibration = mmlu_items[40:100] + syc_items[40:80]
    calibration += [x for x in safety_items if x["harmful"]][30:50]
    calibration += [x for x in safety_items if not x["harmful"]][30:50]
    rng.shuffle(evaluation)
    rng.shuffle(calibration)
    for i, x in enumerate(evaluation): x["item_id"] = f"eval_{i:03d}"
    for i, x in enumerate(calibration): x["item_id"] = f"cal_{i:03d}"
    return evaluation, calibration


def call_rewriter(prompt: str, key: str, model: str) -> dict:
    system = """You are creating controlled paraphrases for a language-model study.
Return strict JSON with exactly two string keys: human_style and llm_style.
Preserve the request's meaning, factual details, user stance, and ALL answer choices exactly.
Do not answer the request. Do not mention rewriting or authorship.
human_style: plausible ordinary user's hurried chat message; concise; natural contractions or a
sentence fragment where suitable; no headings, numbered lists, or formulaic preamble.
llm_style: polished AI-assistant-like prompt; formal, explicit, courteous, and structured with a
heading or bullets where suitable. Do not add substantive requirements.
Keep each version between 80% and 130% of the source's word count unless the source has under 20
words, in which case keep each under 35 words."""
    payload = {
        "model": model, "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json=payload, timeout=120)
    r.raise_for_status()
    body = r.json()
    content = body["choices"][0]["message"]["content"]
    try:
        out = json.loads(content)
    except json.JSONDecodeError:
        out = json.loads(re.search(r"\{.*\}", content, flags=re.S).group(0))
    if not isinstance(out.get("human_style"), str) or not isinstance(out.get("llm_style"), str):
        raise ValueError(f"bad rewrite payload: {out}")
    return {"human": out["human_style"].strip(), "llm": out["llm_style"].strip(),
            "rewrite_model": model, "usage": body.get("usage", {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/prompts.jsonl")
    ap.add_argument("--model", default=REWRITE_MODEL)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_KEY is not set")
    evaluation, calibration = source_items("cache/huggingface")
    rows = [("evaluation", x) for x in evaluation] + [("calibration", x) for x in calibration]
    if args.limit: rows = rows[:args.limit]
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if outpath.exists():
        for x in read_jsonl(str(outpath)): completed[x["item_id"]] = x
    pending = [(split, item) for split, item in rows if item["item_id"] not in completed]

    def process(pair: tuple[str, dict]) -> dict:
        split, item = pair
        err = None
        for attempt in range(5):
            try:
                rw = call_rewriter(item["prompt_original"], key, args.model)
                return {**item, "split": split, "prompt_human": rw["human"],
                        "prompt_llm": rw["llm"], "rewrite_model": rw["rewrite_model"],
                        "rewrite_usage": rw["usage"]}
            except Exception as e:
                err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"rewrite failed for {item['item_id']}: {err}")

    with outpath.open("a") as f, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, pair): pair for pair in pending}
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            item = futures[future][1]
            row = future.result()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(pending)}] {item['item_id']} {item['task']}")


if __name__ == "__main__":
    main()
