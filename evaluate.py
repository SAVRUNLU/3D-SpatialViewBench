"""Evaluate LLMs on 3D-SpatialViewBench.

Sequentially queries each model on every item in dataset.json, saves raw
outputs and aggregate metrics.
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://evren-llmapi.ssyz.org.tr/v1"
MODELS = ["deepseek-v4.1-flash", "glm-5.3", "gemma-4-31b"]
LEVELS = ["Level 1", "Level 2", "Level 3"]

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"
RAW_PATH = RESULTS_DIR / "raw_predictions.json"
METRICS_PATH = RESULTS_DIR / "summary_metrics.csv"

SLEEP_BETWEEN_CALLS = 1.5
MAX_RETRIES = 5
RETRY_BACKOFF = 5.0
MAX_RETRY_WAIT = 60.0
RETRY_STATUS_CODES = {429, 503}

SYSTEM_PROMPT = (
    "You are an expert in 3D spatial reasoning. You will be given a scene "
    "description and a multiple-choice question about it.\n"
    "Work carefully through the geometry, then respond with ONLY a JSON object "
    "(no markdown fences, no extra text) with exactly these keys:\n"
    '  "selected_option": one of "A", "B", "C", "D"\n'
    '  "reasoning": a concise step-by-step explanation of how you arrived at the answer.\n'
    "Example: {\"selected_option\": \"A\", \"reasoning\": \"...\"}"
)


def build_user_prompt(item):
    options = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    return (
        f"Scene description:\n{item['scene_description']}\n\n"
        f"Question:\n{item['question']}\n\n"
        f"Options:\n{options}\n\n"
        "Respond with the JSON object only."
    )


def parse_response(text):
    """Return (selected_option, reasoning) or None if not valid JSON."""
    if text is None:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    option = obj.get("selected_option")
    if not isinstance(option, str):
        return None
    option = option.strip().upper()
    if option not in {"A", "B", "C", "D"}:
        return None
    return option, obj.get("reasoning")


def call_model(client, model, user_prompt):
    """Call the API with retry on 429/503. Returns (text, latency_seconds)."""
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            latency = time.perf_counter() - start
            return resp.choices[0].message.content, latency
        except Exception as exc:  # openai.APIStatusError and friends
            status = getattr(exc, "status_code", None)
            if status not in RETRY_STATUS_CODES or attempt == MAX_RETRIES:
                raise
            wait = min(RETRY_BACKOFF * (2 ** (attempt - 1)), MAX_RETRY_WAIT)
            print(f"    [{status}] retry {attempt}/{MAX_RETRIES - 1} in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def evaluate_model(client, model, dataset):
    records = []
    total = len(dataset)
    for idx, item in enumerate(dataset, start=1):
        print(f"[{model}] {idx}/{total} {item['id']} ...", flush=True)
        record = {
            "model": model,
            "id": item["id"],
            "difficulty": item["difficulty"],
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "raw_output": None,
            "selected_option": None,
            "reasoning": None,
            "json_compliant": False,
            "correct": False,
            "latency_seconds": None,
            "error": None,
        }
        try:
            text, latency = call_model(client, model, build_user_prompt(item))
            record["raw_output"] = text
            record["latency_seconds"] = round(latency, 3)
            parsed = parse_response(text)
            if parsed is not None:
                record["selected_option"], record["reasoning"] = parsed
                record["json_compliant"] = True
                record["correct"] = parsed[0] == item["correct_answer"]
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    ERROR: {record['error']}", flush=True)
        records.append(record)
        if idx < total or model != MODELS[-1]:
            time.sleep(SLEEP_BETWEEN_CALLS)
    return records


def compute_metrics(model, records):
    n = len(records)
    acc = sum(r["correct"] for r in records) / n if n else 0.0
    compliant = sum(r["json_compliant"] for r in records) / n if n else 0.0
    latencies = [r["latency_seconds"] for r in records if r["latency_seconds"] is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    row = {
        "model": model,
        "Overall Accuracy": round(acc, 4),
        "JSON Compliance %": round(compliant * 100, 2),
        "Avg Latency (s)": round(avg_latency, 3),
        "Errors": sum(1 for r in records if r["error"]),
    }
    for level in LEVELS:
        subset = [r for r in records if r["difficulty"] == level]
        row[f"{level} Accuracy"] = (
            round(sum(r["correct"] for r in subset) / len(subset), 4) if subset else 0.0
        )
    return row


def main():
    api_key = os.environ.get("EVREN_API_KEY")
    if not api_key:
        sys.exit("EVREN_API_KEY environment variable is not set.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} items; evaluating {len(MODELS)} models.")

    all_records = {}
    for model in MODELS:
        all_records[model] = evaluate_model(client, model, dataset)
        with open(RAW_PATH, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "model",
        "Overall Accuracy",
        "Level 1 Accuracy",
        "Level 2 Accuracy",
        "Level 3 Accuracy",
        "JSON Compliance %",
        "Avg Latency (s)",
        "Errors",
    ]
    metrics = [compute_metrics(m, all_records[m]) for m in MODELS]

    with open(METRICS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    print("\nSummary:")
    for row in metrics:
        print(row)
    print(f"\nRaw outputs: {RAW_PATH}\nMetrics:     {METRICS_PATH}")


if __name__ == "__main__":
    main()
