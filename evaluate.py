"""
CK3 Modding LLM AutoTune — Evaluation Script (DO NOT MODIFY)
======================================================
Queries LM Studio's API with CK3 modding prompts, then scores the output.
This is the ground truth metric for the research loop.

Usage:
    uv run evaluate.py                  # run full evaluation
    uv run evaluate.py --quick          # run 3 prompts only
    uv run evaluate.py --compare A B    # compare two sample files

The metric is val_score (0-1, higher is better).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

from prepare import (
    EVAL_PROMPTS, LMSTUDIO_BASE_URL, CACHE_DIR,
    evaluate_all_outputs, evaluate_output,
)

OUTPUT_DIR = os.path.join(CACHE_DIR, "output")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")

SYSTEM_MSG = (
    "You are a CK3 modding expert. Write valid Crusader Kings III script code. "
    "Use correct scope chains, proper trigger/effect placement, and follow vanilla patterns. "
    "Only output the script code, no explanations."
)


def query_lmstudio(prompts: list[dict], temperature: float = 0.7) -> dict[str, str]:
    """Query LM Studio API and return outputs keyed by prompt ID."""
    from openai import OpenAI
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")

    outputs = {}
    for prompt in prompts:
        pid = prompt["id"]
        print(f"  [{pid}] Generating...", end=" ", flush=True)
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model="loaded-model",
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt["instruction"]},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            generated = response.choices[0].message.content
            dt = time.time() - t0
            outputs[pid] = generated
            print(f"{len(generated)} chars in {dt:.1f}s")
        except Exception as e:
            print(f"FAILED: {e}")
            outputs[pid] = ""

    return outputs


def run_evaluation(quick: bool = False) -> dict:
    """Run full evaluation and save results."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    prompts = EVAL_PROMPTS[:3] if quick else EVAL_PROMPTS

    print("=" * 60)
    print(f"CK3 Modding LLM AutoTune - Evaluation ({'quick' if quick else 'full'})")
    print(f"  Prompts: {len(prompts)}")
    print(f"  API:     {LMSTUDIO_BASE_URL}")
    print("=" * 60)
    print()

    outputs = query_lmstudio(prompts)
    metrics = evaluate_all_outputs(outputs)

    # Save
    timestamp = int(time.time())
    result = {
        "timestamp": timestamp,
        "metrics": metrics,
        "outputs": outputs,
    }
    result_file = os.path.join(SAMPLES_DIR, f"eval_{timestamp}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print results
    print()
    print("=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  val_score:       {metrics.get('val_score', 0):.6f}")
    print(f"  syntax_score:    {metrics.get('syntax_score', 0):.6f}")
    print(f"  keyword_score:   {metrics.get('keyword_score', 0):.6f}")
    print(f"  structure_score: {metrics.get('structure_score', 0):.6f}")
    print()

    if "per_prompt" in metrics:
        print("  Per-prompt breakdown:")
        for r in metrics["per_prompt"]:
            status = "PASS" if r["total_score"] >= 0.7 else "WARN" if r["total_score"] >= 0.4 else "FAIL"
            print(f"    {r['prompt_id']:30s}  {r['total_score']:.3f}  [{status}]")

    print()
    print(f"  Saved to: {result_file}")

    # Print summary line (for grep in training loop)
    print()
    print("---")
    print(f"val_score: {metrics.get('val_score', 0):.6f}")

    return metrics


def compare_results(file_a: str, file_b: str):
    """Compare two evaluation result files."""
    with open(file_a, 'r') as f:
        a = json.load(f)
    with open(file_b, 'r') as f:
        b = json.load(f)

    ma = a.get("metrics", a)
    mb = b.get("metrics", b)

    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"  {'Metric':20s}  {'A':>10s}  {'B':>10s}  {'Delta':>10s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")

    for key in ["val_score", "syntax_score", "keyword_score", "structure_score"]:
        va = ma.get(key, 0)
        vb = mb.get(key, 0)
        delta = vb - va
        arrow = "+" if delta > 0 else ""
        print(f"  {key:20s}  {va:>10.4f}  {vb:>10.4f}  {arrow}{delta:>9.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CK3 Modding LLM AutoTune - Evaluation")
    parser.add_argument("--quick", action="store_true", help="Run only 3 eval prompts")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two result files")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        run_evaluation(quick=args.quick)
