"""
Hyperparameter sweep for CK3 modding LLM fine-tuning experiments.
Runs train.py with different configs, logs results, keeps the best.

General-purpose — works with any train.py that reads CK3_TRAIN_OVERRIDES
env var and outputs metrics.json.

Usage:
    python sweep.py                     # run default grid
    python sweep.py --dry-run           # preview experiments
    python sweep.py --grid grid.json    # custom grid from file
    python sweep.py --max-experiments 3 # limit runs (for testing)
    python sweep.py --time-budget 600   # 10 min per experiment
"""

import os
import sys
import json
import time
import shutil
import signal
import argparse
import subprocess
import itertools
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Default grid — sensible ranges for CK3 fine-tuning on 20GB VRAM
# ---------------------------------------------------------------------------

DEFAULT_GRID = {
    "LEARNING_RATE": [1e-4, 2e-4, 5e-4],
    "CLM_MIX_RATIO": [0.0, 0.1, 0.2, 0.3],
    "GRADIENT_ACCUMULATION": [4, 8],
    "NUM_EPOCHS": [1, 3],
    "LORA_DROPOUT": [0.0, 0.05],
}
# 3 × 4 × 2 × 2 × 2 = 96 combos × 5 min = ~8 hours

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
TRAIN_SCRIPT = SCRIPT_DIR / "train.py"
CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "ck3_modding_llm_autotune"
OUTPUT_DIR = CACHE_DIR / "output"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
ADAPTER_DIR = OUTPUT_DIR / "lora_adapter"
BEST_DIR = OUTPUT_DIR / "best"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_stop_requested = False


def _signal_handler(sig, frame):
    """Handle Ctrl+C gracefully — finish current experiment, then stop."""
    global _stop_requested
    if _stop_requested:
        print("\n\nForce quit.")
        sys.exit(1)
    _stop_requested = True
    print("\n\nStopping after current experiment... (Ctrl+C again to force)")


signal.signal(signal.SIGINT, _signal_handler)


def generate_combos(grid: dict) -> list[dict]:
    """Generate all combinations from a grid of lists."""
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def run_single_experiment(overrides: dict, time_budget: int = 300) -> dict:
    """
    Run a single training experiment with config overrides.
    Returns metrics dict or error dict.
    """
    env = os.environ.copy()
    # Pass overrides + time budget
    full_overrides = {**overrides, "TIME_BUDGET": time_budget}
    env["CK3_TRAIN_OVERRIDES"] = json.dumps(full_overrides)

    cmd = [sys.executable, str(TRAIN_SCRIPT)]

    print(f"\n  Running: {json.dumps(overrides, indent=None)}")
    t_start = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=time_budget + 600,  # generous timeout (budget + 10 min for eval)
        )
        elapsed = time.time() - t_start

        # Try to read metrics
        if METRICS_FILE.exists():
            with open(METRICS_FILE, "r") as f:
                metrics = json.load(f)
            metrics["elapsed_s"] = round(elapsed, 1)
            metrics["status"] = "ok"

            # Print key results
            vs = metrics.get("val_score", 0)
            print(f"  val_score: {vs:.6f}  ({elapsed:.0f}s)")
            return metrics
        else:
            print(f"  FAILED: no metrics.json produced ({elapsed:.0f}s)")
            # Try to get error from stderr
            stderr_tail = result.stderr[-500:] if result.stderr else "no stderr"
            return {
                "val_score": 0.0,
                "status": "error",
                "error": stderr_tail,
                "elapsed_s": round(elapsed, 1),
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t_start
        print(f"  TIMEOUT after {elapsed:.0f}s")
        return {
            "val_score": 0.0,
            "status": "timeout",
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"  ERROR: {e}")
        return {
            "val_score": 0.0,
            "status": "error",
            "error": str(e),
            "elapsed_s": round(elapsed, 1),
        }


def append_results(results_file: Path, overrides: dict, metrics: dict):
    """Append a row to results.tsv."""
    write_header = not results_file.exists()

    with open(results_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("timestamp\tval_score\tsyntax\tkeyword\tstructure\tstatus\telapsed_s\tconfig\n")

        row = "\t".join([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            f"{metrics.get('val_score', 0.0):.6f}",
            f"{metrics.get('syntax_score', 0.0):.6f}",
            f"{metrics.get('keyword_score', 0.0):.6f}",
            f"{metrics.get('structure_score', 0.0):.6f}",
            metrics.get("status", "unknown"),
            f"{metrics.get('elapsed_s', 0):.0f}",
            json.dumps(overrides, separators=(",", ":")),
        ])
        f.write(row + "\n")


def save_best(best_score: float, current_score: float) -> bool:
    """If current score beats best, copy adapter to best/ dir. Returns True if new best."""
    if current_score <= best_score:
        return False

    if not ADAPTER_DIR.exists():
        return False

    BEST_DIR.mkdir(parents=True, exist_ok=True)

    # Copy adapter files
    for f in ADAPTER_DIR.iterdir():
        shutil.copy2(f, BEST_DIR / f.name)

    print(f"  New best! {current_score:.6f} > {best_score:.6f} — saved to {BEST_DIR}")
    return True


def print_summary(all_results: list[dict]):
    """Print a summary table of all experiments."""
    if not all_results:
        print("\nNo experiments completed.")
        return

    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)

    # Sort by val_score descending
    sorted_results = sorted(all_results, key=lambda r: r.get("val_score", 0), reverse=True)

    # Header
    print(f"{'#':>3}  {'val_score':>10}  {'syntax':>8}  {'keyword':>8}  {'struct':>8}  {'time':>6}  config")
    print("-" * 80)

    for i, r in enumerate(sorted_results, 1):
        config_str = json.dumps(r.get("overrides", {}), separators=(",", ":"))
        if len(config_str) > 40:
            config_str = config_str[:37] + "..."
        print(
            f"{i:>3}  "
            f"{r.get('val_score', 0):>10.6f}  "
            f"{r.get('syntax_score', 0):>8.4f}  "
            f"{r.get('keyword_score', 0):>8.4f}  "
            f"{r.get('structure_score', 0):>8.4f}  "
            f"{r.get('elapsed_s', 0):>5.0f}s  "
            f"{config_str}"
        )

    best = sorted_results[0]
    print(f"\nBest: val_score={best.get('val_score', 0):.6f}")
    print(f"Config: {json.dumps(best.get('overrides', {}), indent=2)}")

    if BEST_DIR.exists():
        print(f"Best adapter saved at: {BEST_DIR}")


def load_grid(grid_path: str) -> dict:
    """Load grid from JSON file."""
    with open(grid_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sweep(grid: dict, time_budget: int = 300, max_experiments: int = 0,
          results_file: Path = None, dry_run: bool = False) -> list[dict]:
    """
    Run a full grid sweep. Returns list of all experiment results.
    This function is also called by the MCP server.
    """
    if results_file is None:
        results_file = SCRIPT_DIR / "results.tsv"

    combos = generate_combos(grid)
    if max_experiments > 0:
        combos = combos[:max_experiments]

    total = len(combos)
    print(f"\nSweep: {total} experiments, ~{total * time_budget // 60} min estimated")
    print(f"Grid: {json.dumps(grid, indent=2)}")
    print(f"Results: {results_file}")

    if dry_run:
        print(f"\n{'#':>3}  config")
        print("-" * 60)
        for i, combo in enumerate(combos, 1):
            print(f"{i:>3}  {json.dumps(combo, separators=(',', ':'))}")
        print(f"\nDry run — no experiments executed.")
        return []

    all_results = []
    best_score = 0.0

    for i, combo in enumerate(combos, 1):
        if _stop_requested:
            print(f"\nStopped by user after {i-1}/{total} experiments.")
            break

        print(f"\n{'='*60}")
        print(f"Experiment {i}/{total}")
        print(f"{'='*60}")

        metrics = run_single_experiment(combo, time_budget=time_budget)
        metrics["overrides"] = combo

        # Log results
        append_results(results_file, combo, metrics)
        all_results.append(metrics)

        # Track best
        score = metrics.get("val_score", 0.0)
        if save_best(best_score, score):
            best_score = score

    print_summary(all_results)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for CK3 LLM AutoTune")
    parser.add_argument("--dry-run", action="store_true", help="Preview experiments without running")
    parser.add_argument("--grid", type=str, help="Path to grid JSON file (default: built-in grid)")
    parser.add_argument("--max-experiments", type=int, default=0, help="Max experiments to run (0=all)")
    parser.add_argument("--time-budget", type=int, default=300, help="Seconds per experiment (default 300)")
    parser.add_argument("--results", type=str, default=None, help="Results file path (default: results.tsv)")
    args = parser.parse_args()

    grid = load_grid(args.grid) if args.grid else DEFAULT_GRID
    results_file = Path(args.results) if args.results else SCRIPT_DIR / "results.tsv"

    sweep(
        grid=grid,
        time_budget=args.time_budget,
        max_experiments=args.max_experiments,
        results_file=results_file,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
