"""
Sweep MCP Server — exposes hyperparameter sweep tools for Claude Code.

General-purpose training sweep server. Works with any project that has
a train.py reading CK3_TRAIN_OVERRIDES and outputting metrics.json.

Register in Claude Code settings:
{
  "mcpServers": {
    "ck3-sweep": {
      "command": "python",
      "args": ["sweep_mcp/server.py"]
    }
  }
}
"""

import os
import sys
import json
from pathlib import Path

from fastmcp import FastMCP

# Add parent dir to path so we can import sweep
sys.path.insert(0, str(Path(__file__).parent.parent))
from sweep import (
    sweep as run_sweep_func,
    run_single_experiment,
    generate_combos,
    DEFAULT_GRID,
    BEST_DIR,
    OUTPUT_DIR,
    SCRIPT_DIR,
)

mcp = FastMCP("ck3-sweep")

RESULTS_FILE = SCRIPT_DIR / "results.tsv"


@mcp.tool()
def run_sweep(
    grid: dict = None,
    time_budget: int = 300,
    max_experiments: int = 0,
) -> str:
    """
    Run a grid search over hyperparameter combinations.
    Each experiment trains for time_budget seconds, evaluates, and logs results.

    Args:
        grid: Dict of param_name -> list of values. Default grid used if not provided.
              Example: {"LEARNING_RATE": [1e-4, 2e-4], "CLM_MIX_RATIO": [0.1, 0.2]}
        time_budget: Seconds per experiment (default 300 = 5 min).
        max_experiments: Max experiments to run (0 = all combos).

    Returns:
        JSON string with all experiment results sorted by val_score.
    """
    if grid is None:
        grid = DEFAULT_GRID

    results = run_sweep_func(
        grid=grid,
        time_budget=time_budget,
        max_experiments=max_experiments,
        results_file=RESULTS_FILE,
    )

    # Sort by val_score descending
    results.sort(key=lambda r: r.get("val_score", 0), reverse=True)

    summary = {
        "total_experiments": len(results),
        "best_val_score": results[0].get("val_score", 0) if results else 0,
        "best_config": results[0].get("overrides", {}) if results else {},
        "results": results,
    }
    return json.dumps(summary, indent=2)


@mcp.tool()
def run_experiment(overrides: dict, time_budget: int = 300) -> str:
    """
    Run a single training experiment with specific config overrides.

    Args:
        overrides: Config overrides. Example: {"LEARNING_RATE": 1e-4, "CLM_MIX_RATIO": 0.2}
        time_budget: Seconds for training (default 300 = 5 min).

    Returns:
        JSON string with experiment metrics (val_score, syntax_score, etc.).
    """
    from sweep import append_results

    metrics = run_single_experiment(overrides, time_budget=time_budget)
    metrics["overrides"] = overrides

    # Log to results.tsv
    append_results(RESULTS_FILE, overrides, metrics)

    return json.dumps(metrics, indent=2)


@mcp.tool()
def get_results(sort_by: str = "val_score", limit: int = 50) -> str:
    """
    Read results.tsv and return experiment history.

    Args:
        sort_by: Column to sort by (default: val_score). Options: val_score, syntax, keyword, structure, elapsed_s.
        limit: Max rows to return (default 50).

    Returns:
        JSON string with sorted results table.
    """
    if not RESULTS_FILE.exists():
        return json.dumps({"error": "No results.tsv found. Run some experiments first."})

    rows = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= len(header):
                row = dict(zip(header, parts))
                # Parse numeric fields
                for field in ["val_score", "syntax", "keyword", "structure", "elapsed_s"]:
                    if field in row:
                        try:
                            row[field] = float(row[field])
                        except ValueError:
                            pass
                rows.append(row)

    # Sort
    sort_col = sort_by if sort_by in header else "val_score"
    rows.sort(key=lambda r: r.get(sort_col, 0), reverse=True)

    if limit > 0:
        rows = rows[:limit]

    return json.dumps({
        "total_rows": len(rows),
        "sort_by": sort_col,
        "results": rows,
    }, indent=2)


@mcp.tool()
def get_best_config() -> str:
    """
    Return the configuration that produced the highest val_score.

    Returns:
        JSON string with best config, score, and adapter path.
    """
    if not RESULTS_FILE.exists():
        return json.dumps({"error": "No results.tsv found."})

    best_score = 0.0
    best_config = {}

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        config_idx = header.index("config") if "config" in header else -1
        score_idx = header.index("val_score") if "val_score" in header else -1

        if score_idx < 0 or config_idx < 0:
            return json.dumps({"error": "results.tsv has unexpected format"})

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(score_idx, config_idx):
                try:
                    score = float(parts[score_idx])
                    config = json.loads(parts[config_idx])
                    if score > best_score:
                        best_score = score
                        best_config = config
                except (ValueError, json.JSONDecodeError):
                    continue

    return json.dumps({
        "best_val_score": best_score,
        "best_config": best_config,
        "adapter_path": str(BEST_DIR) if BEST_DIR.exists() else None,
    }, indent=2)


@mcp.tool()
def get_status() -> str:
    """
    Check current status — is a sweep running, what results exist.

    Returns:
        JSON string with status info.
    """
    # Count results
    num_results = 0
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            num_results = sum(1 for _ in f) - 1  # subtract header

    has_adapter = BEST_DIR.exists() and any(BEST_DIR.iterdir()) if BEST_DIR.exists() else False

    return json.dumps({
        "results_file": str(RESULTS_FILE),
        "num_experiments": max(0, num_results),
        "has_best_adapter": has_adapter,
        "best_adapter_path": str(BEST_DIR) if has_adapter else None,
        "output_dir": str(OUTPUT_DIR),
    }, indent=2)


@mcp.tool()
def resume_best(time_budget: int = 28800) -> str:
    """
    Run extended training on the best configuration found.
    Uses --long --resume to continue from the best adapter.

    Args:
        time_budget: Training duration in seconds (default 28800 = 8 hours).

    Returns:
        JSON string with final metrics after extended training.
    """
    import subprocess
    import shutil

    # First, restore best adapter as the active one
    if BEST_DIR.exists():
        adapter_dir = OUTPUT_DIR / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        for f in BEST_DIR.iterdir():
            shutil.copy2(f, adapter_dir / f.name)
        print(f"Restored best adapter from {BEST_DIR}")

    # Get best config
    best_config = {}
    if RESULTS_FILE.exists():
        best_score = 0.0
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            config_idx = header.index("config") if "config" in header else -1
            score_idx = header.index("val_score") if "val_score" in header else -1
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > max(score_idx, config_idx):
                    try:
                        score = float(parts[score_idx])
                        config = json.loads(parts[config_idx])
                        if score > best_score:
                            best_score = score
                            best_config = config
                    except (ValueError, json.JSONDecodeError):
                        continue

    # Run with best config + long + resume
    env = os.environ.copy()
    overrides = {**best_config, "TIME_BUDGET": time_budget}
    env["CK3_TRAIN_OVERRIDES"] = json.dumps(overrides)

    cmd = [sys.executable, str(SCRIPT_DIR / "train.py"), "--resume"]

    print(f"Starting extended training ({time_budget}s) with config: {best_config}")
    result = subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR), capture_output=True, text=True)

    # Read final metrics
    metrics_file = OUTPUT_DIR / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        return json.dumps({"status": "completed", "metrics": metrics}, indent=2)
    else:
        return json.dumps({
            "status": "error",
            "stderr": result.stderr[-500:] if result.stderr else "unknown error",
        }, indent=2)


if __name__ == "__main__":
    mcp.run()
