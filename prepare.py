"""
CK3 Modding LLM AutoTune — Data Preparation (DO NOT MODIFY)
=====================================================
Extracts all vanilla CK3 script files into structured training data.
Trains a BPE tokenizer on CK3 script syntax.
Provides evaluation utilities for train.py.

Usage:
    python prepare.py                    # full prep (extract + tokenizer)
    python prepare.py --extract-only     # just extract data
    python prepare.py --stats            # print dataset statistics

Data is stored in ~/.cache/ck3_modding_llm_autotune/.
"""

import os
import sys
import re
import json
import random
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 8192          # context length (matches LM Studio config)
TIME_BUDGET = 300           # training time budget in seconds (5 minutes)
VAL_FRACTION = 0.05         # fraction of data held out for validation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VANILLA_PATH = Path("D:/SteamLibrary/steamapps/common/Crusader Kings III/game")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ck3_modding_llm_autotune")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# LM Studio API (OpenAI-compatible)
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

# Content categories and their directory patterns
CONTENT_TYPES = {
    "event": ["events"],
    "decision": ["common/decisions"],
    "interaction": ["common/character_interactions"],
    "on_action": ["common/on_action"],
    "scripted_trigger": ["common/scripted_triggers"],
    "scripted_effect": ["common/scripted_effects"],
    "trait": ["common/traits"],
    "modifier": ["common/modifiers", "common/event_modifiers", "common/static_modifiers"],
    "casus_belli": ["common/casus_belli_types"],
    "culture": ["common/culture/cultures"],
    "faith": ["common/religion/religions"],
    "scheme": ["common/schemes"],
    "activity": ["common/activities/activity_types"],
    "council_task": ["common/council_tasks"],
    "lifestyle": ["common/lifestyles"],
    "law": ["common/laws"],
    "building": ["common/buildings"],
    "men_at_arms": ["common/men_at_arms_types"],
    "court_position": ["common/court_positions"],
    "dynasty_legacy": ["common/dynasty_legacies"],
    "innovation": ["common/culture/innovations"],
    "perk": ["common/lifestyle_perks"],
}

# Directories to skip entirely
SKIP_DIRS = {"gfx", "gui", "fonts", "music", "sound", "dlc", "tools",
             "content_source", "map_data", "common/coat_of_arms",
             "common/genes", "common/portrait_modifiers"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_comments(text: str) -> str:
    """Remove CK3 script comments (# to end of line), preserving quoted strings."""
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            in_string = not in_string
            result.append(ch)
        elif ch == '#' and not in_string:
            while i < len(text) and text[i] != '\n':
                i += 1
            continue
        else:
            result.append(ch)
        i += 1
    return ''.join(result)


def extract_top_level_blocks(text: str) -> list[tuple[str, str, str]]:
    """
    Extract top-level named blocks from CK3 script text.
    Returns list of (block_name, block_text, comment_header).
    comment_header = any comments immediately above the block.
    """
    blocks = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'^(\w[\w.]*)\s*=\s*\{', line)
        if m:
            name = m.group(1)
            # Gather comment header (lines above starting with #)
            header_lines = []
            j = i - 1
            while j >= 0 and lines[j].strip().startswith('#'):
                header_lines.insert(0, lines[j])
                j -= 1

            depth = line.count('{') - line.count('}')
            block_lines = [lines[i]]
            i += 1
            while i < len(lines) and depth > 0:
                depth += lines[i].count('{') - lines[i].count('}')
                block_lines.append(lines[i])
                i += 1
            comment_header = '\n'.join(header_lines) if header_lines else ""
            blocks.append((name, '\n'.join(block_lines), comment_header))
        else:
            i += 1
    return blocks


def relative_path(file_path: Path) -> str:
    try:
        return str(file_path.relative_to(VANILLA_PATH)).replace('\\', '/')
    except ValueError:
        return str(file_path)


def categorize_path(rel_path: str) -> str:
    """Determine content category from relative path."""
    for cat, dirs in CONTENT_TYPES.items():
        for d in dirs:
            if rel_path.startswith(d):
                return cat
    return "other"


# ---------------------------------------------------------------------------
# Localization Loader
# ---------------------------------------------------------------------------

def load_localization(lang: str = "english") -> dict[str, str]:
    """Load all localization strings into a dict."""
    loc_dir = VANILLA_PATH / "localization" / lang
    loc_map = {}
    if not loc_dir.exists():
        print(f"  Warning: localization dir not found: {loc_dir}")
        return loc_map

    for yml_file in loc_dir.rglob("*.yml"):
        try:
            text = yml_file.read_text(encoding='utf-8-sig', errors='replace')
        except Exception:
            continue
        for line in text.split('\n'):
            line = line.strip()
            m = re.match(r'^(\w+):\d*\s+"(.+)"', line)
            if m:
                loc_map[m.group(1)] = m.group(2)

    print(f"  Loaded {len(loc_map):,} localization keys")
    return loc_map


# ---------------------------------------------------------------------------
# Stage 1: Raw Corpus
# ---------------------------------------------------------------------------

def build_raw_corpus() -> int:
    """Extract every relevant .txt script file into raw_corpus.jsonl."""
    output_file = os.path.join(DATA_DIR, "raw_corpus.jsonl")
    count = 0

    all_files = sorted(VANILLA_PATH.rglob("*.txt"))

    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in all_files:
            rel = relative_path(file_path)
            parts = Path(rel).parts
            if any(p in SKIP_DIRS for p in parts):
                continue

            try:
                text = file_path.read_text(encoding='utf-8-sig', errors='replace')
            except Exception as e:
                print(f"  Skipping {rel}: {e}")
                continue

            if len(text.strip()) == 0:
                continue

            category = categorize_path(rel)
            record = {
                "text": text,
                "path": rel,
                "category": category,
                "lines": text.count('\n') + 1,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1

    print(f"  Extracted {count:,} files -> {output_file}")
    return count


# ---------------------------------------------------------------------------
# Stage 2: Structured Blocks
# ---------------------------------------------------------------------------

def build_structured_blocks() -> int:
    """Parse top-level blocks from categorized script files."""
    output_file = os.path.join(DATA_DIR, "structured_blocks.jsonl")
    count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for category, dirs in CONTENT_TYPES.items():
            for subdir in dirs:
                dir_path = VANILLA_PATH / subdir
                if not dir_path.exists():
                    continue
                for file_path in sorted(dir_path.rglob("*.txt")):
                    try:
                        text = file_path.read_text(encoding='utf-8-sig', errors='replace')
                    except Exception:
                        continue

                    clean = strip_comments(text)
                    blocks = extract_top_level_blocks(clean)

                    for name, block_text, comment_header in blocks:
                        record = {
                            "name": name,
                            "category": category,
                            "source_file": relative_path(file_path),
                            "code": block_text,
                            "comment_header": comment_header,
                            "lines": block_text.count('\n') + 1,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        count += 1

    print(f"  Extracted {count:,} blocks -> {output_file}")
    return count


# ---------------------------------------------------------------------------
# Stage 3: Instruction Pairs (for SFT)
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATES = {
    "event": [
        "Write a CK3 event called '{name}'",
        "Create the CK3 event '{name}' that {loc_desc}",
        "Write CK3 script for event '{name}'",
    ],
    "decision": [
        "Write a CK3 decision called '{name}'",
        "Create a CK3 decision '{name}' that {loc_desc}",
    ],
    "interaction": [
        "Write a CK3 character interaction called '{name}'",
        "Create a CK3 interaction '{name}'",
    ],
    "scripted_trigger": [
        "Write a CK3 scripted trigger called '{name}'",
    ],
    "scripted_effect": [
        "Write a CK3 scripted effect called '{name}'",
    ],
    "trait": [
        "Define a CK3 trait called '{name}'",
        "Write the CK3 trait definition for '{name}' that {loc_desc}",
    ],
    "on_action": [
        "Write a CK3 on_action hook called '{name}'",
    ],
}

DEFAULT_TEMPLATES = [
    "Write the CK3 script block '{name}'",
    "Create a CK3 {category} called '{name}'",
]


def build_instruction_pairs() -> int:
    """Generate (instruction, completion) pairs for SFT training."""
    blocks_file = os.path.join(DATA_DIR, "structured_blocks.jsonl")
    output_file = os.path.join(DATA_DIR, "instruction_pairs.jsonl")

    if not os.path.exists(blocks_file):
        print("  Error: structured_blocks.jsonl not found. Run full extraction first.")
        return 0

    print("  Loading localization...")
    loc_map = load_localization()

    count = 0
    random.seed(42)

    with open(blocks_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            block = json.loads(line)
            name = block["name"]
            category = block["category"]
            code = block["code"]

            if block["lines"] < 3:
                continue

            # Try to find localization description
            loc_desc = ""
            for suffix in ["", "_desc", "_tooltip", "_title"]:
                key = name + suffix
                if key in loc_map:
                    loc_desc = loc_map[key]
                    loc_desc = re.sub(r'#\w\s*', '', loc_desc)
                    loc_desc = re.sub(r'\[.*?\]', '', loc_desc)
                    loc_desc = loc_desc.strip()
                    if loc_desc:
                        break
            if not loc_desc:
                loc_desc = f"is a {category} definition"

            templates = INSTRUCTION_TEMPLATES.get(category, DEFAULT_TEMPLATES)
            template = random.choice(templates)
            instruction = template.format(name=name, category=category, loc_desc=loc_desc)

            system_msg = (
                "You are a CK3 modding expert. Write valid Crusader Kings III script code. "
                "Use correct scope chains, proper trigger/effect placement, and follow vanilla patterns. "
                "Only output the script code, no explanations."
            )

            record = {
                "system": system_msg,
                "instruction": instruction,
                "completion": code,
                "category": category,
                "source_file": block["source_file"],
                "name": name,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1

    print(f"  Generated {count:,} instruction pairs -> {output_file}")
    return count


# ---------------------------------------------------------------------------
# Stage 4: Continued Pretraining data (raw CK3 text for CLM)
# ---------------------------------------------------------------------------

def build_clm_dataset() -> int:
    """
    Build a dataset for continued pretraining (causal language modeling).
    Each document is a complete CK3 script file, formatted as plain text.
    This teaches the model CK3 syntax at a fundamental level.
    """
    raw_file = os.path.join(DATA_DIR, "raw_corpus.jsonl")
    output_file = os.path.join(DATA_DIR, "clm_dataset.jsonl")

    if not os.path.exists(raw_file):
        print("  Error: raw_corpus.jsonl not found. Run full extraction first.")
        return 0

    count = 0
    with open(raw_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            rec = json.loads(line)
            # Add file path as context header
            text = f"# File: {rec['path']}\n# Category: {rec['category']}\n\n{rec['text']}"
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
            count += 1

    print(f"  Generated {count:,} CLM documents -> {output_file}")
    return count


# ---------------------------------------------------------------------------
# Stage 5: Localization pairs
# ---------------------------------------------------------------------------

def build_loc_pairs() -> int:
    """Generate instruction pairs for localization patterns."""
    output_file = os.path.join(DATA_DIR, "loc_pairs.jsonl")
    loc_map = load_localization()
    count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        prefixes: dict[str, list[tuple[str, str]]] = {}
        for key, value in loc_map.items():
            parts = key.rsplit('_', 1)
            prefix = parts[0] if len(parts) > 1 else key
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append((key, value))

        for prefix, entries in prefixes.items():
            if len(entries) < 2 or len(entries) > 20:
                continue

            loc_block = "\n".join(f'  {k}:0 "{v}"' for k, v in sorted(entries))
            record = {
                "system": "You are a CK3 modding expert. Write valid CK3 localization in YML format.",
                "instruction": f"Write CK3 localization entries for '{prefix}'",
                "completion": f"l_english:\n{loc_block}",
                "category": "localization",
                "name": prefix,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1

    print(f"  Generated {count:,} loc pairs -> {output_file}")
    return count


# ---------------------------------------------------------------------------
# Evaluation Prompts (fixed benchmark — DO NOT CHANGE)
# ---------------------------------------------------------------------------

EVAL_PROMPTS = [
    {
        "id": "eval_event_basic",
        "instruction": "Write a CK3 event where the player character discovers a hidden treasure map in their court. Include a title, description, trigger block, and two options.",
        "category": "event",
        "checks": ["title", "desc", "trigger", "option", "=", "{", "}"],
    },
    {
        "id": "eval_decision",
        "instruction": "Write a CK3 decision that lets a ruler host a grand tournament. Include is_shown, is_valid, and effect blocks.",
        "category": "decision",
        "checks": ["is_shown", "is_valid", "effect", "=", "{", "}"],
    },
    {
        "id": "eval_scripted_trigger",
        "instruction": "Write a CK3 scripted trigger called 'is_valid_knight_candidate' that checks if a character is an adult, not imprisoned, has prowess >= 8, and is not incapable.",
        "category": "scripted_trigger",
        "checks": ["is_adult", "is_imprisoned", "prowess", "has_trait", "=", "{", "}"],
    },
    {
        "id": "eval_on_action",
        "instruction": "Write a CK3 on_action that fires when a character gains a new trait, checking if they are a ruler and triggering an event.",
        "category": "on_action",
        "checks": ["trigger", "events", "is_ruler", "=", "{", "}"],
    },
    {
        "id": "eval_interaction",
        "instruction": "Write a CK3 character interaction for challenging someone to a duel. Include is_shown, is_valid_showing_failures_only, on_accept, and on_decline blocks.",
        "category": "interaction",
        "checks": ["is_shown", "is_valid", "on_accept", "=", "{", "}"],
    },
    {
        "id": "eval_scope_chain",
        "instruction": "Write a CK3 scripted effect that saves the root character's liege's primary title holder's faith as a scope, then checks if the root character's culture has the same heritage as the scope target's culture.",
        "category": "scripted_effect",
        "checks": ["save_scope_as", "root", "liege", "primary_title", "culture", "=", "{", "}"],
    },
    {
        "id": "eval_create_character",
        "instruction": "Write a CK3 scripted effect that creates a new female character aged 20-30 with the 'brilliant_strategist' education trait, assigns her to the root character's court, and makes her a knight.",
        "category": "scripted_effect",
        "checks": ["create_character", "female", "age", "trait", "add_courtier", "=", "{", "}"],
    },
    {
        "id": "eval_loc",
        "instruction": "Write CK3 localization entries for an event called 'tournament_champion' with a title, description, and two option labels. Use proper loc key format.",
        "category": "localization",
        "checks": ["l_english:", ":0", '"'],
    },
]


def evaluate_output(output: str, prompt: dict) -> dict:
    """
    Score a generated CK3 code output against the evaluation prompt.
    Returns a dict with scores and details.

    Scoring:
    - syntax_score: 0-1, checks balanced braces and basic CK3 patterns
    - keyword_score: 0-1, fraction of expected keywords found
    - structure_score: 0-1, checks for proper block nesting
    - total_score: weighted average (syntax 0.4, keywords 0.3, structure 0.3)
    """
    # Keyword check
    checks = prompt.get("checks", [])
    found = sum(1 for kw in checks if kw in output)
    keyword_score = found / max(len(checks), 1)

    # Syntax: balanced braces
    open_braces = output.count('{')
    close_braces = output.count('}')
    brace_balance = 1.0 if open_braces == close_braces and open_braces > 0 else 0.0

    # Syntax: has = assignments
    has_assignments = 1.0 if '=' in output else 0.0

    # Syntax: no obvious errors
    has_python = 1.0 if ('def ' in output or 'import ' in output or 'class ' in output) else 0.0

    syntax_score = (brace_balance * 0.5 + has_assignments * 0.5) * (1.0 - has_python * 0.5)

    # Structure: proper nesting (at least one name = { ... } pattern)
    has_block = bool(re.search(r'\w+\s*=\s*\{', output))
    # Check for proper scope patterns
    has_scope_pattern = bool(re.search(r'(root|scope:|this|prev|trigger|effect|limit)', output))
    structure_score = (0.5 if has_block else 0.0) + (0.5 if has_scope_pattern else 0.0)

    total = syntax_score * 0.4 + keyword_score * 0.3 + structure_score * 0.3

    return {
        "prompt_id": prompt["id"],
        "syntax_score": round(syntax_score, 4),
        "keyword_score": round(keyword_score, 4),
        "structure_score": round(structure_score, 4),
        "total_score": round(total, 4),
        "brace_balanced": open_braces == close_braces,
        "keywords_found": found,
        "keywords_total": len(checks),
    }


def evaluate_all_outputs(outputs: dict[str, str]) -> dict:
    """
    Evaluate all generated outputs against the fixed benchmark.
    outputs: dict mapping prompt_id -> generated text
    Returns aggregate metrics.
    """
    results = []
    for prompt in EVAL_PROMPTS:
        pid = prompt["id"]
        if pid in outputs:
            result = evaluate_output(outputs[pid], prompt)
            results.append(result)

    if not results:
        return {"val_score": 0.0, "num_prompts": 0}

    avg_total = sum(r["total_score"] for r in results) / len(results)
    avg_syntax = sum(r["syntax_score"] for r in results) / len(results)
    avg_keyword = sum(r["keyword_score"] for r in results) / len(results)
    avg_structure = sum(r["structure_score"] for r in results) / len(results)

    return {
        "val_score": round(avg_total, 6),
        "syntax_score": round(avg_syntax, 6),
        "keyword_score": round(avg_keyword, 6),
        "structure_score": round(avg_structure, 6),
        "num_prompts": len(results),
        "per_prompt": results,
    }


# ---------------------------------------------------------------------------
# Dataset loading utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_instruction_dataset() -> tuple[list[dict], list[dict]]:
    """
    Load instruction pairs and split into train/val.
    Returns (train_records, val_records).
    """
    pairs_file = os.path.join(DATA_DIR, "instruction_pairs.jsonl")
    loc_file = os.path.join(DATA_DIR, "loc_pairs.jsonl")

    records = []
    for filepath in [pairs_file, loc_file]:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    records.append(json.loads(line))

    if not records:
        print(f"ERROR: No training data found in {DATA_DIR}. Run prepare.py first.")
        sys.exit(1)

    random.seed(42)
    random.shuffle(records)
    split_idx = int(len(records) * (1 - VAL_FRACTION))
    return records[:split_idx], records[split_idx:]


def load_clm_dataset() -> tuple[list[dict], list[dict]]:
    """Load CLM dataset and split into train/val."""
    clm_file = os.path.join(DATA_DIR, "clm_dataset.jsonl")
    if not os.path.exists(clm_file):
        return [], []

    records = []
    with open(clm_file, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    random.seed(42)
    random.shuffle(records)
    split_idx = int(len(records) * (1 - VAL_FRACTION))
    return records[:split_idx], records[split_idx:]


def format_chatml(example: dict) -> str:
    """Format an instruction pair into ChatML format (Qwen-compatible)."""
    system = example.get("system", "You are a CK3 modding expert.")
    instruction = example["instruction"]
    completion = example["completion"]
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{completion}<|im_end|>"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_stats():
    """Print dataset statistics."""
    print("\nDataset Statistics:")
    print("=" * 60)
    for name in ["raw_corpus", "structured_blocks", "instruction_pairs", "loc_pairs", "clm_dataset"]:
        filepath = os.path.join(DATA_DIR, f"{name}.jsonl")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {name:25s}: {count:>8,} records ({size_mb:>7.1f} MB)")
        else:
            print(f"  {name:25s}: not found")

    # Category breakdown
    blocks_file = os.path.join(DATA_DIR, "structured_blocks.jsonl")
    if os.path.exists(blocks_file):
        cats = {}
        with open(blocks_file, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                cat = rec["category"]
                cats[cat] = cats.get(cat, 0) + 1
        print("\n  Category breakdown:")
        for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"    {cat:25s}: {n:>6,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CK3 Modding LLM AutoTune - Data Preparation")
    parser.add_argument("--extract-only", action="store_true", help="Only extract raw corpus")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        sys.exit(0)

    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("CK3 Modding LLM AutoTune - Data Preparation")
    print("=" * 60)
    print(f"Vanilla path: {VANILLA_PATH}")
    print(f"Cache dir:    {CACHE_DIR}")
    print()

    if not VANILLA_PATH.exists():
        print(f"ERROR: Vanilla game path not found: {VANILLA_PATH}")
        sys.exit(1)

    print("[1/5] Extracting raw corpus...")
    build_raw_corpus()
    print()

    print("[2/5] Extracting structured blocks...")
    build_structured_blocks()
    print()

    print("[3/5] Building instruction pairs...")
    build_instruction_pairs()
    print()

    print("[4/5] Building CLM dataset...")
    build_clm_dataset()
    print()

    print("[5/5] Building localization pairs...")
    build_loc_pairs()
    print()

    print_stats()
    print()
    print("Done! Ready to train.")
