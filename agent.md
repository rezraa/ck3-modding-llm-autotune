# CK3 Modding LLM AutoTune

This is an experiment to train a local LLM to natively understand CK3 modding.

## Goal

Fine-tune a local model so that CK3 scripting is as natural to it as Python or C++.
The model must understand ALL aspects of CK3 modding at equal footing — no content type
is more important than another. The goal is nuanced, complete understanding:

- Scope chains and transitions (root, prev, scope:, dot-chains)
- Trigger vs effect placement (any_ for triggers, every_/random_/ordered_ for effects)
- Events (namespaced IDs, title/desc/trigger/option blocks, hidden events, event chains)
- Decisions (is_shown, is_valid, cost, effect, ai_check_interval)
- Character interactions (is_shown, is_valid, on_accept, on_decline, auto_accept)
- On_actions (trigger blocks, event lists, weight/multiplier)
- Scripted triggers and effects (reusable blocks, parameter passing)
- Traits (opposites, modifiers, flags, inheritance)
- Localization (YML format, [scope.GetFunction] syntax, NOT [scope:X])
- create_character (dynasty vs dynasty_house, employer vs location+add_courtier)
- Modifiers, cultures, faiths, CBs, schemes, activities, buildings, MaA, laws
- Proper file structure and naming conventions

All of this without needing RAG or reference docs — it should just know.

## Strategy

The sweep grid handles ALL config-expressible dimensions automatically (~96 combos overnight).
You (the agent) only intervene for **structural code changes** that can't be a grid value.

### What the sweep handles (don't touch these manually):
- Learning rate, CLM mix ratio, gradient accumulation
- Number of epochs, LoRA dropout
- Any other numerical hyperparameter added to the grid

### What you handle (structural changes):
- Data composition: add game logs, wiki data, change content type ratios
- Training strategy: SFT only vs CLM vs mixed, curriculum learning
- Prompt formatting: different system prompts, ChatML templates
- Data augmentation: generate variations, rephrase instructions
- New data sources in prepare.py

### Combined loop:
1. Read results.tsv to see what's been tried
2. Make a creative/structural change to train.py or prepare.py
3. Quick 5-min test: `python train.py` — does it crash? Reasonable score?
4. Call `run_sweep(grid)` to optimize numerical knobs for the new structure
5. Analyze results — did the structural change help?
6. Keep or discard, repeat

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar8`). The branch `experiment/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b experiment/<tag>` from current main.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — data extraction, evaluation metrics, dataset loading.
   - `train.py` — the file you modify. Model selection, LoRA config, hyperparameters, data formatting.
   - `sweep.py` — standalone grid search. Also used by sweep MCP tools.
   - `export.py` — GGUF export (standalone, not part of training loop).
4. **Verify data exists**: Check `~/.cache/ck3_modding_llm_autotune/data/` for extracted datasets. If not, run `python prepare.py`.
5. **Run baseline eval**: `python train.py --eval-only` to get the pre-training baseline score.
6. **Initialize results.tsv**: Create with header and baseline entry.
7. **Confirm and go**.

## Experimentation

Each experiment fine-tunes the model and evaluates it. Default training budget is **5 minutes**.

**What you CAN do:**
- Modify `train.py` — everything is fair game: base model, LoRA rank/alpha, learning rate, data mix ratio, formatting, batch size, etc.
- Modify `prepare.py` — add new data sources (game logs, wiki), change extraction logic, add data augmentation
- Use sweep MCP tools to run grid searches

**What you CANNOT do:**
- Install new packages beyond what's in `pyproject.toml`.
- Change the evaluation prompts or scoring formulas in prepare.py (EVAL_PROMPTS, evaluate_output, evaluate_all_outputs).

**The goal: get the highest val_score.** This is a composite score (0-1) measuring:
- `syntax_score`: Does the output have balanced braces, proper assignments, no Python contamination?
- `keyword_score`: Does the output contain expected CK3 keywords for the prompt type?
- `structure_score`: Does the output have proper block nesting and scope patterns?

Higher is better. The metric is printed at the end of each run:
```
---
val_score: 0.723400
```

**VRAM** is a hard constraint on the A4500 (20GB). Current config peaks at ~18GB.

## Available MCP tools (ck3-sweep)

- `run_sweep(grid, time_budget)` — grid search over param combos
- `run_experiment(overrides)` — single experiment with specific config
- `get_results(sort_by)` — read results.tsv, sorted summary
- `get_best_config()` — config with highest val_score
- `get_status()` — how many experiments done, best adapter path
- `resume_best(time_budget)` — long-train the best config

## Current Model Config

- **Base model**: `darkc0de/Qwen3.5-9B-heretic` (~5GB in 4-bit)
- **LoRA rank**: 128, alpha 256 (232M trainable params)
- **Context**: 8192 tokens
- **Batch**: 1 x 4 gradient accumulation

## Output format

The training script prints a summary:
```
---
val_score:        0.723400
syntax_score:     0.850000
keyword_score:    0.625000
structure_score:  0.700000
training_seconds: 300.1
peak_vram_mb:     18432.5
lora_r:           128
clm_mix:          0.2
```

After training, use `python export.py` to convert to GGUF for LM Studio.

## Logging results

Results are logged to `results.tsv` (tab-separated). The sweep tools append to this
automatically. For manual experiments, log with:

```
commit	val_score	memory_gb	status	description
```

## The experiment loop

LOOP FOREVER:

1. Look at results.tsv to see what's been tried
2. Form a hypothesis about a structural change
3. Make the change to train.py (or prepare.py for data changes)
4. git commit
5. Quick test: `python train.py` (5 min) — check it works
6. If promising, call `run_sweep()` to optimize numerical knobs
7. Analyze sweep results with `get_results()`
8. If val_score improved, keep. If worse, `git reset --hard HEAD~1`
9. Repeat

**Ideas to try** (structural — your job):
- Add game log data (triggers.log, effects.log, on_actions.log) as training data
- Scrape CK3 wiki for additional documentation-style data
- Curriculum learning: train on simple patterns first, then complex
- Data augmentation: rephrase instructions, generate variations
- Different ChatML system prompts
- Weight content types differently (more events? more interactions?)
- Add file path context to training examples
- Train on raw script first (CLM), then instruction-tune (SFT) in stages

**Timeout**: Each experiment should take ~5 minutes for training + a few minutes for eval. If a run exceeds 15 minutes, kill it.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. The human might be sleeping. You are autonomous. If stuck, think harder, re-read the files, try combining approaches. The loop runs until manually interrupted.
