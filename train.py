"""
CK3 Modding LLM AutoTune — Training Script
====================================
Fine-tunes a local LLM on CK3 modding data using QLoRA via unsloth.
Evaluates directly in-process. For GGUF export, use export.py.

THIS IS THE FILE THE AGENT MODIFIES.
Everything is fair game: model choice, LoRA config, hyperparameters,
data formatting, training schedule, etc.

Usage:
    python train.py                    # train with defaults (5 min)
    python train.py --eval-only        # evaluate current model
    python train.py --resume           # continue from existing adapter
    python train.py --long             # 8-hour overnight run
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["CC"] = "C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/cl.exe"

import sys
import json
import time
import argparse

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from prepare import (
    MAX_SEQ_LEN, DATA_DIR, CACHE_DIR,
    EVAL_PROMPTS,
    load_instruction_dataset, load_clm_dataset,
    format_chatml, evaluate_all_outputs,
)

# ---------------------------------------------------------------------------
# Configuration (AGENT MAY MODIFY)
# ---------------------------------------------------------------------------

# Base model — darkc0de's heretic fine-tune of Qwen3.5-9B
# ~5GB in 4-bit, leaves ~15GB for LoRA/optimizer on 20GB A4500
BASE_MODEL = "darkc0de/Qwen3.5-9B-heretic"
# Alternatives:
#   "brayniac/Qwen3.5-35B-A3B-heretic"  # ~17.5GB in 4-bit — OOMs on 20GB
#   "Qwen/Qwen3.5-4B"                   # ~2.5GB in 4-bit, fastest iteration

# Keep default 8192 from prepare.py — rank 128 LoRA needs the VRAM headroom
# (16K + rank 128 spilled to shared RAM on 20GB A4500)

# LoRA Configuration
# 9B leaves tons of VRAM — go aggressive for max learning capacity
LORA_R = 128                 # LoRA rank (aggressive — max learning capacity)
LORA_ALPHA = 256             # LoRA alpha (typically 2x rank)
LORA_DROPOUT = 0.0           # 0 enables unsloth fast patching
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training Hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 1               # per-device batch size (minimize for VRAM)
GRADIENT_ACCUMULATION = 4    # effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION (lower = faster steps)
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"

# Training time budget (seconds)
# Default 300s (5 min) is for agent mode — fast feedback loop
# Use --long for 8-hour overnight runs
TIME_BUDGET = 300            # 5 minutes (agent mode)

# Output paths
OUTPUT_DIR = os.path.join(CACHE_DIR, "output")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Data mix: fraction of CLM (continued pretraining) vs SFT (instruction pairs)
# 0.0 = SFT only, 1.0 = CLM only, 0.3 = 30% CLM + 70% SFT
CLM_MIX_RATIO = 0.2

# ---------------------------------------------------------------------------
# Config overrides via environment variable (used by sweep.py / MCP)
# ---------------------------------------------------------------------------
_overrides = os.environ.get("CK3_TRAIN_OVERRIDES")
if _overrides:
    import json as _json
    for _k, _v in _json.loads(_overrides).items():
        if _k in globals():
            globals()[_k] = _v
            print(f"  Override: {_k} = {_v}")

# ---------------------------------------------------------------------------
# Data Formatting (AGENT MAY MODIFY)
# ---------------------------------------------------------------------------

def format_training_example(example: dict) -> dict:
    """
    Format a training example. The agent can modify this to experiment
    with different prompt formats, system messages, etc.
    """
    if "instruction" in example:
        # SFT pair
        text = format_chatml(example)
    else:
        # CLM document (raw CK3 script)
        text = example["text"]
    return {"text": text}


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

def setup_model(resume=False):
    """Load base model with unsloth and apply LoRA. If resume=True, load existing adapter."""
    print(f"Loading model: {BASE_MODEL}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    if resume and os.path.exists(adapter_dir):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        print(f"Resumed LoRA adapter from: {adapter_dir}")
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation — direct inference (model still in GPU memory, no GGUF needed)
# ---------------------------------------------------------------------------

def evaluate_direct(model, tokenizer) -> dict:
    """
    Evaluate the trained model directly in-process.
    No external server needed — the model is already in memory from training.
    Generates CK3 code for each eval prompt and scores it.
    """
    FastLanguageModel.for_inference(model)

    system_msg = (
        "You are a CK3 modding expert. Write valid Crusader Kings III script code. "
        "Use correct scope chains, proper trigger/effect placement, and follow vanilla patterns. "
        "Only output the script code, no explanations."
    )

    outputs = {}
    for prompt in EVAL_PROMPTS:
        pid = prompt["id"]
        try:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt["instruction"]},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the generated tokens (skip the input)
            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            outputs[pid] = generated
            print(f"  {pid}: {len(generated)} chars generated")
        except Exception as e:
            print(f"  {pid}: FAILED - {e}")
            outputs[pid] = ""

    # Switch back to training mode in case caller needs it
    FastLanguageModel.for_training(model)

    metrics = evaluate_all_outputs(outputs)

    # Save generated outputs for review
    samples_dir = os.path.join(OUTPUT_DIR, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    timestamp = int(time.time())
    samples_file = os.path.join(samples_dir, f"eval_{timestamp}.json")
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "outputs": outputs}, f, indent=2, ensure_ascii=False)
    print(f"  Samples saved to: {samples_file}")

    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(resume=False):
    """Main training loop with time budget enforcement."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # Load data
    print("\n" + "=" * 60)
    print("Loading training data...")
    print("=" * 60)

    sft_train, sft_val = load_instruction_dataset()
    clm_train, clm_val = load_clm_dataset()

    print(f"  SFT pairs: {len(sft_train):,} train, {len(sft_val):,} val")
    print(f"  CLM docs:  {len(clm_train):,} train, {len(clm_val):,} val")

    # Mix datasets
    if CLM_MIX_RATIO > 0 and clm_train:
        n_clm = int(len(sft_train) * CLM_MIX_RATIO / (1 - CLM_MIX_RATIO))
        n_clm = min(n_clm, len(clm_train))
        import random
        random.seed(42)
        clm_sample = random.sample(clm_train, n_clm)
        all_train = sft_train + clm_sample
        all_val = sft_val + clm_val[:100]
        print(f"  Mixed: {len(all_train):,} train ({len(sft_train)} SFT + {n_clm} CLM)")
    else:
        all_train = sft_train
        all_val = sft_val

    # Format into text
    train_formatted = [format_training_example(ex) for ex in all_train]
    val_formatted = [format_training_example(ex) for ex in all_val]

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)

    # Setup model
    print("\n" + "=" * 60)
    print("Setting up model...")
    print("=" * 60)
    model, tokenizer = setup_model(resume=resume)

    # Training arguments
    training_args = SFTConfig(
        output_dir=CHECKPOINTS_DIR,
        # Training
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        max_seq_length=MAX_SEQ_LEN,
        # Precision
        bf16=True,
        # Eval
        eval_strategy="steps",
        eval_steps=50,
        per_device_eval_batch_size=1,
        # Saving
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        # Logging
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        # Data
        dataset_text_field="text",
        packing=True,
        seed=42,
    )

    # Time budget callback
    t_start = time.time()

    from transformers import TrainerCallback
    class TimeBudgetCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.time() - t_start
            if elapsed > TIME_BUDGET:
                print(f"\n  Time budget reached ({TIME_BUDGET}s). Stopping training.")
                control.should_training_stop = True

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )
    trainer.add_callback(TimeBudgetCallback())

    # Train
    print("\n" + "=" * 60)
    print(f"Starting training (budget: {TIME_BUDGET}s)")
    print(f"  Model:    {BASE_MODEL}")
    print(f"  LoRA:     r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  LR:       {LEARNING_RATE}")
    print(f"  Batch:    {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Epochs:   {NUM_EPOCHS}")
    print(f"  Seq len:  {MAX_SEQ_LEN}")
    print(f"  CLM mix:  {CLM_MIX_RATIO}")
    print("=" * 60 + "\n")

    trainer.train()

    training_time = time.time() - t_start

    # Save LoRA adapter
    adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nLoRA adapter saved to: {adapter_dir}")

    # Evaluate directly (model still in memory — no external server needed)
    print("\n" + "=" * 60)
    print("Evaluating trained model...")
    print("=" * 60)
    metrics = evaluate_direct(model, tokenizer)

    # Save metrics
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    all_metrics = {
        "model": BASE_MODEL,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
        "epochs": NUM_EPOCHS,
        "clm_mix": CLM_MIX_RATIO,
        "training_time_s": round(training_time, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        **metrics,
    }

    metrics_file = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)

    # Print summary
    print("\n---")
    print(f"val_score:        {metrics.get('val_score', 0.0):.6f}")
    print(f"syntax_score:     {metrics.get('syntax_score', 0.0):.6f}")
    print(f"keyword_score:    {metrics.get('keyword_score', 0.0):.6f}")
    print(f"structure_score:  {metrics.get('structure_score', 0.0):.6f}")
    print(f"training_seconds: {training_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"lora_r:           {LORA_R}")
    print(f"clm_mix:          {CLM_MIX_RATIO}")

    return all_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CK3 Modding LLM AutoTune - Training")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the model")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing LoRA adapter")
    parser.add_argument("--long", action="store_true", help="Override time budget to 8 hours for overnight runs")
    args = parser.parse_args()

    if args.long:
        global TIME_BUDGET
        TIME_BUDGET = 28800  # 8 hours
        print(f"Long mode: TIME_BUDGET set to {TIME_BUDGET}s (8 hours)")

    if args.eval_only:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model, tokenizer = setup_model()
        # Load LoRA adapter if one exists
        adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
        if os.path.exists(adapter_dir):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir)
            print(f"Loaded LoRA adapter from: {adapter_dir}")
        metrics = evaluate_direct(model, tokenizer)
        print("\n---")
        print(f"val_score:        {metrics.get('val_score', 0.0):.6f}")
        print(f"syntax_score:     {metrics.get('syntax_score', 0.0):.6f}")
        print(f"keyword_score:    {metrics.get('keyword_score', 0.0):.6f}")
        print(f"structure_score:  {metrics.get('structure_score', 0.0):.6f}")
        return

    metrics = train(resume=args.resume)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
