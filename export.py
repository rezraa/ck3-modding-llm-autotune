"""
CK3 Modding LLM AutoTune — GGUF Export
================================
Merges a LoRA adapter into the base model and exports the full model as GGUF for LM Studio.
Standalone script — no training code, just model loading + export.

Usage:
    python export.py                        # export latest adapter (q4_k_m)
    python export.py --quant q5_k_m         # different quantization
    python export.py --adapter PATH         # specific adapter path
"""

import os
os.environ["HF_HOME"] = "D:/huggingface_cache"

import argparse
from pathlib import Path

from unsloth import FastLanguageModel

from prepare import MAX_SEQ_LEN, CACHE_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL = "darkc0de/Qwen3.5-9B-heretic"

# Default paths
OUTPUT_DIR = os.path.join(CACHE_DIR, "output")
DEFAULT_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")
DEFAULT_GGUF_DIR = os.path.join(OUTPUT_DIR, "gguf")

# Available quantizations: q4_k_m, q5_k_m, q8_0, f16
DEFAULT_QUANT = "q4_k_m"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_gguf(adapter_dir: str, gguf_dir: str, quant: str):
    """Load base model + LoRA adapter, merge, and export to GGUF."""

    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    # Load LoRA adapter
    if os.path.exists(adapter_dir):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"Loaded LoRA adapter from: {adapter_dir}")
    else:
        print(f"WARNING: No adapter found at {adapter_dir} — exporting base model")

    os.makedirs(gguf_dir, exist_ok=True)

    print(f"\nExporting to GGUF ({quant})...")
    print(f"Output dir: {gguf_dir}")

    model.save_pretrained_gguf(
        gguf_dir,
        tokenizer,
        quantization_method=quant,
    )

    # List exported files
    for f in sorted(Path(gguf_dir).glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

    print(f"\nGGUF export complete! Load in LM Studio from: {gguf_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CK3 Modding LLM AutoTune - GGUF Export")
    parser.add_argument("--quant", type=str, default=DEFAULT_QUANT,
                        help=f"Quantization method (default: {DEFAULT_QUANT}). Options: q4_k_m, q5_k_m, q8_0, f16")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER_DIR,
                        help=f"Path to LoRA adapter directory (default: {DEFAULT_ADAPTER_DIR})")
    parser.add_argument("--output", type=str, default=DEFAULT_GGUF_DIR,
                        help=f"Output directory for GGUF files (default: {DEFAULT_GGUF_DIR})")
    args = parser.parse_args()

    export_gguf(
        adapter_dir=args.adapter,
        gguf_dir=args.output,
        quant=args.quant,
    )


if __name__ == "__main__":
    main()
