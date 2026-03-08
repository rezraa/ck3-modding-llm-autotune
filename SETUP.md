# CK3 Modding LLM AutoTune — Setup & Run Instructions

## Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (tested on RTX A4500 20GB)
- Visual Studio Build Tools with "Desktop development with C++" (for Triton kernel compilation)

## One-Time Setup

### 1. Install PyTorch with CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

### 2. Install dependencies
```bash
pip install unsloth openai datasets peft trl bitsandbytes accelerate xformers fastmcp
```

Verify:
```bash
python -c "from unsloth import FastLanguageModel; print('unsloth OK')"
```

### 3. Extract CK3 training data
```bash
cd ck3-modding-llm-autotune
python prepare.py
```

This extracts ~32K instruction pairs from vanilla CK3 files. Only needs to run once.
Data is stored in `~/.cache/ck3_modding_llm_autotune/data/`.

## Running

### Overnight sweep (recommended first run)
```bash
python sweep.py --dry-run              # preview 96 experiments
python sweep.py                        # run full grid (~8 hours)
python sweep.py --max-experiments 6    # quick test (~30 min)
python sweep.py --grid grid.json       # custom grid
```

### Deep train the best config
```bash
python train.py --long --resume        # 8-hour run on best adapter
```

### Export to GGUF for LM Studio
```bash
python export.py                       # default q4_k_m quantization
python export.py --quant q5_k_m        # higher quality
python export.py --quant q8_0          # highest quality (larger file)
python export.py --adapter PATH        # specific adapter
```

### Agent mode (Claude Code iterates autonomously)
```bash
# In Claude Code, open the ck3-modding-llm-autotune folder and say:
Read agent.md and let's kick off a new CK3 research experiment!
```

Claude will loop: edit train.py → train (5 min) → evaluate → keep/discard → repeat.

### Other commands
```bash
python train.py                        # single 5-min experiment
python train.py --eval-only            # evaluate existing adapter
python prepare.py --stats              # print dataset statistics
```

### Sweep MCP (Claude calls sweep tools)
Add to Claude Code MCP settings:
```json
{
  "mcpServers": {
    "ck3-sweep": {
      "command": "python",
      "args": ["sweep_mcp/server.py"]
    }
  }
}
```

## Output
- **LoRA adapter**: `~/.cache/ck3_modding_llm_autotune/output/lora_adapter/`
- **Best adapter** (from sweep): `~/.cache/ck3_modding_llm_autotune/output/best/`
- **GGUF model**: `~/.cache/ck3_modding_llm_autotune/output/gguf/`
- **Eval samples**: `~/.cache/ck3_modding_llm_autotune/output/samples/`
- **Sweep results**: `results.tsv`

## Troubleshooting

### PyTorch says CUDA not available
You have a CPU-only torch installed. Fix:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### OOM during training
Current config (9B + rank 128 + 8192 context) peaks at ~18GB on 20GB VRAM. If OOM:
- Reduce `LORA_R` (128 → 64)
- Reduce `MAX_SEQ_LEN` (remove the override, falls back to prepare.py default)
- Switch `BASE_MODEL` to `"Qwen/Qwen3.5-4B"`

### "Failed to find C compiler"
Install Visual Studio Build Tools with "Desktop development with C++" workload. Or add to train.py:
```python
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # disables Triton, ~10-20% slower
```

### GGUF export takes forever / crashes
The 16-bit merge is CPU/RAM intensive (~18GB). Make sure you have enough system RAM and the process isn't killed by Windows.
