# safetensors2GGUF

Converts Safetensors / CKPT model checkpoints to GGUF format for use with
**llama.cpp** and **ComfyUI-GGUF**.

Supports direct Python quantization (F32 / F16 / BF16 / Q8_0) and
K-quant quantization via a bundled `llama-quantize` binary (Q6_K, Q5_K_M,
Q4_K_M, Q4_K_S, Q3_K_M, Q2_K).  A Gradio web UI with file browser,
quantization selector, live size estimate, and cancel button is included.

## Supported Architectures

| Architecture | Format |
|---|---|
| FLUX.1 | Diffusers |
| Stable Diffusion 3 | Diffusers |
| SDXL | Diffusers / Non-Diffusers |
| SD 1.x | Diffusers / Non-Diffusers |
| HiDream | Diffusers |
| HunyuanVideo | Diffusers |
| Wan | Diffusers |
| LTXV | Diffusers |
| Cosmos | Diffusers |
| AuraFlow | Diffusers |
| Lumina 2 | Diffusers |

## Installation

```bash
# Install uv (https://docs.astral.sh/uv/) then:
uv sync
```

Dependencies are maintained in `pyproject.toml` and locked in `uv.lock`.
This project does not maintain a separate `requirements.txt`.

<details>
<summary>Using pip instead</summary>

```bash
pip install gguf torch safetensors tqdm gradio
```

</details>

## Web UI (recommended)

Double-click **`start_gui.bat`** — the browser opens automatically.

Or from the terminal:

```bash
uv run python gui.py
```

The UI provides:
- **File browser** to select the source model without typing paths
- **Quantization dropdown** with 10 curated levels (see table below)
- **Live status bar** with percentage embedded in text; scroll-blocked so streaming updates never hijack the viewport
- **Automatic pipeline** — K-quants trigger a 2-step convert → quantize run;
  5D tensor insertion (HunyuanVideo / Wan) is chained automatically when needed

## Quantization Levels

| Key | Bits | Backend | Notes |
|---|---|---|---|
| `F32` | 32 | Python | Full precision, largest |
| `F16` | 16 | Python | Half precision, standard default |
| `BF16` | 16 | Python | Brain float, best for BF16 source models |
| `Q8_0` | 8 | Python | Very high quality |
| `Q6_K` | 6 | llama-quantize | Very high quality |
| `Q5_K_M` | 5 | llama-quantize | High quality |
| `Q4_K_M` | 4 | llama-quantize | **Recommended** — good quality / size balance |
| `Q4_K_S` | 4 | llama-quantize | Good quality, smaller |
| `Q3_K_M` | 3 | llama-quantize | Moderate quality |
| `Q2_K` | 2 | llama-quantize | Smallest, lowest quality |

K-quants (marked *llama-quantize*) require `llama-quantize.exe`.
The default path is detected automatically from:
`H:\ComfyUI-Easy-Install\Add-Ons\Tools\llama.cpp\llama-quantize.exe`

A custom path can be set in the **Advanced** section of the Web UI.

## CLI Usage

```bash
# Convert to F16 GGUF (auto-detects output name)
uv run python convert.py --src model.safetensors

# Specify output path
uv run python convert.py --src model.safetensors --dst model-F16.gguf --overwrite
```

### Options

| Option | Description |
|---|---|
| `--src` | Source file (`.safetensors`, `.ckpt`, `.pt`, `.bin`, `.pth`) |
| `--dst` | Output GGUF path — auto-generated when omitted |
| `--overwrite` | Skip confirmation if output already exists |

### Fix Pad Tokens (Lumina 2 — existing GGUFs)

If ComfyUI raises *size mismatch for x_pad_token*, the GGUF was converted before
the shape fix was introduced.  Repair it with:

```bash
uv run python fix_pad_tokens.py --src model.gguf --dst model-fixed.gguf
```

| Option | Description |
|---|---|
| `--src` | Source GGUF (1D pad tokens) |
| `--dst` | Output GGUF path |
| `--overwrite` | Skip confirmation if output exists |

New conversions are unaffected — `convert.py` stores pad tokens as `[1, D]` automatically.

### 5D Tensor Post-processing (HunyuanVideo / Wan — CLI only)

When using the Web UI, 5D tensor insertion is applied automatically after
llama-quantize.  For manual CLI workflows:

```bash
# 1. Convert to F16
uv run python convert.py --src model.safetensors

# 2. Quantize with llama-quantize (external step)
llama-quantize.exe model-F16.gguf model-Q8_0.gguf Q8_0

# 3. Insert 5D tensors
uv run python fix_5d_tensors.py --src model-Q8_0.gguf --dst model-Q8_0-fixed.gguf
```

## Known Issues

See [Issues Analysis](docs/issues_analysis.md) for common errors and their fixes.

## Running Tests

```bash
uv run pytest
```

## License

Apache-2.0
