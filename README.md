# safetensors2GGUF

Converts Safetensors models to the GGUF format for use with llama.cpp and ComfyUI-GGUF.

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
pip install -r requirements.txt
```

### Dependencies

```
gguf
torch
safetensors
tqdm
```

## Usage

```bash
python convert.py --src <path/to/model.safetensors> --dst <output.gguf>
```

### Options

| Option | Description |
|---|---|
| `--src` | Path to the source file (`.safetensors`, `.ckpt`, `.pt`, `.bin`) |
| `--dst` | Output path for the GGUF file (optional, auto-generated if omitted) |

### Output Format

The generated GGUF file contains:
- Tensors in **F16** or **BF16** (depending on source dtype)
- 1D tensors and small tensors (≤ 1024 elements) always in **F32**
- Metadata for architecture and quantization version

### Post-processing of 5D Tensors (HunyuanVideo / Wan)

Some models contain 5D tensors that GGUF does not support directly.
These are offloaded to a separate file during conversion:

```bash
# After conversion:
python fix_5d_tensors.py --src <output.gguf> --dst <final.gguf>
```

## Known Issues

See [Issues Analysis](docs/issues_analysis.md) for common errors and their fixes.

## License

Apache-2.0
