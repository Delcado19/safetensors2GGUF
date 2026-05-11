# Architecture

## Conversion Pipeline

```
Source file (.safetensors / .ckpt / .pt)
    │
    ▼
load_state_dict()
    │  - Loads tensors into memory
    │  - Detects and strips state-dict prefixes (e.g. "model.diffusion_model.")
    │
    ▼
detect_arch()
    │  - Matches keys in the state dict against keys_detect of each architecture class
    │  - Raises AssertionError when no architecture is detected
    │
    ▼
handle_tensors()
    │  - Iterates over all tensors
    │  - Filters keys_ignore
    │  - Converts dtype: BF16 → F32, float8 → F16
    │  - Decides quantization type:
    │      1D or ≤ 1024 elements or keys_hiprec → F32
    │      BF16 source → BF16
    │      otherwise → F16
    │  - Reshape for SD1/SDXL (shape_fix): (H,W) → (n//256, 256)
    │      stores orig_shape as metadata field
    │  - Handles 5D tensors: offload instead of write
    │
    ▼
GGUFWriter
    │  - Writes header (arch, file_type, quantization_version)
    │  - Writes KV metadata
    │  - Writes tensor data
    │
    ▼
Output file (.gguf)
```

## Architecture Classes

Each supported model architecture inherits from `ModelTemplate`:

```python
class ModelTemplate:
    arch = "invalid"       # GGUF architecture string
    shape_fix = False      # True only for SD1/SDXL
    keys_detect = []       # Key lists for detection
    keys_banned = []       # Invalid variants (e.g. reference format)
    keys_hiprec = []       # Tensors that require F32
    keys_ignore = []       # Tensors to skip
```

### Detection Logic

`keys_detect` is a list of tuples. A model is detected when **all** keys
of a tuple are present in the state dict:

```python
# Flux is detected when ONE of these tuples matches completely:
keys_detect = [
    ("transformer_blocks.0.attn.norm_added_k.weight",),          # Diffusers
    ("double_blocks.0.img_attn.proj.weight",),                    # Non-Diffusers
]
# At the same time: reference format is banned:
keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight"]
```

## 5D Tensor Handling

GGUF supports at most 4-dimensional tensors. Models like HunyuanVideo and Wan
occasionally contain 5D tensors (e.g. RoPE frequencies).

**Two-step process:**

1. `convert.py`: The 5D tensor is **not** written to the GGUF file; instead it is
   offloaded to `fix_5d_tensors_<arch>.safetensors`.

2. `fix_5d_tensors.py`: Reads the fully quantized GGUF file and inserts the
   offloaded tensor as F32.

## Quantization Decision Tree

```
Tensor
 ├─ in keys_ignore?          → skip
 ├─ ndim > 4?                → offload (5D fix)
 ├─ ndim == 1?               → F32
 ├─ n_params ≤ 1024?         → F32
 ├─ key in keys_hiprec?      → F32
 ├─ shape_fix applicable?    → reshape + write orig_shape metadata
 └─ source dtype?
      BF16 → BF16
      otherwise → F16
```
