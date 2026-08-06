# Safetensors Output Mode + Text-Encoder GGUF Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second output format ("Safetensors → Safetensors", quantized to FP16/FP8/NVFP4) alongside the existing GGUF path, and add a third pipeline that converts bare single-file text-encoder checkpoints (Qwen3, Mistral, T5/UMT5) to GGUF via `llama.cpp`'s `convert_hf_to_gguf.py`.

**Architecture:** Reuse the existing `load_state_dict()` / `detect_arch()` / `_quant_type_for()` machinery from `convert.py` for the new safetensors writer — only the output backend changes (safetensors.torch.save_file instead of GGUFWriter). Text-encoder conversion is a fully separate module that shells out to `convert_hf_to_gguf.py` (found inside the user's ComfyUI-Easy-Install Python environment) after assembling a temp directory of user-supplied weights + HuggingFace-downloaded config/tokenizer files. GUI gets a renamed "Convert → GGUF" tab (was "Convert") and two new tabs, "Convert → Safetensors" and "Convert Text Encoder → GGUF".

**Tech Stack:** Python 3.10+, torch 2.11 (native `float8_e4m3fn`, `float4_e2m1fn_x2` dtypes), `safetensors`, `huggingface_hub` (new dependency), Gradio, pytest.

## Global Constraints

- Target ComfyUI compatibility for every new format — verified against `city96/ComfyUI-GGUF`'s `comfy/quant_ops.py` `QUANT_ALGOS` registry (see research below). Do not invent formats ComfyUI cannot load.
- No plain/generic FP4 mode — ComfyUI has no generic 4-bit loader; only `nvfp4` (uint8-packed + `weight_scale` + `weight_scale_2`) is registered. FP4 = NVFP4 only.
- "Mixed" variants reuse the existing `keys_hiprec` / 1D / ≤1024-element / >4D high-precision rule from `convert.py:_quant_type_for` — do not invent a second mixed-precision mechanism.
- Base-model repo for text-encoder conversion is always a manual GUI text field (HF repo ID) — no auto-detection heuristic.
- `convert_hf_to_gguf.py` is invoked via the ComfyUI-Easy-Install embedded Python interpreter as a subprocess (it already has `transformers`/`torch`/`mistral_common`), mirroring the existing `run_quantize()` subprocess pattern in `quantize.py`. It is not vendored into this repo.
- New dependency: `huggingface_hub` (added to `pyproject.toml`), used only for `hf_hub_download`.

---

## Part A — Safetensors → Safetensors quantized output

### Task 1: `safetensors_quant.py` — dtype registry and per-tensor quantize function

**Files:**
- Create: `safetensors_quant.py`
- Test: `tests/test_safetensors_quant.py`

**Interfaces:**
- Produces:
  - `SAFETENSORS_DTYPE_CHOICES: list[tuple[str, str]]` — `(display label, key)` pairs for the GUI dropdown, keys: `"F16"`, `"F16_MIXED"`, `"FP8"`, `"FP8_MIXED"`, `"NVFP4"`, `"NVFP4_MIXED"`.
  - `quantize_tensor_st(data: torch.Tensor, key: str, model_arch, target_key: str) -> dict[str, torch.Tensor]` — returns a dict of `{tensor_name: tensor}` to merge into the output state dict (1 entry for F16/FP8-unscaled, 3 entries `{key, key+".weight_scale", key+".input_scale"}` for FP8 scaled and 2-3 entries for NVFP4 — see Task 2/3).
  - `is_hiprec_st(key: str, data: torch.Tensor, model_arch, old_dtype: torch.dtype) -> bool` — thin wrapper around `convert._quant_type_for`'s hiprec predicate (1D / ≤1024 elems / >4D / keys_hiprec), reused so "mixed" variants match GGUF's existing rule exactly.

- [ ] **Step 1: Write the failing tests for the dtype registry**

```python
"""Tests for safetensors_quant.py — dtype registry and per-tensor quantization."""

from __future__ import annotations

import torch

from safetensors_quant import (
    SAFETENSORS_DTYPE_CHOICES,
    is_hiprec_st,
    quantize_tensor_st,
)
from models.architectures import ModelFlux, ModelLumina2


class TestRegistry:
    def test_choices_is_list_of_tuples(self):
        assert isinstance(SAFETENSORS_DTYPE_CHOICES, list)
        for label, key in SAFETENSORS_DTYPE_CHOICES:
            assert isinstance(label, str) and label
            assert isinstance(key, str) and key

    def test_expected_keys_present(self):
        keys = {k for _, k in SAFETENSORS_DTYPE_CHOICES}
        assert keys == {"F16", "F16_MIXED", "FP8", "FP8_MIXED", "NVFP4", "NVFP4_MIXED"}

    def test_no_duplicate_keys(self):
        keys = [k for _, k in SAFETENSORS_DTYPE_CHOICES]
        assert len(keys) == len(set(keys))


class TestHiprec:
    def test_1d_tensor_is_hiprec(self):
        data = torch.zeros(64, dtype=torch.float32)
        assert is_hiprec_st("some.weight", data, ModelFlux(), torch.float32)

    def test_small_2d_tensor_is_hiprec(self):
        data = torch.zeros(4, 4, dtype=torch.float32)
        assert is_hiprec_st("some.weight", data, ModelFlux(), torch.float32)

    def test_large_2d_tensor_is_not_hiprec(self):
        data = torch.zeros(64, 64, dtype=torch.float32)
        assert not is_hiprec_st("some.weight", data, ModelFlux(), torch.float32)

    def test_keys_hiprec_key_is_hiprec(self):
        data = torch.zeros(64, 64, dtype=torch.bfloat16)
        assert is_hiprec_st("x_pad_token", data, ModelLumina2(), torch.bfloat16)


class TestQuantizeTensorF16:
    def test_f16_plain_casts_dtype(self):
        data = torch.randn(64, 64, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "F16")
        assert set(out.keys()) == {"block.weight"}
        assert out["block.weight"].dtype == torch.float16

    def test_f16_mixed_keeps_hiprec_tensor_f32(self):
        data = torch.randn(4, 4, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "F16_MIXED")
        assert out["block.bias"].dtype == torch.float32

    def test_f16_mixed_casts_large_tensor_f16(self):
        data = torch.randn(64, 64, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "F16_MIXED")
        assert out["block.weight"].dtype == torch.float16
```

- [ ] **Step 2: Run tests to verify they fail (module doesn't exist yet)**

Run: `uv run pytest tests/test_safetensors_quant.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'safetensors_quant'`

- [ ] **Step 3: Implement the registry, hiprec wrapper, and F16/F16_MIXED path**

```python
"""Quantized Safetensors → Safetensors output backend.

Companion to convert.py's GGUF writer: reuses the same architecture detection
and high-precision-tensor rule (keys_hiprec / 1D / <=1024 elems / >4D) but
writes a plain .safetensors file instead of a GGUF container. See
docs/superpowers/plans/2026-08-05-safetensors-output-and-text-encoder-support.md
for the ComfyUI-compatibility research this format registry is based on
(comfy/quant_ops.py QUANT_ALGOS in city96/ComfyUI-GGUF).
"""

from __future__ import annotations

import torch

QUANTIZATION_THRESHOLD = 1024

# Ordered choices for the GUI dropdown: (display label, key)
SAFETENSORS_DTYPE_CHOICES: list[tuple[str, str]] = [
    ("F16       — Half precision",                                    "F16"),
    ("F16 mixed — Half precision, hiprec tensors stay F32",            "F16_MIXED"),
    ("FP8       — float8_e4m3fn, scaled (ComfyUI scaled-fp8 format)",  "FP8"),
    ("FP8 mixed — FP8 scaled, hiprec tensors stay F32",                "FP8_MIXED"),
    ("NVFP4     — Nvidia 4-bit blockscaled (16-elem blocks)",          "NVFP4"),
    ("NVFP4 mixed — NVFP4, hiprec tensors stay F32",                   "NVFP4_MIXED"),
]

_MIXED_KEYS = {"F16_MIXED", "FP8_MIXED", "NVFP4_MIXED"}
_BASE_KEY = {
    "F16": "F16", "F16_MIXED": "F16",
    "FP8": "FP8", "FP8_MIXED": "FP8",
    "NVFP4": "NVFP4", "NVFP4_MIXED": "NVFP4",
}


def is_hiprec_st(key: str, data: torch.Tensor, model_arch, old_dtype: torch.dtype) -> bool:
    """Return True if ``key`` must stay high-precision (F32), mirroring
    convert._quant_type_for's rule so 'mixed' safetensors output matches the
    existing GGUF mixed-precision behaviour exactly."""
    if old_dtype not in (torch.float32, torch.bfloat16):
        return False
    n_dims = data.dim()
    if n_dims == 1:
        return True
    if data.numel() <= QUANTIZATION_THRESHOLD:
        return True
    if any(x in key for x in model_arch.keys_hiprec):
        return True
    return False


def quantize_tensor_st(
    data: torch.Tensor, key: str, model_arch, target_key: str
) -> dict[str, torch.Tensor]:
    """Quantize one tensor for safetensors output.

    Returns a dict of {tensor_name: tensor} — one entry for F16/FP8-unscaled,
    multiple entries (weight + scale tensors) for FP8-scaled and NVFP4.
    """
    old_dtype = data.dtype
    base = _BASE_KEY[target_key]
    mixed = target_key in _MIXED_KEYS

    if mixed and is_hiprec_st(key, data, model_arch, old_dtype):
        return {key: data.to(torch.float32)}

    if base == "F16":
        return {key: data.to(torch.float16)}

    if base == "FP8":
        from safetensors_quant_fp8 import quantize_fp8_scaled
        return quantize_fp8_scaled(data, key)

    if base == "NVFP4":
        from safetensors_quant_nvfp4 import quantize_nvfp4
        return quantize_nvfp4(data, key)

    raise ValueError(f"Unknown target_key: {target_key!r}")
```

- [ ] **Step 4: Run tests, expect the F16 tests to pass and FP8/NVFP4 tests to fail with ImportError**

Run: `uv run pytest tests/test_safetensors_quant.py -v`
Expected: `TestRegistry` and `TestHiprec` and `TestQuantizeTensorF16` PASS. (FP8/NVFP4 tests are added in Tasks 2–3.)

- [ ] **Step 5: Commit**

```bash
git add safetensors_quant.py tests/test_safetensors_quant.py
git commit -m "feat: add safetensors output dtype registry (F16/F16-mixed)"
```

---

### Task 2: FP8 scaled quantization (ComfyUI `scaled_fp8` convention)

**Files:**
- Create: `safetensors_quant_fp8.py`
- Test: `tests/test_safetensors_quant_fp8.py`

**Interfaces:**
- Consumes: nothing from other new modules — pure torch.
- Produces: `quantize_fp8_scaled(data: torch.Tensor, key: str) -> dict[str, torch.Tensor]`, `QUANT_METADATA_FP8: dict` (per-layer `{"format": "float8_e4m3fn"}` entry template used when assembling the file's `_quantization_metadata`).

ComfyUI's `scaled_fp8` convention (confirmed against `comfy/quant_ops.py` in `city96/ComfyUI-GGUF` — see plan header): a `<layer>.weight_scale` float32 scalar per quantized layer, computed so that `weight_fp8 * weight_scale ≈ original_weight`. `float8_e4m3fn` has a representable range of ±448; the scale is `max(|weight|) / 448`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for safetensors_quant_fp8.py — scaled FP8 quantization."""

from __future__ import annotations

import torch

from safetensors_quant_fp8 import quantize_fp8_scaled


class TestQuantizeFp8Scaled:
    def test_returns_weight_and_scale(self):
        data = torch.randn(64, 64, dtype=torch.float32) * 10
        out = quantize_fp8_scaled(data, "block.weight")
        assert set(out.keys()) == {"block.weight", "block.weight.weight_scale"}

    def test_weight_dtype_is_fp8_e4m3fn(self):
        data = torch.randn(64, 64, dtype=torch.float32)
        out = quantize_fp8_scaled(data, "block.weight")
        assert out["block.weight"].dtype == torch.float8_e4m3fn

    def test_scale_dtype_is_float32_scalar(self):
        data = torch.randn(64, 64, dtype=torch.float32)
        out = quantize_fp8_scaled(data, "block.weight")
        scale = out["block.weight.weight_scale"]
        assert scale.dtype == torch.float32
        assert scale.numel() == 1

    def test_dequant_reconstructs_within_fp8_tolerance(self):
        torch.manual_seed(0)
        data = torch.randn(128, 128, dtype=torch.float32) * 5
        out = quantize_fp8_scaled(data, "block.weight")
        recon = out["block.weight"].to(torch.float32) * out["block.weight.weight_scale"]
        # FP8 e4m3fn has ~2 decimal digits of mantissa precision
        assert torch.allclose(recon, data, atol=data.abs().max().item() * 0.1)

    def test_zero_tensor_does_not_divide_by_zero(self):
        data = torch.zeros(16, 16, dtype=torch.float32)
        out = quantize_fp8_scaled(data, "block.weight")
        assert torch.isfinite(out["block.weight"].to(torch.float32)).all()
        assert torch.isfinite(out["block.weight.weight_scale"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_safetensors_quant_fp8.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'safetensors_quant_fp8'`

- [ ] **Step 3: Implement scaled FP8 quantization**

```python
"""Scaled FP8 quantization matching ComfyUI's scaled_fp8 checkpoint convention.

Format reference: city96/ComfyUI-GGUF comfy/quant_ops.py QUANT_ALGOS["float8_e4m3fn"].
Per layer: <name> stored as torch.float8_e4m3fn, <name>.weight_scale as a
float32 scalar such that original ≈ stored.to(float32) * weight_scale.
"""

from __future__ import annotations

import torch

_FP8_MAX = 448.0  # float8_e4m3fn representable magnitude


def quantize_fp8_scaled(data: torch.Tensor, key: str) -> dict[str, torch.Tensor]:
    """Quantize one tensor to scaled float8_e4m3fn.

    Returns {key: fp8_tensor, f"{key}.weight_scale": float32 scalar}.
    """
    amax = data.abs().max()
    scale = (amax / _FP8_MAX) if amax > 0 else torch.tensor(1.0, dtype=torch.float32)
    scale = scale.to(torch.float32)
    scaled = (data.to(torch.float32) / scale).clamp(-_FP8_MAX, _FP8_MAX)
    return {
        key: scaled.to(torch.float8_e4m3fn),
        f"{key}.weight_scale": scale.reshape(1),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_safetensors_quant_fp8.py -v`
Expected: PASS

- [ ] **Step 5: Wire FP8 into `safetensors_quant.py` and add the registry-level FP8 tests from Task 1**

Add to `tests/test_safetensors_quant.py`:

```python
class TestQuantizeTensorFp8:
    def test_fp8_returns_weight_and_scale(self):
        data = torch.randn(64, 64, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "FP8")
        assert "block.weight" in out
        assert "block.weight.weight_scale" in out
        assert out["block.weight"].dtype == torch.float8_e4m3fn

    def test_fp8_mixed_keeps_hiprec_tensor_f32_unscaled(self):
        data = torch.randn(4, 4, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "FP8_MIXED")
        assert set(out.keys()) == {"block.bias"}
        assert out["block.bias"].dtype == torch.float32
```

Run: `uv run pytest tests/test_safetensors_quant.py tests/test_safetensors_quant_fp8.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add safetensors_quant.py safetensors_quant_fp8.py tests/test_safetensors_quant.py tests/test_safetensors_quant_fp8.py
git commit -m "feat: add ComfyUI-compatible scaled FP8 safetensors quantization"
```

---

### Task 3: NVFP4 quantization (Nvidia block-scaled 4-bit)

**Files:**
- Create: `safetensors_quant_nvfp4.py`
- Test: `tests/test_safetensors_quant_nvfp4.py`

**Interfaces:**
- Produces: `quantize_nvfp4(data: torch.Tensor, key: str) -> dict[str, torch.Tensor]`.

Format (confirmed against `comfy/quant_ops.py` `QUANT_ALGOS["nvfp4"]`): weight packed as `torch.uint8` (2× 4-bit E2M1 values per byte, last dim halved), `<name>.weight_scale` as `float8_e4m3fn` per 16-element block along the last dim, `<name>.weight_scale_2` as a single `float32` global scale. Reconstruction: `value ≈ e2m1_decode(nibble) * weight_scale[block].float() * weight_scale_2`. This mirrors `gguf/quants.py`'s `NVFP4` class already vendored via the `gguf` dependency (same E2M1 kvalues table `(0,1,2,3,4,6,8,12,...)`), reused here instead of re-derived, per YAGNI.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for safetensors_quant_nvfp4.py — Nvidia NVFP4 block-scaled quantization."""

from __future__ import annotations

import torch

from safetensors_quant_nvfp4 import quantize_nvfp4


class TestQuantizeNvfp4:
    def test_returns_three_tensors(self):
        data = torch.randn(32, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        assert set(out.keys()) == {
            "block.weight", "block.weight.weight_scale", "block.weight.weight_scale_2",
        }

    def test_weight_is_packed_uint8_half_last_dim(self):
        data = torch.randn(4, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        w = out["block.weight"]
        assert w.dtype == torch.uint8
        assert w.shape == (4, 16)  # 32 elems / 2 per byte

    def test_scale_is_fp8_e4m3fn_per_16_block(self):
        data = torch.randn(4, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        scale = out["block.weight.weight_scale"]
        assert scale.dtype == torch.float8_e4m3fn
        assert scale.shape == (4, 2)  # 32 elems / 16-block

    def test_scale_2_is_global_float32_scalar(self):
        data = torch.randn(4, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        s2 = out["block.weight.weight_scale_2"]
        assert s2.dtype == torch.float32
        assert s2.numel() == 1

    def test_raises_on_non_multiple_of_16_last_dim(self):
        data = torch.randn(4, 17, dtype=torch.float32)
        try:
            quantize_nvfp4(data, "block.weight")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_dequant_reconstructs_within_nvfp4_tolerance(self):
        torch.manual_seed(0)
        data = torch.randn(8, 64, dtype=torch.float32) * 3
        out = quantize_nvfp4(data, "block.weight")
        from safetensors_quant_nvfp4 import dequantize_nvfp4  # test-only helper
        recon = dequantize_nvfp4(out, "block.weight")
        assert recon.shape == data.shape
        # 4-bit float has coarse steps; allow generous relative tolerance
        assert torch.allclose(recon, data, atol=data.abs().max().item() * 0.35)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_safetensors_quant_nvfp4.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'safetensors_quant_nvfp4'`

- [ ] **Step 3: Implement NVFP4 quantization, reusing `gguf.quants.NVFP4`'s E2M1 kvalues table**

```python
"""Nvidia NVFP4 block-scaled quantization for safetensors output.

Format reference: city96/ComfyUI-GGUF comfy/quant_ops.py QUANT_ALGOS["nvfp4"]
(group_size=16, storage_t=uint8, params={weight_scale, weight_scale_2}) — the
same TensorRT-Model-Optimizer convention ComfyUI's native NVFP4 loader expects.
The E2M1 decode table is the same one gguf.quants.NVFP4 uses internally, kept
in sync by using the exact same kvalues tuple (avoids two independent, and
possibly diverging, 4-bit float codebooks in this repo).
"""

from __future__ import annotations

import torch

GROUP_SIZE = 16
# e2m1 values doubled — identical table to gguf.quants.NVFP4.kvalues
_KVALUES = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.float32
)
_FP8_MAX = 448.0


def _nearest_e2m1_index(x: torch.Tensor) -> torch.Tensor:
    """Map float values (already divided by the block scale) to the nearest
    of the 16 E2M1 codebook entries; returns int64 indices 0..15."""
    diffs = (x.unsqueeze(-1) - _KVALUES.to(x.device)).abs()
    return diffs.argmin(dim=-1)


def quantize_nvfp4(data: torch.Tensor, key: str) -> dict[str, torch.Tensor]:
    """Quantize one 2D+ tensor to NVFP4 (uint8-packed, 16-elem block scale + global scale).

    Raises ValueError if the last dimension is not a multiple of 16.
    """
    if data.shape[-1] % GROUP_SIZE != 0:
        raise ValueError(
            f"NVFP4 requires last dim to be a multiple of {GROUP_SIZE}, got {data.shape[-1]}"
        )
    x = data.to(torch.float32)
    *lead, last = x.shape
    n_blocks_per_row = last // GROUP_SIZE
    blocks = x.reshape(*lead, n_blocks_per_row, GROUP_SIZE)

    global_amax = x.abs().max()
    scale_2 = (global_amax / (_FP8_MAX * 6.0)) if global_amax > 0 else torch.tensor(1.0)
    scale_2 = scale_2.to(torch.float32)

    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    block_scale = (block_amax / 6.0 / scale_2).clamp(min=1e-12)
    block_scale_fp8 = block_scale.to(torch.float8_e4m3fn)

    normalized = blocks / (block_scale_fp8.to(torch.float32) * scale_2)
    idx = _nearest_e2m1_index(normalized)  # (*lead, n_blocks, 16) -> index per elem

    idx = idx.reshape(*lead, n_blocks_per_row, GROUP_SIZE // 2, 2)
    packed = (idx[..., 0] | (idx[..., 1] << 4)).to(torch.uint8)
    packed = packed.reshape(*lead, last // 2)

    return {
        key: packed,
        f"{key}.weight_scale": block_scale_fp8.squeeze(-1),
        f"{key}.weight_scale_2": scale_2.reshape(1),
    }


def dequantize_nvfp4(tensors: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    """Reverse quantize_nvfp4 — used by tests to verify round-trip accuracy."""
    packed = tensors[key]
    block_scale = tensors[f"{key}.weight_scale"].to(torch.float32)
    scale_2 = tensors[f"{key}.weight_scale_2"]

    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    *lead, half = packed.shape
    idx = torch.stack([lo, hi], dim=-1).reshape(*lead, half * 2)
    idx = idx.reshape(*lead, half * 2 // GROUP_SIZE, GROUP_SIZE)

    values = _KVALUES[idx]
    values = values * block_scale.unsqueeze(-1) * scale_2
    return values.reshape(*lead, half * 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_safetensors_quant_nvfp4.py -v`
Expected: PASS

- [ ] **Step 5: Wire NVFP4 into `safetensors_quant.py` registry tests**

Add to `tests/test_safetensors_quant.py`:

```python
class TestQuantizeTensorNvfp4:
    def test_nvfp4_returns_packed_and_two_scales(self):
        data = torch.randn(32, 32, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.weight", ModelFlux(), "NVFP4")
        assert set(out.keys()) == {
            "block.weight", "block.weight.weight_scale", "block.weight.weight_scale_2",
        }

    def test_nvfp4_mixed_keeps_hiprec_tensor_unpacked(self):
        data = torch.randn(4, 4, dtype=torch.float32)
        out = quantize_tensor_st(data, "block.bias", ModelFlux(), "NVFP4_MIXED")
        assert set(out.keys()) == {"block.bias"}
        assert out["block.bias"].dtype == torch.float32
```

Run: `uv run pytest tests/test_safetensors_quant.py tests/test_safetensors_quant_nvfp4.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add safetensors_quant.py safetensors_quant_nvfp4.py tests/test_safetensors_quant.py tests/test_safetensors_quant_nvfp4.py
git commit -m "feat: add NVFP4 block-scaled safetensors quantization"
```

---

### Task 4: `convert_to_safetensors()` — end-to-end file writer

**Files:**
- Create: `convert_safetensors.py`
- Test: `tests/test_convert_safetensors.py`

**Interfaces:**
- Consumes: `load_state_dict`, `detect_arch`, `strip_prefix` from `convert.py`; `quantize_tensor_st` from `safetensors_quant.py`.
- Produces: `convert_to_safetensors(path, dst_path=None, target_key="FP8", overwrite=False, on_progress=None, on_log=None, cancel_event=None) -> (dst_path, model_arch)`.

1D pad tokens, 5D tensors, and shape_fix rearranging are **not** replicated here — those exist solely to satisfy GGUF/`llama-quantize` constraints (K-quant block alignment, GGUF's 4D tensor limit). A plain safetensors file has none of those constraints, so tensors are written as-is after quantization. `nan_to_num` clamping is still applied (same corrupted-checkpoint protection as the GGUF path).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for convert_safetensors.py — safetensors-to-safetensors quantized output."""

from __future__ import annotations

import torch
from safetensors.torch import load_file, save_file

from convert_safetensors import convert_to_safetensors


def _write_minimal_flux(tmp_path):
    src = tmp_path / "model.safetensors"
    sd = {
        "double_blocks.0.img_attn.proj.weight": torch.randn(64, 64, dtype=torch.float32),
        "double_blocks.0.img_attn.proj.bias": torch.randn(64, dtype=torch.float32),
    }
    save_file(sd, str(src))
    return src


class TestConvertToSafetensors:
    def test_writes_output_file(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        dst, arch = convert_to_safetensors(str(src), target_key="F16", overwrite=True)
        assert dst.endswith(".safetensors")
        import os
        assert os.path.isfile(dst)
        arch is not None and arch.arch == "flux"

    def test_output_tensor_dtype_matches_target(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="F16", overwrite=True)
        out = load_file(dst)
        assert out["double_blocks.0.img_attn.proj.weight"].dtype == torch.float16

    def test_fp8_output_includes_scale_tensors(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="FP8", overwrite=True)
        out = load_file(dst)
        assert "double_blocks.0.img_attn.proj.weight.weight_scale" in out

    def test_quantization_metadata_written(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        dst, _ = convert_to_safetensors(str(src), target_key="FP8", overwrite=True)
        from safetensors import safe_open
        with safe_open(dst, framework="pt") as f:
            meta = f.metadata()
        assert meta is not None and "_quantization_metadata" in meta

    def test_refuses_overwrite_without_flag(self, tmp_path):
        src = _write_minimal_flux(tmp_path)
        dst_path = str(tmp_path / "out.safetensors")
        convert_to_safetensors(str(src), dst_path=dst_path, target_key="F16", overwrite=True)
        try:
            convert_to_safetensors(str(src), dst_path=dst_path, target_key="F16", overwrite=False)
            assert False, "expected OSError"
        except OSError:
            pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_convert_safetensors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'convert_safetensors'`

- [ ] **Step 3: Implement the writer**

```python
"""Safetensors → Safetensors quantized output.

Sibling to convert.py's GGUF writer: same architecture detection and tensor
loading, different output backend. Unlike GGUF output, no 5D side-car export,
no shape_fix rearrange, no 1D-pad-token unsqueeze — those exist only to
satisfy GGUF/llama-quantize constraints that plain safetensors doesn't have.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file

from convert import load_state_dict
from models.architectures import detect_arch
from safetensors_quant import quantize_tensor_st

_TARGET_TO_QUANT_FORMAT = {
    "FP8": "float8_e4m3fn", "FP8_MIXED": "float8_e4m3fn",
    "NVFP4": "nvfp4", "NVFP4_MIXED": "nvfp4",
}


def convert_to_safetensors(
    path,
    dst_path=None,
    target_key="FP8",
    overwrite=False,
    on_progress=None,
    on_log=None,
    cancel_event=None,
):
    """Convert a model checkpoint to a quantized .safetensors file.

    Args:
        path: Source model file path.
        dst_path: Output path; auto-generated as ``<src>-<target_key>.safetensors`` when None.
        target_key: One of safetensors_quant.SAFETENSORS_DTYPE_CHOICES' keys.
        overwrite: Skip existence check when True.
        on_progress: Optional callback(idx, total, key).
        on_log: Optional callback(msg); prints when None.
        cancel_event: Optional threading.Event; raises RuntimeError("cancelled") when set.

    Returns:
        (dst_path, model_arch)
    """
    def _log(msg):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    state_dict = load_state_dict(path)
    model_arch = detect_arch(state_dict)
    _log(f"INFO:  Architecture: {model_arch.arch}")

    if dst_path is None:
        dst_path = f"{os.path.splitext(path)[0]}-{target_key}.safetensors"

    if os.path.isfile(dst_path) and not overwrite:
        raise OSError(f"Output exists and overwrite is disabled: {dst_path}")

    out_tensors: dict[str, torch.Tensor] = {}
    layer_formats: dict[str, dict] = {}
    quant_format = _TARGET_TO_QUANT_FORMAT.get(target_key)

    items = list(state_dict.items())
    total = len(items)
    for idx, (key, data) in enumerate(items):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")
        if on_progress:
            on_progress(idx + 1, total, key)
        if any(x in key for x in model_arch.keys_ignore):
            continue

        data = torch.nan_to_num(data, nan=0.0, posinf=65504.0, neginf=-65504.0)
        quantized = quantize_tensor_st(data, key, model_arch, target_key)
        out_tensors.update(quantized)
        if quant_format and len(quantized) > 1:
            layer_formats[key] = {"format": quant_format}

    metadata = {"comfy.gguf_source_arch": model_arch.arch}
    if layer_formats:
        metadata["_quantization_metadata"] = json.dumps(
            {"format_version": "1.0", "layers": layer_formats}
        )

    _log(f"INFO:  Writing {len(out_tensors)} tensors → {dst_path}")
    save_file(out_tensors, dst_path, metadata=metadata)
    _log(f"INFO:  Done → {dst_path}")
    return dst_path, model_arch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_convert_safetensors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add convert_safetensors.py tests/test_convert_safetensors.py
git commit -m "feat: add safetensors-to-safetensors quantized conversion"
```

---

### Task 5: GUI — rename Convert tab, add "Convert → Safetensors" tab

**Files:**
- Modify: `gui.py:1` (module docstring), `gui.py:802` (Blocks title), `gui.py:808` (`gr.Tab("Convert")` → `gr.Tab("Convert → GGUF")`)
- Modify: `gui.py:24-36` (imports — add `from convert_safetensors import convert_to_safetensors` and `from safetensors_quant import SAFETENSORS_DTYPE_CHOICES`)
- Test: `tests/test_gui.py` (extend existing file)

**Interfaces:**
- Consumes: `convert_to_safetensors` (Task 4), `SAFETENSORS_DTYPE_CHOICES` (Task 1).
- Produces: nothing new consumed elsewhere — this is the leaf UI task.

- [ ] **Step 1: Write the failing test — new tab exists and dropdown has the right choices**

Check `tests/test_gui.py` first for the existing pattern used to assert on tab/component presence (likely inspecting `app.blocks` or launching in test mode); mirror that pattern. Minimal version if none exists:

```python
def test_convert_safetensors_tab_present():
    from gui import build_app
    app = build_app()
    label_texts = []
    for block in app.blocks.values():
        label = getattr(block, "label", None)
        if label:
            label_texts.append(label)
    # Dropdown label used in the new tab must be present
    assert any("Output format" == t for t in label_texts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gui.py -k convert_safetensors_tab_present -v`
Expected: FAIL — no such tab/dropdown yet

- [ ] **Step 3: Rename the Convert tab and add the new tab**

In `gui.py`, change:

```python
            with gr.Tab("Convert"):
```

to:

```python
            with gr.Tab("Convert → GGUF"):
```

Update the module docstring at `gui.py:1` from `"""Web UI for safetensors2GGUF — Gradio frontend for convert, quantize, and fix_5d."""` to `"""Web UI for safetensors2GGUF — Gradio frontend for GGUF convert, safetensors convert, quantize, and fix_5d."""`.

Add the import block:

```python
from convert_safetensors import convert_to_safetensors
from safetensors_quant import SAFETENSORS_DTYPE_CHOICES
```

Insert a new tab immediately after the (renamed) `Convert → GGUF` tab's closing block (before `# ── Fix Pad Tokens ──` at `gui.py:887`):

```python
            # ── Convert → Safetensors ──────────────────────────────────────
            with gr.Tab("Convert → Safetensors"):
                gr.Markdown(
                    "Convert a **Safetensors / CKPT** model checkpoint to a quantized "
                    "**Safetensors** file — no GGUF, no llama-quantize.  FP8 uses "
                    "ComfyUI's `scaled_fp8` convention (per-layer `weight_scale`); "
                    "NVFP4 uses Nvidia's 16-block scaled format — both load natively "
                    "in ComfyUI without the GGUF loader node."
                )
                with gr.Column(elem_classes=["card"]):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            st_src_path = gr.Textbox(
                                label="Source model",
                                placeholder="model.safetensors / .ckpt / .pt / .bin / .pth",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_st_src_btn = gr.Button("Browse", size="sm")
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            st_dst_path = gr.Textbox(
                                label="Output path",
                                placeholder="Auto-generated next to source",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_st_dst_btn = gr.Button("Browse", size="sm")
                    st_format_dropdown = gr.Dropdown(
                        choices=SAFETENSORS_DTYPE_CHOICES,
                        value="FP8",
                        label="Output format",
                    )
                    overwrite_st = gr.Checkbox(label="Overwrite existing output", value=False)

                st_convert_btn = gr.Button("▶  Convert", variant="primary", elem_id="st-convert-btn")
                st_status = gr.Textbox(
                    value="Ready", show_label=False, interactive=False,
                    lines=1, max_lines=1, elem_id="st-status",
                )
                st_log = gr.Textbox(
                    label="Log", lines=10, max_lines=10,
                    interactive=False, autoscroll=False, elem_id="st-log",
                )

                def _browse_st_src():
                    return browse_model()

                def _browse_st_dst():
                    result = _browse(_MODEL_TYPES)
                    return result

                def _run_st_convert(src, dst, fmt, overwrite):
                    if not src or not os.path.isfile(src):
                        yield "Error: source file not found", ""
                        return
                    log_lines: list[str] = []

                    def _on_log(msg):
                        log_lines.append(msg)

                    try:
                        dst_final, _ = convert_to_safetensors(
                            src, dst_path=(dst or None), target_key=fmt,
                            overwrite=overwrite, on_log=_on_log,
                        )
                        yield f"Done → {dst_final}", "\n".join(log_lines)
                    except Exception as exc:
                        yield f"Error: {exc}", "\n".join(log_lines)

                browse_st_src_btn.click(_browse_st_src, outputs=st_src_path)
                browse_st_dst_btn.click(_browse_st_dst, outputs=st_dst_path)
                st_convert_btn.click(
                    _run_st_convert,
                    inputs=[st_src_path, st_dst_path, st_format_dropdown, overwrite_st],
                    outputs=[st_status, st_log],
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gui.py -k convert_safetensors_tab_present -v`
Expected: PASS

- [ ] **Step 5: Manually launch the GUI and click through both tabs**

Run: `uv run python gui.py`, open the browser, confirm "Convert → GGUF" and "Convert → Safetensors" tabs both render, pick a small local test checkpoint, run an F16 safetensors conversion, confirm the output file appears and loads via `safetensors.torch.load_file`.

- [ ] **Step 6: Commit**

```bash
git add gui.py tests/test_gui.py
git commit -m "feat: add Convert -> Safetensors tab, rename Convert -> Convert -> GGUF"
```

---

## Part B — Architecture coverage (Flux.2, Z-Image, Qwen-Image-Edit)

### Task 6: Verify and document existing coverage — no new detection code

**Files:**
- Modify: `README.md:11-25` (Supported Architectures table)
- Modify: `docs/architecture.md` (append verification notes)

**Rationale (do not re-derive — already confirmed against real local checkpoints during planning):**
- **Flux.2** (klein 9B, Dev): `detect_arch()` already returns `arch="flux"` — confirmed against `snofsSexNudesAndOtherFunStuff_distilledV12Fp8.safetensors`. ComfyUI-GGUF itself reuses `arch="flux"` for Flux.2 (no distinct `flux2` tag exists upstream), so this is correct, not a false positive. No code change.
- **Z-Image** (Turbo, Base): `detect_arch()` already returns `arch="lumina2"` — confirmed against `jibMixZIT_v10.safetensors` (identical NextDiT tensor names: `cap_embedder`, `context_refiner`, `noise_refiner`, `x_pad_token`, `cap_pad_token`). Already covered by the existing `test_lumina2` test in `tests/test_convert.py`. No code change.
- **Qwen-Image-Edit 2511**: covered by the existing `ModelQwenImage` class (added in commit `400f861`). No local raw-safetensors source was available to re-verify against the specific 2511 checkpoint revision; flagged as "verify when a raw safetensors source becomes available", not blocking.

- [ ] **Step 1: Update README's Supported Architectures table**

In `README.md`, extend the table at line 11 to add:

```markdown
| Flux.2 (klein / Dev) | Diffusers (shares `flux` arch tag with Flux.1) |
| Z-Image (Turbo / Base) | Diffusers (shares `lumina2` arch tag) |
| Qwen-Image / Qwen-Image-Edit (incl. 2511) | Diffusers |
```

- [ ] **Step 2: Append a verification note to `docs/architecture.md`**

Add a short section documenting that Flux.2 and Z-Image required no new `ModelTemplate` subclass because their tensor-key layout is identical to Flux.1 and Lumina2 respectively, with a pointer to which local checkpoint files were used to confirm it (so future maintainers know this was empirically verified, not assumed).

- [ ] **Step 3: Commit**

```bash
git add README.md docs/architecture.md
git commit -m "docs: confirm Flux.2 and Z-Image are already covered by flux/lumina2 arch detection"
```

---

## Part C — Text-Encoder → GGUF conversion

### Task 7: `text_encoder_convert.py` — locate `convert_hf_to_gguf.py` and the embedded Python interpreter

**Files:**
- Create: `text_encoder_convert.py`
- Test: `tests/test_text_encoder_convert.py`

**Interfaces:**
- Produces:
  - `find_convert_script() -> Path | None` — searches ComfyUI-Easy-Install roots for `python_embeded/Lib/site-packages/llama_cpp/bin/convert_hf_to_gguf.py`.
  - `find_embedded_python() -> Path | None` — searches the same roots for `python_embeded/python.exe`.
  - `TEXT_ENCODER_OUTTYPES: list[tuple[str, str]]` — GUI dropdown choices mapped to `convert_hf_to_gguf.py --outtype` values: `[("F16", "f16"), ("BF16", "bf16"), ("Q8_0", "q8_0"), ("F32", "f32")]` (the script's own `--outtype` flag already handles these; no separate llama-quantize step needed for text encoders since `convert_hf_to_gguf.py` quantizes directly for these types).

Reuses the same Easy-Install root discovery pattern as `quantize.py:_easy_install_roots()` — do not duplicate that logic; import and call it.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for text_encoder_convert.py — locating convert_hf_to_gguf.py and embedded Python."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from text_encoder_convert import (
    TEXT_ENCODER_OUTTYPES,
    find_convert_script,
    find_embedded_python,
)


class TestOuttypes:
    def test_is_list_of_tuples(self):
        assert isinstance(TEXT_ENCODER_OUTTYPES, list)
        for label, value in TEXT_ENCODER_OUTTYPES:
            assert isinstance(label, str) and label
            assert isinstance(value, str) and value


class TestFindConvertScript:
    def test_returns_none_when_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COMFYUI_EASY_INSTALL_HOME", str(tmp_path / "nonexistent"))
        with patch("text_encoder_convert._easy_install_roots", return_value=[tmp_path]):
            assert find_convert_script() is None

    def test_finds_script_under_easy_install_root(self, tmp_path):
        script_dir = tmp_path / "python_embeded" / "Lib" / "site-packages" / "llama_cpp" / "bin"
        script_dir.mkdir(parents=True)
        script = script_dir / "convert_hf_to_gguf.py"
        script.write_text("# stub")
        with patch("text_encoder_convert._easy_install_roots", return_value=[tmp_path]):
            found = find_convert_script()
            assert found == script


class TestFindEmbeddedPython:
    def test_finds_python_exe_under_easy_install_root(self, tmp_path):
        py_dir = tmp_path / "python_embeded"
        py_dir.mkdir(parents=True)
        py_exe = py_dir / "python.exe"
        py_exe.write_text("stub")
        with patch("text_encoder_convert._easy_install_roots", return_value=[tmp_path]):
            found = find_embedded_python()
            assert found == py_exe
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_text_encoder_convert.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'text_encoder_convert'`

- [ ] **Step 3: Implement discovery functions**

```python
"""Text-encoder (LLM/T5) checkpoint -> GGUF conversion.

Bare single-file text-encoder safetensors (as ComfyUI's models/text_encoders/
folder holds — no accompanying config.json/tokenizer files) cannot be
converted by llama.cpp's convert_hf_to_gguf.py directly: that script hard-
requires config.json + tokenizer files (see llama.cpp ModelBase.load_hparams
and its --remote fetch list). There is no tensor-shape-only fallback in
either llama.cpp or ComfyUI-GGUF's public tooling.

Workflow implemented here:
  1. User supplies the local weights file + a HuggingFace repo ID for the
     *base* model (manual field -- no auto-detection, see plan header).
  2. Download that base repo's config.json + tokenizer files via
     huggingface_hub.
  3. Assemble a temp directory: downloaded config/tokenizer + the local
     weights renamed to what convert_hf_to_gguf.py expects.
  4. Run convert_hf_to_gguf.py as a subprocess, using the ComfyUI-Easy-
     Install embedded Python interpreter (it already has transformers /
     torch / mistral_common installed; this repo's own venv does not need
     those heavy deps as a result).
"""

from __future__ import annotations

from pathlib import Path

from quantize import _easy_install_roots

CONVERT_SCRIPT_RELATIVE = Path("python_embeded") / "Lib" / "site-packages" / "llama_cpp" / "bin" / "convert_hf_to_gguf.py"
EMBEDDED_PYTHON_RELATIVE = Path("python_embeded") / "python.exe"

TEXT_ENCODER_OUTTYPES: list[tuple[str, str]] = [
    ("F32", "f32"),
    ("F16", "f16"),
    ("BF16", "bf16"),
    ("Q8_0", "q8_0"),
]


def find_convert_script() -> Path | None:
    """Return the path to convert_hf_to_gguf.py under a discoverable
    ComfyUI-Easy-Install root, or None if not found."""
    for root in _easy_install_roots():
        candidate = root / CONVERT_SCRIPT_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def find_embedded_python() -> Path | None:
    """Return the path to the ComfyUI-Easy-Install embedded python.exe,
    or None if not found."""
    for root in _easy_install_roots():
        candidate = root / EMBEDDED_PYTHON_RELATIVE
        if candidate.is_file():
            return candidate
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_text_encoder_convert.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add text_encoder_convert.py tests/test_text_encoder_convert.py
git commit -m "feat: locate convert_hf_to_gguf.py and embedded python for text-encoder conversion"
```

---

### Task 8: Add `huggingface_hub` dependency and config/tokenizer fetch helper

**Files:**
- Modify: `pyproject.toml:5-11` (add `huggingface_hub` to `dependencies`)
- Modify: `text_encoder_convert.py` (add fetch function)
- Test: `tests/test_text_encoder_convert.py` (extend)

**Interfaces:**
- Consumes: `huggingface_hub.hf_hub_download`.
- Produces: `fetch_base_config_files(repo_id: str, dest_dir: Path, on_log=None) -> list[str]` — downloads `config.json` and, best-effort, whichever tokenizer files exist in the repo (`tokenizer.json`, `tokenizer_config.json`, `tokenizer.model`, `special_tokens_map.json`) into `dest_dir`; returns the list of filenames actually downloaded. Raises if `config.json` itself is missing (that one is mandatory, tokenizer files are attempted individually so a repo missing one variant doesn't abort the whole fetch).

- [ ] **Step 1: Update `pyproject.toml`**

```toml
dependencies = [
    "gguf",
    "torch",
    "safetensors",
    "tqdm",
    "gradio>=6.14.0",
    "huggingface_hub",
]
```

Run: `uv sync`

- [ ] **Step 2: Write the failing test**

```python
class TestFetchBaseConfigFiles:
    def test_downloads_config_and_available_tokenizer_files(self, tmp_path):
        from text_encoder_convert import fetch_base_config_files

        calls = []

        def _fake_download(repo_id, filename, local_dir):
            calls.append(filename)
            if filename not in ("config.json", "tokenizer.json"):
                raise Exception("not found")  # simulate missing optional file
            out = Path(local_dir) / filename
            out.write_text("{}")
            return str(out)

        with patch("text_encoder_convert.hf_hub_download", side_effect=_fake_download):
            downloaded = fetch_base_config_files("Qwen/Qwen3-8B", tmp_path)

        assert "config.json" in downloaded
        assert "tokenizer.json" in downloaded
        assert (tmp_path / "config.json").is_file()

    def test_raises_if_config_json_missing(self, tmp_path):
        from text_encoder_convert import fetch_base_config_files

        def _always_fail(repo_id, filename, local_dir):
            raise Exception("404")

        with patch("text_encoder_convert.hf_hub_download", side_effect=_always_fail):
            try:
                fetch_base_config_files("bad/repo", tmp_path)
                assert False, "expected RuntimeError"
            except RuntimeError:
                pass
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_text_encoder_convert.py -k FetchBaseConfigFiles -v`
Expected: FAIL — `fetch_base_config_files` / `hf_hub_download` not defined in module

- [ ] **Step 4: Implement the fetch helper**

Add to `text_encoder_convert.py`:

```python
from huggingface_hub import hf_hub_download

_MANDATORY_FILES = ("config.json",)
_OPTIONAL_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
)


def fetch_base_config_files(repo_id: str, dest_dir: Path, on_log=None) -> list[str]:
    """Download config.json + whichever tokenizer files exist for repo_id into dest_dir.

    config.json is mandatory (RuntimeError if missing); tokenizer files are
    best-effort since repos vary in which ones they ship.
    """
    def _log(msg):
        if on_log:
            on_log(msg)

    downloaded: list[str] = []
    for filename in _MANDATORY_FILES:
        try:
            hf_hub_download(repo_id, filename, local_dir=str(dest_dir))
            downloaded.append(filename)
        except Exception as exc:
            raise RuntimeError(f"Required file {filename!r} not found in {repo_id!r}: {exc}") from exc

    for filename in _OPTIONAL_TOKENIZER_FILES:
        try:
            hf_hub_download(repo_id, filename, local_dir=str(dest_dir))
            downloaded.append(filename)
            _log(f"INFO:  Downloaded {filename}")
        except Exception:
            continue  # optional — not every repo ships every tokenizer variant

    return downloaded
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_text_encoder_convert.py -k FetchBaseConfigFiles -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock text_encoder_convert.py tests/test_text_encoder_convert.py
git commit -m "feat: fetch base-model config/tokenizer files from HuggingFace for text-encoder conversion"
```

---

### Task 9: `convert_text_encoder()` — assemble temp dir and run the subprocess

**Files:**
- Modify: `text_encoder_convert.py` (add the orchestration function)
- Test: `tests/test_text_encoder_convert.py` (extend)

**Interfaces:**
- Consumes: `find_convert_script`, `find_embedded_python`, `fetch_base_config_files` (this module); `hf_hub_download` (huggingface_hub).
- Produces: `convert_text_encoder(weights_path: str, base_repo_id: str, dst_path: str | None = None, outtype: str = "f16", on_log=None, cancel_event=None) -> str` — returns the output GGUF path. Raises `FileNotFoundError` if the convert script or embedded python isn't found, `RuntimeError` on non-zero subprocess exit.

The temp directory needs the weights file named so `convert_hf_to_gguf.py` recognizes it — that script auto-discovers `model.safetensors` or `model.safetensors.index.json` in the target directory (single-shard HF layout). Copy (not move) the user's file to `<tmpdir>/model.safetensors` so the original is untouched.

- [ ] **Step 1: Write the failing test**

```python
class TestConvertTextEncoder:
    def test_raises_when_convert_script_not_found(self, tmp_path):
        from text_encoder_convert import convert_text_encoder

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stub")
        with patch("text_encoder_convert.find_convert_script", return_value=None):
            try:
                convert_text_encoder(str(weights), "Qwen/Qwen3-8B")
                assert False, "expected FileNotFoundError"
            except FileNotFoundError:
                pass

    def test_runs_subprocess_with_expected_args(self, tmp_path):
        from text_encoder_convert import convert_text_encoder

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stub")
        script = tmp_path / "convert_hf_to_gguf.py"
        script.write_text("# stub")
        py_exe = tmp_path / "python.exe"
        py_exe.write_text("stub")

        with patch("text_encoder_convert.find_convert_script", return_value=script), \
             patch("text_encoder_convert.find_embedded_python", return_value=py_exe), \
             patch("text_encoder_convert.fetch_base_config_files", return_value=["config.json"]), \
             patch("text_encoder_convert.subprocess.Popen") as mock_popen:
            mock_proc = mock_popen.return_value
            mock_proc.stdout = iter(["INFO: done\n"])
            mock_proc.wait.return_value = 0
            mock_proc.returncode = 0

            out = convert_text_encoder(
                str(weights), "Qwen/Qwen3-8B", dst_path=str(tmp_path / "out.gguf"),
                outtype="f16",
            )

        assert out == str(tmp_path / "out.gguf")
        called_cmd = mock_popen.call_args[0][0]
        assert str(py_exe) == called_cmd[0]
        assert str(script) == called_cmd[1]
        assert "--outtype" in called_cmd
        assert "f16" in called_cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_text_encoder_convert.py -k ConvertTextEncoder -v`
Expected: FAIL — `convert_text_encoder` not defined

- [ ] **Step 3: Implement the orchestration function**

Add to `text_encoder_convert.py`:

```python
import shutil
import subprocess
import tempfile


def convert_text_encoder(
    weights_path: str,
    base_repo_id: str,
    dst_path: str | None = None,
    outtype: str = "f16",
    on_log=None,
    cancel_event=None,
) -> str:
    """Convert a bare single-file text-encoder checkpoint to GGUF.

    Downloads config.json/tokenizer files for base_repo_id, assembles a temp
    HF-style model directory with the local weights, then runs
    convert_hf_to_gguf.py via the ComfyUI-Easy-Install embedded Python.
    """
    def _log(msg):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    script = find_convert_script()
    if script is None:
        raise FileNotFoundError(
            "convert_hf_to_gguf.py not found — expected under a ComfyUI-Easy-Install "
            f"root at {CONVERT_SCRIPT_RELATIVE}"
        )
    py_exe = find_embedded_python()
    if py_exe is None:
        raise FileNotFoundError(
            f"Embedded python.exe not found — expected under a ComfyUI-Easy-Install "
            f"root at {EMBEDDED_PYTHON_RELATIVE}"
        )

    if dst_path is None:
        dst_path = f"{weights_path.rsplit('.', 1)[0]}-{outtype}.gguf"

    with tempfile.TemporaryDirectory(prefix="s2g_text_encoder_") as tmpdir:
        tmp_path = Path(tmpdir)
        _log(f"INFO:  Fetching config/tokenizer for {base_repo_id}…")
        fetch_base_config_files(base_repo_id, tmp_path, on_log=_log)

        weights_dst = tmp_path / "model.safetensors"
        shutil.copy2(weights_path, weights_dst)

        cmd = [
            str(py_exe), str(script), str(tmp_path),
            "--outfile", dst_path,
            "--outtype", outtype,
        ]
        _log(f"INFO:  $ {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                proc.wait()
                raise RuntimeError("cancelled")
            _log(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"convert_hf_to_gguf.py exited with code {proc.returncode}")

    return dst_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_text_encoder_convert.py -k ConvertTextEncoder -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add text_encoder_convert.py tests/test_text_encoder_convert.py
git commit -m "feat: assemble temp HF dir and run convert_hf_to_gguf.py for text-encoder conversion"
```

---

### Task 10: GUI — "Convert Text Encoder → GGUF" tab

**Files:**
- Modify: `gui.py` (imports, new tab after "Convert → Safetensors")
- Test: `tests/test_gui.py` (extend)

**Interfaces:**
- Consumes: `convert_text_encoder`, `TEXT_ENCODER_OUTTYPES`, `find_convert_script`, `find_embedded_python` from `text_encoder_convert.py`.

- [ ] **Step 1: Write the failing test**

```python
def test_text_encoder_tab_present():
    from gui import build_app
    app = build_app()
    label_texts = [getattr(b, "label", None) for b in app.blocks.values()]
    assert any("Base model HF repo ID" == t for t in label_texts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gui.py -k text_encoder_tab_present -v`
Expected: FAIL

- [ ] **Step 3: Add the import and the tab**

Add import:

```python
from text_encoder_convert import (
    TEXT_ENCODER_OUTTYPES,
    convert_text_encoder,
    find_convert_script,
    find_embedded_python,
)
```

Insert a new tab after the "Convert → Safetensors" tab block from Task 5:

```python
            # ── Convert Text Encoder → GGUF ─────────────────────────────────
            with gr.Tab("Convert Text Encoder → GGUF"):
                gr.Markdown(
                    "Convert a **bare single-file text-encoder checkpoint** (Qwen3, "
                    "Mistral, T5/UMT5, …) to GGUF.  Requires the base model's "
                    "`config.json`/tokenizer files — downloaded automatically from "
                    "the HuggingFace repo ID you provide (not the fine-tuned "
                    "checkpoint's repo, which usually doesn't have one — the "
                    "**original base model's** repo).  Runs `convert_hf_to_gguf.py` "
                    "from your ComfyUI-Easy-Install Python environment."
                )
                script_found = find_convert_script()
                py_found = find_embedded_python()
                te_setup_info = (
                    f"convert_hf_to_gguf.py: {script_found or 'NOT FOUND'}\n"
                    f"embedded python.exe: {py_found or 'NOT FOUND'}"
                )
                with gr.Column(elem_classes=["card"]):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, elem_classes=["path-input"]):
                            te_src_path = gr.Textbox(
                                label="Text-encoder weights file",
                                placeholder="qwen3_8b_abliterated.safetensors",
                                lines=1, max_lines=1,
                            )
                        with gr.Column(scale=0, min_width=124, elem_classes=["browse-col"]):
                            browse_te_src_btn = gr.Button("Browse", size="sm")
                    te_base_repo = gr.Textbox(
                        label="Base model HF repo ID",
                        placeholder="e.g. Qwen/Qwen3-8B",
                        lines=1, max_lines=1,
                        info="The ORIGINAL base model's repo (config.json/tokenizer source), not the fine-tune's.",
                    )
                    te_dst_path = gr.Textbox(
                        label="Output path",
                        placeholder="Auto-generated next to source",
                        lines=1, max_lines=1,
                    )
                    te_outtype = gr.Dropdown(
                        choices=TEXT_ENCODER_OUTTYPES, value="f16", label="Output type",
                    )
                    te_setup = gr.Textbox(
                        label="Setup", value=te_setup_info, lines=2, max_lines=2, interactive=False,
                    )

                te_convert_btn = gr.Button("▶  Convert", variant="primary", elem_id="te-convert-btn")
                te_status = gr.Textbox(
                    value="Ready", show_label=False, interactive=False,
                    lines=1, max_lines=1, elem_id="te-status",
                )
                te_log = gr.Textbox(
                    label="Log", lines=10, max_lines=10,
                    interactive=False, autoscroll=False, elem_id="te-log",
                )

                def _browse_te_src():
                    return browse_model()

                def _run_te_convert(src, repo_id, dst, outtype):
                    if not src or not os.path.isfile(src):
                        yield "Error: source file not found", ""
                        return
                    if not repo_id:
                        yield "Error: base model HF repo ID is required", ""
                        return
                    log_lines: list[str] = []

                    def _on_log(msg):
                        log_lines.append(msg)

                    try:
                        out = convert_text_encoder(
                            src, repo_id, dst_path=(dst or None), outtype=outtype, on_log=_on_log,
                        )
                        yield f"Done → {out}", "\n".join(log_lines)
                    except Exception as exc:
                        yield f"Error: {exc}", "\n".join(log_lines)

                browse_te_src_btn.click(_browse_te_src, outputs=te_src_path)
                te_convert_btn.click(
                    _run_te_convert,
                    inputs=[te_src_path, te_base_repo, te_dst_path, te_outtype],
                    outputs=[te_status, te_log],
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gui.py -k text_encoder_tab_present -v`
Expected: PASS

- [ ] **Step 5: Manual smoke test against a real local file**

Run: `uv run python gui.py`, open "Convert Text Encoder → GGUF", point it at `G:\ComfyUI-Easy-Install\ComfyUI\models\text_encoders\Z-Image Turbo\...` is already GGUF — instead use a real bare safetensors text encoder if available (e.g. `C:\ComfyUI-Models\models\text_encoders\Wan 2.2\umt5_xxl_fp8_e4m3fn_scaled.safetensors`) with base repo ID `google/umt5-xxl`, confirm the temp dir gets assembled and `convert_hf_to_gguf.py` starts (full run may take minutes — confirming it starts without an immediate error is sufficient for this smoke test).

- [ ] **Step 6: Commit**

```bash
git add gui.py tests/test_gui.py
git commit -m "feat: add Convert Text Encoder -> GGUF tab"
```

---

### Task 11: README + CHANGELOG updates

**Files:**
- Modify: `README.md` (new sections for Safetensors output and Text-Encoder conversion)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a "Safetensors Output" section to README** documenting the dtype table (F16/F16-mixed/FP8/FP8-mixed/NVFP4/NVFP4-mixed), the ComfyUI-compatibility rationale (scaled_fp8 / nvfp4 conventions), and that there is intentionally no unscaled/generic FP4 mode.

- [ ] **Step 2: Add a "Text-Encoder Conversion" section to README** documenting the base-repo-ID requirement, the ComfyUI-Easy-Install dependency (`convert_hf_to_gguf.py` + embedded `python.exe`), and that this is a separate subprocess pipeline, not part of the DiT architecture system.

- [ ] **Step 3: Add CHANGELOG entries** under an `[Unreleased]` heading for: safetensors output mode, NVFP4/FP8 quantization, text-encoder GGUF conversion, Flux.2/Z-Image support confirmation, new `huggingface_hub` dependency.

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: document safetensors output mode and text-encoder conversion"
```

---

## Self-Review Notes

- **Spec coverage:** F16/F16-mixed/FP8/FP8-mixed/NVFP4/NVFP4-mixed → Tasks 1–4. GUI rename + new safetensors tab → Task 5. Flux.2/Z-Image/Qwen-Image-Edit coverage → Task 6. Text-encoder GGUF conversion (manual repo-ID field, embedded-python discovery, huggingface_hub) → Tasks 7–10. Docs → Task 11.
- **Dropped from original ask, with reasoning recorded in-conversation:** generic/unscaled FP4 (ComfyUI has no loader for it — would produce files nothing can open); "NVFP8"/"NVFP6" (not real formats, no upstream spec or loader).
- **Type consistency check:** `quantize_tensor_st` (Task 1) is called with `target_key` matching `SAFETENSORS_DTYPE_CHOICES` keys throughout; `convert_to_safetensors` (Task 4) passes `target_key` straight through unchanged; GUI dropdown (Task 5) uses the same `SAFETENSORS_DTYPE_CHOICES` list as its `choices=`. `convert_text_encoder`'s `outtype` parameter matches `TEXT_ENCODER_OUTTYPES` values (`"f16"` not `"F16"` — lowercase, matching `convert_hf_to_gguf.py`'s own `--outtype` flag convention) consistently across Tasks 7, 9, 10.
