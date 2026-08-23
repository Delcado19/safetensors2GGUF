# Streaming Safetensors Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `convert_to_safetensors()`'s single accumulate-then-`save_file()` write path with a two-pass streaming writer (plan header, then append each tensor's bytes as it's quantized), so peak RAM scales with one tensor instead of the full model.

**Architecture:** Pass 1 computes every output tensor's exact (name, dtype, shape) from source shape/dtype metadata alone — no tensor data touched — using a new `plan_tensor_output()` extracted from the existing `estimate_safetensors_output_size()` branching. That plan becomes the safetensors JSON header, written once up front. Pass 2 re-runs the real `quantize_tensor_st()` per tensor exactly as today, but writes each result's bytes directly to the open file the instant it's produced, instead of accumulating an `out_tensors` dict. Both passes iterate `state_dict` through one shared filter/order helper, so Pass 2's tensors are guaranteed to arrive in the exact order Pass 1 planned them — no seeking, no offset bookkeeping, just sequential appends, checked with a cheap assert per tensor.

**Tech Stack:** Python, PyTorch, the `safetensors` Python package (still used for *reading* — `safe_open`/`load_file` — never for the new write path), pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-streaming-safetensors-writer-design.md`

## Global Constraints

- Streaming only in this plan — no resume/checkpoint (see spec's Scope section; that's a separate future plan).
- `convert_to_safetensors()`'s public signature and return value (`(dst_path, model_arch)`) do not change.
- Every existing test in `tests/test_convert_safetensors.py` and `tests/test_safetensors_quant.py` must stay green, unmodified, after this refactor — they assert on output *content*, not the write mechanism, so they're the primary regression net for "does the new writer produce the same file as the old one."
- No new third-party dependency — the writer is hand-rolled (see spec's "Explicitly not pursued": no other safetensors library streams writes either).
- `uv run pytest` is this project's test runner (see project conventions) — use it for every test-running step below, not bare `pytest`.

---

### Task 1: Reverse torch-dtype→safetensors-dtype-string map + float8 byte sizes

**Files:**
- Modify: `convert.py` (near `_ST_DTYPE_MAP`, convert.py:49)
- Modify: `safetensors_quant.py` (near `_ST_DTYPE_BYTES`, safetensors_quant.py:645)
- Test: `tests/test_safetensors_quant.py`

**Interfaces:**
- Produces: `convert._TORCH_TO_ST_DTYPE: dict[torch.dtype, str]` — inverse of the existing `_ST_DTYPE_MAP`. Later tasks import this from `convert` the same way `safetensors_quant.py` will need it.
- Produces: `safetensors_quant._ST_DTYPE_BYTES` gains `"F8_E4M3": 1, "F8_E5M2": 1` entries.

`_ST_DTYPE_MAP` already maps every safetensors dtype string this project ever writes to a torch dtype (F32/F16/BF16/F64/F8_E4M3/F8_E5M2/I8/I16/I32/I64 — confirm by reading convert.py:49-60 before writing the test). The reverse map is needed because the new planning path decides *torch* dtypes internally (mirroring `quantize_tensor_st`) but must emit *safetensors header* dtype strings.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_safetensors_quant.py — add to file, near the top-level tests
class TestTorchToStDtypeMap:
    def test_reverse_map_covers_every_st_dtype_map_entry(self):
        from convert import _ST_DTYPE_MAP, _TORCH_TO_ST_DTYPE
        for st_name, torch_dtype in _ST_DTYPE_MAP.items():
            if torch_dtype is None:
                continue  # float8 dtypes are None on old PyTorch builds
            assert _TORCH_TO_ST_DTYPE[torch_dtype] == st_name

    def test_f8_dtypes_have_byte_sizes(self):
        from safetensors_quant import _ST_DTYPE_BYTES
        assert _ST_DTYPE_BYTES["F8_E4M3"] == 1
        assert _ST_DTYPE_BYTES["F8_E5M2"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_safetensors_quant.py::TestTorchToStDtypeMap -v`
Expected: FAIL — `ImportError: cannot import name '_TORCH_TO_ST_DTYPE'` (and the byte-size test fails with `KeyError: 'F8_E4M3'`).

- [ ] **Step 3: Implement**

In `convert.py`, immediately after the `_ST_DTYPE_MAP` dict literal (convert.py:49-61 or wherever it ends):

```python
# Reverse of the above -- torch dtype -> safetensors header dtype string.
# Used by the streaming safetensors writer (convert_safetensors.py) and
# safetensors_quant.plan_tensor_output() to build header entries for
# tensors that keep their original dtype (mixed-precision hiprec passthrough,
# passthrough sentinel blobs) instead of one of quantize_tensor_st's own
# fixed output dtypes.
_TORCH_TO_ST_DTYPE: dict = {
    v: k for k, v in _ST_DTYPE_MAP.items() if v is not None
}
```

In `safetensors_quant.py`, edit the existing `_ST_DTYPE_BYTES` dict (safetensors_quant.py:645-648):

```python
_ST_DTYPE_BYTES: dict[str, int] = {
    "F32": 4, "F64": 8, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_safetensors_quant.py::TestTorchToStDtypeMap -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add convert.py safetensors_quant.py tests/test_safetensors_quant.py
git commit -m "feat: add torch-dtype to safetensors-dtype reverse map"
```

---

### Task 2: Extract `_is_hiprec_shape()` — shape-only hiprec predicate

**Files:**
- Modify: `safetensors_quant.py` (`is_hiprec_st`, safetensors_quant.py:481-503)
- Test: `tests/test_safetensors_quant.py`

**Interfaces:**
- Produces: `safetensors_quant._is_hiprec_shape(key: str, shape: tuple[int, ...], old_dtype: torch.dtype, model_arch) -> bool`
- `is_hiprec_st(key, data, model_arch, old_dtype)`'s public signature and behavior are UNCHANGED — it becomes a thin wrapper delegating to `_is_hiprec_shape`. Every existing caller/test keeps working without modification (5 existing test call sites in `tests/test_safetensors_quant.py` — do not touch them).

`is_hiprec_st` only ever calls `data.dim()` and `data.numel()` on the tensor — never reads a value. Both are derivable from a shape tuple alone (`len(shape)`, product of `shape`). This extraction lets the new shape-only planning path (`plan_tensor_output`, Task 4) reuse the exact same hiprec rule instead of re-implementing it — the two must never drift apart, since drift here would make the planned header wrong for `_MIXED` formats.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_safetensors_quant.py
class TestIsHiprecShape:
    def test_matches_is_hiprec_st_for_1d(self):
        from safetensors_quant import _is_hiprec_shape, is_hiprec_st
        from models.architectures import ModelFlux
        import torch
        data = torch.randn(64, dtype=torch.float32)
        assert _is_hiprec_shape("some.bias", (64,), torch.float32, ModelFlux()) == \
            is_hiprec_st("some.bias", data, ModelFlux(), torch.float32)

    def test_matches_is_hiprec_st_for_keys_hiprec_match(self):
        from safetensors_quant import _is_hiprec_shape, is_hiprec_st
        from models.architectures import ModelLumina2
        import torch
        data = torch.randn(4096, 4096, dtype=torch.bfloat16)
        assert _is_hiprec_shape("x_pad_token.weight", (4096, 4096), torch.bfloat16, ModelLumina2()) == \
            is_hiprec_st("x_pad_token.weight", data, ModelLumina2(), torch.bfloat16)

    def test_non_float_dtype_is_never_hiprec(self):
        from safetensors_quant import _is_hiprec_shape
        from models.architectures import ModelFlux
        import torch
        assert _is_hiprec_shape("any.weight", (4096, 4096), torch.int8, ModelFlux()) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_safetensors_quant.py::TestIsHiprecShape -v`
Expected: FAIL — `ImportError: cannot import name '_is_hiprec_shape'`

- [ ] **Step 3: Implement**

Replace `is_hiprec_st` (safetensors_quant.py:481-503) with:

```python
def _is_hiprec_shape(
    key: str, shape: tuple[int, ...], old_dtype: torch.dtype, model_arch
) -> bool:
    """Shape/dtype-only hiprec predicate -- the actual logic behind
    is_hiprec_st(), extracted so the streaming writer's planning pass
    (plan_tensor_output(), see below) can reuse it without a real tensor.
    Never inspects tensor values, only dim()/numel(), both derivable from
    a shape tuple -- see is_hiprec_st's docstring for the F16-dtype-gate
    history this must not regress."""
    if old_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    if len(shape) == 1:
        return True
    numel = 1
    for d in shape:
        numel *= d
    if numel <= QUANTIZATION_THRESHOLD:
        return True
    if any(x in key for x in model_arch.keys_hiprec):
        return True
    return False


def is_hiprec_st(key: str, data: torch.Tensor, model_arch, old_dtype: torch.dtype) -> bool:
    """Return True if ``key`` must stay high-precision (F32), mirroring
    convert._quant_type_for's rule so 'mixed' safetensors output matches the
    existing GGUF mixed-precision behaviour exactly."""
    return _is_hiprec_shape(key, tuple(data.shape), old_dtype, model_arch)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_safetensors_quant.py -v -k "IsHiprecShape or Hiprec"`
Expected: PASS — including the 5 pre-existing `is_hiprec_st` tests, unmodified.

- [ ] **Step 5: Commit**

```bash
git add safetensors_quant.py tests/test_safetensors_quant.py
git commit -m "refactor: extract shape-only _is_hiprec_shape from is_hiprec_st"
```

---

### Task 3: `dequantized_shape_of()` — invert NVFP4's packed last-dim

**Files:**
- Modify: `dequantize.py` (near `dequantize_nvfp4` import, dequantize.py:29)
- Test: `tests/test_dequantize.py` (create if it doesn't already exist — check first with a quick search for other dequantize tests, e.g. inside `tests/test_convert_safetensors.py` or a dedicated file, and add alongside whatever's already there)

**Interfaces:**
- Produces: `dequantize.dequantized_shape_of(fmt: str, packed_shape: tuple[int, ...]) -> tuple[int, ...]`

FP8/INT8-sourced already-quantized ".weight" tensors keep their on-disk shape after dequantization (weight + sidecar scale merge back to the same shape). NVFP4-sourced ones don't: `quantize_nvfp4()` packs 2 values per byte in the last dimension (safetensors_quant_nvfp4.py:141, `packed = packed.reshape(*lead, last // 2)`), and `dequantize_nvfp4()` doubles it back (safetensors_quant_nvfp4.py:185, `values.reshape(*lead, half * 2)`). This function makes that specific inverse available from shape metadata alone, without loading the tensor — needed so Pass 1 can plan the correct post-dequant shape for NVFP4-sourced already-quantized layers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dequantize.py
import pytest


class TestDequantizedShapeOf:
    def test_nvfp4_doubles_last_dim(self):
        from dequantize import dequantized_shape_of
        assert dequantized_shape_of("nvfp4", (64, 32)) == (64, 32 * 2)

    def test_fp8_and_int8_unchanged(self):
        from dequantize import dequantized_shape_of
        assert dequantized_shape_of("float8_e4m3fn", (64, 64)) == (64, 64)
        assert dequantized_shape_of("int8_tensorwise", (64, 64)) == (64, 64)

    def test_unknown_format_raises(self):
        from dequantize import dequantized_shape_of
        with pytest.raises(ValueError):
            dequantized_shape_of("bogus", (64, 64))

    def test_matches_real_dequantize_nvfp4_shape(self):
        # Cross-check against the real quantize/dequantize round trip rather
        # than trusting the packing-math in isolation.
        import torch
        from dequantize import dequantized_shape_of
        from safetensors_quant_nvfp4 import quantize_nvfp4, dequantize_nvfp4

        data = torch.randn(64, 32, dtype=torch.float32)
        packed = quantize_nvfp4(data, "layer.weight")
        real = dequantize_nvfp4(packed, "layer.weight")
        assert dequantized_shape_of("nvfp4", tuple(packed["layer.weight"].shape)) == tuple(real.shape)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dequantize.py::TestDequantizedShapeOf -v`
Expected: FAIL — `ImportError: cannot import name 'dequantized_shape_of'`

- [ ] **Step 3: Implement**

In `dequantize.py`, after the `dequantize_nvfp4` import (dequantize.py:29):

```python
def dequantized_shape_of(fmt: str, packed_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the shape a ".weight" tensor of on-disk quant format ``fmt``
    will have AFTER dequantize_weight() reconstructs it -- from shape alone,
    no tensor data touched. Used by convert_safetensors.py's streaming
    writer Pass 1 to plan already-quantized sources' post-dequant shape.

    Only NVFP4 changes shape on disk (2 values packed per byte, halving the
    last dimension -- see safetensors_quant_nvfp4.quantize_nvfp4's
    `packed.reshape(*lead, last // 2)` and its inverse in
    dequantize_nvfp4's `values.reshape(*lead, half * 2)`, which this
    function's nvfp4 branch mirrors). FP8/INT8 weight+scale sidecars merge
    back to the original shape unchanged.
    """
    if fmt == "nvfp4":
        return (*packed_shape[:-1], packed_shape[-1] * 2)
    if fmt in ("float8_e4m3fn", "int8_tensorwise"):
        return tuple(packed_shape)
    raise ValueError(f"Unsupported quantized format for shape inference: {fmt!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dequantize.py::TestDequantizedShapeOf -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dequantize.py tests/test_dequantize.py
git commit -m "feat: add dequantized_shape_of for NVFP4 packed-shape inference"
```

---

### Task 4: `plan_tensor_output()` — the shared planning function

**Files:**
- Modify: `safetensors_quant.py` (add near `quantize_tensor_st`, safetensors_quant.py:506)
- Test: `tests/test_safetensors_quant.py`

**Interfaces:**
- Consumes: `_is_hiprec_shape` (Task 2), `_TORCH_TO_ST_DTYPE` (Task 1, imported from `convert`), `layer_key` (existing), `_BASE_KEY`/`_MIXED_KEYS` (existing), `_CONVROT_GROUP_SIZE` (existing).
- Produces: `plan_tensor_output(key: str, shape: tuple[int, ...], old_dtype: torch.dtype, model_arch, target_key: str, full_precision_fp8: bool = True, full_precision_nvfp4: bool = True) -> tuple[list[tuple[str, str, tuple[int, ...]]], dict | None]` — later tasks (5, 6) call this exact signature. Return value: `(entries, layer_conf)` where `entries` is `[(output_name, safetensors_dtype_string, output_shape), ...]` in the SAME order `quantize_tensor_st()`'s real output dict would have (verified by Step 1's cross-check test below), and `layer_conf` is the `_quantization_metadata` "layers" fragment for this key (or `None` if it produces no scale sidecar).

This is the core of the whole plan: it must reproduce `quantize_tensor_st()`'s exact branching and every real quantize function's exact output tensor names/dtypes/shapes, using only shape/dtype metadata. Read `quantize_tensor_st()` (safetensors_quant.py:506-642), `quantize_fp8_scaled()` (safetensors_quant_fp8.py), `quantize_int8_tensorwise()`/`quantize_int8_convrot()` (safetensors_quant_int8.py), and `quantize_nvfp4()` (safetensors_quant_nvfp4.py) side by side with the implementation below before touching test cases — the cross-check test in Step 1 is what actually proves this is correct, not code review.

- [ ] **Step 1: Write the failing test — cross-check against the REAL quantize functions**

```python
# tests/test_safetensors_quant.py
class TestPlanTensorOutput:
    """plan_tensor_output() must predict exactly what quantize_tensor_st()
    actually produces, for every branch. Each case below runs BOTH and
    compares -- this is the real correctness proof, not the implementation
    reading right."""

    def _check(self, data, key, model_arch, target_key, **kw):
        from safetensors_quant import plan_tensor_output, quantize_tensor_st
        old_dtype = data.dtype
        real = quantize_tensor_st(data, key, model_arch, target_key)
        planned_entries, planned_layer_conf = plan_tensor_output(
            key, tuple(data.shape), old_dtype, model_arch, target_key, **kw
        )
        assert list(real.keys()) == [name for name, _, _ in planned_entries]
        from convert import _TORCH_TO_ST_DTYPE
        for (name, st_dtype, shape), real_name in zip(planned_entries, real.keys()):
            assert name == real_name
            assert _TORCH_TO_ST_DTYPE[real[real_name].dtype] == st_dtype, (
                f"{name}: planned dtype {st_dtype}, real {real[real_name].dtype}"
            )
            assert tuple(real[real_name].shape) == shape, (
                f"{name}: planned shape {shape}, real {tuple(real[real_name].shape)}"
            )
        return planned_layer_conf

    def test_f16(self):
        import torch
        from models.architectures import ModelFlux
        data = torch.randn(64, 64, dtype=torch.float32)
        conf = self._check(data, "some.weight", ModelFlux(), "F16")
        assert conf is None

    def test_1d_bias_stays_f32(self):
        import torch
        from models.architectures import ModelFlux
        data = torch.randn(64, dtype=torch.float32)
        conf = self._check(data, "some.bias", ModelFlux(), "FP8")
        assert conf is None

    def test_conv_weight_gte_3d_stays_f16(self):
        import torch
        from models.architectures import ModelSDXL
        data = torch.randn(32, 4, 3, 3, dtype=torch.float32)
        conf = self._check(data, "conv.weight", ModelSDXL(), "FP8")
        assert conf is None

    def test_mixed_hiprec_keeps_original_dtype(self):
        import torch
        from models.architectures import ModelLumina2
        data = torch.randn(4096, 4096, dtype=torch.bfloat16)
        conf = self._check(data, "x_pad_token.weight", ModelLumina2(), "FP8_MIXED")
        assert conf is None

    def test_fp8_plain(self):
        import torch
        from models.architectures import ModelFlux
        data = torch.randn(256, 256, dtype=torch.float32)
        conf = self._check(data, "some.weight", ModelFlux(), "FP8")
        assert conf == {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}

    def test_fp8_shape_critical_stays_f16(self):
        import torch
        from models.architectures import ModelFlux
        data = torch.randn(256, 256, dtype=torch.float32)
        arch = ModelFlux()
        key = arch.keys_shape_critical[0] if arch.keys_shape_critical else "txt_in.weight"
        conf = self._check(data, key, arch, "FP8")
        assert conf is None

    def test_int8_tensorwise_odd_infeatures(self):
        import torch
        from models.architectures import ModelFlux
        data = torch.randn(64, 100, dtype=torch.float32)  # 100 % 256 != 0
        conf = self._check(data, "some.weight", ModelFlux(), "INT8")
        assert conf == {"format": "int8_tensorwise"}

    def test_int8_convrot(self):
        import torch
        from models.architectures import ModelFlux
        data = torch.randn(64, 512, dtype=torch.float32)  # 512 % 256 == 0
        conf = self._check(data, "some.weight", ModelFlux(), "INT8")
        assert conf == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}

    def test_nvfp4_plain(self):
        import torch
        from models.architectures import ModelFlux
        data = torch.randn(64, 32, dtype=torch.float32)  # 32 % 16 == 0
        conf = self._check(data, "some.weight", ModelFlux(), "NVFP4")
        assert conf == {"format": "nvfp4", "full_precision_matrix_mult": True}

    def test_nvfp4_non_block_aligned_falls_back_to_f16(self):
        import torch
        from models.architectures import ModelFlux
        data = torch.randn(64, 20, dtype=torch.float32)  # 20 % 16 != 0
        conf = self._check(data, "some.weight", ModelFlux(), "NVFP4")
        assert conf is None

    def test_full_precision_flags_off(self):
        import torch
        from models.architectures import ModelFlux
        data = torch.randn(256, 256, dtype=torch.float32)
        conf = self._check(
            data, "some.weight", ModelFlux(), "FP8", full_precision_fp8=False
        )
        assert conf == {"format": "float8_e4m3fn"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_safetensors_quant.py::TestPlanTensorOutput -v`
Expected: FAIL — `ImportError: cannot import name 'plan_tensor_output'`

- [ ] **Step 3: Implement**

In `safetensors_quant.py`, add immediately before `quantize_tensor_st` (safetensors_quant.py:506):

```python
def plan_tensor_output(
    key: str,
    shape: tuple[int, ...],
    old_dtype: torch.dtype,
    model_arch,
    target_key: str,
    full_precision_fp8: bool = True,
    full_precision_nvfp4: bool = True,
) -> tuple[list[tuple[str, str, tuple[int, ...]]], dict | None]:
    """Plan quantize_tensor_st()'s output for one source tensor, purely from
    shape/dtype metadata -- no tensor data touched, no quantization math run.

    Mirrors quantize_tensor_st()'s branching exactly (same order: mixed+
    hiprec passthrough, F16, <=1D, >=3D conv, then FP8/NVFP4/INT8-specific
    shape_critical/shape-alignment fallbacks) and every real quantize_*
    function's exact output tensor naming/dtype/shape (quantize_fp8_scaled,
    quantize_int8_tensorwise/convrot, quantize_nvfp4) -- kept in sync by
    tests.test_safetensors_quant.TestPlanTensorOutput, which cross-checks
    this function's output against quantize_tensor_st()'s REAL output for
    every branch, not just by code inspection.

    Both estimate_safetensors_output_size() (byte-count only) and
    convert_safetensors.py's streaming Pass 1 (exact header) build on this
    single source of truth so they can't silently drift apart from each
    other or from quantize_tensor_st().

    Returns (entries, layer_conf):
      entries: [(output_name, safetensors_dtype_string, output_shape), ...]
        in the same order quantize_tensor_st()'s real output dict has.
      layer_conf: the _quantization_metadata "layers" fragment for this
        key (format/convrot/full_precision_matrix_mult), or None if this
        key produces no scale sidecar (F16, 1D/hiprec passthrough, >=3D
        conv fallback, or a shape_critical/alignment F16 fallback).
    """
    base = _BASE_KEY[target_key]
    mixed = target_key in _MIXED_KEYS
    from convert import _TORCH_TO_ST_DTYPE

    if mixed and _is_hiprec_shape(key, shape, old_dtype, model_arch):
        return [(key, _TORCH_TO_ST_DTYPE[old_dtype], tuple(shape))], None

    if base == "F16":
        return [(key, "F16", tuple(shape))], None

    if len(shape) <= 1:
        return [(key, "F32", tuple(shape))], None

    if len(shape) >= 3:
        return [(key, "F16", tuple(shape))], None

    # From here on shape is always exactly 2D -- quantize_tensor_st's own
    # <=1D and >=3D branches above already filtered everything else out.
    prefix = layer_key(key)
    shape_critical = any(
        x in key for x in getattr(model_arch, "keys_shape_critical", [])
    )

    if base == "FP8":
        if shape_critical:
            return [(key, "F16", tuple(shape))], None
        entries = [
            (key, "F8_E4M3", tuple(shape)),
            (f"{prefix}.weight_scale", "F32", (1,)),
        ]
        layer_conf = {"format": "float8_e4m3fn"}
        if full_precision_fp8:
            layer_conf["full_precision_matrix_mult"] = True
        return entries, layer_conf

    if base == "NVFP4":
        if shape_critical or shape[-1] % 16 != 0:
            return [(key, "F16", tuple(shape))], None
        out_f, in_f = shape
        k_blocks = in_f // 16
        packed_shape = (out_f, in_f // 2)
        # 2D weights are always swizzled to 128x4-tile alignment (see
        # safetensors_quant_nvfp4._swizzle_block_scale) -- the >=3D branch
        # above already guarantees shape is exactly 2D here, so the
        # "higher-rank, unswizzled" case quantize_nvfp4() itself still
        # defends against is unreachable via this call path; mirrored
        # anyway for robustness against a future change to that filter.
        if len(shape) == 2:
            m_padded = -(-out_f // 128) * 128
            k_padded = -(-k_blocks // 4) * 4
            scale_shape = (m_padded, k_padded)
        else:
            scale_shape = (*shape[:-1], k_blocks)
        entries = [
            (key, "U8", packed_shape),
            (f"{prefix}.weight_scale", "F8_E4M3", scale_shape),
            (f"{prefix}.weight_scale_2", "F32", (1,)),
        ]
        layer_conf = {"format": "nvfp4"}
        if full_precision_nvfp4:
            layer_conf["full_precision_matrix_mult"] = True
        return entries, layer_conf

    if base == "INT8":
        if shape_critical:
            return [(key, "F16", tuple(shape))], None
        out_f, in_f = shape
        if in_f % _CONVROT_GROUP_SIZE == 0:
            entries = [
                (key, "I8", tuple(shape)),
                (f"{prefix}.weight_scale", "F32", (out_f, 1)),
            ]
            layer_conf = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": _CONVROT_GROUP_SIZE,
            }
        else:
            entries = [
                (key, "I8", tuple(shape)),
                (f"{prefix}.weight_scale", "F32", ()),
            ]
            layer_conf = {"format": "int8_tensorwise"}
        return entries, layer_conf

    raise ValueError(f"Unknown target_key: {target_key!r}")
```

Note: `test_int8_tensorwise_odd_infeatures` above uses a 2D shape (64, 100) — `quantize_tensor_st`'s real INT8 branch only attempts ConvRot when `data.dim() == 2` (always true past the >=3D filter) and falls back to tensorwise on `in_features % group_size != 0` via the `except ValueError` in `quantize_int8_convrot`. Confirm this matches by re-reading safetensors_quant.py:629-640 if the cross-check test fails.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_safetensors_quant.py::TestPlanTensorOutput -v`
Expected: PASS — all cases. If any case fails, the mismatch is between this function and the real quantize path — fix `plan_tensor_output()`, not the test (the test asserts against the real function's actual behavior).

- [ ] **Step 5: Commit**

```bash
git add safetensors_quant.py tests/test_safetensors_quant.py
git commit -m "feat: add plan_tensor_output, shape-only prediction of quantize_tensor_st output"
```

---

### Task 5: Refactor `estimate_safetensors_output_size()` onto `plan_tensor_output()`

**Files:**
- Modify: `safetensors_quant.py` (`estimate_safetensors_output_size`, safetensors_quant.py:652-762)
- Test: `tests/test_safetensors_quant.py` (existing `TestEstimateSafetensorsOutputSize` — must pass UNMODIFIED)

**Interfaces:**
- Consumes: `plan_tensor_output()` (Task 4).
- `estimate_safetensors_output_size(path, target_key, model_arch) -> int | None` — signature and behavior unchanged; this task only changes its internals to remove the duplicated branching, now that `plan_tensor_output()` is the single source of truth.

This task's entire purpose is proving `plan_tensor_output()` is a faithful extraction — if this refactor changes any existing `TestEstimateSafetensorsOutputSize` result, the extraction has a bug. Do not modify that test class.

- [ ] **Step 1: Run the existing tests first to record the baseline**

Run: `uv run pytest tests/test_safetensors_quant.py::TestEstimateSafetensorsOutputSize -v`
Expected: PASS (already passing before this task — this step just confirms the starting point).

- [ ] **Step 2: Implement — replace the per-tensor loop body**

Replace the body of the `for name, meta in header.items():` loop in `estimate_safetensors_output_size` (safetensors_quant.py:684-762) with a call into `plan_tensor_output`. Keep the function's header-reading preamble (safetensors_quant.py:667-682) unchanged — only the loop body changes:

```python
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta.get("shape")
        if not shape and shape != []:
            continue
        shape = tuple(shape)
        src_dtype = meta.get("dtype", "F16")
        from convert import _ST_DTYPE_MAP
        old_dtype = _ST_DTYPE_MAP.get(src_dtype)
        if old_dtype is None:
            continue  # unrecognized/unsupported dtype in this header -- skip

        entries, _ = plan_tensor_output(name, shape, old_dtype, model_arch, target_key)
        for _, st_dtype, out_shape in entries:
            n_elems = 1
            for d in out_shape:
                n_elems *= d
            total += n_elems * _ST_DTYPE_BYTES.get(st_dtype, 2)

    return total
```

Delete the now-unused `hiprec_substrings`/`shape_critical_substrings` locals and the `mixed`/`base` locals from the old implementation (safetensors_quant.py:677-683) if `plan_tensor_output` makes them redundant — check by re-reading the function after this edit; keep only what's still referenced (the `base = _BASE_KEY.get(target_key)` / `if base is None: return None` early-exit guard at the top must stay, since `plan_tensor_output` doesn't itself validate `target_key` against `_BASE_KEY`).

- [ ] **Step 3: Run test to verify it still passes**

Run: `uv run pytest tests/test_safetensors_quant.py::TestEstimateSafetensorsOutputSize -v`
Expected: PASS — identical results to Step 1's baseline. If a specific case now differs, read that test's assertion and compare against `plan_tensor_output`'s branch for the same shape/dtype/target_key — the bug is in one of the two, not the test.

- [ ] **Step 4: Run the full safetensors_quant test file**

Run: `uv run pytest tests/test_safetensors_quant.py -v`
Expected: PASS, all tests (this file now has zero duplicated branching between the estimator and the planner).

- [ ] **Step 5: Commit**

```bash
git add safetensors_quant.py
git commit -m "refactor: estimate_safetensors_output_size builds on plan_tensor_output"
```

---

### Task 6: Streaming primitives in `convert_safetensors.py`

**Files:**
- Modify: `convert_safetensors.py` (add helpers, don't wire into `convert_to_safetensors` yet)
- Test: `tests/test_convert_safetensors.py`

**Interfaces:**
- Consumes: `plan_tensor_output` (Task 4), `dequantized_shape_of` (Task 3), `_TORCH_TO_ST_DTYPE` (Task 1, from `convert`).
- Produces:
  - `_iter_output_keys(state_dict, model_arch, quant_skip_keys) -> Iterator[tuple[str, bool]]` — yields `(key, is_passthrough)` in `state_dict` order, applying the existing skip/passthrough/keys_ignore filtering.
  - `_tensor_bytes(tensor: torch.Tensor) -> bytes` — raw little-endian bytes for one tensor.
  - `_build_header(entries: list[tuple[str, str, tuple[int, ...]]], metadata: dict) -> tuple[dict, int]` — returns `(header_dict, total_data_bytes)`.
  - `_write_header(fh, header: dict) -> int` — writes the 8-byte length prefix + JSON header to an already-open binary file handle, returns the number of header bytes written (informational only in this task; not consumed by data offsets since Task 7's writer is purely sequential — see spec's Pass 2).

These are pure/isolated helpers, independently testable before they're wired into the main conversion loop in Task 7 — keeping this task's diff reviewable on its own.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_convert_safetensors.py
class TestStreamingPrimitives:
    def test_iter_output_keys_skips_quant_sidecars_and_ignored_keys(self):
        import torch
        from convert_safetensors import _iter_output_keys
        from models.architectures import ModelTemplate

        state_dict = {
            "a.weight": torch.randn(4, 4),
            "a.weight_scale": torch.randn(1),  # in quant_skip_keys
            "spiece_model": torch.randint(0, 255, (10,), dtype=torch.uint8),
            "ignored.weight": torch.randn(4, 4),
        }
        arch = ModelTemplate()
        arch.keys_ignore = ["ignored"]
        result = list(_iter_output_keys(state_dict, arch, quant_skip_keys={"a.weight_scale"}))
        assert result == [("a.weight", False), ("spiece_model", True)]

    def test_tensor_bytes_roundtrips_bfloat16(self):
        import torch
        from convert_safetensors import _tensor_bytes
        t = torch.tensor([1.5, -2.25], dtype=torch.bfloat16)
        raw = _tensor_bytes(t)
        assert len(raw) == 4  # 2 elements * 2 bytes
        rebuilt = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        assert torch.equal(rebuilt, t)

    def test_tensor_bytes_roundtrips_0dim_scalar(self):
        import torch
        from convert_safetensors import _tensor_bytes
        t = torch.tensor(3.5, dtype=torch.float32)
        raw = _tensor_bytes(t)
        assert len(raw) == 4
        rebuilt = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(())
        assert torch.equal(rebuilt, t)

    def test_build_header_computes_sequential_offsets(self):
        from convert_safetensors import _build_header
        entries = [
            ("a.weight", "F16", (4, 4)),       # 4*4*2 = 32 bytes
            ("a.weight_scale", "F32", (1,)),   # 4 bytes
        ]
        header, total = _build_header(entries, {})
        assert header["a.weight"]["data_offsets"] == [0, 32]
        assert header["a.weight_scale"]["data_offsets"] == [32, 36]
        assert total == 36

    def test_build_header_includes_metadata(self):
        from convert_safetensors import _build_header
        header, _ = _build_header([("a.weight", "F16", (1,))], {"foo": "bar"})
        assert header["__metadata__"] == {"foo": "bar"}

    def test_write_header_roundtrips_via_safe_open(self, tmp_path):
        import struct
        from convert_safetensors import _build_header, _write_header
        header, total = _build_header([("a.weight", "F16", (2,))], {"foo": "bar"})
        path = tmp_path / "hdr_test.safetensors"
        with open(path, "wb") as fh:
            _write_header(fh, header)
            fh.write(b"\x00" * total)  # dummy data section matching declared size
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            assert f.metadata() == {"foo": "bar"}
            assert "a.weight" in f.keys()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_convert_safetensors.py::TestStreamingPrimitives -v`
Expected: FAIL — `ImportError` for each missing name.

- [ ] **Step 3: Implement**

Add to `convert_safetensors.py`, after the existing imports and `_TARGET_TO_QUANT_FORMAT` dict (convert_safetensors.py:38-42):

```python
import struct


def _iter_output_keys(state_dict, model_arch, quant_skip_keys):
    """Yield (key, is_passthrough) from state_dict in order, applying the
    same skip/passthrough/keys_ignore filtering convert_to_safetensors()'s
    main loop always has -- factored out so the streaming writer's planning
    pass (Pass 1) and quantizing pass (Pass 2) iterate identically. Both
    passes MUST see the exact same keys in the exact same order, or Pass 2's
    tensors won't line up with Pass 1's planned header (see convert_
    to_safetensors()'s Pass 2 assert)."""
    for key in state_dict.keys():
        if key in quant_skip_keys:
            continue
        if key.endswith(_PASSTHROUGH_TENSOR_SUFFIXES):
            yield key, True
            continue
        if any(x in key for x in model_arch.keys_ignore):
            continue
        yield key, False


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Raw little-endian bytes for one tensor, in safetensors' flat storage
    convention. Reinterprets through a uint8 view instead of calling
    tensor.numpy() directly -- numpy has no bfloat16/float8_e4m3fn/
    float8_e5m2 support on most builds, and .numpy() raises for those
    dtypes. torch.Tensor.view(dtype) is a pure reinterpret-cast (no data
    copy beyond what .contiguous() already needs) and handles the 0-dim
    scalar case by adding a new trailing dimension of the itemsize ratio,
    per its own documented behavior."""
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _build_header(
    entries: list[tuple[str, str, tuple[int, ...]]], metadata: dict
) -> tuple[dict, int]:
    """Compute the safetensors JSON header (name -> dtype/shape/data_offsets)
    and total data-section byte size, from a flat, ordered list of (name,
    dtype, shape) tuples. Offsets are assigned in list order -- Pass 2
    (convert_to_safetensors) MUST produce tensors in this same order for
    the file to be valid; this function itself doesn't enforce that, the
    assert in Pass 2 does."""
    from safetensors_quant import _ST_DTYPE_BYTES

    header: dict = {}
    offset = 0
    for name, dtype, shape in entries:
        n_elems = 1
        for d in shape:
            n_elems *= d
        size = n_elems * _ST_DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    if metadata:
        header["__metadata__"] = metadata
    return header, offset


def _write_header(fh, header: dict) -> int:
    """Write the safetensors 8-byte little-endian header-length prefix plus
    the JSON header itself to an already-open binary file handle. Pads the
    JSON body with trailing spaces to a multiple of 8 bytes, matching the
    reference Rust safetensors serializer's convention (not required by the
    format spec for correctness -- readers only ever consult the declared
    length prefix -- but matched here for closest byte-level parity with
    files this project's old save_file()-based writer produced). Returns
    the number of header bytes written (prefix + JSON, informational)."""
    body = json.dumps(header).encode("utf-8")
    pad = (-len(body)) % 8
    body += b" " * pad
    fh.write(struct.pack("<Q", len(body)))
    fh.write(body)
    return 8 + len(body)
```

`json` and `torch` are already imported at the top of `convert_safetensors.py` — no new top-level import needed beyond `struct`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_convert_safetensors.py::TestStreamingPrimitives -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add convert_safetensors.py tests/test_convert_safetensors.py
git commit -m "feat: add streaming safetensors writer primitives (unwired)"
```

---

### Task 7: Wire Pass 1 + Pass 2 into `convert_to_safetensors()`

**Files:**
- Modify: `convert_safetensors.py` (`convert_to_safetensors`, convert_safetensors.py:45-274 — this is the core change)
- Test: `tests/test_convert_safetensors.py` (existing tests must pass UNMODIFIED — this is the regression net)

**Interfaces:**
- Consumes: everything from Tasks 1-6 (`plan_tensor_output`, `dequantized_shape_of`, `_iter_output_keys`, `_tensor_bytes`, `_build_header`, `_write_header`, `_TORCH_TO_ST_DTYPE`).
- `convert_to_safetensors(...)`'s signature and return value are UNCHANGED.

This replaces the current single loop (convert_safetensors.py:166-259: accumulate into `out_tensors`, then one `save_file()` call) with: Pass 1 builds `entries`/`layer_formats` via `_iter_output_keys` + `plan_tensor_output` (+ `dequantized_shape_of` for already-quantized sources); the header is written; Pass 2 re-iterates via `_iter_output_keys`, runs the real `quantize_tensor_st()`/passthrough copy per key exactly as today, and writes each tensor's bytes immediately instead of updating a dict.

- [ ] **Step 1: Run the full existing test file first to record the baseline**

Run: `uv run pytest tests/test_convert_safetensors.py -v`
Expected: PASS, all tests (confirms the starting point before this task's rewrite).

- [ ] **Step 2: Implement — replace the function body**

Replace `convert_to_safetensors`'s body from the `out_tensors: dict[str, torch.Tensor] = {}` line through the final `save_file(out_tensors, dst_path, metadata=metadata)` call (convert_safetensors.py:166-258) with:

```python
    quant_format = _TARGET_TO_QUANT_FORMAT.get(target_key)

    # --- Pass 1: plan every output tensor's (name, dtype, shape) and the
    # full _quantization_metadata, from shape/dtype metadata alone -- no
    # tensor data touched. Must use the exact same key set/order as Pass 2
    # below (_iter_output_keys is the shared contract that guarantees this).
    shape_of = getattr(state_dict, "shape_of", None)
    dtype_of = getattr(state_dict, "dtype_of", None)

    def _shape_dtype(k):
        if shape_of is not None:
            return tuple(shape_of(k)), dtype_of(k)
        t = state_dict[k]
        return tuple(t.shape), t.dtype

    entries: list[tuple[str, str, tuple]] = []
    layer_formats: dict[str, dict] = {}

    for key, is_passthrough in _iter_output_keys(state_dict, model_arch, quant_skip_keys):
        shape, dtype = _shape_dtype(key)
        if is_passthrough:
            entries.append((key, _TORCH_TO_ST_DTYPE[dtype], shape))
            continue

        if key in quant_formats:
            shape = dequantized_shape_of(quant_formats[key], shape)
            dtype = torch.float32
        if _FLOAT8_DTYPES and dtype in _FLOAT8_DTYPES:
            dtype = torch.float16

        out_entries, layer_conf = plan_tensor_output(
            key, shape, dtype, model_arch, target_key,
            full_precision_fp8, full_precision_nvfp4,
        )
        entries.extend(out_entries)
        if layer_conf is not None:
            layer_formats[layer_key(key)] = layer_conf

    metadata = {} if model_arch.arch == "invalid" else {"comfy.gguf_source_arch": model_arch.arch}
    if layer_formats:
        metadata["_quantization_metadata"] = json.dumps(
            {"format_version": "1.0", "layers": layer_formats}
        )

    header, _total_data_bytes = _build_header(entries, metadata)

    _log(f"INFO:  Writing {len(entries)} tensors -> {dst_path}")

    # --- Pass 2: re-run the REAL quantization per key, streaming each
    # result's bytes to disk the instant it's produced instead of
    # accumulating a dict -- this is the actual RAM fix. Tensors are
    # written strictly in Pass 1's planned order (guaranteed by using the
    # same _iter_output_keys + plan_tensor_output/quantize_tensor_st
    # branching), so no seeking or offset lookup is needed, just sequential
    # appends after the header.
    total = len(state_dict)
    log_tensor_every = max(1, int(log_tensor_every or 1))
    entry_idx = 0
    idx = 0
    with open(dst_path, "wb") as fh:
        _write_header(fh, header)

        for key, is_passthrough in _iter_output_keys(state_dict, model_arch, quant_skip_keys):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("cancelled")
            if on_progress:
                on_progress(idx + 1, total, key)
            idx += 1

            if is_passthrough:
                data = state_dict[key]
                exp_name, exp_dtype, exp_shape = entries[entry_idx]
                entry_idx += 1
                assert key == exp_name and tuple(data.shape) == exp_shape, (
                    f"Streaming writer plan mismatch for passthrough {key!r}: "
                    f"planned shape {exp_shape}, got {tuple(data.shape)}"
                )
                fh.write(_tensor_bytes(data))
                continue

            data = state_dict[key]
            if key in quant_formats:
                data = dequantize_weight(state_dict, key, quant_formats[key], data)
            old_dtype = data.dtype
            if _FLOAT8_DTYPES and data.dtype in _FLOAT8_DTYPES:
                data = data.to(torch.float16)
            data = torch.nan_to_num(data, nan=0.0, posinf=65504.0, neginf=-65504.0)
            quantized = quantize_tensor_st(data, key, model_arch, target_key)
            if (
                log_tensor_every == 1
                or idx == 1
                or idx == total
                or idx % log_tensor_every == 0
            ):
                _log(f"  {key}  {old_dtype} -> {target_key}")

            for name, tensor in quantized.items():
                exp_name, exp_dtype, exp_shape = entries[entry_idx]
                entry_idx += 1
                assert (
                    name == exp_name
                    and tuple(tensor.shape) == exp_shape
                    and _TORCH_TO_ST_DTYPE[tensor.dtype] == exp_dtype
                ), (
                    f"Streaming writer plan mismatch for {name!r}: planned "
                    f"{exp_dtype}/{exp_shape}, got real "
                    f"{_TORCH_TO_ST_DTYPE[tensor.dtype]}/{tuple(tensor.shape)} -- "
                    "Pass 1 (plan_tensor_output) and Pass 2 (quantize_tensor_st) "
                    "have drifted apart"
                )
                fh.write(_tensor_bytes(tensor))

    _log(f"INFO:  Done -> {dst_path}")
```

Add the two new imports this needs at the top of `convert_safetensors.py` (alongside the existing `from dequantize import ...` and `from safetensors_quant import ...` lines):

```python
from dequantize import (
    _PASSTHROUGH_TENSOR_SUFFIXES,
    _scan_quantized_layers,
    dequantized_shape_of,
    detect_quantized_weight,
    dequantize_weight,
)
from safetensors_quant import filename_suffix_for, layer_key, plan_tensor_output, quantize_tensor_st
from convert import _TORCH_TO_ST_DTYPE
```

(`detect_quantized_weight` was already imported but unused directly in this file before — check whether it's still referenced elsewhere in `convert_safetensors.py` after this edit; if not, it was already dead in the original file and this task doesn't need to remove it, but don't newly introduce an unused import if `detect_quantized_weight` turns out to have been unused already — verify with a quick grep before finalizing this task's diff.)

Remove the now-unused `from safetensors.torch import save_file` import if nothing else in the file calls `save_file` after this change — verify with a grep before deleting.

The trailing "output larger than input" warning block (convert_safetensors.py:261-272, using `os.path.getsize(path)`/`os.path.getsize(dst_path)`) is unchanged — it runs after the `with open(dst_path, "wb") as fh:` block closes, exactly as it ran after the old `save_file()` call.

- [ ] **Step 3: Run the full existing test file to verify it still passes**

Run: `uv run pytest tests/test_convert_safetensors.py -v`
Expected: PASS, every test, unmodified — this is the byte-content regression proof the spec calls for. If any test fails, the new writer's output differs from the old one; do not edit the test to match new (possibly wrong) output — find and fix the discrepancy in Pass 1/Pass 2.

- [ ] **Step 4: Run the full project test suite**

Run: `uv run pytest -v`
Expected: PASS, all tests across the whole project (this touched a widely-used core function — confirm nothing else broke, e.g. `text_encoder_convert.py`'s tests, which call `convert_to_safetensors()` too).

- [ ] **Step 5: Commit**

```bash
git add convert_safetensors.py
git commit -m "feat: stream safetensors output instead of buffering full model in RAM

Replaces convert_to_safetensors()'s accumulate-then-save_file() write
path with a two-pass streaming writer: Pass 1 plans every output
tensor's name/dtype/shape from source metadata alone (plan_tensor_output),
Pass 2 re-runs the real quantization per tensor and appends its bytes to
disk immediately. Peak RAM now scales with one tensor instead of the
full quantized model -- fixes the OOM found quantizing FLUX.2 dev's 32B
diffusion model (see docs/superpowers/specs/2026-08-23-streaming-safetensors-writer-design.md)."
```

---

### Task 8: Streaming-specific correctness tests + docs

**Files:**
- Modify: `tests/test_convert_safetensors.py`
- Modify: `CHANGELOG.md` (`[Unreleased]` → `### Changed` or `### Fixed`, per this project's existing convention — check the file's current `[Unreleased]` section heading style before editing)
- Modify: `docs/architecture.md` if it documents `convert_to_safetensors()`'s write mechanism (check for an existing description before assuming it needs an edit)

**Interfaces:** none new — this task adds tests/docs only, no new production code.

Task 7's regression tests prove the new writer produces the *same* output as the old one for existing fixtures. This task adds tests for behavior the old single-pass writer had no equivalent of: the Pass 1/Pass 2 drift safety assert, and a larger multi-tensor-shape smoke test exercising every format's header/offset math together in one file (existing fixtures are mostly 1-2 tensors).

- [ ] **Step 1: Write the new tests**

```python
# tests/test_convert_safetensors.py
class TestStreamingWriterCorrectness:
    def test_multi_tensor_multi_format_smoke(self, tmp_path):
        # Exercises header/offset math across a mix of tensor shapes in one
        # file -- 1D (bias), 2D non-block-aligned, 2D block-aligned
        # (triggers ConvRot/NVFP4's real packing math, not just fallbacks),
        # and >=3D (conv fallback) -- for every safetensors target format.
        import torch
        from safetensors.torch import save_file, load_file
        from convert_safetensors import convert_to_safetensors

        src = tmp_path / "model.safetensors"
        sd = {
            "double_blocks.0.img_attn.proj.weight": torch.randn(64, 512, dtype=torch.float32),
            "double_blocks.0.img_attn.proj.bias": torch.randn(64, dtype=torch.float32),
            "double_blocks.0.img_attn.qkv.weight": torch.randn(96, 100, dtype=torch.float32),
            "conv_stem.weight": torch.randn(16, 4, 3, 3, dtype=torch.float32),
        }
        save_file(sd, str(src))

        for target_key in ("F16", "FP8", "FP8_MIXED", "INT8", "INT8_MIXED", "NVFP4", "NVFP4_MIXED"):
            dst, _ = convert_to_safetensors(
                str(src), target_key=target_key, overwrite=True
            )
            out = load_file(dst)
            assert "double_blocks.0.img_attn.proj.weight" in out
            assert "double_blocks.0.img_attn.proj.bias" in out
            assert "double_blocks.0.img_attn.qkv.weight" in out
            assert "conv_stem.weight" in out
            # Conv weight is always F16 regardless of target format (>=3D
            # guard, quantize_tensor_st/plan_tensor_output's shared rule).
            assert out["conv_stem.weight"].dtype == torch.float16

    def test_plan_pass2_mismatch_raises(self, tmp_path, monkeypatch):
        # Forces plan_tensor_output to disagree with the real quantize_
        # tensor_st for one call, and confirms the streaming writer's
        # safety assert catches it instead of silently writing misaligned
        # bytes.
        import torch
        from safetensors.torch import save_file
        import convert_safetensors as cs

        src = tmp_path / "model.safetensors"
        save_file(
            {"double_blocks.0.img_attn.proj.weight": torch.randn(64, 64, dtype=torch.float32)},
            str(src),
        )

        real_plan = cs.plan_tensor_output

        def _wrong_plan(*args, **kwargs):
            entries, conf = real_plan(*args, **kwargs)
            # Corrupt the first entry's declared shape so it disagrees with
            # what quantize_tensor_st will actually produce.
            name, dtype, shape = entries[0]
            entries[0] = (name, dtype, tuple(d + 1 for d in shape))
            return entries, conf

        monkeypatch.setattr(cs, "plan_tensor_output", _wrong_plan)
        with pytest.raises(AssertionError, match="plan mismatch"):
            cs.convert_to_safetensors(str(src), target_key="F16", overwrite=True)
```

- [ ] **Step 2: Run tests to verify they fail appropriately, then pass**

Run: `uv run pytest tests/test_convert_safetensors.py::TestStreamingWriterCorrectness -v`
Expected: both PASS against the Task 7 implementation (no separate "fails first" step needed here — these are new coverage for already-implemented behavior, not driving new production code). If `test_plan_pass2_mismatch_raises` doesn't raise, the assert in Task 7's Pass 2 loop isn't wired correctly — revisit that task.

- [ ] **Step 3: Update CHANGELOG.md**

Read the current `[Unreleased]` section's heading style first (`Read CHANGELOG.md`, first ~40 lines), then add an entry under the appropriate heading (likely `### Fixed` or `### Changed`, matching whatever heading this project already uses for internal-mechanism-only changes with no user-facing format/behavior difference):

```markdown
- `convert_to_safetensors()` now streams quantized tensors directly to disk
  instead of buffering the entire output in memory before one `save_file()`
  call — peak RAM now scales with a single tensor instead of the full
  model, fixing an OOM crash found quantizing FLUX.2 dev's 32B diffusion
  model on a 50GB-RAM system. Output content is unchanged (same tensors,
  same dtypes, same `_quantization_metadata`) — this is a write-mechanism
  change only. See `docs/superpowers/specs/2026-08-23-streaming-safetensors-writer-design.md`.
```

- [ ] **Step 4: Check docs/architecture.md**

Run: `grep -n "convert_to_safetensors\|save_file" docs/architecture.md` (or use the Grep tool) — if it describes the old accumulate-then-save_file mechanism specifically (not just "writes a quantized safetensors file"), update that description to mention the two-pass streaming approach. If it only describes the function's purpose/inputs/outputs (not its internal write strategy), no edit needed — don't add detail the file didn't already have at this level for other functions.

- [ ] **Step 5: Run the full test suite one final time**

Run: `uv run pytest -v`
Expected: PASS, all tests.

- [ ] **Step 6: Commit**

```bash
git add tests/test_convert_safetensors.py CHANGELOG.md docs/architecture.md
git commit -m "test: add streaming writer plan/pass2 drift and multi-format smoke tests"
```

---

## Self-Review Notes

- **Spec coverage:** Pass 1 plan (Task 4, 7), header write (Task 6, 7), Pass 2 stream (Task 7), error handling / plan-drift assert (Task 7, tested in Task 8), files-touched list (Tasks 1-3, 6, 7 match the spec's list exactly: `convert_safetensors.py`, `safetensors_quant.py`, `dequantize.py`), byte-equivalence testing (Task 7 relies on unmodified existing tests as the equivalence proof, per spec's testing section), `dequantized_shape_of()` unit test (Task 3). All spec sections have a covering task.
- **Placeholder scan:** no TBD/TODO; the one "verify with a grep before deleting" instruction in Task 7 is a real, specific verification step (not a placeholder) since this plan's author doesn't have the post-Task-6 file state to check import usage against ahead of time.
- **Type consistency:** `plan_tensor_output`'s signature (Task 4) is used identically in Task 5 (estimate) and Task 7 (streaming) — `(key, shape, old_dtype, model_arch, target_key, full_precision_fp8=True, full_precision_nvfp4=True)`. `_iter_output_keys`'s `(key, is_passthrough)` yield shape is used identically in Task 6's test and Task 7's wiring. `entries: list[tuple[str, str, tuple[int, ...]]]` (name, st_dtype, shape) is the consistent element type across `plan_tensor_output`, `_build_header`, and Task 7's Pass 1/Pass 2.
