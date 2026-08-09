# Known Conversion Errors and Fixes

Analysis of GitHub Issues from [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/issues).

---

## 1. `inf`/`NaN` values after conversion

**Error message:**
```
ggml_validate_row_data: found inf value at block 0
llama_model_quantize: failed to quantize: tensor 'norm_final.weight' has invalid data
```

**Cause:** BF16 values in the source model exceed the representable F16 range (> 65504)
after conversion. `llama-quantize` refuses to quantize the file.

**Fix:**
```python
# In handle_tensors(), before dtype conversion:
data = torch.nan_to_num(data, nan=0.0, posinf=65504, neginf=-65504)
```

---

## 2. `size mismatch for x_pad_token` — Lumina 2 pad token shape

**Error message:**
```
size mismatch for x_pad_token: copying a param with shape torch.Size([3840])
from checkpoint, the shape in current model is torch.Size([1, 3840]).
size mismatch for cap_pad_token: copying a param with shape torch.Size([3840])
from checkpoint, the shape in current model is torch.Size([1, 3840]).
```

**Cause:** Older Lumina 2 GGUFs store `x_pad_token` and `cap_pad_token` as 1D tensors
`[3840]`.  ComfyUI's NextDiT model registers them as `nn.Parameter` with shape
`[1, 3840]`, so `load_state_dict` raises a size mismatch.

**Fix for existing GGUFs** (one-time repair):

```bash
uv run python fix_pad_tokens.py --src model.gguf --dst model-fixed.gguf
```

The GUI offers this as the **Fix Pad Tokens** tab.

**Fix for new conversions:** Already handled automatically — `ModelLumina2.keys_unsqueeze`
causes `convert.py` to call `unsqueeze(0)` on these tensors before writing them.

---

## 3. Wrong architecture detection

**Error message:**
```
AssertionError: Unknown model architecture!
# or:
AssertionError: Model architecture not allowed for conversion!
```

**Causes:**
- Model is in **reference format** instead of Diffusers format (has `keys_banned` keys)
- Model is a **merge** whose keys deviate slightly from the standard
- External GGUF file without `general.architecture` field, key matching fails

**Diagnosis:** Check `keys_detect` of the relevant architecture class to see whether
the detection keys are present in the state dict.

---

## 4. `mat1 and mat2 shapes cannot be multiplied` — matrix multiplication error

**Error message:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (NxM and KxM)
```

**Cause A — `shape_fix` reshape without metadata restoration:**
For SD1/SDXL, tensors are reshaped to `(n//256, 256)`. The original shape is stored
as a `comfy.gguf.orig_shape.<key>` metadata field. If this field is missing when
loading, the tensor remains in the wrong shape → matmul fails.

**Cause B — incorrect `shape_fix` usage:**
`shape_fix=True` must only be set for SD1/SDXL. For other architectures
(Flux, SD3, etc.) it produces incompatible tensor shapes.

**Fix:** Enable `shape_fix` only in `ModelSD1` and `ModelSDXL`. Always write
`orig_shape` metadata alongside the reshape.

---

## 5. `GGML_ASSERT(ne[i] > 0)` in llama-quantize

**Error message:**
```
/ggml/src/ggml.c:22112: GGML_ASSERT(info->ne[i] > 0) failed
```

**Cause:** The GGUF file produced by `convert.py` contains tensors with a dimension
of 0, or the shape metadata is invalid for llama.cpp's validator.
Common with merge models that have unusual tensor shapes.

**Note:** The F16 GGUF is created successfully — the error only appears during the
second step (`llama-quantize`).

---

## 6. `gather(): Expected dtype int64 for index`

**Error message:**
```
gather(): Expected dtype int64 for index
```

**Cause:** An index tensor is created with `torch.int32` in the dequantization logic.
PyTorch ≥ 2.6 requires `int64` for `torch.gather`.

**Fix:**
```python
# dequant.py ~line 280:
# before:
qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int32))
# after:
qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int64))
```

---

## 7. `only 0-dimensional arrays can be converted to Python scalars`

**Cause:** The `gguf` library sometimes returns 1D arrays of size 1 instead of true
0D scalars for scalar metadata fields. `.item()` then fails.

**Fix:** When reading metadata fields, check whether the array has size 1 before
calling `.item()`.

---

## 8. Unknown architecture for certain ZIB / Flux.2 UNet variants (open, unfixed)

**Error message:**
```
AssertionError: Unknown model architecture. Checked: ['flux', 'sd3', ...]
```

**Affected models:** Some Z-Image Beyond / Flux.2 community fine-tunes whose
state-dict keys deviate from the standard `keys_detect` patterns.

**Status:** Reported in city96 issue #418, no fix from upstream or this project yet.

**Workaround:** Inspect the state dict with `torch.load` / `load_file` and compare
the actual top-level key names against `ModelFlux.keys_detect` and
`ModelLumina2.keys_detect` in `models/architectures.py`. Add the variant's unique
key as an additional tuple to the matching `keys_detect` list.

---

## 9. NVFP4 safetensors output could crash ComfyUI on hyperparameter-detection tensors (fixed for 8 of 12 architectures)

**Error message (before the fix):**
```
RuntimeError: Error(s) in loading state_dict for NextDiT:
    size mismatch for cap_embedder.0.weight: copying a param with shape torch.Size([2560])
    from checkpoint, the shape in current model is torch.Size([1280]).
```

**Affected models:** Any checkpoint whose architecture-detection tensor (see below)
got NVFP4-packed by this tool's `NVFP4`/`NVFP4_MIXED` safetensors output target.

**Cause:** ComfyUI's `comfy/model_detection.py` infers architecture hyperparameters
directly from specific tensors' raw on-disk shapes, **before** any dequantization
happens — this is a universal pattern across ComfyUI's supported architectures, not a
Lumina2-only quirk (e.g. Flux infers `in_channels` from `img_in.weight.shape[1]`,
Lumina2 infers `cap_feat_dim` from `cap_embedder.1.weight.shape[1]`). NVFP4 packs two
4-bit values per byte, halving the packed tensor's on-disk last dimension, which
corrupts that inference. `FP8`/`FP8_MIXED` output is unaffected (dtype changes, but the
shape stays the same). This is why searching for "ComfyUI NVFP4 broken" turns up
nothing — published NVFP4 checkpoints conventionally leave small embedding/patchify
"glue" layers unquantized anyway (standard practice: negligible size savings,
disproportionate quality/compatibility sensitivity), so nobody publishing real NVFP4
checkpoints hits this. This tool's non-mixed `NVFP4` mode packed everything indiscriminately, which is the actual gap — not a ComfyUI bug.

**Fix:** Added `ModelTemplate.keys_shape_critical` (`models/architectures.py`) — tensors
excluded from NVFP4 packing (falls back to F16) **unconditionally**, regardless of the
`*_MIXED` flag, since this is a shape-safety constraint rather than a precision one.
Every architecture was individually checked against a live ComfyUI install's
`comfy/model_detection.py` (and, where the tensor's own type was ambiguous from the
detection code alone, against the actual model class in `comfy/ldm/`) for tensors whose
raw `.shape[-1]` (or `.shape[1]` for a 2D weight) is read to infer hyperparameters:

| Architecture | Shape-critical tensor | Note |
|---|---|---|
| `ModelFlux` | `img_in.weight` | `in_channels` from `.shape[1]` |
| `ModelQwenImage` | `img_in.weight` | same convention as Flux |
| `ModelSD3` | `context_embedder.weight` | `in_features` from `.shape[1]` |
| `ModelAura` | `cond_seq_linear.weight` | `cond_seq_dim` from `.shape[1]` |
| `ModelLumina2` | `cap_embedder.1.weight` | `cap_feat_dim` from `.shape[1]` |
| `CosmosPredict2` | `x_embedder.proj.1.weight` | Linear inside `PatchEmbed`'s `Sequential` (index 1, after a `Rearrange`) |
| `ModelHyVid` | `txt_in.input_embedder.weight` | `context_in_dim` from `.shape[1]`; `img_in.proj.weight` is a Conv and safe (only its kernel-width last dim is touched, and ComfyUI reads `.shape[1]`/`.shape[2:]` there, not `.shape[-1]`) |
| `ModelWan` | `head.modulation` | 3D `nn.Parameter` `(1, 2, dim)` — `dim` from `.shape[-1]`; not 1D, so the unconditional 1D-skip doesn't already cover it |
| `ModelLTXV` | `transformer_blocks.0.attn2.to_k.weight` | `cross_attention_dim` from `.shape[1]` |
| `ModelHiDream` | *(none — audited, confirmed safe)* | its `unet_config` is entirely hardcoded constants, no shape reads at all |

**Status:** Fixed for `ModelFlux`, `ModelQwenImage`, `ModelSD3`, `ModelAura`,
`ModelLumina2`, `CosmosPredict2`, `ModelHyVid`, `ModelWan`, `ModelLTXV` (9 of 12 — and
`ModelHiDream` confirmed to need no fix, so 10 of 12 are resolved one way or the other).
`ModelSDXL` and `ModelSD1` remain **unaudited**: ComfyUI infers `context_dim` there by
scanning block indices dynamically for the first `attn2.to_k.weight` rather than a
single fixed tensor name, so pinning the exact key needs more work than the DiT
architectures above.

**Workaround (for `ModelSDXL`/`ModelSD1`):** Prefer `FP8`/`FP8_MIXED` over
`NVFP4`/`NVFP4_MIXED` until these are audited, since FP8 never changes tensor shape.

## 10. `AssertionError: Invalid B scale shape` — NVFP4 weight_scale wasn't padded/swizzled for ComfyUI's cuBLAS kernel

**Error message:**
```
AssertionError: Invalid B scale shape
  File "comfy_kitchen/backends/cuda/__init__.py", line 1642, in scaled_mm_nvfp4
    assert block_scale_b.size() == (roundup_n, roundup_sk), "Invalid B scale shape"
```

**Affected models:** Any NVFP4/NVFP4_MIXED-converted checkpoint containing a `Linear`
weight whose output-feature count isn't a multiple of 128 (or whose 16-element block
count isn't a multiple of 4) — e.g. Lumina2/Z-Image's `final_layer.linear.weight`, the
layer this was first reported on. Most transformer block weights happen to have output
widths that are already multiples of 128 (hidden_size, mlp_dim, …), which is why this
went unnoticed until a checkpoint with an odd-sized output projection was tested.

**Cause:** ComfyUI's `comfy_kitchen` CUDA backend (`scaled_mm_nvfp4`, the kernel behind
its native NVFP4 matmul) requires each layer's block-scale tensor pre-padded to
`(roundup(out_features, 128), roundup(in_features // 16, 4))` **and** physically
rearranged into a 128x4 block-interleaved ("swizzled") layout before being read by the
cuBLAS FP4 GEMM — this is the same convention used by vLLM's `swizzle_blockscale()`
(`nvfp4_utils.py`) and TensorRT-Model-Optimizer NVFP4 checkpoint exports. This tool wrote
a plain, unpadded `(out_features, in_features // 16)` block-scale tensor. When
`out_features` isn't already a multiple of 128, the kernel's shape assertion fails
outright (the crash reported here). When it's already a multiple of 128 (the common case
for most transformer block weights), the assertion silently passes but the scale values
are read in the wrong order — every such layer in every prior NVFP4/NVFP4_MIXED
conversion loaded with quietly wrong weights, not just the one layer that happened to
crash.

**Fix:** Added `_swizzle_block_scale()`/`_unswizzle_block_scale()` to
`safetensors_quant_nvfp4.py`, applied unconditionally (not just `*_MIXED`, since this is
a layout-correctness constraint, not a precision one) for 2D weight tensors — the only
case that reaches this kernel path in ComfyUI. Verified by reading `comfy_kitchen`'s CUDA
backend source directly for the exact `roundup_n`/`roundup_sk` formula, then
cross-checked against vLLM's open-source implementation of the identical layout.

## 11. NVFP4 output decoded at exactly half magnitude — every weight quietly wrong

**Symptom:** NVFP4/NVFP4_MIXED output produced pure-noise/garbage images in ComfyUI
even after the shape-crash (#9) and swizzle-layout (#10) fixes above — no error, no
crash, just meaningless output, reported by a user on a fresh Z-Image/Lumina2 conversion.

**Cause:** `safetensors_quant_nvfp4.py`'s `_KVALUES` E2M1 decode table used
`gguf.quants.NVFP4.kvalues`' **doubled** magnitudes (`0, 1, 2, 3, 4, 6, 8, 12, …`), per
its own docstring claim of being "identical" to that table. But gguf's NVFP4 format
doubles its table specifically to compensate for a *different* per-block scale
encoding it uses internally — a custom "unsigned e4m3" (`ue4m3_to_fp32`) that bakes in
a `*0.5` factor on decode (`gguf/quants.py`). This tool never used that custom
encoding — it writes plain `torch.float8_e4m3fn` block scales, exactly what ComfyUI's
`comfy_kitchen` expects, but *without* gguf's compensating halving. The result: every
weight decoded at **exactly half its intended magnitude**, silently, across every 2D
tensor in every NVFP4/NVFP4_MIXED conversion this tool ever produced. This tool's own
round-trip tests (quantize then dequantize with the same table) could never catch it —
using the same wrong table both ways cancels the error out; it only surfaces when
decoded by ComfyUI's real, independent kernel.

**Found by:** Comparing this tool's `quantize_nvfp4()` output against
`comfy_kitchen.tensor.nvfp4.TensorCoreNVFP4Layout`'s own quantize/dequantize path on a
live ComfyUI install (CUDA available), for identical input tensors — controlled
exact-table-value inputs (e.g. a block filled with `6.0`) decoded back to `3.0`,
directly showing the 2x error, before generalizing to full random tensors.

**Fix:** Replaced `_KVALUES` with the standard, undoubled OCP E2M1 magnitudes
(`0, 0.5, 1, 1.5, 2, 3, 4, 6, …`) — the existing `/6.0` divisors in the scale
formulas already assumed this (undoubled) range, so no other change was needed.
Verified against `comfy_kitchen`'s own reference quantize/dequantize on a random
256x128 tensor: mean reconstruction error 0.2857 (ours) vs. 0.2856 (ComfyUI's own),
effectively identical. Added `test_kvalues_are_standard_undoubled_e2m1_magnitudes` and
`test_block_of_exact_table_values_round_trips_exactly` regression tests.

**Note on #9/#10/#11 vs. the user's actual reported symptoms:** the shape-crash,
swizzle-layout, and table-doubling bugs above are all real and independently verified
against `comfy_kitchen`'s own reference implementation on clean synthetic tensors — that
evidence stands regardless of the checkpoint used to discover them. But the *specific*
noise-image reports that prompted #10 and #11 turned out to come from a checkpoint that
was **also** independently corrupted by #12 below (its source file was already
quantized). Both bug classes were real and both needed fixing; #12 doesn't retract #9-11
— only that the specific noise images reported can't be attributed to #10/#11 in
isolation, since #12 was also acting on that same checkpoint. Whether a genuinely clean
checkpoint would have shown visible noise from #10/#11 alone wasn't separately tested.
Once #12 was fixed, a re-test from a genuinely clean checkpoint still produced noise —
that noise is explained by #13 below (nibble order), not by #10/#11.

## 12. Re-quantizing an already-quantized source checkpoint silently corrupted it

**Symptom:** FP8_MIXED output produced a black/white QR-code-like noise pattern in
ComfyUI. Investigation of the output file's `_quantization_metadata` showed 170 of 378
"layers" with a corrupted name — an extra `.weight_scale` suffix (e.g.
`context_refiner.0.attention.out.weight_scale` instead of
`context_refiner.0.attention.out`) — while the other 208 layers had correct names.

**Cause:** The source checkpoint (`juggernautZ_v10ByRundiffusion.safetensors`, believed
by the user to be an unquantized "Original") turned out to be **itself already
quantized**: 170 of its layers use ComfyUI's native `int8_tensorwise` format with
ConvRot rotation (`convrot_groupsize: 64`), stored as raw `I8` weight tensors alongside
their own pre-existing `.weight_scale` (float32) and `.comfy_quant` (uint8 JSON blob)
sidecar tensors — confirmed by inspecting the source file directly (its own
`_quantization_metadata` names the same 170 layers with `format: int8_tensorwise`).

This tool has no concept of a partially-quantized source. `convert_to_safetensors()`
iterates every tensor in the state dict and re-quantizes each one as if it were an
ordinary weight — including the pre-existing `.weight_scale` tensors themselves (a
`[rows, 1]` float32 tensor easily mistaken for a small 2D weight matrix). Since
`layer_key()` only strips a trailing `.weight` and these sidecar keys end in
`.weight_scale`, not `.weight`, they pass through unchanged and get used directly as a
"layer" name when building the new `_quantization_metadata` — producing exactly the
corrupted `<layer>.weight_scale` layer-name pattern observed. The real INT8 weight data
also gets nonsensically re-quantized a second time (INT8 codes treated as raw magnitudes
and re-scaled to FP8/NVFP4), and the pre-existing `.comfy_quant` metadata tensor gets
silently corrupted by the `nan_to_num` dtype coercion applied to every tensor.

**Fix:** Added `_check_not_already_quantized()` in `convert_safetensors.py`, called
before the conversion loop. Two signals: any `.comfy_quant`-suffixed key present
anywhere in the state dict (primary — unambiguous, what real ComfyUI-native quantized
checkpoints always have), or any `.weight` tensor stored as `int8`/`uint8` (secondary,
catches a checkpoint with stripped metadata but still-integer weights). Deliberately
excludes bare `float8` weights from the second check — an isolated float8 tensor with no
sidecar is an existing, deliberately-supported case (coerced to float16, not rejected;
see `test_float8_input_coerced_to_float16`). Verified against both real files in this
report's directory: raises immediately on `juggernautZ_v10ByRundiffusion.safetensors`
(170 `.comfy_quant` sidecars found) and does not raise on `zImageBase_base.safetensors`
(genuinely plain BF16, no metadata) — same directory, so the false-positive risk of an
overly broad guard was checked directly, not just assumed.

**Workaround:** Use a genuinely unquantized source checkpoint (e.g. `zImageBase_base`,
not `juggernautZ_v10ByRundiffusion`, in the case that surfaced this).

## 13. NVFP4 output produced uniform noise even with every value/scale/layout check passing — packed nibbles were swapped for the real inference kernel

**Symptom:** NVFP4_MIXED output produced full-image colored-speckle noise even after
fixes #10-12 above, from a source checkpoint independently confirmed clean (no
pre-existing quantization) with all 413 weight tensors verified correct in value and
scale layout against the original and against `comfy_kitchen`'s own reference decode —
that verification held under this tool's own (as it turned out, wrong) nibble
convention, which is exactly why it didn't catch this bug; see Cause below.

**Cause:** `comfy_kitchen`'s `dequantize_nvfp4()` (a standalone helper) accepts a
`hi_first: bool = True` parameter controlling which half of each packed byte holds the
even-indexed element. But the function actually used during real ComfyUI inference,
`ck.scaled_mm_nvfp4()` (the fused GEMM kernel behind `_handle_nvfp4_linear`), **has no
such parameter at all** — its signature is fixed, with the nibble convention hardcoded
internally to match `hi_first=True`. This tool's `quantize_nvfp4()` packed nibbles in
the *opposite* order (`idx[..., 0]` — the even-indexed element — in the low nibble,
`idx[..., 1]` in the high nibble), i.e. `hi_first=False`.

This bug was invisible to every prior verification in this investigation: this tool's
own `dequantize_nvfp4()` test helper correctly decodes its own packing regardless of
convention (self-consistent by construction), and even the direct comparison against
`ck.dequantize_nvfp4()` "passed" because that standalone helper's `hi_first` argument
was explicitly set to match this tool's (wrong) convention — a discriminator only
insofar as someone thinks to test the *other* value. `scaled_mm_nvfp4`, the function
ComfyUI actually calls, doesn't expose that knob at all, so every weight was
nibble-swapped in real inference despite passing every dequantize-based check.

**Found by:** Realizing `ck.scaled_mm_nvfp4`'s signature has no `hi_first` parameter
(`inspect.signature(ck.scaled_mm_nvfp4)` — only `a, b, tensor_scale_a, tensor_scale_b,
block_scale_a, block_scale_b, bias, out_dtype, alpha`) after auditing all 413 weight
tensors in a real converted file and finding zero anomalies — ruling out every
data-correctness explanation forced looking at the *consumption* path instead of the
*data* itself.

**Fix:** Swapped `quantize_nvfp4()`'s packing to `(idx[..., 1] | (idx[..., 0] << 4))`
(even index in the high nibble) and `dequantize_nvfp4()`'s unpacking to match. Verified
directly against the real `scaled_mm_nvfp4` kernel (not just `dequantize_nvfp4`): a
weight quantized by this tool and multiplied via `scaled_mm_nvfp4` against a
`comfy_kitchen`-natively-quantized input now matches a full `comfy_kitchen`-native
ground truth (both operands quantized by `comfy_kitchen` itself) almost exactly —
13.22% relative mean error (ours) vs. 13.01% (ground truth), the expected error for a
4-bit × 4-bit matmul, not a bug signature. Added
`test_packs_even_index_in_high_nibble_hi_first_true`, which checks a hand-computed
packed byte value directly rather than relying on a self-consistent round trip.

**Lesson:** when a format exposes an explicit configuration parameter (like
`hi_first`) on one code path but not another, verify against the path with *no*
parameter — that's the one whose convention is actually load-bearing, and the
configurable path can silently mask a mismatch by being told what answer to give.
