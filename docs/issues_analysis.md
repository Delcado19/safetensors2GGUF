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

## 14. Non-mixed NVFP4 crashed on Lumina2's pad-token parameters — keys_hiprec doesn't protect outside *_MIXED mode

**Error message:**
```
RuntimeError: Error(s) in loading state_dict for NextDiT:
    size mismatch for x_pad_token: copying a param with shape torch.Size([1, 1920])
    from checkpoint, the shape in current model is torch.Size([1, 3840]).
    size mismatch for cap_pad_token: copying a param with shape torch.Size([1, 1920])
    from checkpoint, the shape in current model is torch.Size([1, 3840]).
```

**Affected models:** Lumina2/Z-Image checkpoints converted with the plain (non-`_MIXED`)
`NVFP4` safetensors target.

**Cause:** `ModelLumina2.keys_hiprec = ["x_pad_token", "cap_pad_token"]` already existed
(for an unrelated bf16 size-doubling issue, #2 above) and does keep these tensors
unquantized — but only under `*_MIXED` targets, since `quantize_tensor_st()` only
consults `keys_hiprec` when `mixed` is `True`. Under plain `NVFP4`, these tensors were
packed like any other 2D weight, halving their last dimension (2 values/byte). Unlike
`cap_embedder.1.weight` (#9), which corrupts ComfyUI's *inferred* architecture config,
`x_pad_token`/`cap_pad_token` are `nn.Parameter`s that ComfyUI's `NextDiT.__init__`
allocates with a shape hardcoded to the (correctly-detected) `dim` — so the mismatch
surfaces as `load_state_dict`'s own strict shape check failing outright, a hard crash
rather than silently-wrong values.

**Fix:** Added `"x_pad_token"` and `"cap_pad_token"` to `ModelLumina2.keys_shape_critical`
alongside `cap_embedder.1.weight` — that list is checked unconditionally (not gated by
`*_MIXED`), which is exactly the protection these two tensors were missing. Added
`test_nvfp4_pad_token_falls_back_to_f16_unconditionally`.

## 15. FP8_MIXED/NVFP4_MIXED remained visibly wrong (black bars, mirrored/wrong poses,
wrong identities) after every on-disk-data bug (#10-#14) was fixed and verified —
root cause is in ComfyUI's runtime, not in the file this tool writes

**Symptom:** With `rayZimageBaseNSFW_v2.safetensors` (a Lumina2/Z-Image checkpoint) and
the official `zImageBase_base.safetensors`, both FP8_MIXED and NVFP4_MIXED output
produced clearly wrong images compared to the unconverted original run with the same
prompt/seed — FP8_MIXED: a large black bar across part of the frame plus mirrored
composition and mismatched poses/identities; NVFP4_MIXED: full-image noise. This
persisted after: (a) a full 413-tensor audit proved every quantized *and*
pass-through weight in the FP8_MIXED file bit-for-bit correct against the original
(max error 3.6%, median 1.4% — expected FP8 rounding noise, not a bug); (b) writing
`full_precision_matrix_mult: true` into every layer's `_quantization_metadata` (forces
ComfyUI to skip the FP8 matmul kernel and dequantize-then-multiply in full precision
instead, while keeping `manual_cast` active) made the black bar disappear but left the
pose/identity mismatch; (c) adding `attention.qkv`/`attention.out` to
`ModelLumina2.keys_hiprec` (protecting a large fraction of the model's parameters —
confirmed via output file size barely shrinking vs. the unprotected conversion) made
no visible difference to the pose/identity drift at all.

**Investigation:** GGUF output for the same checkpoints reliably produces the same
image at lower quality, never a structurally different one — a strong signal that
whatever is wrong is specific to ComfyUI's *native* FP8/NVFP4 quantized-compute path,
not to quantization error in general. Reading `comfy/quant_ops.py`'s `QUANT_ALGOS`
registry (Comfy-Org/ComfyUI) confirmed the mechanism difference: `"float8_e4m3fn"` and
`"nvfp4"` both carry an `input_scale` parameter and no `"quantize_input": False`
override, meaning ComfyUI's `mixed_precision_ops` (`comfy/ops.py`) dynamically
quantizes *activations* too at inference time and dispatches real low-bit tensor-core
GEMM kernels — a fundamentally different, much younger code path than GGUF's
dequantize-to-full-precision-then-standard-matmul. A confirmed GitHub issue
(Comfy-Org/ComfyUI#14595, filed against **Anima**, a similarly new/custom DiT
architecture) shows this exact dynamic-activation-quantization logic silently
mishandles tensors depending on whether they get reshaped to 2D or 3D before a
`torch.nn.functional.linear` call — an architecture-dependent shape bug in ComfyUI's
own runtime, not in anything a checkpoint file controls. `"int8_tensorwise"` and
`"convrot_w4a4"` are the only two `QUANT_ALGOS` entries with `"quantize_input": False`
— weight-only quantization, activations always stay full precision, sidestepping this
code path entirely (much closer to GGUF's robustness).

**Conclusion:** given (a)-(c) above already ruled out on-disk data correctness,
compute-precision-only fallback, and attention-layer sensitivity as the cause, and
given the mechanism difference confirmed in ComfyUI's own source plus a filed bug
report against a structurally similar architecture, the remaining corruption most
plausibly originates in ComfyUI/`comfy_kitchen`'s own dynamic-activation-quantization
runtime path for Lumina2-family architectures — outside what this tool's conversion
output can fix. Per this project's debugging discipline (3+ failed fix attempts on the
same symptom → question the approach, not the next patch), FP8/NVFP4 were removed
from `SAFETENSORS_DTYPE_CHOICES` (the GUI-selectable list) — `safetensors_quant_fp8.py`
and `safetensors_quant_nvfp4.py` are kept and still covered by their existing tests
(they are byte-correct for what they claim to do), just no longer offered as a
diffusion-model conversion target given ComfyUI's own known-buggy consumption path.

**Fix:** Added `safetensors_quant_int8.py` implementing ComfyUI's `int8_tensorwise`
format (`comfy_kitchen.tensor.int8.TensorWiseINT8Layout`): single absmax scalar scale
per tensor, or — where the input dimension is a multiple of `CONVROT_GROUP_SIZE`
(256) — an offline block-Hadamard rotation (ConvRot) followed by per-row INT8
quantization, matching `comfy_kitchen`'s `quantize_int8_convrot_weight`/`_build_hadamard`/
`_rotate_weight` reference algorithms exactly (verified against
`Comfy-Org/comfy-kitchen`'s published source, including the `.comfy_quant` metadata
schema read by `comfy/ops.py`'s `_load_quantized_weight_body`:
`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`). Replaced
`FP8`/`FP8_MIXED`/`NVFP4`/`NVFP4_MIXED` with `INT8`/`INT8_MIXED` in
`SAFETENSORS_DTYPE_CHOICES` and the GUI's Convert → Safetensors tab default/description.
Also extended `ModelLumina2.keys_hiprec` with `adaLN_modulation`/`final_layer` (matching
this project's own per-tensor error audit and two independent community ComfyUI
quantization tools' per-architecture blacklists) alongside the existing
`attention.qkv`/`attention.out` protection — end-to-end image tests (same prompt/seed
against the unconverted checkpoint) confirmed correct pose/composition/no black bars for
both FP8_MIXED and INT8_MIXED once attention was protected, with `adaLN_modulation`
protection additionally correcting a visible lighting/mood shift.

**Follow-up bug found while verifying the fix — hiprec protection made output *larger*
than the unquantized source:** `quantize_tensor_st()`'s `mixed`-branch unconditionally
upcasts every `is_hiprec_st`-protected tensor to `torch.float32`, regardless of the
source dtype. With `keys_hiprec` limited to a handful of small/1D tensors (its original
scope) this cost little. Once it grew to cover `attention.qkv`/`attention.out` plus
`adaLN_modulation`/`final_layer` — a large fraction of the model's parameters — every one
of those bf16-sourced tensors doubled in size (2→4 bytes/element) for zero precision
benefit (ComfyUI casts every loaded weight to its own `compute_dtype` at inference time
regardless of on-disk dtype). Result: `INT8_MIXED` output for `rayZimageBaseNSFW_v2.safetensors`
was **12.57 GB — larger than the 12.31 GB unquantized bf16 source**, defeating the point of
quantizing at all. Found by the user asking to compare output file sizes against the
source after the image-quality fix above looked visually correct. Fixed by casting hiprec
tensors to their original dtype instead of a forced F32 (`data.to(old_dtype)` — a no-op
for float32 sources, since `is_hiprec_st` only ever returns `True` for `old_dtype` in
`(float32, bfloat16)`). Re-verified: same checkpoint now produces an 8.30 GB `INT8_MIXED`
file (~33% smaller than source). Added
`test_mixed_hiprec_tensor_keeps_original_dtype_not_forced_f32` — every prior test of this
code path used a float32 source, which is why the bug was invisible to the existing suite.

**Second follow-up — protection list still narrower than the established community
consensus:** with the black-bar/nibble/pose bugs above fixed and file size back under the
source, end-to-end testing still showed a visible identity/pose difference from the
unconverted original (same prompt/seed) on both the `ray` fine-tune and the official
`zImageBase_base` checkpoint — i.e. checkpoint-independent, ruling out a source-specific
issue. Cross-referencing `tritant/ComfyUI_Kitchen_nvfp4_Converter`'s published
per-architecture blacklist (a community ComfyUI custom node built directly on
`comfy_kitchen`, with a dedicated `"Z-Image-Base"` profile) showed this project's
`keys_hiprec` was still missing five submodules their profile protects: `cap_embedder`,
`x_embedder`, `t_embedder`, `noise_refiner`, `context_refiner` — the layers that build the
initial noise/caption representation *before* the main transformer blocks run, plausibly
more identity/composition-determining than the attention/modulation layers alone. Also
broadened `"attention.qkv"`/`"attention.out"` to a plain `"attention"` substring match (their
profile blacklists the whole submodule, catching `attention.q_norm`/`k_norm` too, though
those are 1D and already covered by the unconditional 1D hiprec rule). Verified all five
substrings exist as real tensor keys in a live Lumina2 checkpoint before adopting. Output
size grew from 8.30 GB to 8.77 GB (still ~29% smaller than the 12.31 GB source) — the
expected cost of protecting more of the model. End-to-end verification (same prompt/seed,
`ray` fine-tune and official `zImageBase_base`) confirmed pose/composition/identity now
match the unconverted original closely.

**Third follow-up — the same gap existed for every other architecture this project
supports that tritant's converter also has a profile for:** `ModelFlux` had *no*
`keys_hiprec` at all, `ModelQwenImage` had none, `ModelWan` only protected `.modulation`,
`CosmosPredict2` (the architecture Anima checkpoints use) only protected `pos_embedder` —
all four were exposed to the same class of *_MIXED accuracy risk Lumina2 had, just never
reported because no one had tested those combinations end-to-end yet. Adopted
`tritant/ComfyUI_Kitchen_nvfp4_Converter`'s published blacklists for all four (union of
the Qwen-Image-Edit-2511/Qwen-Image-2512 profiles for `ModelQwenImage`; Anima's profile —
cross-checked against `Comfy-Org/comfy-quants`' `anima.md`, which independently protects
the same embedder/final-layer/llm_adapter set — for `CosmosPredict2`). Not end-to-end
image-tested (no checkpoints for these architectures were available in this session,
unlike Lumina2) — the substrings are adopted on the strength of matching an established,
comfy_kitchen-based community tool's per-architecture configuration, not verified against
a live conversion+load+render cycle the way the Lumina2 fix was.

**Correction (later session) — the Comfy-Org/ComfyUI#14595 citation above was wrong.**
The Investigation and Conclusion sections above (and CHANGELOG.md, README.md,
docs/architecture.md, gui.py's Convert → Safetensors tab, safetensors_quant.py, and
safetensors_quant_int8.py at the time) cited issue #14595 as "a confirmed
architecture-dependent shape bug" causing the observed corruption. Reading the issue's
own text (fetched directly via `gh api repos/Comfy-Org/ComfyUI/issues/14595`) shows this
is wrong: the issue reports that some MLP GEMMs silently dispatch as bf16 instead of FP8,
costing ~15% of the expected speedup — the author's own words are that "the bf16 fallback
is silent... [but] no relevant logs" beyond a performance trace, i.e. a *performance*
regression, not a correctness one. Silently falling back to bf16 for some shapes is
mathematically still correct, just slower; it cannot produce black bars, mirrored
composition, wrong poses/identities, or full-image noise.

This does **not** reopen the INT8-only decision — the corruption itself was directly
observed (user-confirmed renders across two checkpoints), and three other pieces of
evidence independent of any specific issue number still support the same conclusion:
(1) the `QUANT_ALGOS["quantize_input"]` registry distinction itself — FP8/NVFP4/MXFP8
dynamically quantize *activations*, `int8_tensorwise`/`convrot_w4a4` don't, and activation
quantization is inherently lossy in a way no weight-side `keys_hiprec` list can touch,
which independently explains why broadening the protection list never fixed the
pose/identity drift; (2) the `full_precision_matrix_mult: true` experiment earlier in this
investigation, which removed the black bar and pointed at the quantized-compute path, not
on-disk data; (3) `marcorez8/Z-image-aka-Base-nvfp4`'s own published tiered quality
ratings (a completely independent third-party NVFP4 conversion of the same base model),
which show real, user-rated quality degradation on Z-Image at every attention-protection
level up to and including roughly this project's own `keys_hiprec` scope — external,
real-world corroboration that has nothing to do with #14595. The precise ComfyUI-side
mechanism producing the corruption remains formally unidentified; the decision to ship
INT8-only rests on the above three points, not on a specific bug citation.

Also worth noting for future reference: a maintainer comment on ComfyUI PR #14859 (the
INT4 ConvRot / `convrot_w4a4` addition) states that naive full-int4 quantization of a
model is low quality by the format author's own admission, and that a good INT4
conversion needs a per-layer mix of bit-widths this project's binary
protected/not-protected `keys_hiprec` model cannot express — a relevant constraint if
INT4 ConvRot is ever considered as a future format.

## 16. FP8 checkpoints work fine everywhere else on Civitai/HuggingFace — this
tool's own FP8 output didn't need to be excluded, it needed `full_precision_matrix_mult`

**Symptom:** #15 removed `FP8`/`FP8_MIXED` from `SAFETENSORS_DTYPE_CHOICES` alongside
NVFP4, reasoning that ComfyUI's `QUANT_ALGOS["quantize_input"]` registry marks both
formats as dynamic-activation-quantizing and therefore equally risky. The user pointed
out this doesn't match observed reality: scaled-FP8 checkpoints are extremely common on
Civitai/HuggingFace and load/render correctly in ComfyUI all the time — if FP8's runtime
path were unconditionally as risky as NVFP4's, that shouldn't be true.

**Investigation:** Reading `comfy/utils.py`'s `convert_old_quants()` (the function
ComfyUI uses to load legacy community "scaled_fp8" checkpoints — the format most
Civitai/HuggingFace FP8 releases actually use) shows it sets
`"full_precision_matrix_mult": true` in the layer config it synthesizes for those
checkpoints. `comfy/ops.py`'s `MixedPrecisionOps.Linear.forward()` checks this flag
before dispatching to the quantized-compute branch: when present, it dequantizes the
weight to the model's `compute_dtype` and runs a plain full-precision matmul instead of
FP8 tensor-core GEMM — the same dynamic-activation-quantization code path #15 identified
as the root cause of the Lumina2/Z-Image corruption is skipped entirely for that layer,
regardless of architecture. This explains the discrepancy: circulating community FP8
checkpoints are safe not because FP8 itself is safe, but because the *loader* already
opts them out of the risky path by default. This tool's own FP8 writer
(`safetensors_quant_fp8.py` via `convert_safetensors.py`) never set this flag, so its
output took the same risky path NVFP4 does — the corruption #15 diagnosed for
FP8_MIXED/NVFP4_MIXED was real for both formats, but only NVFP4 lacks a known opt-out.

**Conclusion:** FP8's dynamic-activation-quantization risk is avoidable by construction
(`full_precision_matrix_mult`), architecture-independently — unlike INT8's `keys_hiprec`
bet (a per-architecture, per-layer guess at which tensors need protection) or NVFP4,
which has no equivalent documented safe mode. FP8 does not need to stay excluded; it
needs to default to the same safety mechanism ComfyUI's own loader already relies on for
legacy checkpoints. The tradeoff is losing FP8 tensor-core compute speedup — this makes
FP8 output a storage/VRAM-savings format only, same practical performance profile as
INT8 or GGUF, unless a user explicitly opts out via the new `full_precision_fp8=False`
parameter.

**Fix:** `convert_to_safetensors()` (`convert_safetensors.py`) gained a
`full_precision_fp8: bool = True` parameter; when the target format resolves to
`float8_e4m3fn` (`FP8`/`FP8_MIXED`) and the flag is True (default), every FP8 layer's
`.comfy_quant` config now includes `"full_precision_matrix_mult": true`. Re-added
`FP8`/`FP8_MIXED` to `SAFETENSORS_DTYPE_CHOICES` (`safetensors_quant.py`) and updated
`format_recommendation()` to return an unconditional `"ok"` for FP8, unlike INT8's
per-architecture caution logic — the safety mechanism itself doesn't depend on
`keys_hiprec` or architecture at all. `model_support.py`'s `support_level()` (the data
model behind the "Model Support" GUI tab, see below) marks FP8/FP8_MIXED
`SUPPORT_VERIFIED` for every architecture on the same reasoning.

**Open follow-up — NOT done in this fix:** NVFP4 was not re-investigated here. It has no
known equivalent to `full_precision_matrix_mult` — `comfy/utils.py`'s
`convert_old_quants()` only synthesizes that flag for legacy `scaled_fp8` checkpoints,
not for NVFP4 ones, and nothing in `comfy/quant_ops.py`'s `QUANT_ALGOS["nvfp4"]` entry
suggests an equivalent safe-mode toggle exists. `model_support.py`'s `support_level()`
deliberately keeps NVFP4/NVFP4_MIXED at `SUPPORT_CAUTION` for every architecture pending
that investigation, tracked as a future, separate plan — see its docstring for the full
reasoning. Until then, NVFP4 stays out of `SAFETENSORS_DTYPE_CHOICES` for diffusion-model
output for the same reason #15 removed it.
