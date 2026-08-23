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
| `ModelSD3` | `y_embedder.mlp.0.weight` | `adm_in_channels` from `.shape[1]`; missing entry caused a live NVFP4/NVFP4_MIXED crash on load (`mat1 and mat2 shapes cannot be multiplied`), fixed 2026-08-16 |
| `ModelAura` | `cond_seq_linear.weight` | `cond_seq_dim` from `.shape[1]` |
| `ModelLumina2` | `cap_embedder.1.weight` | `cap_feat_dim` from `.shape[1]` |
| `CosmosPredict2` | `x_embedder.proj.1.weight` | Linear inside `PatchEmbed`'s `Sequential` (index 1, after a `Rearrange`) |
| `ModelHyVid` | `txt_in.input_embedder.weight` | `context_in_dim` from `.shape[1]`; `img_in.proj.weight` is a Conv and safe (only its kernel-width last dim is touched, and ComfyUI reads `.shape[1]`/`.shape[2:]` there, not `.shape[-1]`) |
| `ModelWan` | `head.modulation` | 3D `nn.Parameter` `(1, 2, dim)` — `dim` from `.shape[-1]`; not 1D, so the unconditional 1D-skip doesn't already cover it |
| `ModelLTXV` | `transformer_blocks.0.attn2.to_k.weight` | `cross_attention_dim` from `.shape[1]` |
| `ModelHiDream` | `.ff_i.gate.weight`, `img_emb.emb_pos` | not a shape-inference case (`unet_config` is hardcoded) — these are read via a raw state_dict assign that bypasses `MixedPrecisionOps`' quantized-loading path entirely, same bug class as `safetensors_quant.py`'s FP8-branch CLIP `position_embedding` example. NVFP4 halving `ff_i.gate.weight`'s last dim crashed load (`size mismatch ... [4, 1280] ... [4, 2560]`); plain INT8 produced severely corrupted renders. Fixed 2026-08-17 |

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
`FP8`/`FP8_MIXED` to `SAFETENSORS_DTYPE_CHOICES` (`safetensors_quant.py`).

**Correction (final whole-branch review):** the safety mechanism above only fixes
ComfyUI's *runtime compute* path — it does nothing for precision already lost when a
`keys_hiprec`-sensitive tensor is quantized to e4m3 *on disk*, which plain (non-mixed)
FP8 still does exactly like plain INT8. `format_recommendation()` therefore does **not**
return an unconditional `"ok"` for FP8: it now shares INT8's sensitive-architecture
warn/ok logic, warning on plain FP8 for sensitive architectures and recommending FP8
mixed instead. And because the `full_precision_matrix_mult` conclusion itself comes from
reading ComfyUI's source, not from an actual convert+load+render test with this tool's
own output on any architecture, `model_support.py`'s `support_level()` marks FP8/FP8_MIXED
`SUPPORT_CAUTION` (not `SUPPORT_VERIFIED`) everywhere, pending that render test.

**Second correction (render-test evidence, 2026-08-11):** the prediction in the first
correction above is now directly confirmed, not just theoretical. A same-seed/same-prompt
comparison of Z-Image Base output — full precision vs. this tool's plain `FP8` — showed
the standing figure's outfit (structured leather blazer/skirt vs. a latex catsuit with a
belt) and the entire room background/set dressing (portrait painting + brick wall vs. a
tufted couch + shelving + candles) changed between the two renders, with only the two
figures' poses staying recognizable — the same "wrong identity/composition, pose skeleton
survives" pattern this section's original Symptom described for pre-fix FP8_MIXED and
`#15`'s plain-INT8 corruption. `model_support.py`'s `_RENDER_CONFIRMED_BAD` and
`safetensors_quant.py`'s `_RENDER_CONFIRMED_BAD_PLAIN` now include `("lumina2", "FP8")`:
`support_level()` returns `SUPPORT_BAD` (not `SUPPORT_CAUTION`) for plain FP8 on lumina2,
and `format_recommendation()`'s warn message states the corruption as confirmed rather
than predicted by analogy from INT8. FP8_MIXED's status is unchanged by this — the
comparison only exercised plain FP8, and FP8_MIXED's own `keys_hiprec` protection (mixed
mode) is a different, still-unverified question.

**Third correction (FP8_MIXED render-test evidence, 2026-08-11):** that remaining
question is now answered too. A second same-seed/same-prompt comparison (a different
Z-Image Base prompt/checkpoint pairing) of full precision vs. this tool's `FP8_MIXED`
output showed composition, pose, face, makeup, jewelry, hair, and outfit all preserved —
the only deviation was the color/shape of a single secondary prop (a small creature held
in a jar), judged tolerable render-to-render quantization variance rather than a
correctness failure. This meets the same bar INT8_MIXED was already held to
(`_RENDER_VERIFIED_ARCHES` in the first correction above). `safetensors_quant.py`'s
`_RENDER_VERIFIED_ARCHES` (arch-only) was generalized to `_RENDER_VERIFIED_MIXED`
(`(arch_key, format_key)` pairs, mirroring `_RENDER_CONFIRMED_BAD`'s shape) and now
includes both `("lumina2", "INT8_MIXED")` and `("lumina2", "FP8_MIXED")`.
`support_level()` returns `SUPPORT_VERIFIED` for FP8_MIXED on lumina2;
`format_recommendation()` no longer discloses the "not yet confirmed by a render test"
caveat for that combination.

**Open follow-up — NOT done in this fix:** NVFP4 was not re-investigated here. It has no
known equivalent to `full_precision_matrix_mult` — `comfy/utils.py`'s
`convert_old_quants()` only synthesizes that flag for legacy `scaled_fp8` checkpoints,
not for NVFP4 ones, and nothing in `comfy/quant_ops.py`'s `QUANT_ALGOS["nvfp4"]` entry
suggests an equivalent safe-mode toggle exists. `model_support.py`'s `support_level()`
deliberately keeps NVFP4/NVFP4_MIXED at `SUPPORT_CAUTION` for every architecture pending
that investigation, tracked as a future, separate plan — see its docstring for the full
reasoning. Until then, NVFP4 stays out of `SAFETENSORS_DTYPE_CHOICES` for diffusion-model
output for the same reason #15 removed it.

## 17. NVFP4's "no equivalent safe mode" follow-up from #16, closed — and a
naming pass to align this tool's format keys with ComfyUI/Civitai/HuggingFace convention

**Follow-up (2026-08-13, recorded retroactively):** #16's "Open follow-up" above turned out
to be wrong. Reading `comfy/ops.py`'s `MixedPrecisionOps.Linear.forward()` directly (not
just `convert_old_quants()`, which is FP8-only) shows `full_precision_matrix_mult` is read
generically from any layer's `.comfy_quant` config, regardless of quant format — nothing
gates it to `float8_e4m3fn`. `convert_safetensors.py` already sets it for NVFP4 too
(`full_precision_nvfp4=True`, default on) since that finding. The "NVFP4 has no equivalent
safe mode" conclusion only ever applied to ComfyUI's own legacy-checkpoint upgrade path,
never to this tool's own writer.

**Render-test evidence:** NVFP4/NVFP4_MIXED have since been render-tested clean across
`flux`, `sdxl`, `sd1`, `sd3`, `hidream`, and `aura` (`safetensors_quant.py`'s
`_RENDER_VERIFIED_MIXED`, see each entry's own comment for the specific evidence) — the
render-test bar #16 held FP8 to before re-adding it is equally met here.

**Symptom (2026-08-19, separate from the above):** the user asked whether this tool offers
the "fp8_e4m3fn_scaled" compression seen constantly on Civitai as an alternative to fp16
releases. It does (this tool's `FP8`), but nothing in the GUI or output filenames said so —
`FP8` doesn't read as "scaled fp8_e4m3fn" to someone comparing against a Civitai listing.
Asked to research and align this tool's naming against ComfyUI/Civitai/HuggingFace/
llama.cpp convention broadly, not just for FP8.

**Findings:**
- `F16`/`FP16`, GGUF's `Q4_K_M`/`Q8_0`/etc. (llama.cpp's own naming), and `INT8`/`INT8_MIXED`
  (no established external convention for this tool's tensor-wise/ConvRot method) already
  match or have no external term to align to. Left unchanged.
- `FP8`/`FP8_MIXED` — external convention is `fp8_e4m3fn_scaled` (Comfy-Org's own
  repackaging name, ubiquitous on Civitai/HuggingFace). This tool only writes the e4m3fn
  variant, never e5m2.
- `NVFP4`/`NVFP4_MIXED` — the key already matches NVIDIA's own name. The real risk is
  confusion with **NF4** (bitsandbytes' 4-bit normalfloat, common on Civitai for Flux NF4
  releases) — a different algorithm this tool does not offer. Documented, not renamed.

**Fix:** `SAFETENSORS_DTYPE_CHOICES` (`safetensors_quant.py`) re-gained `NVFP4`/
`NVFP4_MIXED` (closing #16's follow-up) and every label now spells out the external-naming
equivalent or lack thereof. Deliberately did **not** rename the internal target_key strings
(`"FP8"`, `"NVFP4_MIXED"`, etc.) — that identifier is threaded through `_MIXED_KEYS`,
every `_RENDER_VERIFIED_MIXED`/`model_support.py` tuple, and ~330 tests; renaming it would
be a purely cosmetic, repo-wide, high-risk change for no behavioural benefit. Instead, a
new `filename_suffix_for()` in `safetensors_quant.py` maps `target_key` to an
output-filename suffix, identity for every key except `FP8`→`"fp8_e4m3fn_scaled"` and
`FP8_MIXED`→`"fp8_e4m3fn_scaled_mixed"`. Applied everywhere a default/templated output
filename gets built: `convert_to_safetensors()`'s own `dst_path=None` default
(`convert_safetensors.py`), and `gui.py`'s `_resolve_dst_st`/`_resolve_dst_te` (the
Safetensors and Convert Text Encoder tabs' Browse-button/`{ftype}`-template resolution).
`gui.py`'s stale `apply_support_table_selection()` docstring (asserting NVFP4 is
deliberately absent) and `text_encoder_convert.py`'s NVFP4 dropdown label were also
corrected/aligned. Tests added.

---

## 18. `convert.py`'s GGUF path had no dequantization pre-pass for already-quantized
source checkpoints — crashed llama-quantize on a community Wan 2.2 fp8_mixed file

**Symptom (2026-08-20):** quantizing the remaining Wan 2.2 diffusion-model formats (FP8,
FP8_MIXED, INT8, INT8_MIXED, NVFP4, GGUF Q4_K_M — only NVFP4_MIXED was previously
render-verified) from the community `DaSiWa_Wan22_I2V_14B_SnatchKiss_v11_{HIGH,LOW}_
fp8_mixed.safetensors` checkpoints (no bf16 source available). The five safetensors
formats built fine via `convert_to_safetensors()`. The GGUF path crashed:
`llama-quantize` exited with `GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS) failed`
(Windows exit code 3221226505 / `STATUS_STACK_BUFFER_OVERRUN` when run through `quantize.
run_quantize()`, since it swallows llama-quantize's own stderr assert message).

**Root cause:** `convert_safetensors.py` has had `_scan_quantized_layers()` since #12 —
it detects already-quantized `.weight` tensors in the *source* checkpoint (via their
`.weight_scale`/`.comfy_quant` sidecars) and dequantizes them before re-quantizing to the
requested target format. `convert.py`'s GGUF path never had the equivalent. This
particular source checkpoint carries hundreds of 0-dimensional F32
`.weight_scale` tensors (one per quantized Linear layer, Comfy-Org's own fp8_scaled
repackaging convention) that `convert.py`'s `handle_tensors()` wrote straight through as
ordinary weight tensors — a 0-dim tensor is invalid in GGUF/GGML (`n_dims` must be 1–4),
which is what `llama-quantize` asserts on. Beyond the crash, this would also have been
silently *wrong* had it not crashed: the real weight tensors (on-disk as `float8_e4m3fn`)
were being coerced straight to float16 and written out without ever being multiplied by
their scale, i.e. the GGUF's raw fp8 bit patterns reinterpreted as scaled-down fp16
values — an architecture-independent correctness gap that could silently corrupt any
future GGUF conversion of a pre-quantized source checkpoint, not just this one.

**Fix:** moved `_scan_quantized_layers()` (and its `_ALREADY_QUANTIZED_DTYPES`/
`_PASSTHROUGH_TENSOR_SUFFIXES` constants) from `convert_safetensors.py` into
`dequantize.py`, the one module both `convert.py` and `convert_safetensors.py` already
depend on without creating a cycle (`convert_safetensors.py` imports `load_state_dict`
from `convert.py`, so `convert.py` importing back from `convert_safetensors.py` directly
would have been circular). `convert.py`'s `handle_tensors()` now runs the same
scan-and-dequantize pre-pass — skip known scale/comfy_quant sidecar keys, and for any
`.weight` key `_scan_quantized_layers()` flagged, call `dequantize_weight()` before the
existing dtype-coercion/nan-clamp/GGUF-write path. Both call sites now share one
implementation instead of two independently-maintained copies. 346 tests pass (no new
tests added — this is a plumbing fix with existing coverage of `_scan_quantized_layers`/
`dequantize_weight` themselves via `test_convert_safetensors.py`/`test_dequantize.py`;
the GGUF-specific integration only surfaces with a real pre-quantized multi-GB checkpoint,
impractical to fixture).

**Separate footgun found while fixing the above (2026-08-20, workflow not code):**
`convert.py`'s 5D-tensor side-car file is named `fix_5d_tensors_<arch>.safetensors` —
architecture-scoped, not source-file-scoped. Building Wan's GGUF for both HIGH and LOW
noise checkpoints back-to-back (same `arch == "wan"`) without running `fix_5d_tensors.py`
between them silently overwrote HIGH's side-car with LOW's before HIGH's Q4_K_M was ever
fixed — both checkpoints' `patch_embedding.weight` values differ (confirmed:
`max abs diff` ~0.13, not a near-duplicate), so this would have shipped the LOW model's
patch-embedding weight inside the HIGH GGUF and left the LOW GGUF missing its own
patch-embedding entirely (nothing left to insert once already consumed... in this
instance the HIGH `.safetensors` was still on disk, so the correct tensor was
re-extracted and both files were manually re-fixed; not always recoverable in general).
No code change made — the GUI's own per-conversion "Fix 5D Tensors" step already forces
sequential single-file handling, so this only bites scripted/batch use outside the GUI.
Noted here so a future batch script (or CLI wrapper) applies `fix_5d_tensors()`
immediately after each GGUF build instead of batching the whole side-car step to the end.

---

## 19. Plain INT8's tensor-wise scale was a `(1,)`-shaped tensor, not a true scalar —
crashed ComfyUI's low-VRAM dynamic-requantize path

**Symptom (2026-08-20):** a live SDXL `INT8`/`INT8_MIXED` render (the next architecture
queued for testing after Wan) crashed with
`RuntimeError: output with shape [1] doesn't match the broadcast shape [640, 1]`, deep in
`comfy_kitchen`'s tensor-copy machinery (`comfy/ops.py`'s `cast_bias_weight` ->
`resolve_cast_module_with_vbar` -> `post_cast` -> `orig.copy_(y)` ->
`comfy_kitchen.tensor.base.py`'s `_handle_copy_`).

**Root cause:** `safetensors_quant_int8.py`'s `quantize_int8_tensorwise()` wrote its
per-tensor scale as `scale.reshape(1)` — a 1-element 1D tensor, `.dim() == 1`. Reading the
installed `comfy_kitchen` package's `tensor/int8.py` directly:
`TensorWiseINT8Layout.requantize_kwargs()` derives `"per_channel": bool(is_weight and
(convrot or params.scale.dim() > 0))` — it uses `.dim() > 0`, not `.numel() > 1`, to
detect ConvRot's per-row layout. Our `(1,)`-shaped "scalar" scale satisfies `dim() > 0`,
so it's misread as per-channel. This is harmless for the ordinary matmul path
(`_handle_int8_linear_tensorwise`/`_handle_int8_mm_tensorwise`, both of which correctly
check `.numel() == 1`, not `.dim()`) — the bug only fires when `comfy/ops.py`'s low-VRAM
"vbar" offload path calls `requantize_from_float()` under VRAM pressure (`want_requant`/
`update_weight` in `post_cast`), which uses `requantize_kwargs()`'s (wrong) `per_channel`
result to allocate a fresh `[out_features, 1]` per-row scale, then tries to copy it back
into the original `(1,)`-shaped resident buffer. Explains why this was never caught by
any of the render tests backing `wan`/`hidream`/`aura`/`flux`'s existing `INT8`/
`INT8_MIXED` `✓ Verified` status — none of those test sessions happened to trigger a
low-VRAM offload.

**Fix:** `quantize_int8_tensorwise()` now writes `scale` directly (a genuine 0-dim
tensor from `.abs().max()`, never reshaped) instead of `scale.reshape(1)` — matches
`comfy_kitchen`'s own scalar convention and sidesteps the `dim() > 0` misdetection.
`safetensors.torch.save_file`/`load_file` round-trip 0-dim tensors correctly (verified
directly). ConvRot's per-row `[out_f, 1]` scale (`quantize_int8_convrot()`) is unaffected
— it's supposed to be 2D. `dequantize_weight()`/`_scan_quantized_layers()`/
`convert_safetensors.py`'s `.comfy_quant`-sidecar `convrot` flag all already keyed off
`.numel()`, not `.dim()`, so no other code needed to change. Test added
(`test_plain_int8_scale_is_a_true_scalar_not_reshaped`). 348 tests pass.

**Not yet done:** every `INT8`/`INT8_MIXED` safetensors file already built and deployed
before this fix (currently `wan`, `hidream`, `aura`, `flux` per `_RENDER_VERIFIED_MIXED`)
still carries the old `(1,)`-shaped scale on disk and would hit the same crash under
low-VRAM offload. Rebuilding them is a user decision (each is a multi-GB re-quantization),
not done automatically as part of this fix.

## 20. Plain NVFP4 crashed with `IndexError` on a genuine 0-dim scalar tensor —
`quantize_tensor_st()`'s 1D-only guard didn't cover 0-dim

**Symptom (2026-08-20):** the first real Qwen-Image-Edit-2511 batch conversion (source:
Comfy-Org's official `fp8_scaled` repackage, dequantized via `_scan_quantized_layers`
before requantizing) crashed partway through plain `NVFP4` with
`IndexError: tuple index out of range` at `safetensors_quant_nvfp4.py`'s
`if data.shape[-1] % GROUP_SIZE != 0:` — the source checkpoint carries a bare 0-dim
scalar tensor, `__index_timestep_zero__`, whose `.shape` is an empty tuple.

**Root cause:** `safetensors_quant.py`'s `quantize_tensor_st()` only special-cased
`data.dim() == 1` (biases/norm weights) before dispatching to the format-specific
quantizer; a true 0-dim scalar fell through that check and reached `quantize_nvfp4()`
directly, which raises `IndexError` (not the `ValueError` the NVFP4 branch's
`except ValueError` fallback catches) on an empty shape tuple. `FP8`/`INT8` happened to
survive the same tensor only by accident — each has its own unrelated `dim()` check
further down that also happens to catch 0-dim (FP8 has none — see below; INT8's
`data.dim() != 2` routes it to the tensor-wise path, which works on any shape). This was
the first architecture whose source checkpoint carried a genuine 0-dim metadata tensor;
no prior architecture (Flux/SD3/AuraFlow/HiDream/Lumina2/Cosmos/SDXL/SD1/Wan) triggered
this path.

**Fix:** widened the guard from `data.dim() == 1` to `data.dim() <= 1` — a 0-dim scalar
has the exact same "no accuracy or size benefit to scale-quantizing" rationale already
documented for 1D tensors, so it belongs in the same unconditional F32 passthrough, not a
per-format patch. Test added (`test_nvfp4_non_mixed_0dim_scalar_does_not_crash`).

## 21. `.input_scale` activation-quant sidecars weren't in bug #18's dequant skip-set —
crashed `llama-quantize.exe` on GGUF Q4_K_M

**Symptom (2026-08-20):** the same Qwen-Image-Edit-2511 batch conversion's GGUF `Q4_K_M`
step ran the F16-GGUF intermediate export cleanly, then `llama-quantize.exe` crashed with
exit code `3221226505` (`0xC0000409`, `STATUS_STACK_BUFFER_OVERRUN`) partway through
quantizing.

**Root cause:** Comfy-Org's `fp8_scaled` repackage carries a per-layer `.input_scale`
sidecar (a dynamic-activation-quantization scale used by the runtime FP8 compute kernel)
alongside every `.weight_scale`, in addition to the already-handled `.weight_scale`
family. `dequantize.py`'s `_scan_quantized_layers()` skip-set (added in bug #18's fix)
only covered `.weight_scale`/`.weight_scale_2`/`.comfy_quant`/`.scale_weight`/
`scaled_fp8` — `.input_scale` wasn't on that list, so ~1600 orphaned 0-dim `.input_scale`
tensors (meaningless once the weight is dequantized back to float — they scaled a
runtime *activation*, not this checkpoint's weights) were carried straight through into
the GGUF writer, exactly the same crash class bug #18's own docstring already documents
for un-skipped scale sidecars ("crashes llama-quantize with
`GGML_ASSERT(n_dims >= 1 && n_dims <= 4)`" — here manifesting as a stack-buffer-overrun
instead, likely from the sheer volume of unexpected 0-dim tensors rather than a single
one). `convert_safetensors.py`'s safetensors-output formats (FP8/INT8/NVFP4_MIXED) built
from the same source earlier in this session carry the same orphaned tensors but didn't
crash — ComfyUI's safetensors loader silently ignores unrecognized extra keys; the
external `llama-quantize.exe` binary does not.

**Fix:** added `.input_scale` to `_scan_quantized_layers()`'s skip set. Test added
(`test_input_scale_is_skipped_not_carried_through`). Formats already built earlier in
this same batch run (FP8/INT8/INT8_MIXED/NVFP4_MIXED safetensors) still carry the inert
extra tensors on disk — harmless (ComfyUI ignores them) but not rebuilt, since only the
GGUF path actually crashed.

**Correction (2026-08-20, same day) — this fix was necessary but not sufficient; GGUF
`Q4_K_M` still crashes for `qwen_image` with the exact same `llama-quantize.exe` exit
code (`0xC0000409`) after the `.input_scale` fix.** Root cause is architectural, not
another leftover sidecar: `city96/ComfyUI-GGUF`'s `tools/lcpp.patch` — the patch that
makes `llama-quantize` understand diffusion-model GGUFs at all (see
`docs/building-llama-quantize.md`) — only adds 11 `llm_arch` enum values (FLUX, SD1,
SDXL, SD3, AURA, LTXV, HYVID, WAN, HIDREAM, COSMOS, LUMINA2). `qwen_image` is not one of
them — confirmed against the current patch on `main` (zero matches for "qwen") and
against the still-open, unanswered upstream issue
[city96/ComfyUI-GGUF#347](https://github.com/city96/ComfyUI-GGUF/issues/347) asking for
exactly this. With an unrecognized `general.architecture` string, `llama-quantize`
doesn't take the early-return path the patch adds for known image archs and instead runs
into undefined behavior trying to process an MMDiT checkpoint as if it might be a
text/LLM model — manifesting as the stack-buffer-overrun crash, not a clean error
message.

**Conclusion: GGUF `Q4_K_M` (and every other llama-quantize K-quant) is currently
structurally unsupported for `qwen_image` with this tool's llama-quantize dependency —
not a bug in this tool's own code, an upstream gap.** The `.input_scale` fix above is
still correct and stays (a real, separate bug that would have hit any architecture
carrying dynamic-FP8 activation scales, once one exists that the patch *does* support),
but does not unblock Qwen-Image GGUF on its own. The only current workaround is a
pre-quantized community GGUF built with different/unpublished tooling (e.g.
`city96/Qwen-Image-gguf` on Hugging Face) — not reproducible with this project's public
`llama-quantize` binary. No code change follows from this until upstream adds
`qwen_image` support to the patch (tracked by watching #347) or this project builds its
own patch extension — out of scope for now.

## 22. Qwen2.5-VL-7B GGUF text-encoder crashes on any image-conditioning workflow —
`llama.cpp`'s converter drops the vision tower entirely

**Symptom (2026-08-23):** render-testing Qwen-Image-Edit-2511 (FP8 diffusion model +
`qwen2.5_vl_7b_huihui_abliterated_q4_k_m.gguf` text encoder), `TextEncodeQwenImageEdit`
crashed:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (780x1280 and 3840x1280)
```
raised inside `comfy/text_encoders/qwen_vl.py`'s vision-tower forward pass
(`self.qkv(hidden_states)`), reached via `llama.py:preprocess_embed` →
`self.visual(image, grid)`.

**Root cause:** `text_encoder_convert.py` shells out to plain `llama.cpp`'s
`convert_hf_to_gguf.py` without `--mmproj`, so only the `Qwen2VLModel` (`TextModel`)
class runs. Verified directly against the built file with `gguf.GGUFReader`: **0 of 339
tensors** start with `visual.` — the entire vision tower is absent, versus 650
`visual.*` tensors present in the equivalent NVFP4_MIXED safetensors build from the same
source. `--mmproj` mode does export a vision tower (`Qwen2VLVisionModel` in
`.llama.cpp/conversion/qwenvl.py`), but into a *separate* mmproj GGUF file using
llama.cpp's own multimodal-runtime tensor names (`v.blk.N.attn_q/k/v.weight`, QKV
pre-split) — a different naming scheme than ComfyUI-GGUF's `CLIPLoaderGGUF`, which
expects the vision tower inline in the same file under ComfyUI's own names
(`visual.blocks.N.attn.qkv.weight`, fused). Neither mode this tool's pipeline can drive
produces a file `CLIPLoaderGGUF` can load correctly for image-conditioning use.

With the vision tower missing, ComfyUI still constructs the `visual` submodule from its
own Python architecture definition (independent of GGUF content) and loads whatever
partial state the GGUF sidecar *does* provide; the resulting `qkv` weight ends up
transposed relative to what `torch.nn.functional.linear` expects, producing the shape
mismatch above rather than a clean "missing key" error.

**Conclusion: GGUF text encoders for Qwen2.5-VL-7B (and any Qwen-VL-family model) are
structurally broken for image-conditioning workflows — not a bug in this tool's own
quantization code, a gap in what `llama.cpp`'s public converter can produce for
ComfyUI-GGUF's loader.** Text-only encoding (no image input) was not tested and may
still work, since only the language-model tower matters there — but that doesn't apply
to Qwen-Image-**Edit**, which always feeds a reference image through the vision tower.
**Practical guidance: use a safetensors format (FP8/INT8/NVFP4/etc.) for any Qwen-VL
text encoder used in an edit/image-conditioning workflow; GGUF is only safe for
pure-text Qwen-VL use, if that.** No code change made — same category as bug #21's
correction (upstream tooling gap, not fixable inside this project's own quantization
code without a from-scratch GGUF writer matching ComfyUI's exact key/layout
conventions, out of scope).

---

## 23. FLUX.2 dev's Mistral text encoder failed to load with `json.decoder.JSONDecodeError`
— a second sentinel-tensor byte-blob, same bug class as `spiece_model` (#18's era), new name

**Symptom (2026-08-23):** loading a batch-built `mistral_3_small_flux2_fp8_mixed.safetensors`
(FLUX.2 dev's text encoder, this tool's own FP8_MIXED output) in ComfyUI's `CLIPLoader`
crashed:
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```
raised inside `comfy/text_encoders/flux.py`'s `load_mistral_tokenizer()` ->
`from_tekken_json()` (`comfy/text_encoders/bpe_tokenizer.py`), which calls `json.loads()`
directly on a tensor's raw bytes.

**Root cause:** `mistral_3_small_flux2_*.safetensors` embeds the raw Tekken tokenizer JSON
as a top-level 1D `uint8` tensor named `tekken_model` — the same "self-contained tokenizer
blob" pattern already documented for Comfy-Org's `spiece_model` (Wan/UMT5's SentencePiece
model, see #18's era in `dequantize.py`'s `_PASSTHROUGH_TENSOR_SUFFIXES` docstring), just a
different sentinel name this project had never encountered before this session. Verified
directly: source `tekken_model` is `torch.uint8`, shape `(19399895,)`, starting bytes
`b'{\n    "config": {...'` (valid JSON); the built FP8_MIXED output has the same key at
`torch.float32` — every other tensor in `convert_to_safetensors()`'s main loop goes through
`nan_to_num`/`quantize_tensor_st()`, both of which assume floating-point weight data, and
neither `_PASSTHROUGH_TENSOR_SUFFIXES` (only covered `spiece_model`) nor any other guard
exempted this new sentinel name. Affects **all 6 safetensors formats** built from this
source before the fix (FP8/FP8_MIXED/INT8/INT8_MIXED/NVFP4/NVFP4_MIXED all share the same
conversion path with no format-specific exemption), not just the one that happened to be
tested first.

**Fix:** added `"tekken_model"` to `dequantize.py`'s `_PASSTHROUGH_TENSOR_SUFFIXES` (now
`("spiece_model", "tekken_model")`) — copied through byte-identical, same as `spiece_model`.
Also added it defensively to `text_encoder_convert.py`'s `_GGUF_INCOMPATIBLE_TENSOR_SUFFIXES`
for consistency, though this family's GGUF path is currently blocked earlier anyway (see
the correction to this doc's earlier `mistral-small-3.2-24b` GGUF entry in `model_support.py`
— `detect_text_encoder_family()` fails outright on FLUX.2's bespoke 30-layer trim before
this stripping code would even run). Test added
(`test_preserves_tekken_model_sentinel_tensor_byte_identical`, mirroring the existing
`spiece_model` test). All 6 already-built `mistral_3_small_flux2_*.safetensors` files
rebuilt after the fix.

**How to apply:** any new text-encoder family with its own self-contained tokenizer
convention (a raw tokenizer file embedded as a byte-blob tensor) needs its sentinel key name
added to `_PASSTHROUGH_TENSOR_SUFFIXES` before a real batch run, not discovered after the
fact by a ComfyUI load failure — check for suspiciously-named top-level `uint8`/`int8`
tensors with no `.weight` suffix when onboarding a new family, the same way `keys_hiprec`
gets checked proactively per [[feedback_check_keys_hiprec_first]].

---

Findings noted for future reference that did not lead to a code change, because the
evidence didn't point at a defect in this tool's output.

### HiDream-I1 text encoders: subject placement drift on a spatially complex prompt, not correlated with quantization strength

**Observation (2026-08-18):** render-testing HiDream-I1's Llama-3.1-8B and T5-XXL text
encoders (see `_TE_RENDER_VERIFIED`'s docstring in `model_support.py` for the pass/fail
evidence) with a spatially complex two-subject prompt ("a woman stands beside a grand
piano, leaning toward a seated pianist") showed the standing subject's position relative
to the piano shift between some formats: in the original source checkpoint and this
tool's plain `FP8` Llama-3.1-8B output, she stands clearly in front of the piano case
(her legs occlude it, floor tiles visible beneath her); in this tool's `F16` and `INT8`
Llama-3.1-8B output, her body instead overlaps the piano's keybed with no visible
separation, as if standing inside the instrument. On the T5-XXL side, `NVFP4` and
`NVFP4_MIXED` placed her correctly (matching the source); `F16`, `FP8`, `FP8_MIXED`,
`INT8`, and `INT8_MIXED` did not.

**Why this isn't classified as a defect:** the pattern doesn't correlate with
quantization strength in either direction. `F16` — the least lossy of this tool's
outputs, effectively a precision re-save of the source tensors — reproduced the
misplacement, while the more lossy `FP8` did not; `NVFP4`/`NVFP4_MIXED` (normally the
formats under the most scrutiny) were the *correct* ones on the T5-XXL side. A defect
tied to this tool's quantization code would be expected to at least trend with format
lossiness; this doesn't. All renders used a fixed seed and only the text-encoder weights
varied, so the effect is real (not seed noise) — but it looks like inherent sensitivity
of this specific, decision-boundary-adjacent multi-subject spatial prompt to small
numerical perturbations in the text-encoder output, not a correctness bug in any one
format. The simpler single-subject prompt used for the actual pass/fail render-test
evidence (a lone humanoid figure, no relative-placement instruction) showed zero
compositional deviation across all 14 format combinations tested.

**Action:** none taken. Not used to change any format's support-level classification in
`model_support.py`. Recorded here in case the same pattern reappears on another
architecture or prompt and turns out to be systematic after all.

### Wan 2.2 I2V test renders show ghosting/outfit deformation from ~frame 48 (~3s @16fps) onward, identically across every format including the baseline

**Observation (2026-08-20):** reviewing the `wan` format-coverage renders (see #18's
`Verified` entry) frame-by-frame beyond the blink-timing check, frames past ~48/81 show a
translucent double-exposure "ghost" trailing the character's silhouette during the
prompted head-turn ("looking around"), worsening by frame 60 into visible outfit
deformation (backpack straps/color corrupted, shape distorted). Confirmed present
identically in the `NVFP4_MIXED` baseline as well as `FP8`/`INT8`/`NVFP4`/etc. — same
severity, same onset frame, across every format tested. Since the effect doesn't vary by
format at all, it isn't a quantization defect: `_RENDER_VERIFIED_MIXED`'s `wan` entries
stand as recorded (the formats are all equally faithful to whatever the diffusion model
itself produces here).

**Likely cause (not investigated further, out of scope for this task):** the "Format
Testing" workflow's `KSamplerAdvanced` nodes use `steps=4, cfg=8` with no visible
Lightning/speed LoRA in the workflow graph — 4 steps is very low for Wan 2.2 video
diffusion outside a distilled/accelerated setup (which normally also runs `cfg=1`, not
8). Fast prompted motion (a head turn) is a plausible trigger for exactly this kind of
temporal-coherence breakdown under-sampled diffusion produces. Not fixed or reproduced
against a corrected workflow — recorded so it isn't mistaken for a per-format artifact if
someone reruns this test workflow later.
