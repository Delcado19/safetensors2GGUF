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
