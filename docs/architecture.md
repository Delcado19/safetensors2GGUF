# Architecture

## Conversion Pipeline

```
Source file (.safetensors / .ckpt / .pt)
    │
    ▼
load_state_dict()
    │  - .safetensors → returns a _LazyStateDict wrapper around safe_open()
    │      that materializes tensors on demand via get_tensor() — keeps peak
    │      RAM bounded to a single tensor instead of the full state-dict
    │      (required for >RAM checkpoints, e.g. 19 GiB FP8 Qwen-Image-Edit
    │      on a 16 GiB system)
    │  - .ckpt / .pt / .bin / .pth → torch.load (eager); no streaming API
    │  - Detects and strips state-dict prefixes (e.g. "model.diffusion_model.")
    │      via shared _build_key_map helper that drives both paths
    │
    ▼
detect_arch()
    │  - Matches keys against keys_detect of each architecture class
    │      (uses dict-style __contains__; works identically against
    │       _LazyStateDict and eager dict)
    │  - Raises AssertionError when no architecture is detected
    │
    ▼
handle_tensors()
    │  - Iterates state_dict.items() directly (no list materialization)
    │      so _LazyStateDict streams one tensor at a time
    │  - Filters keys_ignore
    │  - Converts dtype: BF16 → F32, float8 → F16
    │  - Clamps inf/NaN → 0 / ±65504 (prevents llama-quantize validation failures)
    │  - Decides quantization type:
    │      1D or ≤ 1024 elements or keys_hiprec → F32
    │      BF16 source → BF16
    │      otherwise → F16
    │  - Reshape for SD1/SDXL (shape_fix): (H,W) → (n//256, 256)
    │      stores orig_shape as metadata field
    │  - Handles 5D tensors: offload instead of write
    │
    ▼
GGUFWriter (use_temp_file=True)
    │  - Writes header (arch, file_type, quantization_version)
    │  - Writes KV metadata
    │  - Spills converted tensor bytes to a SpooledTemporaryFile
    │      (256 MiB RAM buffer, remainder on disk in %TEMP%) so the
    │      writer never accumulates the full F16/BF16 state-dict in RAM
    │  - Final write copies temp file into the output GGUF
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
    keys_unsqueeze = []    # 1D tensors reshaped to [1, D] before writing
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

### Architecture Ordering in `arch_list`

The list is iterated in order and the first matching architecture wins. This
matters whenever two architectures share tensor names that one of them treats
as banned. Concrete case: Qwen-Image shares
`transformer_blocks.0.attn.norm_added_k.weight` with Flux and
`transformer_blocks.0.attn.add_q_proj.weight` with SD3. Both keys are listed
as `keys_banned` on those classes (reference-format guard). Without ordering,
a Qwen-Image checkpoint would be misclassified as a banned Flux/SD3 reference
file and abort with "Model architecture not allowed for conversion". To avoid
this, `ModelQwenImage` is placed before `ModelFlux` and `ModelSD3` in
`arch_list` so its more specific detection keys
(`time_text_embed.timestep_embedder.linear_2.weight` +
`transformer_blocks.0.attn.norm_added_q.weight` +
`transformer_blocks.0.img_mlp.net.0.proj.weight`) match first.

## 5D Tensor Handling

GGUF supports at most 4-dimensional tensors. Models like HunyuanVideo and Wan
occasionally contain 5D tensors (e.g. RoPE frequencies).

**Two-step process:**

1. `convert.py`: The 5D tensor is **not** written to the GGUF file; instead it is
   offloaded to `fix_5d_tensors_<arch>.safetensors`.

2. `fix_5d_tensors.py`: Reads the fully quantized GGUF file and inserts the
   offloaded tensor as F32.

## SDXL Component Extraction

`component_extract.py` handles the optional VAE/CLIP extraction path used by
the Web UI.  It does not change the GGUF conversion pipeline.

For SDXL checkpoints, the relevant embedded groups are:

| Component | Embedded checkpoint prefix | Export schema |
|---|---|---|
| VAE | `first_stage_model.*` | prefix removed |
| CLIP-L | `conditioner.embedders.0.transformer.*` | prefix removed; `position_ids` skipped |
| CLIP-G | `conditioner.embedders.1.model.*` | OpenCLIP keys mapped to Comfy/HF keys |

The CLIP-G export splits packed `attn.in_proj_weight` / `attn.in_proj_bias`
into separate Q/K/V tensors and transposes `text_projection` so the output
matches the `clip_g.safetensors` key layout used by ComfyUI.

The analyzer compares exportable embedded tensors with local references under
the selected ComfyUI `models` root:

- `models/vae/sdxlVAE.safetensors`
- `models/clip/clip_l.safetensors`
- `models/clip/clip_g.safetensors`

## Safetensors Output Pipeline

`convert_safetensors.py` is a sibling to the GGUF writer above: it reuses the
same `load_state_dict()` / `detect_arch()` machinery, but writes a plain
quantized `.safetensors` file instead of a GGUF container, so ComfyUI can load
the result natively (no GGUF loader node needed).

```
Source file (.safetensors / .ckpt / .pt)
    │
    ▼
load_state_dict() + detect_arch()      (shared with the GGUF pipeline above)
    │
    ▼
convert_to_safetensors() in convert_safetensors.py
    │  - _scan_quantized_layers(): pre-scan for already-quantized .weight
    │      tensors (dequantize.py's detect_quantized_weight(), preferring the
    │      per-layer .comfy_quant JSON sidecar, falling back to a
    │      dtype+weight_scale heuristic) — Automatic Dequantization, mirrors
    │      Starnodes ComfyUI Model Converter. Each detected layer is
    │      reconstructed to float32 (dequantize_weight()) before the normal
    │      pipeline below runs, instead of refusing outright. Only an
    │      int8/uint8 weight with NO recognizable scale sidecar still raises
    │      (unrecoverable — no scale to reconstruct magnitude from)
    │  - Pass 1 iterates the shared _iter_output_keys() order and uses
    │      plan_tensor_output() plus shape/dtype metadata to build the
    │      safetensors header and _quantization_metadata without loading
    │      tensor data
    │  - Filters keys_ignore
    │  - Coerces float8 source tensors → float16 before nan_to_num
    │      (nan_to_num has no float8 kernel)
    │  - Clamps inf/NaN → 0 / ±65504
    │  - No 5D side-car export, no shape_fix reshape, no 1D-pad-token
    │      unsqueeze — those exist only to satisfy GGUF/llama-quantize
    │      constraints that a plain safetensors file doesn't have
    │
    ▼
quantize_tensor_st() in safetensors_quant.py       (per tensor)
    │  - target_key one of F16 / F16_MIXED / FP8 / FP8_MIXED / INT8 / INT8_MIXED
    │      (GUI-selectable; NVFP4/NVFP4_MIXED still exist and are still
    │      correct, but are not offered — see "Why INT8/FP8, not NVFP4/MXFP8"
    │      below)
    │  - *_MIXED: keys_hiprec / 1D / ≤1024-elem tensors stay F32 — mirrors
    │      convert.py's _quant_type_for exactly, one mixed-precision rule
    │      shared by both output pipelines, not reinvented per format
    │  - INT8 (both mixed and non-mixed): 1D tensors (biases, norm weights)
    │      always stay F32 regardless of the mixed flag — a per-layer
    │      weight_scale on a 1D tensor has no consumer in ComfyUI's
    │      scaled-quant loader and would silently load back wrong
    │  - INT8 → safetensors_quant_int8.quantize_int8_convrot() when the
    │      tensor is 2D and in_features % 256 == 0: Hadamard-rotates the
    │      weight in 256-wide blocks, then quantizes row-wise (int8 +
    │      per-row float32 weight_scale); otherwise
    │      quantize_int8_tensorwise(): plain int8 + a single scalar
    │      weight_scale (ComfyUI int8_tensorwise convention)
    │
    ▼
Streaming safetensors writer with _quantization_metadata
    │  - Pass 2 reuses the exact same _iter_output_keys() order, runs the
    │      real quantize_tensor_st() per tensor, asserts each emitted tensor
    │      matches Pass 1's planned name/dtype/shape, then appends its raw
    │      bytes directly after the already-written header. Peak output RAM
    │      scales with one tensor instead of the full quantized model.
    │  - Per-layer {"format": "int8_tensorwise", "convrot": true,
    │      "convrot_groupsize": 256} JSON (convrot fields only present when
    │      that tensor actually takes the ConvRot path), written only for
    │      layers that actually produce scale-tensor siblings
    │  - FP8/FP8_MIXED layers additionally get "full_precision_matrix_mult":
    │      true by default (full_precision_fp8=True param on
    │      convert_to_safetensors()) — makes ComfyUI's
    │      MixedPrecisionOps.Linear.forward() (comfy/ops.py) dequantize and
    │      run a plain full-precision matmul instead of the FP8
    │      quantized-compute path, for every layer unconditionally. See
    │      "Why INT8/FP8, not NVFP4/MXFP8" below and docs/issues_analysis.md #16
    │
    ▼
Output file (.safetensors)
```

Format reference (verified against `Comfy-Org/ComfyUI`'s `comfy/quant_ops.py`
`QUANT_ALGOS` registry and `Comfy-Org/comfy-kitchen`'s
`tensor/int8.py`/`tensor/int8_utils.py` source, so ComfyUI can actually load
the output): `int8_tensorwise` uses a plain per-layer `weight_scale` float32
scalar (or, with ConvRot, a per-row scale plus the block-Hadamard weight
rotation `comfy_kitchen`'s kernel un-rotates on load), matching ComfyUI's
native `TensorWiseINT8Layout` loader.

**Why INT8/FP8, not NVFP4/MXFP8:** `safetensors_quant_fp8.py`/`safetensors_quant_nvfp4.py`
implement ComfyUI's scaled-FP8/NVFP4 conventions correctly (byte-verified
against ComfyUI's own kernels, `docs/issues_analysis.md` #10-#13). ComfyUI's
`QUANT_ALGOS["float8_e4m3fn"]`/`["nvfp4"]`/`["mxfp8"]` all leave
`"quantize_input"` at its `True` default, so ComfyUI dynamically quantizes
*activations* too at inference time — which produced visibly wrong output
(black bars, mirrored/wrong poses, wrong identities, full-image noise) on
Lumina2/Z-Image checkpoints even after this tool's on-disk weight data was
verified byte-correct. Activation quantization is inherently lossy in a way
no weight-side protection list can compensate for. `int8_tensorwise` is one
of only two `QUANT_ALGOS` entries marked `"quantize_input": False` —
weight-only quantization, avoiding that path entirely (much closer to
GGUF's dequantize-then-standard-matmul robustness). See
`docs/issues_analysis.md` #15, including its Correction note — an earlier
draft misattributed the corruption to a specific ComfyUI GitHub issue
(Comfy-Org/ComfyUI#14595) that turned out to be a performance-only bug (silent
bf16 fallback for some GEMM shapes, mathematically correct just slower); that
citation has been retracted, the INT8 decision itself is unaffected.

FP8 is no longer excluded alongside NVFP4/MXFP8, though: `comfy/utils.py`'s
`convert_old_quants()` — how ComfyUI loads the scaled-FP8 checkpoints already
common on Civitai/HuggingFace — sets `"full_precision_matrix_mult": true` for
those legacy checkpoints, which makes `MixedPrecisionOps.Linear.forward()`
skip the quantized-compute branch and run a plain full-precision matmul
instead, architecture-independently. `convert_to_safetensors()` now writes
that same flag by default (see the pipeline diagram above), making this
tool's own FP8 output as safe as the checkpoints already circulating in the
wild — no per-architecture `keys_hiprec` bet the way INT8 needs. NVFP4/MXFP8
have no known equivalent safe-mode flag; that investigation is explicitly
**not done** — tracked as an open follow-up in `docs/issues_analysis.md` #16
and `model_support.py`'s `support_level()` docstring — so they stay excluded
until it is. Full FP8 discovery/fix writeup: `docs/issues_analysis.md` #16.

## Model Support Tab

`model_support.py` and `gui.py` split the "Model Support" GUI tab the same
way the rest of this codebase splits data from presentation: `model_support.py`
is the data model, `gui.py` is rendering + interaction.

`model_support.py` owns:
- `MODEL_DISPLAY_NAMES`: public model name per internal `models.architectures.*.arch`
  key (e.g. `"lumina2" → "Z-Image / Lumina-Image 2.0 Family (lumina2)"`) — an
  editorial mapping, documented inline as such, not something ComfyUI or any
  upstream project defines.
- `TABLE_FORMATS`: the ordered (label, format-key) columns the table renders —
  GGUF first (collapses every K-quant level into one column, since K-quants
  are a uniform post-processing step on top of a working F16 GGUF conversion,
  not an architecture-dependent choice), then every GUI-selectable
  safetensors format, then NVFP4 (implemented but not GUI-offered).
- `support_level(arch_key, keys_hiprec_nonempty, format_key)`: the pure
  function computing one of `SUPPORT_VERIFIED`/`SUPPORT_CAUTION`/`SUPPORT_UNKNOWN`
  for one (architecture, format) pair, from facts already in
  `models/architectures.py` (`keys_hiprec` presence) plus this project's own
  render-testing history (`_RENDER_VERIFIED_ARCHES`, currently just `lumina2`).
  Its docstring is the canonical explanation of every branch's reasoning —
  see the module itself rather than duplicating it here.
- `build_support_table()`: one row per `models.architectures.arch_list` entry,
  ready for the GUI to render directly without re-deriving per-architecture
  sensitivity itself.

`gui.py` owns everything about turning that data into an interactive table
and reusing it elsewhere in the UI:
- Renders `build_support_table()` as a read-only `gr.Dataframe(datatype="html")`
  (`_support_table_rows_for_dataframe()`/`_support_table_cell_html()`), with
  colored ✓/⚠/✗/? cells reusing the app's own `--s2g-accent`/`--s2g-warn`/
  `--s2g-danger`/`--s2g-muted` CSS tokens, plus a legend
  (`_SUPPORT_TABLE_LEGEND_HTML`) explaining the four-state scheme (✗ "Known
  issue" split out from ⚠ "Caution" so "never tested" and "tested and
  confirmed wrong" read as distinct signals).
- `apply_support_table_selection(evt)`: the table's `.select()` handler —
  switches `main_tabs` to the matching Convert tab and pre-selects that
  format in its dropdown. Uses `gr.update(selected=...)` rather than the
  `gr.Tabs(selected=...)` constructor call the original plan for this task
  specified, due to a Gradio version mismatch in this project's pinned
  dependency — functionally equivalent, just the update-object form instead
  of a fresh component. The GGUF column always resolves to `Q4_K_M` (this
  tool's own "recommended ★" default), since a single cell can't express a
  specific K-quant level; clicking the Model-name column is a no-op.
- `annotate_safetensors_choices()`/`annotate_gguf_choices()`: reuse
  `support_level()` (via the shared `_annotate_choices_for_arch()`) to prefix
  a ⚠ or ✗ onto any dropdown entry that's `SUPPORT_CAUTION`/`SUPPORT_BAD` for
  the detected source checkpoint's architecture, wired as `.change()`
  handlers on the source-path inputs. Purely informational — every entry
  stays selectable.

**Text Encoder Support table:** a second support table below the main one,
one row per vendored text-encoder family (`TEXT_ENCODER_FAMILY_DISPLAY_NAMES`
in `model_support.py`) rather than one row per `models.architectures.arch_list`
entry — text encoders aren't in that list and have no `keys_hiprec`-style risk
model (see the Text-Encoder Conversion Pipeline section below), but
`detect_text_encoder_family()` (see that section) makes per-family
identification possible the same way `detect_arch()` does for diffusion
models, so this table now mirrors `build_support_table()`'s per-row design
instead of collapsing into one generic row. `model_support.py` defines
`TEXT_ENCODER_TABLE_FORMATS` (GGUF — collapsing every direct outtype/K-quant,
see below — plus FP8/FP8_MIXED/NVFP4/NVFP4_MIXED),
`text_encoder_support_level(family, format_key)` (three-state: `SUPPORT_BAD`
only for a `_TE_RENDER_CONFIRMED_BAD` pair — actually render-tested and
confirmed to produce broken/garbage output, not merely "differs from the
unquantized baseline"; `SUPPORT_VERIFIED` for a `_TE_RENDER_VERIFIED` pair;
`SUPPORT_CAUTION` otherwise, i.e. untested), and
`build_text_encoder_support_table()`. As of 2026-08-12, `qwen3-4b` is
`SUPPORT_VERIFIED` on every column (GGUF direct outtypes/K-quants and all 4
safetensors formats convert+load+render-tested in ComfyUI, no format-specific
defect found across 4 seeds each); every other vendored family is
`SUPPORT_CAUTION` (untested). The GGUF column stays collapsed rather than
splitting direct-outtype vs. K-quant: testing found no defect distinguishing
them for `qwen3-4b` (F16/Q8_0/Q6_K all showed the same rate of an unrelated,
seed-driven base-model prompt-adherence artifact — see the "Corrected" entry
in CHANGELOG.md for what that artifact turned out to be and why it isn't
evidence against any format).
`gui._text_encoder_support_rows_for_dataframe()` renders it as an N-row
`gr.Dataframe`, reusing `_support_table_cell_html()`.
`apply_text_encoder_support_table_selection(evt)` mirrors
`apply_support_table_selection()` but switches to the Convert Text Encoder
tab (`main_tabs` id 2) and, unlike the diffusion-model table, never no-ops
on NVFP4 — it's a real `TEXT_ENCODER_FORMAT_CHOICES` entry, not excluded the
way it is for diffusion-model output. Both support tables share one traffic-
light color legend (`_SUPPORT_TABLE_LEGEND_HTML`): literal red/yellow/green
CSS custom properties (`--s2g-support-good/-caution/-bad`) rather than the
app's teal `--s2g-accent` branding color, kept separate so this doesn't
recolor the rest of the UI.

## Text-Encoder Conversion Pipeline

`text_encoder_convert.py` converts a bare single-file HF/Transformers-style
text-encoder checkpoint (Qwen3, T5/UMT5, Mistral, …) to GGUF (direct outtypes
or K-quants) or to a quantized safetensors file (FP8/NVFP4). `convert_text_encoder_any()`
dispatches on the selected `TEXT_ENCODER_FORMAT_CHOICES` key to one of three
backends:

- **Direct GGUF outtypes** (F32/F16/BF16/Q8_0): `convert_text_encoder()`, the
  original subprocess pipeline below — no DiT architecture-detection involved.
- **K-quants** (Q6_K…Q2_K): `convert_text_encoder_kquant()` runs the F16 GGUF
  pipeline to a temp intermediate, then a second llama-quantize pass (see below).
- **Safetensors** (FP8/FP8_MIXED/NVFP4/NVFP4_MIXED): `convert_text_encoder_to_safetensors()`
  reuses `convert_safetensors.convert_to_safetensors()` with
  `_TEXT_ENCODER_MODEL_ARCH`, a `models.architectures.ModelTemplate()`
  instance (text encoders aren't in `arch_list`) — the only place this
  module imports `models.architectures` or `convert.py`. Also passes
  `strip_prefixes=False`: `load_state_dict()`'s default prefix-stripping
  rule assumes a leading "model." wraps a diffusion UNet inside a larger
  checkpoint, but a standalone text encoder's "model." is its own genuine
  HF module path (e.g. Qwen3's `model.layers.0....`) — stripping it broke
  ComfyUI's own text-encoder architecture detection (`comfy/sd.py`'s
  `detect_te_model()`) on the output file. `_TEXT_ENCODER_MODEL_ARCH` also
  sets `keys_shape_critical = ["embed_tokens", "shared", "token_embedding",
  "wte", "lm_head"]` — NVFP4 halves a tensor's on-disk last dimension,
  which corrupts an `nn.Embedding` table loaded via a plain (non-dequant)
  `load_state_dict` the same way it corrupts a diffusion model's raw
  hyperparameter-inference tensors (`safetensors_quant.py`'s existing
  `keys_shape_critical` guard, previously only ever populated for
  diffusion-model architectures).

```
Source weights (.safetensors, bare file — no config.json/tokenizer)
  + Base model HF repo ID (manual field, e.g. "Qwen/Qwen3-8B")
    │
    ▼
ensure_llama_cpp() / find_convert_script()
    │  - Clones ggml-org/llama.cpp (shallow, --depth 1) into .llama.cpp/ next
    │      to this repo on first use, override via S2G_LLAMA_CPP_HOME env var
    │  - Skipped if convert_hf_to_gguf.py is already present in that dir
    │  - Raises RuntimeError if git is missing or the clone fails
    │  - convert_hf_to_gguf.py's own imports (transformers, sentencepiece,
    │      protobuf) are satisfied by this repo's own venv — declared as
    │      regular pyproject.toml dependencies, not a separate environment
    │
    ▼
If base_repo_id is blank, detect_text_encoder_family(state_dict) fingerprints
the weights first (see "Vendored Text-Encoder Configs" below) and resolves
straight to a vendored family — fetch_base_config_files() below is only
reached when base_repo_id was given explicitly.

fetch_base_config_files(repo_id, dest_dir)
    │  - If repo_id is one of _VENDORED_REPOS, copies config.json + tokenizer
    │      files from text_encoder_configs/<name>/ instead — no network call
    │  - Otherwise downloads config.json (mandatory, RuntimeError if missing)
    │      and best-effort tokenizer files (tokenizer.json, tokenizer_config.json,
    │      tokenizer.model, special_tokens_map.json, tekken.json, spiece.model)
    │      from HuggingFace Hub via huggingface_hub.hf_hub_download
    │
    ▼
convert_text_encoder() assembles a temp directory
    │  - Copies (not moves) the source weights to <tmpdir>/model.safetensors
    │      so convert_hf_to_gguf.py's HF-layout auto-discovery finds them,
    │      and the user's original file is left untouched
    │  - Runs convert_hf_to_gguf.py as a subprocess of sys.executable — this
    │      tool's own interpreter, no external ComfyUI Python needed
    │  - Streams subprocess stdout to on_log line by line; polls
    │      cancel_event, terminates the subprocess and raises
    │      RuntimeError("cancelled") if set
    │  - Raises RuntimeError on non-zero subprocess exit
    │
    ▼
Output file (.gguf)
```

**K-quant second pass** (`convert_text_encoder_kquant()`): after the F16 GGUF
above, `ensure_plain_llama_quantize()` builds a **plain, unpatched**
llama-quantize once via `cmake -B .llama.cpp/build-quantize` from the same
auto-cloned checkout (needs `cmake` + a C++ compiler; cached after the first
build) and `quantize.run_quantize()` (shared with the diffusion-model
pipeline) runs it against the F16 intermediate. This binary is intentionally
separate from the City96-patched `llama-quantize` used for diffusion-model
GGUFs (see [building-llama-quantize.md](building-llama-quantize.md)) — that
patch is documented as unsafe for LLM/text GGUFs.

**Safetensors branch** (`convert_text_encoder_to_safetensors()`): loads the
checkpoint via `convert.load_state_dict()` and quantizes it with the same
`safetensors_quant_fp8.py`/`safetensors_quant_nvfp4.py` backends used for
diffusion models, but with a generic `ModelTemplate()` (empty `keys_hiprec`/
`keys_shape_critical`) instead of a detected architecture. This is safe
because ComfyUI's text-encoder loaders (`comfy/text_encoders/*.py`) build
models from fixed config presets keyed by filename/CLIPType, not by inferring
hyperparameters from checkpoint tensor shapes — so there's no analogue to the
DiT shape-corruption risk documented in
[issues_analysis.md](issues_analysis.md) #9, and no per-architecture audit is
needed here. No HuggingFace download or llama.cpp involved.

`convert_hf_to_gguf.py` in current llama.cpp imports its per-architecture
model classes from a sibling `conversion/` package (~90 files, one per
architecture family) rather than being a single self-contained script —
vendoring it would mean tracking that whole package. A shallow git clone
avoids that maintenance burden and stays current with upstream automatically.

The base-model repo ID is always a manual text field — there is no
auto-detection heuristic mapping a bare checkpoint to its base model's HF
repo. Per-family automation (SDXL CLIP key mapping, Qwen multimodal `mmproj`
pairing) is explicitly **not** implemented; the generic `convert_hf_to_gguf.py`
path handles whatever base architectures llama.cpp's converter itself
supports, and unsupported/non-standard architectures require manual key
mapping outside this tool.

### Future Work: Per-Family Text-Encoder Automation

The generic HF-to-GGUF path above is what's implemented today. The following
remains future work, not yet automated by this tool: SDXL CLIP-L and CLIP-G
remain special cases because they require checkpoint-specific extraction and
key mapping before they can be considered for GGUF conversion — OpenAI's
`clip-vit-large-patch14` is the likely standard CLIP-L reference for this
branch of future work, handled as a CLIP-specific conversion case rather than
assuming the same generic path as Qwen/T5-style text encoders.

### Planned Text Encoder Families

| Family | Models | Diffusers / config class | Converter work |
|---|---|---|---|
| CLIP | SDXL CLIP-L, SDXL CLIP-G | `CLIPTextModel`, `CLIPTextModelWithProjection` | Extract or load CLIP-only weights, map keys, preserve projection tensors |
| T5 | FLUX.1, FLUX.1 Kontext | `T5EncoderModel` | HF-to-GGUF path for T5-style encoder weights and tokenizer |
| Qwen2.5-VL 7B | Qwen-Image, Qwen-Image-Edit | `Qwen2_5_VLForConditionalGeneration` / ComfyUI `qwen_2.5_vl_7b_fp8_scaled.safetensors` | Multimodal Qwen-VL path; preserve paired mmproj handling for GGUF |
| Qwen3 4B | Z-Image, Z-Image-Turbo, FLUX.2 [klein] 4B | `Qwen3Model`, `Qwen3ForCausalLM` | HF-to-GGUF path; preserve Qwen tokenizer/chat-template assumptions |
| Qwen3 8B | FLUX.2 [klein] 9B | Qwen3 8B text encoder files in ComfyUI layout | Same Qwen path, but keep model-pair compatibility checks |
| Mistral / Ministral3 | ERNIE-Image, ERNIE-Image-Turbo, FLUX.2 [dev] | `Mistral3Model`, `Ministral3ForCausalLM`, Mistral Small 3.2 24B | Separate Mistral-family detection and conversion path |

Compatibility rules that the implementation should enforce:

- Do not quantize `.safetensors` text encoders directly.  Convert to GGUF first,
  then run a compatible `llama-quantize`.
- Existing `.gguf` text encoders can be re-quantized directly, but should write
  new output files and avoid overwriting sources by default.
- Qwen-Image / Qwen-Image-Edit use `qwen_2.5_vl_7b_fp8_scaled.safetensors` in
  ComfyUI's text-encoder path; a GGUF variant also needs matching multimodal
  projection (`mmproj`) handling, not only the main language weights.
- FLUX.2 [klein] 4B and 9B are not interchangeable: the 4B model uses a Qwen3
  4B text encoder; the 9B model uses a Qwen3 8B text encoder.
- ERNIE-Image needs both the main `Mistral3Model` text encoder and the
  `Ministral3ForCausalLM` prompt-enhancer path represented in the analysis UI.
- CLIP-L / CLIP-G should remain separate from Qwen/T5/Mistral handling because
  SDXL checkpoints may embed modified CLIP weights that need extraction,
  comparison, and key remapping before conversion.

### Vendored Text-Encoder Configs

`text_encoder_configs/<name>/` vendors `config.json` + tokenizer files for the
base repos above, so `fetch_base_config_files()` can skip HuggingFace Hub
entirely for known repo IDs (see `_VENDORED_REPOS` in `text_encoder_convert.py`).
Vendored so far: `clip-l` (`openai/clip-vit-large-patch14`), `clip-bigg`
(`laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`), `t5-xxl` (`google/t5-v1_1-xxl`),
`qwen3-4b` (`Qwen/Qwen3-4B`), `qwen3-8b` (`Qwen/Qwen3-8B`), `qwen2.5-vl-7b`
(`Qwen/Qwen2.5-VL-7B-Instruct`), `mistral-small-3.2-24b`
(`mistralai/Mistral-Small-3.2-24B-Instruct-2506`), and `ernie-image-pe` — the
`Ministral3ForCausalLM` prompt-enhancer, pulled from the `pe/` subfolder of
`baidu/ERNIE-Image` (not from `_VENDORED_REPOS`/auto-selected yet, since that
repo ID is ambiguous between its `pe/` and `text_encoder/` subfolders).

Any repo ID not in `_VENDORED_REPOS` still falls through to the live
HuggingFace download — vendoring is an optimization/availability hedge, not a
replacement for the download path.

### Base-Model Family Auto-Detection

`detect_text_encoder_family(state_dict)` in `text_encoder_convert.py` lets
`convert_text_encoder()` skip the manual "Base model HF repo ID" field
entirely for the 8 vendored families, the same way diffusion-model conversion
identifies its architecture from the source file alone (`detect_arch()` in
`models/architectures.py`) rather than asking the user.

Key-name matching (`detect_arch()`'s approach) doesn't work here: Qwen3,
Mistral, and Ministral3 are all Llama-style decoders with near-identical key
names (`model.layers.N.self_attn.q_proj.weight`, …) regardless of model size.
Instead, `detect_text_encoder_family()` reads `(hidden_size, num_hidden_layers,
vocab_size)` straight off the checkpoint's own embedding tensor shape (`*embed_tokens.weight`,
`*token_embedding.weight`, or `shared.weight`, matched by suffix to cover
Llama-style/CLIP/T5 naming) and layer count (highest `layers.N.`/`block.N.`
index + 1), then looks that triple up in `_FAMILY_SIGNATURES`:

| Signature (hidden, layers, vocab) | Family |
|---|---|
| (768, 12, 49408) | clip-l |
| (1280, 32, 49408) | clip-bigg |
| (4096, 24, 32128) | t5-xxl |
| (2560, 36, 151936) | qwen3-4b |
| (4096, 36, 151936) | qwen3-8b |
| (3584, 28, 152064) | qwen2.5-vl-7b |
| (5120, 40, 131072) | mistral-small-3.2-24b |
| (3072, 26, 131072) | ernie-image-pe |

These triples were verified unique against each family's real `config.json`
(source of the numbers above), and in particular resolve the Ministral3
(ERNIE-Image prompt-enhancer) vs. Mistral-Small-3.2-24B ambiguity that made
`baidu/ERNIE-Image` unsuitable as a `_VENDORED_REPOS` key — the two are
distinguishable by shape even though their key names alone are not.

For a `_LazyStateDict` (the safetensors loading path), shape is read via its
`shape_of()` accessor straight from the safetensors header — no tensor data
is materialized just to detect the family, consistent with the streaming
design documented for `_LazyStateDict` elsewhere in this file.

If no signature matches, `convert_text_encoder()` raises `RuntimeError`
telling the user to supply the base repo ID manually — auto-detection only
covers this tool's documented candidate families, not arbitrary base models.

### Estimated Output Size (all three Convert tabs)

Every Convert tab (GGUF, Safetensors, Text Encoder) shows an "Estimated
output" line under its format dropdown, but each uses a different mechanism
because each output backend's size depends on different inputs:

- **GGUF tab**: `quantize.estimate_output_size()` — parses the safetensors
  header (no tensor data loaded) and replicates `convert.py`'s own
  unconditional "1D or ≤`QUANTIZATION_THRESHOLD`-element tensor → always F32"
  rule, counting F32-forced and quantizable tensors separately per
  `_QUANT_BYTES_PER_ELEM`. Falls back to `SIZE_RATIOS[quant_key] × source
  size` for non-safetensors sources.
- **Safetensors tab**: `safetensors_quant.estimate_safetensors_output_size()`
  — a header-only estimator that instead replicates `quantize_tensor_st()`'s
  branching exactly: `is_hiprec_st`'s 1D/small/`keys_hiprec` gate (only
  relevant in `mixed` mode — this is *why* GGUF's simpler rule can't be
  reused here, the `*_MIXED` formats' size genuinely depends on the detected
  architecture's `keys_hiprec`), `keys_shape_critical` F16 fallback, NVFP4's
  block-scale swizzle-padding (`_swizzle_block_scale` pads to 128×4 tile
  alignment for 2D weights — can matter a lot for < 128-row tensors), and
  INT8 ConvRot's one-F32-scale-per-output-row overhead. Cross-checked
  byte-for-byte against real `quantize_tensor_st()` output in
  `tests/test_safetensors_quant.py::TestEstimateSafetensorsOutputSize`.
  `gui.py` supplies the architecture via the same `_detected_arch_or_none()`
  detection the dropdown ⚠/✗ annotations already use.
- **Text Encoder tab**: FP8/FP8_MIXED/NVFP4/NVFP4_MIXED reuse the Safetensors
  tab's estimator with the fixed `_TEXT_ENCODER_MODEL_ARCH` (no per-checkpoint
  detection for text encoders). GGUF-family formats (F32/F16/BF16/Q8_0/
  K-quants) deliberately use the plain `SIZE_RATIOS` ratio rather than
  `estimate_output_size()`'s per-tensor analysis: that analysis's F32-forcing
  heuristic was calibrated for `convert.py`'s own GGUF writer (diffusion
  models), not llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize` (what
  text-encoder GGUF output actually goes through) — `SIZE_RATIOS`' own
  reference data (Llama-3-8B via `llama-quantize`) is the better-matched
  source for an LLM-style text encoder regardless.

All three share one `_fmt_size()`/percentage-line formatting helper in
`gui.py` for display consistency, but the size-estimate KEY point is: **a
static percentage label is only trustworthy for a fixed-ratio format** (a
pure per-element cast — F16, FP8, INT8, NVFP4). `*_MIXED` variants have no
single fixed ratio (it depends on how much of a given checkpoint matches
`keys_hiprec`), which is why they're estimated live from the actual header
rather than given a static `% smaller` label in any dropdown's display text.

Primary model references:

- Stability AI SDXL Base model card:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>
- Stability AI SDXL `model_index.json` pipeline component map:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/model_index.json>
- Stability AI SDXL `text_encoder_2` config:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/text_encoder_2/config.json>
- `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` model card:
  <https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k>
- `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` file tree:
  <https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/tree/main>
- `openai/clip-vit-large-patch14` model card:
  <https://huggingface.co/openai/clip-vit-large-patch14>
- `openai/clip-vit-large-patch14` file tree:
  <https://huggingface.co/openai/clip-vit-large-patch14/tree/main>
- `Qwen/Qwen-Image` model card:
  <https://huggingface.co/Qwen/Qwen-Image>
- `Qwen/Qwen-Image-Edit` model card:
  <https://huggingface.co/Qwen/Qwen-Image-Edit>
- `Qwen/Qwen2.5-VL-7B-Instruct` model card:
  <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>
- ComfyUI Qwen-Image setup guide:
  <https://docs.comfy.org/tutorials/image/qwen/qwen-image>
- ComfyUI Qwen-Image-Edit setup guide:
  <https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit>
- `Tongyi-MAI/Z-Image-Turbo` model card:
  <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>
- `Tongyi-MAI/Z-Image-Turbo` `model_index.json`:
  <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/model_index.json>
- `Tongyi-MAI/Z-Image-Turbo` text encoder config:
  <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/text_encoder/config.json>
- Black Forest Labs FLUX.1 Kontext model card:
  <https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev>
- Black Forest Labs FLUX.2 inference repository:
  <https://github.com/black-forest-labs/flux2>
- ComfyUI FLUX.2 [klein] setup guide:
  <https://docs.comfy.org/tutorials/flux/flux-2-klein>
- `baidu/ERNIE-Image-Turbo` model card:
  <https://huggingface.co/baidu/ERNIE-Image-Turbo>
- `baidu/ERNIE-Image-Turbo` `model_index.json`:
  <https://huggingface.co/baidu/ERNIE-Image-Turbo/blob/main/model_index.json>
- ComfyUI ERNIE-Image setup guide:
  <https://docs.comfy.org/tutorials/image/ernie-image/ernie-image>

Candidate non-CLIP text encoder reference:

- `huihui-ai/Huihui-Qwen3-4B-abliterated-v2` model card:
  <https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-abliterated-v2>
- `huihui-ai/Huihui-Qwen3-4B-abliterated-v2` file tree:
  <https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-abliterated-v2/tree/main>

Tooling references:

- llama.cpp `convert_hf_to_gguf.py`:
  <https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py>
- ComfyUI-GGUF:
  <https://github.com/city96/ComfyUI-GGUF>

## City96 llama.cpp Patch Reference

`city96/ComfyUI-GGUF/tools/lcpp.patch` is a primary implementation reference for
image GGUF quantization support.  It should not be treated as just a historical
build helper.  The patch describes the bridge between ComfyUI image-model GGUFs
and `llama.cpp` / `llama-quantize`.

Relevant implementation knowledge in the patch:

- Image architecture registration: adds diffusion architectures to the
  llama.cpp architecture registry so GGUF files with non-LLM `general.architecture`
  values can be recognized.
- Image tensor quantization policy: defines model-aware tensor classification
  so large weights can be K-quantized while norms, embeddings, positional
  tensors, scalar tensors, and sensitive special tensors can remain higher
  precision.
- Metadata bypasses: avoids failing image GGUFs on LLM-only expectations such
  as vocabulary, tokenizer, rope, or attention metadata.
- Shape compatibility: includes targeted handling for image-model tensors whose
  stored shape or name length would otherwise fail llama.cpp validation.

For current llama.cpp, expect a forward-port rather than a clean direct apply.
The old patch targets older files such as `src/llama.cpp`; modern llama.cpp
splits related logic across files such as `llama-arch.*`, `llama-model.*`, and
`llama-quant.*`.  Future work should extract the architecture list and tensor
rules from City96's patch and map them onto the current source layout.

Patch references:

- ComfyUI-GGUF tools guide:
  <https://github.com/city96/ComfyUI-GGUF/tree/main/tools>
- City96 `lcpp.patch`:
  <https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/tools/lcpp.patch>
- Current llama.cpp source layout:
  <https://github.com/ggml-org/llama.cpp/tree/master/src>

## Future Extension: Checkpoint GGUF Workflows

The current conversion pipeline treats one source file as one GGUF target.  A
future checkpoint-level workflow would treat a checkpoint as a component graph:

```
checkpoint
  -> analyze component prefixes and architectures
  -> split into diffusion model / VAE / text encoders / auxiliary components
  -> choose conversion and quantization per component
  -> write a mixed output layout or manifest
  -> validate with a ComfyUI loader path
```

Two product shapes are possible:

1. **Checkpoint decomposer / repacker**
   - Reads a monolithic checkpoint and extracts known components.
   - Presents component-specific conversion options.
   - Converts supported components to GGUF, leaves unsupported or loss-sensitive
     components as safetensors.
   - Writes a manifest or bundle that records which file backs each component.

2. **Mixed checkpoint loader node**
   - Loads a checkpoint-like manifest or folder layout.
   - Dispatches `.gguf` components through GGUF loaders and `.safetensors`
     components through native ComfyUI loaders.
   - Avoids forcing GGUF tensors back into a safetensors checkpoint container.

The loader-node approach is likely the lower-risk release target.  Repacking a
single physical checkpoint would need a stable container format for mixed
GGUF/safetensors payloads, plus loader support for per-component metadata.
Until that exists, the safer abstraction is a folder/manifest layout that keeps
the original file formats intact.

Implementation requirements:

- Reuse `component_extract.py` analysis for SDXL VAE / CLIP detection.
- Add architecture-aware component classification for diffusion models and
  text encoders before exposing quantization choices.
- Keep quantization choices component-scoped; for example, Q4 diffusion model,
  Q6 text encoder, untouched VAE.
- Never overwrite the source checkpoint or source component files by default.
- Include compatibility checks for model-pair constraints, especially FLUX.2
  [klein] 4B vs 9B text encoder pairing and ERNIE prompt-enhancer handling.
- Validate output through an actual ComfyUI-compatible loader path, not only by
  checking file existence.

## Quantization Decision Tree

```
Tensor
 ├─ in keys_ignore?          → skip
 ├─ dtype coercion           float8 → F16, BF16 → F32
 ├─ inf/NaN clamping         nan→0, ±inf→±65504
 ├─ key in keys_unsqueeze?   unsqueeze(0): [D] → [1, D]
 ├─ ndim > 4?                → offload (5D fix)
 ├─ ndim == 1?               → F32
 ├─ n_params ≤ 1024?         → F32
 ├─ key in keys_hiprec?      → F32
 ├─ shape_fix applicable?    → reshape + write orig_shape metadata
 └─ source dtype?
      BF16 → BF16
      otherwise → F16
```

## Architecture Coverage Verification

Three architectures require no new code because their tensor-key layout matches existing families:

**Flux.2 (klein / Dev)**: Detected as `arch="flux"` — tensor names and structure are identical to Flux.1. Verified against `snofsSexNudesAndOtherFunStuff_distilledV12Fp8.safetensors` from the planning checkpoint set. ComfyUI-GGUF also reuses `arch="flux"` for Flux.2 variants, confirming this is correct classification, not a false positive.

**Z-Image (Turbo / Base)**: Detected as `arch="lumina2"` — shares the identical NextDiT tensor-key layout with Lumina2 (e.g., `cap_embedder`, `context_refiner`, `noise_refiner`, `x_pad_token`, `cap_pad_token`). Verified against `jibMixZIT_v10.safetensors`. Coverage is validated by the existing `test_lumina2` test in `tests/test_convert.py`.

**Qwen-Image / Qwen-Image-Edit (incl. 2511)**: Covered by the existing `ModelQwenImage` class (introduced in commit `400f861`). The 2511 revision checkpoint could not be re-verified against a raw `.safetensors` source due to availability constraints; only a pre-quantized GGUF was available during planning. Future maintainers should re-verify against a raw safetensors source when one becomes available, though detection correctness is not blocked on this.
