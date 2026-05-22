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

## Future Extension: Text Encoder to GGUF

Text Encoder conversion is a separate future track from SDXL component
extraction.  For HF/Transformers-style text encoders such as Qwen or T5, the
likely pipeline is:

```
HF repo / local text encoder folder
  -> validate config, tokenizer, and safetensors shards
  -> llama.cpp convert_hf_to_gguf.py
  -> compatible llama-quantize binary
  -> ComfyUI models/text_encoders output
```

The tool should keep this path source-driven rather than trying to quantize
`.safetensors` files directly.  Local `.gguf` files can already be re-quantized;
local `.safetensors` text encoders need a conversion step first.  SDXL CLIP-L
and CLIP-G remain special cases because they require checkpoint-specific
extraction and key mapping before they can be considered for GGUF conversion.
OpenAI's `clip-vit-large-patch14` is the likely standard CLIP-L reference for
this branch of future work; it should be handled as a CLIP-specific conversion
case rather than assuming the same path as Qwen/T5-style text encoders.

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
