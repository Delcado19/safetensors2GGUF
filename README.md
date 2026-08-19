![safetensors2GGUF](assets/safetensors2GGUF.jpg)
# safetensors2GGUF

Converts Safetensors / CKPT diffusion model checkpoints to **GGUF** for use with
**llama.cpp** and **ComfyUI-GGUF**, or to a quantized **Safetensors** file that
ComfyUI can load natively without the GGUF loader node. Also converts bare
single-file **text-encoder** checkpoints (Qwen3, T5/UMT5, Mistral, …) to GGUF.

- **GGUF output** — direct Python quantization (F32 / F16 / BF16 / Q8_0) and
  K-quant quantization via a bundled `llama-quantize` binary (Q6_K, Q5_K_M,
  Q4_K_M, Q4_K_S, Q3_K_M, Q2_K).
- **Safetensors output** — F16, scaled FP8, and ComfyUI-compatible tensor-wise
  INT8 (ConvRot-rotated where possible), each with a "mixed" variant that
  keeps critical layers at F32.
- **Model Support tab** — a read-only table showing which quantization
  formats are verified/caution/unknown for each detectable architecture;
  click a cell to jump to the matching Convert tab with that format
  pre-selected.
- **Text-encoder → GGUF** — converts bare text-encoder checkpoints that lack
  `config.json`/tokenizer files, using a base model's HuggingFace repo ID; runs
  fully standalone, auto-cloning `llama.cpp` on first use.

A Gradio web UI with file browser, quantization selector, live size estimate,
and cancel button is included for all three pipelines.

## Supported Architectures

| Architecture | Format |
|---|---|
| FLUX.1 | Diffusers |
| FLUX.2 (klein / Dev) | Diffusers (shares `flux` arch tag with Flux.1) |
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
| Z-Image (Turbo / Base) | Diffusers (shares `lumina2` arch tag) |
| Qwen-Image / Qwen-Image-Edit (incl. 2511) | Diffusers |

## Installation

```bash
git clone https://github.com/Delcado19/safetensors2GGUF.git
cd safetensors2GGUF

# Install uv (https://docs.astral.sh/uv/) then:
uv sync
```

Dependencies are maintained in `pyproject.toml` and locked in `uv.lock`.
This project does not maintain a separate `requirements.txt`.

<details>
<summary>Using pip instead</summary>

```bash
pip install gguf torch safetensors tqdm gradio huggingface_hub transformers sentencepiece protobuf
```

</details>

`uv sync` covers everything except GGUF K-quants:

- **GGUF K-quants** (`Q6_K`, `Q5_K_M`, …) need a `llama-quantize` binary —
  see [llama-quantize Sources](#llama-quantize-sources) below. F32/F16/BF16/Q8_0,
  the Safetensors output mode, and text-encoder conversion all work with
  `uv sync` alone.
- **Text-encoder → GGUF conversion** needs `git` on `PATH`: it auto-clones
  `llama.cpp` (for its `convert_hf_to_gguf.py`) into `.llama.cpp/` on first
  use — no ComfyUI installation required. See
  [Text-Encoder Conversion](#text-encoder-conversion) below.

## Web UI (recommended)

Double-click **`start_gui.bat`** — the browser opens automatically.

Or from the terminal:

```bash
uv run python gui.py
```

The UI provides:
- **File browser** to select the source model without typing paths
- **Quantization dropdown** with 10 curated levels (see table below)
- **Live status bar** with percentage embedded in text; scroll-blocked so streaming updates never hijack the viewport
- **Automatic pipeline** — K-quants trigger a 2-step convert → quantize run;
  5D tensor insertion (HunyuanVideo / Wan) is chained automatically when needed
- **SDXL component extraction** — analyze embedded VAE / CLIP-L / CLIP-G
  against local standard files, then export selected components to the ComfyUI
  `models\vae` and `models\clip` folders
- **Advanced performance controls** — pass a fixed thread count to
  `llama-quantize` and keep GUI tensor logging throttled during conversion
- **llama-quantize picker** — the path is auto-detected or selected with a
  native file dialog; manual path typing is disabled
- **Model Support tab** — a colored table of every detectable architecture x
  quantization format, marked verified / caution / unknown, with a legend.
  Click a cell to jump to the matching Convert tab with that format
  pre-selected (the GGUF column pre-selects `Q4_K_M`, since it stands for
  every K-quant level collapsed into one cell)
- **Download from HF tab** — enter a HuggingFace repo ID and a target folder,
  click Download. Multi-shard checkpoints (`model-00001-of-0000N.safetensors`,
  …) are downloaded into a throwaway temp folder, merged into one tensor dict,
  and written out as a single `<repo-name>.safetensors` file — the shards are
  deleted afterwards, so the target folder only ever ends up with the one
  merged file the Convert tabs expect as input

## Quantization Levels

| Key | Bits | Backend | Notes |
|---|---|---|---|
| `F32` | 32 | Python | Full precision, largest |
| `F16` | 16 | Python | Half precision, standard default |
| `BF16` | 16 | Python | Brain float, best for BF16 source models |
| `Q8_0` | 8 | Python | Very high quality |
| `Q6_K` | 6 | llama-quantize | Very high quality |
| `Q5_K_M` | 5 | llama-quantize | High quality |
| `Q4_K_M` | 4 | llama-quantize | **Recommended** — good quality / size balance |
| `Q4_K_S` | 4 | llama-quantize | Good quality, smaller |
| `Q3_K_M` | 3 | llama-quantize | Moderate quality |
| `Q2_K` | 2 | llama-quantize | Smallest, lowest quality |

K-quants (marked *llama-quantize*) require a City96/ComfyUI-GGUF compatible
`llama-quantize` binary.  Upstream `ggml-org/llama.cpp` release binaries are
not selected automatically because they do not include the image-model patch
required for architectures such as Lumina 2.

The path is detected automatically from:
1. `LLAMA_QUANTIZE_PATH`
2. The ComfyUI Easy-Install bundled path:
`H:\ComfyUI-Easy-Install\Add-Ons\Tools\llama.cpp\llama-quantize.exe`
3. `COMFYUI_EASY_INSTALL_HOME` plus `Add-Ons\Tools\llama.cpp\llama-quantize.exe`
4. The system `PATH`

If no compatible binary is found, use the **Browse** button in the **Advanced**
section and select the `llama-quantize` executable.  The path field is
read-only on purpose, so users do not have to type or escape Windows paths.

### llama-quantize Sources

Recommended sources:

| OS | Source |
|---|---|
| Windows | ComfyUI Easy-Install bundled `Add-Ons\Tools\llama.cpp\llama-quantize.exe` |
| macOS / Linux | Build `llama-quantize` from `city96/ComfyUI-GGUF` using `tools/lcpp.patch` |

For a self-build (required on Linux/macOS, and on Windows without Easy-Install),
see **[docs/building-llama-quantize.md](docs/building-llama-quantize.md)** for
exact prerequisites and commands per OS — including a Visual Studio path and an
MSYS2/MinGW-w64 path that doesn't require installing Visual Studio at all.

The City96 patch is also a primary implementation reference, not only a build
step.  It documents how image GGUF architectures are registered in
`llama.cpp`, how `llama-quantize` should classify image-model tensors, and which
LLM-specific metadata assumptions must be bypassed for diffusion models.

## CLI Usage

```bash
# Convert to F16 GGUF (auto-detects output name)
uv run python convert.py --src model.safetensors

# Specify output path
uv run python convert.py --src model.safetensors --dst model-F16.gguf --overwrite
```

### Options

| Option | Description |
|---|---|
| `--src` | Source file (`.safetensors`, `.ckpt`, `.pt`, `.bin`, `.pth`) |
| `--dst` | Output GGUF path — auto-generated when omitted |
| `--overwrite` | Skip confirmation if output already exists |

### Fix Pad Tokens (Lumina 2 — existing GGUFs)

If ComfyUI raises *size mismatch for x_pad_token*, the GGUF was converted before
the shape fix was introduced.  Repair it with:

```bash
uv run python fix_pad_tokens.py --src model.gguf --dst model-fixed.gguf
```

| Option | Description |
|---|---|
| `--src` | Source GGUF (1D pad tokens) |
| `--dst` | Output GGUF path |
| `--overwrite` | Skip confirmation if output exists |

New conversions are unaffected — `convert.py` stores pad tokens as `[1, D]` automatically.

### SDXL Component Extraction

The **Extract Components** tab inspects SDXL checkpoints that bundle UNet, VAE,
CLIP-L, and CLIP-G in one `.safetensors` file.  It compares embedded components
with local standard references when present:

| Component | Embedded prefix | Local reference |
|---|---|---|
| VAE | `first_stage_model.*` | `models\vae\sdxlVAE.safetensors` |
| CLIP-L | `conditioner.embedders.0.*` | `models\clip\clip_l.safetensors` |
| CLIP-G | `conditioner.embedders.1.*` | `models\clip\clip_g.safetensors` |

After analysis, select the components to export.  VAE files are written to
`models\vae`; CLIP-L and CLIP-G files are written to `models\clip`.  CLIP-G is
converted from the embedded OpenCLIP key layout to the Comfy/HF key layout,
including Q/K/V split and `text_projection` transposition.

### 5D Tensor Post-processing (HunyuanVideo / Wan)

The Web UI applies this automatically after llama-quantize — no manual step
needed there.  For manual CLI workflows:

```bash
# 1. Convert to F16
uv run python convert.py --src model.safetensors

# 2. Quantize with llama-quantize (external step)
llama-quantize.exe model-F16.gguf model-Q8_0.gguf Q8_0

# 3. Insert 5D tensors
uv run python fix_5d_tensors.py --src model-Q8_0.gguf --dst model-Q8_0-fixed.gguf
```

## Benchmarking llama-quantize

Use the benchmark helper to compare compatible `llama-quantize`
binaries on the same source GGUF:

```bash
uv run python tools/benchmark_llama_quantize.py \
  --src model-F16.gguf \
  --quant Q4_K_M \
  --threads 8 \
  --exe H:\ComfyUI-Easy-Install\Add-Ons\Tools\llama.cpp\llama-quantize.exe \
  --exe path\to\another\patched\llama-quantize.exe
```

The benchmark writes temporary output GGUFs and removes them after timing.  Use
an F16, BF16, F32, or Q8_0 source GGUF for meaningful measurements.

## Safetensors Output

The **Convert → Safetensors** tab in the Web UI produces quantized `.safetensors`
files as an alternative to GGUF.  This is useful
when you want ComfyUI-compatible weights without the GGUF container format.

| Key | Format | Backend | Notes |
|---|---|---|---|
| `F16` | Half precision | Python | Standard default, smallest non-quantized size |
| `F16_MIXED` | Half precision, high-precision tensors stay F32 | Python | Matches GGUF K-quant behavior for critical layers |
| `FP8` | Scaled float8_e4m3fn (ComfyUI convention) | Python | Full-precision compute by default (`full_precision_matrix_mult`) — see below |
| `FP8_MIXED` | FP8, high-precision tensors stay F32 | Python | Same full-precision-compute default as `FP8` |
| `INT8` | int8_tensorwise, ConvRot-rotated where possible (ComfyUI convention) | Python | Per-layer `weight_scale`; weight-only quantization, no runtime activation quant |
| `INT8_MIXED` | INT8/ConvRot, high-precision tensors stay F32 | Python | Aggressive 8-bit quantization with protection |
| `NVFP4` | NVIDIA FP4, block-scaled | Python | Full-precision compute by default (`full_precision_nvfp4`); needs a Blackwell GPU (RTX 50-series/B200) to run — see below |
| `NVFP4_MIXED` | NVFP4, high-precision tensors stay F32 | Python | Same full-precision-compute default as `NVFP4` |

**Naming vs. the community:** `FP8`/`FP8_MIXED` write files named
`<model>-fp8_e4m3fn_scaled(_mixed).safetensors` — the same "scaled fp8"
convention Civitai/Comfy-Org releases use, so you can tell at a glance it's
the same compression as a Civitai `fp8_e4m3fn_scaled` upload (this tool only
writes the e4m3fn variant, never e5m2). `NVFP4` is NVIDIA's own name and
**not** the same algorithm as **NF4** (bitsandbytes' 4-bit normalfloat, the
format behind Civitai's "Flux NF4" releases) — this tool does not offer NF4.

**Estimated output size:** once a source model is selected, the tab shows an
"Estimated output" size under the format dropdown, computed from the actual
safetensors header (no tensor data loaded) rather than a fixed ratio — needed
because `*_MIXED` formats' size depends on the detected architecture's
protected-tensor list, not just the target dtype.

**Per-architecture recommendation badge:** once a source model is selected,
the tab shows a colored hint under the format dropdown — analogous to the
GGUF tab's static "recommended ★" label on `Q4_K_M`, but per-architecture.
Unlike GGUF K-quants (a uniform quality/size tradeoff regardless of
architecture), plain `INT8` has measurably corrupted output (wrong
poses/identities) on attention-sensitive DiT architectures in this project's
own testing, while being fine on architectures with no protected layers at
all. The badge discloses confidence honestly: only Lumina2/Z-Image's
protection scope has been confirmed by an actual convert+load+render cycle;
other protected architectures show the same recommendation but note it's
matched against a community reference, not yet render-verified.

**ComfyUI version requirement:** loading `INT8`/`INT8_MIXED` output needs
**ComfyUI v0.25.0+** — `comfy-kitchen`'s int8/ConvRot optimizations landed in
`comfy-kitchen` 0.2.9 (commit `ade4dfd`), first bundled in ComfyUI v0.25.0
(2026-06-16). Older ComfyUI versions either won't have the `int8_tensorwise`
loader at all or will use an unoptimized/pre-ConvRot version of it.

**Why FP8 is safe now, and why NVFP4/MXFP8 still aren't offered:** ComfyUI's
native FP8/NVFP4/MXFP8 formats dynamically quantize *activations* too at
inference time (`QUANT_ALGOS["quantize_input"]` defaults `True`), which
produced visibly wrong output (black bars, wrong poses/identities, full-image
noise) on Lumina2/Z-Image checkpoints even after this tool's own on-disk data
was verified byte-correct — activation quantization is inherently lossy in a
way no weight-side protection list can compensate for. `int8_tensorwise` is
one of only two `QUANT_ALGOS` entries ComfyUI marks `"quantize_input": False`
— weight-only quantization, activations always stay full precision, avoiding
that lossy path entirely. See [docs/issues_analysis.md](docs/issues_analysis.md)
#15 (including its Correction note — an earlier draft of this investigation
misattributed the corruption to a specific ComfyUI GitHub issue that turned
out to be a performance-only bug; that citation has been retracted).

FP8 turned out not to need this exclusion: `comfy/utils.py`'s
`convert_old_quants()` — the function ComfyUI uses to load the scaled-FP8
checkpoints already common on Civitai/HuggingFace — sets
`"full_precision_matrix_mult": true` for those legacy checkpoints, which makes
`comfy/ops.py`'s `MixedPrecisionOps.Linear.forward()` skip the risky
quantized-compute branch entirely and run a plain full-precision matmul
instead, architecture-independently. `convert_to_safetensors()` now writes
that same flag by default for every FP8/FP8_MIXED layer it produces
(`full_precision_fp8=True`), so this tool's own FP8 output *should* be as
safe as the checkpoints already circulating in the wild — no per-architecture
bet the way INT8's `keys_hiprec` is, for the runtime compute path. That
conclusion comes from reading ComfyUI's source, though, not from an actual
convert+load+render test with this tool's own output — so the Model Support
tab still shows FP8_MIXED as **? Unknown** on every architecture except
`lumina2` and `flux`. `lumina2`: a same-seed/prompt comparison confirmed
composition, identity, and outfit preserved (only a single secondary prop's
color/shape deviated, judged tolerable variance) — the same bar INT8_MIXED
already met, so `lumina2` + FP8_MIXED shows **✓ Verified** (see
`docs/issues_analysis.md` #16's third correction). `flux` (FLUX.2 Klein 9B):
3 same-seed/prompt comparisons (5 seeds total) showed zero visible deviation
from baseline for FP8, FP8_MIXED, INT8, *and* plain INT8 — a stronger result
than `lumina2`'s, so all four show **✓ Verified** there, unlike `lumina2`
where plain FP8/INT8 stayed on the ✗ side. `full_precision_matrix_mult` also only fixes the
*runtime* compute path — it does nothing for precision already lost when a
tensor is quantized to e4m3 *on disk*, so plain (non-mixed) FP8 carries the
same keys_hiprec sensitive-architecture risk plain INT8 does, and the
format-recommendation badge on Convert → Safetensors warns for it
identically. That risk is no longer just theoretical for `lumina2`: a
separate same-seed/prompt comparison showed plain FP8 output there with a
different outfit and background than the unquantized render (only the pose
skeleton matching) — the Model Support tab shows plain FP8 on `lumina2` as
**✗ Known issue**, not Caution (see #16's second correction). The tradeoff
on top of all that is no FP8
tensor-core compute speedup; FP8 output is a storage/VRAM-savings format
only unless a user explicitly opts out (`full_precision_fp8=False`). See
[docs/issues_analysis.md](docs/issues_analysis.md) #16 for the full
investigation. NVFP4 has a fix now too (2026-08-13): reading ComfyUI's
actual `comfy/ops.py` source found `full_precision_matrix_mult` is read
generically per-layer, not FP8-specific — `convert_to_safetensors()` now
sets it for `nvfp4` layers as well (`full_precision_nvfp4=True`). A first
`flux` render test with this alone still crashed ComfyUI outright
(`RuntimeError: mat1 and mat2 shapes cannot be multiplied`) — a second bug,
`ModelFlux.keys_shape_critical` missing `txt_in.weight`/
`vector_in.in_layer.weight`, the same #9 shape-inference class `img_in`
already guarded against. Both fixed, re-converted, re-rendered: `flux` +
NVFP4/NVFP4_MIXED now show **✓ Verified**, same bar FP8/INT8 already
cleared. `lumina2` was re-converted and re-tested with the same fix
(2026-08-13; its own `keys_shape_critical` was audited against ComfyUI's
Lumina2/Z-Image detection code and already covered every raw shape read
there, so no `flux`-style gap existed to fix) — no more full-image noise on
any of 3 tested prompts, but one of the three still showed a full
composition/pose/outfit change from baseline. That's a real improvement over
the pre-fix "full-image noise" evidence, but when it does fail it fails
exactly like plain FP8/INT8 above (full composition/identity swap, not a
minor detail) — and both of those are **✗ Known issue** on a single failing
test each, no "but other tests were clean" exception. Holding NVFP4 to a
looser bar for showing the identical failure mode wasn't consistent, so it
stays **✗ Known issue** too rather than **✓ Verified** or **⚠ Caution**.
**Update (2026-08-19, docs/issues_analysis.md #17):** NVFP4/NVFP4_MIXED are
now offered in the Convert → Safetensors dropdown. Since the investigation
above, `keys_shape_critical`/`keys_hiprec` coverage was audited and fixed
architecture-by-architecture (`flux`, `sdxl`, `sd1`, `sd3`, `hidream`,
`aura`), and each was re-converted and render-tested clean at a fixed seed —
see `safetensors_quant.py`'s `_RENDER_VERIFIED_MIXED` for the evidence
behind each entry. `lumina2`'s NVFP4/NVFP4_MIXED — the specific case
described above — remains **✗ Known issue**; that finding wasn't
re-investigated or superseded.

**Already-quantized sources:** if you point Convert → Safetensors at a
checkpoint that's already quantized (e.g. a published ComfyUI-native
`int8_tensorwise`/ConvRot or scaled-FP8 release), it's automatically
dequantized first and then cleanly re-quantized to your chosen target format
— see `dequantize.py`. The one case that still fails is an int8/uint8 weight
with no recognizable `weight_scale`/`scale_weight`/`comfy_quant` sidecar at all, since
there's no way to reconstruct the original magnitude without a scale.

## Model Support Tab

The **Model Support** tab shows, for every architecture this tool detects, a
color-coded support level for each output format (GGUF, F16/F16_MIXED,
FP8/FP8_MIXED, INT8/INT8_MIXED, NVFP4/NVFP4_MIXED), using four states:
**✓ Verified** (actually converted, loaded, and rendered correctly in
ComfyUI with this tool's own output, no visible deviation from the
uncompressed baseline — e.g. `flux` + NVFP4/NVFP4_MIXED, after fixing the
two bugs that caused its earlier composition drift, see above), **⚠ Caution**
(actually render-tested and showing some visible-but-tolerable deviation
from the uncompressed baseline — not broken but not clean either; currently
empty, no combination has earned exactly this bar), **✗ Known issue**
(actually render-tested and confirmed to produce wrong output on at least
one real prompt — full composition/identity swap, not a minor detail — e.g.
plain INT8, plain FP8, or NVFP4/NVFP4_MIXED on `lumina2`; see
`docs/issues_analysis.md` #15/#16), or **? Unknown** (never actually
rendered, no evidence either way — most (architecture, format) cells are
here). Caution used to mean "supported but untested", the same thing
Unknown means today — split apart 2026-08-13 because "never tried" and
"tried, and it does something different" call for different reactions from
a user picking a format: the former is a request for a test report, the
latter a request to judge whether the tradeoff is acceptable for a given
use case. Click a format cell to switch to the matching Convert tab with
that format pre-selected — the GGUF column collapses every K-quant level
into one cell, since a working F16 GGUF conversion carries over to all of
them uniformly, so clicking it defaults to `Q4_K_M`. NVFP4/NVFP4_MIXED
cells behave like any other Safetensors format now (see the Update note
above) — clicking one switches to Convert → Safetensors with that format
pre-selected. Once a source checkpoint is selected on either Convert
tab, its format dropdown also gets a ⚠, ✗, or ? prefix on any entry that's
Caution, Known-issue, or Unknown for the detected architecture (`gui.py`'s
`annotate_safetensors_choices()`/`annotate_gguf_choices()`) — informational
only, every entry stays selectable. The underlying support data (which model
names map to which internal architecture key, and the four-state logic
itself) lives in `model_support.py` and is documented there as an explicit
editorial judgment call, open to correction.

Below the per-architecture table is a second **Text Encoder Support** table,
one row per vendored base-model family (Qwen3-4B/8B, Qwen2.5-VL-7B,
Mistral-Small-3.2-24B, ERNIE-Image's Ministral3 prompt-enhancer, T5-XXL,
CLIP-L, CLIP-bigG, Pile-T5-XL, Llama-3.1-8B — the same families
`detect_text_encoder_family()` matches, see [Text-Encoder
Conversion](#text-encoder-conversion) below), covering the
formats that tab offers (GGUF — collapsing every direct outtype and K-quant —
plus FP8/FP8_MIXED/INT8/INT8_MIXED/NVFP4/NVFP4_MIXED). As of 2026-08-12,
**Qwen3-4B** shows **✓ Verified** across every column — GGUF plus all six
safetensors formats (convert+load+render-tested in ComfyUI, no
format-specific defect found). INT8/INT8_MIXED come with one sharp
gotcha, not a quality caveat: this format only loads correctly through
ComfyUI's native **`CLIPLoader`** node. Loading it through ComfyUI-GGUF's
**`CLIPLoaderGGUF`** node instead — an easy mistake, since that's the node
most Z-Image workflows already use for the GGUF text-encoder slot —
produces full-image structured-noise garbage (that node expects an actual
`.gguf` file and has no ConvRot/INT8-safetensors decode path). This was
briefly misdiagnosed as a broken conversion after a same-day batch of
renders all used the wrong loader node; loading the identical file through
`CLIPLoader` renders cleanly. **Qwen3-8B** (FLUX.2 Klein 9B's own text
encoder) shows **✓ Verified** across every column — GGUF plus all six
safetensors formats. FP8/FP8_MIXED/NVFP4/NVFP4_MIXED got 3 same-seed
comparisons, INT8/INT8_MIXED got 2, GGUF (Q5_K_M) got 2: zero visible
deviation for the safetensors formats, minor conditioning drift (a
face-tattoo detail, a small ornament gap) but identical subject and
composition for GGUF — comparable to the conditioning noise Qwen3-4B's own
GGUF K-quants already showed, not a format-specific defect. Notably
NVFP4/NVFP4_MIXED stayed clean here even though quantizing the FLUX.2 Klein
DiT itself with those same formats *did* show visible composition drift in
the same testing session — text-encoder quantization only perturbs the
conditioning vector fed to an otherwise full-precision DiT, not the
sampling trajectory's own numerics.
**Llama-3.1-8B** and **T5-XXL** (HiDream-I1's Llama-3.1-8B-Instruct and T5-XXL
text encoders) show **✓ Verified** for FP8/FP8_MIXED/INT8/INT8_MIXED/
NVFP4/NVFP4_MIXED, render-tested clean at a fixed seed against the
unquantized baseline. **Pile-T5-XL** (AuraFlow's `aura_t5`) shows **✓
Verified** for GGUF only — its FP8/INT8/NVFP4 safetensors formats show **✗
Known-issue**: ComfyUI's `AuraT5Model` loader never got the
`*_quantization_metadata` wiring other families' loaders have, so those
formats render broken output structurally, not from a conversion defect.
**CLIP-L**/**CLIP-bigG** show **✗ Known-issue** on every quantized format for
the same class of reason (see `model_support.py`'s docstring for the
per-family evidence behind every cell) — this applies to HiDream-I1 too,
which uses CLIP-L/CLIP-G as 2 of its 4 text encoders: F16 safetensors is
therefore the only safe format for those two, even though the other two
(Llama-3.1-8B/T5-XXL, above) are fully Verified. HiDream-I1 is not
end-to-end quantizable across all four of its text encoders. Every other family still shows
**? Unknown** across every column (never actually rendered). Both this table and the main one share one
color legend — literal red/yellow/green, not reused from the app's teal
accent color. Clicking a cell jumps to the Convert Text Encoder tab with that
format pre-selected — unlike the diffusion-model table, NVFP4 isn't excluded
here since it's a real, selectable format for text encoders.

## Text-Encoder Conversion

The **Convert Text Encoder** tab converts
bare HF/Transformers text-encoder checkpoints (Qwen, T5, CLIP, Mistral variants) to
GGUF **or** quantized safetensors. This is separate from SDXL CLIP-L/CLIP-G
extraction and runs entirely in this tool's own Python environment — no
ComfyUI installation required.

The format dropdown covers three backends:

| Formats | Backend | Needs base repo ID? | Extra prerequisites |
|---|---|---|---|
| `F32`/`F16`/`BF16`/`Q8_0` | `convert_hf_to_gguf.py` (llama.cpp, auto-cloned) | Yes | `git` |
| `Q6_K`…`Q2_K` (K-quants) | Same, then a **plain** `llama-quantize` second pass | Yes | `git`, plus `cmake` + a C++ compiler ([install instructions](docs/building-llama-quantize.md#text-encoder-k-quants-build-automatically--you-dont-need-this-guide-for-them) — the build itself is automatic) |
| `FP8`/`FP8_MIXED`/`INT8`/`INT8_MIXED`/`NVFP4`/`NVFP4_MIXED` | This tool's own safetensors quantizer (`safetensors_quant*.py`) | **No** | None — no llama.cpp, no download |

The K-quant path deliberately builds its **own** plain llama-quantize from the
auto-cloned llama.cpp checkout via `cmake` (cached under `.llama.cpp/build-quantize/`,
built once) — **not** the City96-patched binary used for diffusion-model GGUFs
(see [Building llama-quantize](docs/building-llama-quantize.md)), since that patch
is documented as unsafe for LLM/text GGUFs.

### Workflow

For **GGUF formats** (direct outtypes and K-quants), text-encoder conversion requires:

1. **Source weights file**: A single `.safetensors` checkpoint (bare model file, not
   a directory). Since standalone text-encoder `.safetensors` files lack `config.json`
   and tokenizer files, you must also provide:

2. **Base model HF repo ID** (optional): The base model's Hugging Face repository
   (not a fine-tune's repo). Leave this blank and the tool fingerprints your
   weights' own tensor shapes (`detect_text_encoder_family()`) against its
   vendored candidate families — Qwen3-4B/8B, Mistral-Small-3.2-24B, CLIP-L/bigG,
   T5-XXL, Qwen2.5-VL-7B, ERNIE-Image PE — and uses the matching one automatically,
   no download needed. Only set this manually for base models outside that list,
   e.g.:
   - For a Qwen-Image text encoder: `Qwen/Qwen2.5-VL-7B-Instruct`
   - For a Z-Image text encoder: A Qwen3 4B base model
   - For FLUX.2 klein: A Qwen3 4B or 8B base model

   The repo ID must have `config.json` (mandatory) and tokenizer files
   (`tokenizer.json`, `tokenizer.model`, etc.). For the vendored families these
   are copied from `text_encoder_configs/` in this repo; for anything else
   they're fetched from HuggingFace and assembled with your weights into a
   temporary HF-style directory.

3. **Output path and format**: Choose an output filename and one of the formats above.

For **FP8/FP8_MIXED/INT8/INT8_MIXED/NVFP4/NVFP4_MIXED**, only the source
weights file and an output path are needed — the base repo ID field is
ignored.

Like the GGUF and Safetensors tabs, this tab shows an "Estimated output"
size under the format dropdown once a source file is selected.

### Implementation

**GGUF path:**
1. Clones `llama.cpp` into `.llama.cpp/` next to this repo if not already
   present (skipped on subsequent runs).
2. Assembles a temporary directory with your source weights (renamed to
   `model.safetensors` to preserve the original file) and downloaded
   config/tokenizer files.
3. Runs `convert_hf_to_gguf.py` with this tool's own Python interpreter
   (`transformers`/`sentencepiece`/`protobuf` are regular dependencies in
   `pyproject.toml`, installed by `uv sync`).
4. For K-quants: builds a plain `llama-quantize` from the same clone (`cmake`,
   cached after the first run) and runs it as a second pass on the F16 output.
5. Returns the output at your chosen path.

**Safetensors path (FP8/NVFP4):** loads the checkpoint, applies the same
per-tensor FP8-scaled/NVFP4-block-scaled quantization used for diffusion models
(`safetensors_quant_fp8.py`/`safetensors_quant_nvfp4.py`), and writes a
ComfyUI-native quantized `.safetensors` file — no architecture-specific tensor
protection is needed here (unlike diffusion DiTs, ComfyUI's text-encoder loaders
build models from fixed config presets rather than inferring hyperparameters
from checkpoint tensor shapes, so there's no analogous shape-corruption risk;
see [docs/issues_analysis.md](docs/issues_analysis.md) #9 for that class of bug).

This is a subprocess/library pipeline, not part of the DiT architecture detection
system. Per-family text-encoder handling (e.g., SDXL CLIP key mapping, Qwen mmproj
pairing) is not automated — the generic HF-to-GGUF conversion handles supported
standard architectures. Unsupported or non-standard architectures require manual
key mapping.

Override the clone location with the `S2G_LLAMA_CPP_HOME` environment variable
if you already have a llama.cpp checkout elsewhere and want to reuse it.

### Reference: Encoder Family per Model Family

Candidate models for text-encoder GGUF conversion:

| Model family | Text encoder family | Status |
|---|---|---|
| SDXL 1.0 | CLIP-L + OpenCLIP-bigG | CLIP-specific extraction and key mapping (not generic HF-to-GGUF route) |
| Qwen-Image / Qwen-Image-Edit | Qwen2.5-VL 7B + mmproj | Multimodal text/image encoder; keep mmproj paired with the GGUF encoder |
| Z-Image / Z-Image-Turbo | Qwen3 4B | Standard HF/Transformers path via `convert_hf_to_gguf.py` |
| FLUX.1 / FLUX.1 Kontext | CLIP-L + T5-XXL | Existing Flux dual-encoder layout |
| FLUX.2 [klein] 4B | Qwen3 4B | Must keep 4B encoder paired with 4B model |
| FLUX.2 [klein] 9B | Qwen3 8B | Must keep 8B encoder paired with 9B model |
| FLUX.2 [dev] | Mistral Small 3.2 24B | Separate Mistral text-encoding path |
| ERNIE-Image / Turbo | Mistral3 + Ministral3 PE | Requires text encoder and prompt enhancer handling |

Primary model references:

- Stability AI SDXL Base model card:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>
- Stability AI SDXL pipeline component map:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/model_index.json>
- Stability AI SDXL `text_encoder_2` config:
  <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/text_encoder_2/config.json>
- LAION OpenCLIP ViT-bigG/14 model card:
  <https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k>
- LAION OpenCLIP ViT-bigG/14 repository files:
  <https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/tree/main>
- OpenAI CLIP-L model card:
  <https://huggingface.co/openai/clip-vit-large-patch14>
- OpenAI CLIP-L repository files:
  <https://huggingface.co/openai/clip-vit-large-patch14/tree/main>
- Qwen-Image model card:
  <https://huggingface.co/Qwen/Qwen-Image>
- Qwen-Image-Edit model card:
  <https://huggingface.co/Qwen/Qwen-Image-Edit>
- Qwen2.5-VL 7B model card:
  <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>
- ComfyUI Qwen-Image setup guide:
  <https://docs.comfy.org/tutorials/image/qwen/qwen-image>
- ComfyUI Qwen-Image-Edit setup guide:
  <https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit>
- Tongyi-MAI Z-Image-Turbo model card:
  <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>
- Tongyi-MAI Z-Image-Turbo pipeline component map:
  <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/model_index.json>
- Tongyi-MAI Z-Image-Turbo text encoder config:
  <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/text_encoder/config.json>
- Black Forest Labs FLUX.1 Kontext model card:
  <https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev>
- Black Forest Labs FLUX.2 inference repository:
  <https://github.com/black-forest-labs/flux2>
- ComfyUI FLUX.2 [klein] setup guide:
  <https://docs.comfy.org/tutorials/flux/flux-2-klein>
- Baidu ERNIE-Image-Turbo model card:
  <https://huggingface.co/baidu/ERNIE-Image-Turbo>
- Baidu ERNIE-Image-Turbo pipeline component map:
  <https://huggingface.co/baidu/ERNIE-Image-Turbo/blob/main/model_index.json>
- ComfyUI ERNIE-Image setup guide:
  <https://docs.comfy.org/tutorials/image/ernie-image/ernie-image>

Candidate non-CLIP text encoder reference:

- Huihui Qwen3 4B model card:
  <https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-abliterated-v2>
- Huihui Qwen3 4B repository files:
  <https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-abliterated-v2/tree/main>

Tooling references:

- llama.cpp HF-to-GGUF converter:
  <https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py>
- ComfyUI-GGUF text encoder loader support:
  <https://github.com/city96/ComfyUI-GGUF>

## Possible Future Extensions

### Image GGUF Quantization Reference

City96's `tools/lcpp.patch` should be treated as the implementation reference
for image-model GGUF quantization support.  Even if it needs forward-porting for
newer `llama.cpp` versions, it captures the expected integration points:

- image architecture registration for families such as SD1, SDXL, SD3, Flux,
  HunyuanVideo, Wan, HiDream, Cosmos, and Lumina 2
- tensor classification rules for which weights may use K-quants and which
  tensors should remain higher precision
- loader and metadata adjustments so `llama-quantize` does not reject image
  GGUFs because they do not look like LLM checkpoints
- shape/name handling needed by ComfyUI-GGUF-compatible image models

Patch references:

- ComfyUI-GGUF tools guide:
  <https://github.com/city96/ComfyUI-GGUF/tree/main/tools>
- City96 `lcpp.patch`:
  <https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/tools/lcpp.patch>
- Current llama.cpp source layout:
  <https://github.com/ggml-org/llama.cpp/tree/master/src>

### Checkpoint GGUF Workflow

A later release could target full checkpoint workflows instead of only single
model files.  Two related directions are useful:

1. **Checkpoint decomposer / repacker**: split a monolithic checkpoint into its
   components, let the user choose quantization per component, convert each
   supported component, then write a new checkpoint manifest or bundle.
2. **Checkpoint loader node**: provide a ComfyUI node that accepts a checkpoint
   layout containing both regular `.safetensors` components and GGUF
   components, then loads each part through the correct backend.

The decomposer path should reuse SDXL component analysis for VAE / CLIP and add
model-family-aware handling for diffusion models and text encoders.  The loader
path is likely safer than trying to repack every component into one physical
file, because GGUF and safetensors have different metadata and loading
assumptions.

Implementation notes for future releases:

- Analyze first, convert second: identify UNet/DiT, VAE, CLIP, T5, Qwen, and
  Mistral-family components before presenting quantization options.
- Store conversion choices per component and keep source checkpoints untouched.
- Support mixed output layouts such as GGUF diffusion model + safetensors VAE +
  GGUF text encoder.
- Treat "repack into checkpoint" as a manifest/bundle problem unless ComfyUI
  gains a stable native format for mixed GGUF/safetensors checkpoints.
- Validate the resulting layout with a ComfyUI loader path rather than only
  checking that files were written.

## Known Issues

See [Issues Analysis](docs/issues_analysis.md) for common errors and their fixes.

## Running Tests

```bash
uv run pytest
uv run ruff check .
```

## License

Apache-2.0
