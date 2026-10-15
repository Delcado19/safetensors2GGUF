# FLUX Q6_K full-model validation

Date: 2026-10-15. Branch: `research/unified-converter-inventory`.
Scope: an installed complete checkpoint through the existing float-GGUF
preparation stage, then the research NumPy Q6_K path and pinned native quantizer.
The controlled conversion comparison is file/numeric validation. A subsequent
user-generated PNG pair provides scoped loader/inference and visual evidence below.

## Result

**PASS: all 201 tensors match the pinned native reference in name, type, shape
and complete payload bytes.** All typed metadata values and effective alignment
agree. The output contains 115 Q6_K tensors and 86 F32 tensors. Evidence:
[`data/flux-q6-full-model-comparison.json`](../data/flux-q6-full-model-comparison.json).

This completes the first full-checkpoint numeric parity gate for the explicit
FLUX research path. Numeric parity alone does not establish runtime
compatibility; the supplied PNG evidence below covers one concrete workflow.

## Exact input and preparation

Source: `G:\ComfyUI-Easy-Install\ComfyUI\models\diffusion_models\Flux.2 klein\9B - Base\flux-2-klein-base-9b.safetensors`.
The installed source contains 201 BF16 tensors. Its canonical header matches the
previous installed-model audit. The source remains untouched; all model output,
spool files and logs belong under
`I:\safetensors2GGUF-flux-q6-full-2026-10-15`.

The existing `convert.convert_file` prepares `prepared-F16.gguf` with explicit
`target_quant=F16`. Existing source reconstruction, name mapping and precision
protection remain in control. The resulting 19,939,776,800-byte GGUF contains
105 F32 and 96 F16 tensors, with `general.architecture = flux`.

The native quantizer-stage policy quantizes 19 of the prepared F32 matrices as
well: a prepared F32 tag alone is not a native protection flag. The research
path reproduces that existing behavior; it does not silently merge the Python
preparation protections with the native GGUF policy. Format/stage-specific
protection rules must remain explicit during consolidation.

The first preparation invocation successfully published the complete GGUF before
its final console status message raised `UnicodeEncodeError` for an arrow under
Windows cp1252. The prepared file was independently opened and checked before
reuse. Subsequent commands run with Python `-X utf8`; no second checkpoint copy
or unverified partial file was substituted. `prepare.log` retains the original
traceback so the invocation's nonzero exit is not confused with an ordinary
successful return. The writer had closed and atomically published before that
final logging failure.

This comparison starts from the **same prepared file** for both quantizers.
It does not establish equivalence with directly quantizing the original BF16
weights: the common F16 preparation stage is part of the existing native route.

## Compared paths

The first full Python attempt processed all 201 tensors, then exposed a Windows
large-file bug during readback: explicit alignment was a NumPy uint32 scalar,
and offsets beyond 2 GiB overflowed C long during modulo. Atomic cleanup removed
that attempt's temporary file. No failed output was accepted as evidence.

The shared research reader now converts alignment to a Python int at the read
boundary and owns mapping cleanup. This applies to the requantizer, container
checks and file comparison. Regression tests exercise offsets above 2 GiB, 4 GiB
and 1 TiB, including misalignment, without allocating huge test files. The actual
checkpoint run is repeated with the guard; no numerical encoder change is made.

- Python: `scripts.requantize_flux_q6.requantize`, shared NumPy Q6_K mathematics,
  existing FLUX quantizer-stage protections and row-width fallback.
- Native: frozen City96-patched llama.cpp b3962, verified commit/archive/patch/
  executable checksums, invoked with `Q6_K` and one worker thread.
- Comparison: `scripts.compare_gguf_outputs.compare_files`, typed metadata and
  ordered tensor names/types/shapes plus complete payload SHA256 values. Copy
  windows are at most 1 MiB, including file hashing; no full-tensor Python byte
  strings are constructed. Mappings close on every exit path.

An absent versus explicit default alignment key is normalized, while effective
alignment must agree. Whole-file hashes are recorded but are not an equality
criterion: metadata ordering/explicit defaults can change container bytes.

Original checkpoint and prepared-file SHA256 values are collected separately,
with unchanged source size/mtime around hashing. Native/Python stage reports and
logs remain beside the model files on `I:\`. The numeric comparison report is
retained in `data/flux-q6-full-model-comparison.json`.

Python and native conversion ran concurrently to reduce test turnaround. Their
elapsed wall times are observations under that shared load, not an isolated
speed benchmark. File mappings/page-cache behavior means bounded buffers do not
establish a process peak-RSS guarantee. The independent full-checkpoint hashing
job also ran during conversion.

The corrected Python attempt reused the prepared input and completed after the
native job had finished. Its elapsed time includes independent readback and
whole-file hashing. Do not treat the two durations as a controlled backend
benchmark.

The successful corrected Python run took approximately 1,136 seconds including
readback and hashing. The native run took approximately 442 seconds under the
different shared-load conditions described above. These timings are not evidence
of a speed improvement. Output sizes and exact file/tensor hashes are in the
report; metadata layout differences mean whole-file hashes need not match.

## Reproduce the final comparison

```bash
python -X utf8 -m scripts.compare_gguf_outputs --left "I:/safetensors2GGUF-flux-q6-full-2026-10-15/python-Q6_K.gguf" --right "I:/safetensors2GGUF-flux-q6-full-2026-10-15/native-Q6_K.gguf" --output fresh-full-model-report.json
```

The command writes a fresh JSON report atomically and exits nonzero on semantic
mismatch. It refuses to overwrite existing reports or either input. Reproduction
of the conversions uses the commands and pinned native build in the
[research-path guide](flux-q6-research-path.md), always with fresh output paths.

## User-generated ComfyUI comparison

The user supplied two PNGs on 2026-10-15: Python Q6_K at 17:39:38 and the
original safetensors at 17:44:06. Both embedded executed prompts and full UI
workflows are retained with PNG SHA256 values in
[`data/flux-q6-user-render-comparison.json`](../data/flux-q6-user-render-comparison.json).
PNG chunk CRCs were checked when extracting metadata. Original images remain
at their recorded local paths; they are not copied into the repository.

Only the executed model-loader nodes and the model connection into the LoRA
loader differ: `UnetLoaderGGUF` loads `python-Q6_K.gguf`, while `UNETLoader`
loads `Flux.2 klein/9B - Base/flux-2-klein-base-9b.safetensors` with default dtype.
All other executed node definitions match, including:

- Seed `738334089135969`, 896 x 1344, 34 steps, `dpmpp_sde`, CFG 4.0,
  `Flux2Scheduler`.
- Identical positive/negative prompts; `qwen3-8b-heretic_fp8_e4m3fn.safetensors`
  text encoder and `flux2-vae.safetensors` VAE.
- Two enabled diffusion LoRAs through rgthree Power Lora Loader:
  `F2k-9b_Base - NiceGirls UltraReal.safetensors` at 0.4 and
  `F2k-9b_Base - Lenovo UltraReal.safetensors` at 0.7.

Visual assessment of the supplied pair: subject identity/appearance, expression,
pose, silhouette and clothing remain close. Individual hair strands, clothing
folds and hand details vary; the poster illustration/text and background books
change visibly. No obvious added corruption or clear material degradation of the
main subject is apparent in this pair. This is a qualitative judgment, not a
quality metric or a guarantee for other prompts/checkpoints.

This establishes successful image generation through the named GGUF node with
both LoRAs enabled in this specific workflow. Exact installed loader revision,
execution logs, resource use and offload behavior were not captured. Matching
saved settings do not prove deterministic execution or isolate quantization from
loader/precision effects. There was no separate native-Q6_K render: full tensor
byte parity already exists, but is not a claim of pixel-identical inference.
No additional render was started by the agent.

The user subsequently reported converting FLUX.2 klein Base 9B to Q6_K through
the application. No new artifact hashes or output PNG were supplied for that
application run; this report is not a second independently measured parity test.

## Application integration boundary

`conversion_service._pipeline` already prepares K-quants as F16 GGUF with
`apply_unsqueeze=False` and calls `quantize.run_quantize` for the second stage.
The research path belongs at that second-stage boundary, subject to explicit
FLUX/Q6_K eligibility, rather than in `quantize.PYTHON_PRECISIONS`: that registry
selects direct first-stage conversion for every architecture.

Application integration preserves cancellation/progress, output validation/
overwrite behavior and intermediate cleanup through this boundary.
The research requantizer now supports an explicit `overwrite=True` /
`--overwrite` transaction, replacing an existing output only after successful
readback. Tests retain the old destination on cancellation and late invalid
values and reject source aliases even with overwrite enabled. This satisfies
the overwrite service contract. An explicit experimental Workbench/API route
now reuses the existing F16 preparation and exposes this writer only for
FLUX/Q6_K; native remains the default. See the [integration guide](flux-q6-research-path.md#workbenchapi-integration).
Do not retry native conversion implicitly after arbitrary
Python failures; cancellation, disk/write failures and malformed inputs must
remain visible. Native remains the existing default and unsupported model/preset
combinations must keep their established route. A backend selection is separate
from a quantization format, not a new global precision entry.

## Remaining gates

Do not make the experimental backend the default from this single checkpoint comparison.
Targeted offload, Linux execution, isolated memory/throughput measurements,
broader architecture/checkpoint/LoRA coverage and a real browser interaction
check remain separate gates. Small service/API contract tests pass. The tested native backend remains intact.
