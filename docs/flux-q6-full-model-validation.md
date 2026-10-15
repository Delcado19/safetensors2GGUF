# FLUX Q6_K full-model numeric validation

Date: 2026-10-15. Branch: `research/unified-converter-inventory`.
Scope: an installed complete checkpoint through the existing float-GGUF
preparation stage, then the research NumPy Q6_K path and pinned native quantizer.
This is file/numeric validation, not loader, inference or image-quality evidence.

## Result

**PASS: all 201 tensors match the pinned native reference in name, type, shape
and complete payload bytes.** All typed metadata values and effective alignment
agree. The output contains 115 Q6_K tensors and 86 F32 tensors. Evidence:
[`data/flux-q6-full-model-comparison.json`](../data/flux-q6-full-model-comparison.json).

This completes the first full-checkpoint numeric parity gate for the explicit
FLUX research path. It does not enable production dispatch or establish runtime
compatibility beyond this file-level comparison.

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

## Remaining gates

Do not enable production dispatch solely from a numeric comparison. Installed
loader behavior, complete model instantiation, inference/image quality, LoRA,
offload, Linux execution and isolated memory/throughput measurements remain
separate gates. No render was started. The tested native backend remains intact.
