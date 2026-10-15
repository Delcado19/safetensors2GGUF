# Shared Q5_K mathematical pilot

Date: 2026-10-15. Branch: `research/unified-converter-inventory`.
Implementation: [`scripts/reference_q5_k.py`](../scripts/reference_q5_k.py).
Evidence: [`data/q5-pilot-comparison.json`](../data/q5-pilot-comparison.json).

## Scope

A bounded NumPy implementation ports unweighted `make_qkx2_quants` and
`quantize_row_q5_K_ref` from frozen MIT-licensed llama.cpp b3962
`c8c07d658a6cefc5a50cfdf6be7d726503612303`. The preserved GGML MIT notice is
referenced in the source. This is block mathematics only: 256 finite F32 values
become 176 Q5_K bytes. Existing application/native backends and format registries
are unchanged by this pilot; its seven-format Krea allowlist is not expanded.

**Q5_K is not Q5_K_M.** The latter is a model-level precision policy that can
promote selected tensors to Q6_K, preserve others and apply shape fallbacks.
Those rules, model adapters and GGUF metadata are separate from the block codec;
the FLUX file-policy follow-up below implements one explicit research preset.

## Compatibility details

- Groups of 32 values fit a weighted affine scale and nonpositive minimum using
  native scalar accumulation order and float32 operations. The candidate search
  updates the minimum after a winning fit, affecting later candidates.
- Group scales/minima are packed into 6-bit metadata; block scale/minimum use
  little-endian FP16. Zero stored scales retain initial fitted codes.
- Native maximum-minimum reduction starts at positive zero and accepts only
  strictly greater values. Preserve that sign: NumPy can otherwise retain -0
  and change header bytes on zero/positive-only probes. The first diagnostic run
  found eight affected cases (one or two bytes each); the corrected run passes.
- Input dtype/row size must be explicit and valid. Do not silently cast, flatten
  incompatible rows, accept nonfinite values or emit overflowing arithmetic.
  The entire finite float32 range is not guaranteed by this prototype.
- `iter_q5_k` handles strided logical C order with bounded chunk copies and
  independently owned payloads. Consumers stop/close between chunks to cancel.
  Later invalid chunks can fail after earlier output; file consumers must publish
  atomically. NumPy error-state settings do not leak across a yield.

## Measured results

**PASS: 166 inputs, 498 byte comparisons against all three frozen native
references, plus 332 independent chunk-size comparisons (1 and 3 vs default 256).**
All bytes agree. There are 145 synthetic edge/random inputs and 21 tiny installed
weight slices across seven families, from plain F16/BF16 sources. Original models
were only read; header hashes and source stability were checked by the existing
sampling helper. Native source/archive/executable checksums are verified first.

Positive-only ramp 1..2 RMSE is 0.01387811, matching the native reference. It does
not exhibit frozen Molbal's approximately 0.061 saturation. This is not an image
quality ranking or evidence for a complete Q5_K_M checkpoint.

Unit tests replay synthetic golden hashes without native executables and cover
signed-zero headers, positive-ramp reconstruction, read-only/strided/ND input,
bounded independent chunks, bad limits/dtypes/shapes, nonfinite/overflow rejection,
early close, late failure and NumPy error-state restoration. Q6_K retains its
existing checker API and tests; `check_q6_pilot` now explicitly selects either
block format with Q6_K still the default.

## Reproduce

```bash
python -m scripts.check_q6_pilot --format Q5_K --models-root "G:/ComfyUI-Easy-Install/ComfyUI/models" --output fresh-q5-pilot.json
```

Omit `--models-root` for synthetic probes only. Existing pinned native binaries
must be available from the prior offline reference build. The output report must
be new and publishes atomically; no downloads or inference are started.

Broader model comparisons, loader/image
quality, importance matrices, Linux and performance measurements remain open.
Application integration stays restricted to the separately validated FLUX preset.


## FLUX Q5_K_M file-policy follow-up

The explicit research writer now accepts `--preset Q5_K_M`; its default remains
Q6_K. The Workbench/API now offers both through the explicit experimental
`python_flux` backend; native remains the default. This adds no Krea K-quant support.

Frozen City96 promotes attention V, fused QKV and selected feedforward names to
Q6_K; other eligible matrices use Q5_K. Architecture/general exclusions preserve
source float bytes, and incompatible row widths fall back to F16. These image
rules deliberately do not use the text quantizer's layer-selection heuristics.

`data/flux-q5-path-comparison.json` records eight live schematic comparisons
against the verified patched b3962 executable: F16/F32 sources, both insertion
orders and aligned/unaligned rows. All 196 tensor payload/type/shape comparisons,
metadata and alignment match. Both encoder hashes and the writer hash are kept.
Artifacts are under `I:/safetensors2GGUF-flux-q5-fixtures-final-2026-10-15`.
Original models are untouched. Tests cover promotion aliases, mixed output,
unknown-preset rejection and preservation of existing output on cancellation or
late invalid data. The subsequent [complete FLUX.2 klein Base 9B comparison](flux-q5-full-model-validation.md)
also passes all 201 tensors and metadata. Supplied Python/native Q5 renders are
pixel-identical with two LoRAs. A follow-up original with matching seed/settings
shows Visible drift in pose/composition, while remaining coherent in this scene.
Broader quality and Linux execution remain untested.

```bash
python -m scripts.requantize_flux_q6 --source prepared-F16.gguf --output model-Q5_K_M.gguf --preset Q5_K_M
python -m scripts.check_flux_q6_path --preset Q5_K_M --artifacts /fresh/artifact-directory --output /fresh/report.json
```

The historical script name is retained to preserve existing Q6 callers.

The [Q4 pilot](shared-q4-pilot.md) now shares this affine fitter/packer and
stream validator with its own native search grid and payload layout. Q5 was
rerun against all three references after sharing; its report records the new
source hash while earlier model reports retain their historical run hashes.
