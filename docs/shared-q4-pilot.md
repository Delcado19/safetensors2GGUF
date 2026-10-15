# Shared Q4_K mathematical pilot

Date: 2026-10-15. Branch: `research/unified-converter-inventory`.

## Implementation and result

The research-only NumPy Q4_K encoder ports unweighted
`quantize_row_q4_K_ref` from preserved llama.cpp b3962
`c8c07d658a6cefc5a50cfdf6be7d726503612303` (`ggml/src/ggml-quants.c`, MIT).
It reuses the Q5 pilot's affine fitting, scale/minimum packing, validation and
bounded streaming implementation; it does not duplicate a second affine fitter.

Q4 uses codes 0-15, a -1.0 starting search offset and 21 candidates at 0.1
spacing. Q5 retains codes 0-31, a -0.5 offset and 16 candidates. Both preserve
native sequential float32 accumulation, candidate-minimum updates, native
rounding, positive-zero header minima and initial codes when FP16 scales underflow.
Q4 packs two four-bit codes per byte and omits Q5's separate high-bit plane:
256 values produce 144 bytes rather than 176.

[`data/q4-pilot-comparison.json`](../data/q4-pilot-comparison.json) records
**166 inputs, 498 exact native comparisons and 332 exact chunk comparisons**.
The same three frozen native builds are used as for Q5/Q6, with archive/commit
and executable SHA256 checks before execution. All comparisons pass. Decoder
agreement is also retained. Inputs include 145 synthetic edge/random cases and
21 small read-only slices from seven installed floating-point families: CLIP-L,
CLIP-bigG, FLUX, Lumina2, SDXL, Qwen3-4B and Qwen3-8B. Input/payload hashes,
shapes, source-header provenance and RMSE are recorded. Both the Q4 wrapper and
shared Q5 implementation hashes are retained, avoiding wrapper-only provenance.

The positive ramp from 1 to 2 has RMSE about 0.02299; the frozen Molbal saturation
problem is absent. This is a numeric probe, not an image-quality prediction.
Q5 was rerun after sharing the implementation: all 166 inputs / 498 native and
332 chunk checks still pass. Its report was refreshed with the new source hash;
earlier full-model reports retain hashes of the exact code used for those runs.

## Contracts and limits

### CPU performance follow-up (2026-10-16)

The shared affine fitter now copies each bounded input window to column-contiguous
storage. Its repeated column accesses become contiguous without changing native
sequential float32 accumulation or candidate ordering. Both complete 166-input
Q4/Q5 pilots were rerun against all three pinned native references after this
change; their current reports retain the updated implementation hashes.

`data/affine-layout-benchmark.json` records seven alternating baseline/optimized
runs on the same 1,048,576-value prepared FLUX slice, 256 blocks per chunk.
Median Q4 time decreased from 0.17458 s to 0.13673 s (about 22%); Q5 decreased
from 0.13635 s to 0.11058 s (about 19%). Both output hashes remain identical.
These measurements include encoding and hashing, exclude loading/file I/O/readback,
and do not establish full-model speed or native-speed parity. The earlier full-model
reports describe the older encoder and retain their historical source hashes.

Profiling this Q4 slice placed about 94% of encoder time in affine fitting.
Larger chunk sizes offered little improvement, and an experimental cumulative-sum
rewrite was slower, so neither was adopted. Native-speed conversion remains an
open target; fused compiled fitting and bounded parallel execution need separate
parity, cancellation, packaging and memory measurements before adoption.

`iter_q4_k` accepts explicit nonempty float32 rows divisible by 256, preserves
strided logical C order and only copies bounded windows. Each yielded chunk owns
its data. Consumers must use atomic publication and close/stop between chunks
for cancellation, because nonfinite validation is incremental. Arithmetic
overflow is rejected; the entire finite float32 range is not guaranteed.
`quantize_q4_k` allocates complete packed output with preserved leading dimensions.

Tests replay all synthetic native hashes with chunk sizes 1/3/256, verify source
provenance and retained seven-family coverage, and check strided ordering,
independent chunks, non-mutation, early close, late NaN, invalid shapes/dtypes and
chunk-size rejection. Existing Q5/Q6 and application tests remain in place.

Q4_K_S/Q4_K_M are model-level mixed policies, not block encodings. A subsequent
[FLUX research file path](flux-q4-research-path.md) now implements both and matches
native in 16 schematic conversions (400 tensors), including order-sensitive
first-four-V promotion and incompatible V fallback consuming a slot. No Q4
application dispatch or ComfyUI render was added in this block/preset phase.
A subsequent [complete FLUX Q4_K_M comparison](flux-q4m-full-model-validation.md)
also matches all 201 tensors and metadata; Q4_K_S remains schematic-only.
Native Q4 presets and the Krea allowlist remain unchanged. Linux native execution,
importance matrices and controlled performance/peak-memory measurements are open.

## Reproduce

```bash
python -m scripts.check_q6_pilot --format Q4_K --models-root /path/to/ComfyUI/models --output /fresh/q4-report.json
```

Omit `--models-root` for synthetic probes only. The existing pinned native build
reports/binaries are required; this command performs no downloads or inference
and refuses to replace an existing report. Installed weights are never modified.
