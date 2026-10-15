# FLUX Q6_K research file path

Date: 2026-10-15. Branch: `research/unified-converter-inventory`.
Implementation: [`scripts/requantize_flux_q6.py`](../scripts/requantize_flux_q6.py).
Evidence: [`data/flux-q6-path-comparison.json`](../data/flux-q6-path-comparison.json).

## Scope

The first architecture-specific integration connects the shared NumPy Q6_K
encoder to a real FLUX GGUF quantizer stage. It accepts an already prepared,
single-file little-endian F16/F32 FLUX GGUF, preserves metadata/tensor order and
writes a mixed-precision Q6_K GGUF atomically. Production Workbench/CLI dispatch,
format registries and the existing native backend are unchanged.

This is not a safetensors importer or a new model adapter. Original checkpoint
reconstruction, prefix stripping, precision protection, SD shape fixes and
architecture metadata remain the existing converter's responsibility. The new
path starts at the same float-GGUF boundary as `llama-quantize`.

The FLUX policy is adapted from the preserved Apache-2.0 City96 patch at
`6ea2651e7df66d7585f6ffee804b20e92fb38b8a`, together with MIT-licensed b3962's
general exclusions. Source/license attribution is recorded in the implementation;
the numerical encoder retains its separate GGML MIT attribution.

## Behavior

- Eligible canonical 2D weight tensors use Q6_K. Q6_K does not use the ordered
  S/M promotions; those other presets remain outside this implementation.
- FLUX input/conditioning/final-layer protections and b3962's general exclusions
  preserve source bytes and dtype. Non-weight and non-2D tensors are copied.
- Eligible matrices whose row width is not divisible by 256 fall back to F16,
  matching the image patch. This can convert F32 to F16; it is an explicit native
  policy fallback, not a blanket promise to preserve all source precision.
- Standard/custom metadata fields retain their types and values. File type and
  quantization version are updated. An explicit alignment field is emitted,
  retaining the input alignment (including custom 64-byte alignment).
- Flat memmap windows contain at most 65,536 values. Only each quantized window
  is expanded to float32; protected tensors use bounded byte copies. Output
  readback checks tensor names/shapes/types/alignment/size and SHA256 per tensor.
- Atomic output, cancellation between windows/tensors, unchanged source size/mtime
  and existing-file/source-alias rejection protect output publication. Reader
  mappings and writer handles close on success and failure, including Windows.

Only canonical F16/F32 inputs are accepted. Other architectures, quantized/BF16
input, shards, sidecars, empty tensors, ranks above four and leading unit
dimensions requiring GGML rank canonicalization are rejected. Nonfinite tensor
values and overflowing arithmetic fail instead of producing silent corruption.
Source-stat checks are a change guard, not a lock against hostile concurrent
writers; do not modify input while converting.

## Measured file parity

Eight schematic FLUX files passed comparison against the exact pinned,
City96-patched b3962 quantizer: **196 tensor comparisons, no differences in tensor
types, logical shapes or payload bytes**. All compared metadata values agree;
comparison excludes the optional explicit default-alignment field and GGUF
container bookkeeping. This does not claim whole-file binary identity.

Cases cover the cross product of F16/F32 input, forward/reversed order and
aligned/unaligned row widths. Fixtures include attention, fused QKV, feed-forward,
input/conditioning/final-layer protections, normalization, non-2D tensors and
general exclusions. The checker verifies archive/commit/patch/executable hashes
before invoking the native reference. It requires a new artifact directory and
records failed comparisons with a nonzero exit status.

Final artifacts: `I:\safetensors2GGUF-flux-q6-path-provenance-2026-10-15` contains
24 small GGUF files (eight inputs plus native/Python outputs) and eight native
logs. The report records SHA256 values for each input and both output files.
Earlier diagnostic directories are retained separately. No installed model was
modified or converted in these file-level runs. The earlier complete
weight-tensor numerical tests remain documented in the
[shared Q6_K pilot](shared-q6-pilot.md).

Unit coverage also checks custom metadata arrays/types, custom alignment,
unchanged protected/source bytes, cancellation, existing output, invalid source,
architecture/namespace/rank rejection and refusal to requantize a Q6_K input.
CI replays retained results without requiring the external native executable.

## Reproduce

```bash
# Explicit research path; source must already be a supported prepared FLUX GGUF.
python -m scripts.requantize_flux_q6 --source "I:/prepared-flux-F16.gguf" --output "I:/research-flux-Q6_K.gguf"

# Repeat the small native-versus-Python file comparison into fresh paths.
python -m scripts.check_flux_q6_path --artifacts "I:/new-flux-q6-check" --output new-flux-q6-report.json
```

The conversion prints a JSON report with tensor decisions/checksums. The
comparison uses the retained native build paths in
`data/converter-policy-comparison.json`; rebuild references explicitly if needed.
Neither tool implicitly downloads or updates upstream tools/models.

## Remaining gates

One complete installed FLUX.2 klein Base 9B checkpoint now passes
[full-model numeric validation](flux-q6-full-model-validation.md): all 201 tensor
payloads/types/shapes and metadata match the native reference after common float
preparation. Loader behavior, image quality, LoRA/offload behavior and isolated
peak memory/throughput still need separate evidence.
This change authorizes no renders. BF16 input, other architectures, shape repair,
text-encoder metadata/policies and other quantization formats remain separate
work. Keep the tested native production path available.
