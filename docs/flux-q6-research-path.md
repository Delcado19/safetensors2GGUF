# FLUX Q6_K research file path

Date: 2026-10-15. Branch: `research/unified-converter-inventory`.
Implementation: [`scripts/requantize_flux_q6.py`](../scripts/requantize_flux_q6.py).
Evidence: [`data/flux-q6-path-comparison.json`](../data/flux-q6-path-comparison.json).

## Scope

The first architecture-specific integration connects the shared NumPy Q6_K
encoder to a real FLUX GGUF quantizer stage. It accepts an already prepared,
single-file little-endian F16/F32 FLUX GGUF, preserves metadata/tensor order and
writes a mixed-precision Q6_K GGUF atomically. The Workbench/API now exposes it
as an explicit experimental backend for FLUX safetensors and Q6_K/Q5_K_M. Native
remains the default; global format registries and text encoders are unchanged.

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
  and source-alias rejection protect output publication. Existing output is
  refused by default; explicit `--overwrite` / `overwrite=True` atomically replaces
  it only after validation. Failures/cancellation preserve the previous output. Reader
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

## Workbench/API integration

Choose diffusion GGUF, Q6_K or Q5_K_M, then **Advanced settings > GGUF converter >
Python - FLUX only (experimental)**. The executable and thread controls are
disabled for this backend; their values are not submitted. The existing
**Keep intermediate F16 file** and overwrite choices still apply. Other formats,
containers and text encoders use their normal routes.

API clients explicitly send `"gguf_backend": "python_flux"` in `/api/inspect`
and `/api/jobs`. Omitting the field selects `native`. The API rejects Python
requests for other architectures/formats/containers, non-safetensors sources,
non-default precision profiles or native executable/thread settings before work.
The service revalidates architecture independently of the API. No implicit
native retry follows Python errors or cancellation.

Preparation reuses `convert_file(..., target_quant=F16, apply_unsqueeze=False)`.
Output publication uses the checked atomic Python writer; a failed/cancelled
job preserves any previous destination. A unique intermediate lives beside the
output and is removed on success/failure unless the user asked to retain it.
Cleanup errors are logged rather than hiding the conversion result. No new
full-model conversion or render was performed for the integration itself.

Small end-to-end tests exercise real preparation/writing and an API job without
native discovery/invocation, protected F32 values, cancellation, disk/write
failure, keep-intermediate behavior, overwrite and rejection boundaries. The
native default and existing Krea/shape-fix paths retain their own tests.

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
preparation. A user-generated PNG pair additionally confirms generation through
`UnetLoaderGGUF` with two enabled LoRAs and matching saved settings against the
original checkpoint; main-subject appearance remains close, while background
details differ. See the full-model guide for exact settings and limits. Targeted
offload, Linux and isolated peak memory/throughput still need separate evidence.
This change authorizes no renders. BF16 input, other architectures, shape repair,
text-encoder metadata/policies and other quantization formats require separate
validation. The [Q5_K_M research file follow-up](shared-q5-pilot.md#flux-q5_k_m-file-policy-follow-up)
now includes full-checkpoint numeric and user-supplied inference evidence and
is available through the experimental backend. The [Q4 file presets](flux-q4-research-path.md)
remain schematic research exports without application dispatch.
Keep the tested native production path available.

The legacy API identifier `python_flux_q6` retains Q6_K-only behavior. The shared
`python_flux` identifier supports both Q6_K and Q5_K_M. Q5 uses the validated
mixed writer and shows an inline warning about observed pose/composition drift
versus the original with two LoRAs. Native remains the default; other presets,
architectures and custom precision profiles are rejected before preparation.
