# Native reference block comparison

Date: 2026-10-15. Research branch: `research/unified-converter-inventory`.
Evidence: [machine-readable report](../data/converter-native-comparison.json).

## Scope and method

Built three small CPU executables from our verified source archives. The harness
calls unmodified GGML `quantize_row_*_ref` and `dequantize_row_*` functions directly.
It does not run `llama-quantize`, GGUF metadata processing, model selection, S/M
preset policies, importance matrices or inference. Source archives are unchanged;
builds and extracted regular files are retained locally under ignored
`build/converter-reference/run-*`. Nonessential archive symlinks are not copied.

| Source | Full revision |
| --- | --- |
| b3962 | `c8c07d658a6cefc5a50cfdf6be7d726503612303` |
| Installed text reference | `687e7789271ec1276e3470f158428e11a4f80b6f` |
| Newer frozen text reference | `8bb3cf0d216a5ae48360179f2931daf284be4d01` |

The City96 patch is not applied: it changes model/policy, shape handling and
GGML tensor-name capacity, not the `ggml-quants.c` block encoders probed here.
These are the original b3962 math kernels shared by that diffusion path; this is
not an execution test of the complete patched converter.

Tested on Windows x64 with MSVC 19.51.36260.0 and CMake 4.3.1-msvc1, Release builds.
Optional GPU/BLAS/OpenMP backends and explicit AVX/AVX2/FMA/F16C options are off;
no timing claim is made. The report records exact source/archive hashes,
executable hashes, compiler identity, cache flags, build commands and harness
hashes. Binary hashes identify the tested files, not a guarantee of reproducible
binary bytes across rebuilds (timestamps/toolchains may differ).

## Results

The same 12 float32 vectors as the Python probe phase were tested in ten formats:
Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K and Q6_K. The direct Q3_K
block format is distinct from the application's Q3_K_M model policy.

- **360 cases:** 120 per source, comprising 95 encoded and 25 rejected row-width
  cases per source. Row rejection is an explicit harness guard, not an upstream
  fallback; those policy paths are intentionally bypassed.
- **All 95 encoded cases produce identical bytes across all three revisions.**
  Each encoding is repeated to check determinism. This covers these exact input
  vectors and reference functions, not all possible tensors or optimized kernels.
- Native dequantization and the installed gguf-py decoder agree exactly on every
  encoded case (maximum observed difference: zero).
- Seven additional real protocol rejection checks pass on each binary: unknown
  format, negative/excessive dimensions, incompatible rows, truncated/extra input
  and non-finite floats. Windows binary-pipe mode preserves payload bytes.

On the positive ramp from 1 to 2, all three native references give the following:

| Format | Native RMSE | Frozen Molbal RMSE |
| --- | --- | --- |
| Q2_K | 0.039040 | Not implemented |
| Q3_K | 0.019269 | Not implemented |
| Q4_K | 0.022988 | 1.468111 |
| Q5_K | 0.013878 | 1.468125 |
| Q6_K | 0.009717 | 0.014219 |

Native Q4_K reconstructs approximately 1.025 to 2.001; native Q5_K reconstructs
approximately 1.010 to 1.974. The native references do not exhibit the positive
ramp saturation observed in frozen Molbal. Molbal Q6_K also produces different
bytes on this probe; valid layout does not imply the same scale fitting or
rounding. Details and the affine-clipping explanation are in
[the initial analysis](converter-reference-analysis.md).

## Interpretation

These measurements support maintaining one shared mathematical core for the
tested standard/K block formats. They do not justify merging model adapters,
precision policies or runtime loaders: those are separate layers. No production
backend has been changed or added. Do not adopt Molbal's Q4_K/Q5_K math unchanged;
use the native references as the next implementation/validation baseline.

No general quality ranking follows from these tiny tensors. Importance-weighted
encoders, SIMD/GPU paths, unusual magnitudes/shapes and real models need separate
evidence. The existing Krea format allowlist remains unchanged.

## Reproduce offline

Install the project's locked Python environment plus a local C/C++ compiler and
CMake. No model weights or upstream checkout/download is needed. On Windows use
a Visual Studio CMake generator or a developer shell; if CMake is not on PATH,
pass its executable with `--cmake`.

```bash
python -m scripts.compare_native_reference_vectors --output data/converter-native-comparison.json
# Or explicitly select the locally installed CMake executable:
python -m scripts.compare_native_reference_vectors --cmake /path/to/cmake --output data/converter-native-comparison.json
```

Each run verifies snapshots and creates fresh build directories; it never
overwrites an existing source/build tree. Build stdout/stderr stay in
`build-0.log` and `build-1.log` alongside each extracted tree. A failing build or
unexpected numerical/protocol error aborts without publishing a new report.
Windows build execution was verified; Linux build/runtime is not yet verified.
These research probes do not complete production offline dependency packaging.

Next: add model/preset decision vectors and design a shared encoder against the
native baseline. Preserve the working production routes until those gates pass.
