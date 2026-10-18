# Offline research CPU kernel packages

Updated: 2026-10-18. The Workbench/API now exposes an opt-in `compiled_flux`
backend. Native remains the default.

## Ready-to-use Windows integration

Select **Bundled CPU kernel - FLUX only (experimental)** for FLUX safetensors
and Q2_K, Q3_K_M, Q4_K_S, Q4_K_M, Q5_K_M or Q6_K. Windows x86_64 ships the verified package
under `third_party/cpu-kernels/win32-x86_64`; no compiler, CMake, upstream fetch
or external quantizer is used at runtime. Distributions must retain this folder,
the pinned reference identity/license, adapter sources and golden data used by
the existing package verifier. Package building below is for maintainers only.

Before float preparation the backend validates package integrity/platform and
replays native self-tests. Missing, incompatible or corrupt packages fail
explicitly; there is no automatic build or fallback. It reuses F16 preparation,
progress/cancellation, overwrite protection, atomic output and intermediate cleanup.
API clients select `gguf_backend: "compiled_flux"`; thread/executable overrides
are rejected. Existing Python backend contracts remain unchanged.

Real DLL API fixtures for the original four presets match the NumPy writer
byte-for-byte. Q2_K/Q3_K_M API fixtures match independent pinned native tensor
payloads. All six K presets match native payloads, tensor layout and metadata
across 48 F16/F32 fixtures, including reversed order and incompatible V rows.
F32/F16/BF16/Q8_0 retain their existing compiler-free direct writer; they do not
need this K-quant backend. Q3_K_M promotes the first two eligible V tensors to
Q5_K and later V, fused QKV and FFN-down tensors to Q4_K; incompatible V rows
consume the ordered counter before falling back to F16, matching City96.

The shipped Windows DLL statically links the MSVC runtime. Its dependency table
contains no Visual C++ or Universal CRT runtime DLLs; installing a compiler or
VC runtime redistributable is not part of kernel use. This static dependency
check does not replace clean-machine application qualification.
Full-model native-payload evidence covers Q4_K_M, Q5_K_M and Q6_K on the documented
FLUX checkpoint; Q2_K application/readback and supplied inference evidence are
recorded below. Linux/macOS packages, clean-machine runtime qualification and an overall
application installer remain open. This integration alone does not make every
other converter backend ready to use.

## Build and use

The package builder verifies and extracts the complete pinned b3962 source
archive already tracked under `third_party/converter_references`, then builds
the existing generic CPU C ABI. It never fetches upstream code. It requires the
project Python environment, CMake and a C/C++ compiler already installed.
Use fresh, disjoint package and work directories; neither is overwritten.

```text
uv run --offline python -m scripts.compiled_kernel_package --output build/kernel-package --work build/kernel-build
```

If CMake is not on PATH, add `--cmake` with its executable path. The installed
Windows validation used Visual Studio 18 Community's bundled CMake at:

```text
C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
```

Build logs, extracted sources and CMake output remain under the work directory.
The package contains:

- `converter-kernel.dll` on Windows, or the platform `.so`/`.dylib` name.
- `kernel.c`, `CMakeLists.txt` and the upstream MIT `LICENSE`.
- `kernel.json`: host system/machine/pointer width, pinned source/archive identity,
  every packaged file's SHA256, compiler/cache/build commands and self-test record.

Before publication the builder executes ten C boundary checks and **725 native
golden comparisons**: 145 synthetic vectors for each of Q2_K through Q6_K. No
installed model or external reference executable is needed for these self-tests.
The golden data are the project's previously measured native reference reports.
The manifest is written after successful build/self-tests/file copies; an
interrupted or failed copy is not a usable package. Existing output is retained
and failed build logs are preserved, without implicit repair or deletion.

Verify an explicitly trusted package without rebuilding:

```text
uv run --offline python -m scripts.compiled_kernel_package --verify --output build/kernel-package
```

This checks host compatibility, source/adapter/license identity and file hashes
before loading native code, then replays all self-tests. Python `-O` is rejected
because it would disable the assertion-based C guard tests. The checks need the
matching repository, golden reports and pinned source manifest; the package is
not a standalone installer or a signed distribution. Checksums establish
integrity, not publisher authenticity: only select packages you trust.

Use the package with the bounded research writer:

```text
uv run --offline python -m scripts.requantize_flux_q6 --source PREPARED_FLOAT_GGUF --output NEW_GGUF --preset Q5_K_M --kernel-package build/kernel-package
```

`--kernel-package` replaces the manual library-path/hash pair. The existing
`--kernel-library` plus `--kernel-sha256` route remains available, and mixing the
two selections is rejected. An output cannot alias any package file or manifest.
Successful JSON output identifies the selected package and library SHA256.
Without a kernel selection Q4/Q5/Q6 still use NumPy; Q2/Q3 reject missing kernels; no build/search/fallback occurs during conversion.

## Actual validation and limits

The historical `data/compiled-kernel-package-validation.json` records a fresh Windows x86_64
offline build, 435 passing goldens and all ten guards. Actual subprocess CLI
conversions for Q6_K/Q5_K_M/Q4_K_S/Q4_K_M match complete NumPy files, covering
mixed types, reversed tensor order and an early incompatible V-row fallback.
A modified library copy is rejected before loading. Unit tests additionally
reject incompatible platforms, source/schema/file-list changes, adapter/license
changes, existing/overlapping build destinations and conflicting CLI arguments.

The C math and generic CPU flags are unchanged from the
[full-model validation](flux-compiled-kernel-full-model-validation.md). Its
reports retain hashes of the exact earlier writer version. Packaging does not
change weights; the large models were not converted again for this CLI change.

Only Windows/MSVC has been built/executed here. Linux/macOS paths are written
but untested. The current Windows DLL is shipped with a static C runtime; clean-machine
application installation and non-Windows runtime dependencies remain unqualified. Repeating the
build is supported, but bit-identical binaries are not promised (compiler/linker
timestamps and paths may differ). Maintainer build files remain in ignored local directories; the Windows runtime
package is tracked under `third_party/cpu-kernels`; tracked source archives and build instructions remain the
project-controlled source of truth. Workbench integration is available for the
shipped Windows backend; other platform binaries remain open.


## Extended K-preset evidence (2026-10-18)

`data/q2-q3-kernel-goldens.json` records 290 independent b3962 block encodings
for the two new block types. `data/compiled-kernel-levels-validation.json` records
the new package manifest, 48 native-equal file comparisons (1,192 tensor payloads),
two prepared API fixtures, DLL imports and current source hashes. Earlier reports
keep their historical hashes and timing scopes. The schematic checks did not
reconvert a large checkpoint; the subsequent application run is recorded below.
No new image quality claim is made for Q2_K/Q3_K_M.

Repeat file validation with the verified patched native-reference manifest:

```text
uv run --offline python -m scripts.check_flux_q6_path --preset Q3_K_M --kernel-package third_party/cpu-kernels/win32-x86_64 --references VERIFIED_REFERENCES.json --artifacts NEW_ARTIFACT_DIRECTORY --output NEW_REPORT.json
```

Repeat for each of Q2_K, Q3_K_M, Q4_K_S, Q4_K_M, Q5_K_M and Q6_K. The independent
reference build is maintainer tooling in `scripts.compare_converter_policies`;
normal conversion never builds or fetches it. Old packages whose adapter hashes
predate this extension are rejected rather than silently used as new packages.


## Full FLUX.2 Q2_K application conversion (2026-10-18)

The integrated application pipeline (`compiled_flux`, Q2_K) successfully converted
installed FLUX.2 klein Base 9B safetensors without compilation or downloads.
The source is now under `G:\ComfyUI-Easy-Install\ComfyUI\models\diffusion_models\Flux.2 Klein 9B-base`.
The historical result was `I:\safetensors2GGUF-flux2-q2-bundled-2026-10-18\flux-2-klein-base-9b-Q2_K.gguf`
(deleted in the approved cleanup below; its logs and JSON remain):
3,356,285,248 bytes, 201 tensors (115 Q2_K, 86 protected F32). Pipeline time was
662.291 s, including F16 preparation, quantization and internal readback; original
source hashing before the pipeline and an additional output hash afterward are
outside that timing. Source size/mtime remained unchanged and the intermediate
was cleaned up. `data/flux2-q2-bundled-conversion.json` retains paths, source/output
SHA256, kernel identity, current code hashes and verification scope; the full log
and runtime status remain in the historical output directory. This is successful conversion and
internal validation, not independent full-model native parity or inference proof.
Subsequent user-supplied ComfyUI evidence is recorded below; broader Q2 quality
coverage remains open.


## User-supplied Q2_K ComfyUI comparison (2026-10-18)

Two supplied 768x1024 RGB PNGs retain valid chunk CRCs and full workflows/prompts
in `data/flux-q2-user-render-comparison.json`. Both use seed 572241167616859,
34 Flux2Scheduler steps, er_sde, CFG 4, matching positive/negative prompts, VAE
and Qwen3-8B text encoder. Both LoRAs are disabled. The original model feeds
node 113 from UNETLoader 108. Q2 feeds it through switch 116 (active B) from
UnetLoaderGGUF 118; the unused original loader remaining in the saved graph does
not feed that active model path. The installed GGUF has SHA256
`b35a4fce26a4fdeafdadf40cd28128f1fc494f3c01bb5791e1ba98d8e47c9411`, exactly the
converted output. No new render was initiated by the agent.

This supports successful GGUF inference in the supplied setup, with **Visible
drift** versus the original: body stance/leg placement, face appearance and car
camera angle/composition change materially. Q2 looks less natural in the face
and coarser in skin/leg shading in this pair; the main scene remains recognizable
without gross image collapse. Differences exceed minor lighting or texture
variation. This is one no-LoRA pair, not broad quality parity or evidence of an
encoder defect; precision and loader differ together. Native Q2 render parity,
LoRA, forced offload, memory/throughput and other prompts remain untested.


## Supplied distilled-model Q2 export comparison (2026-10-18)

The user supplied three renders of `snofsSexNudesAndOtherFunStuff_distilledV12Fp8`:
original, Q2_K and a separately named Q2_K-Python export. All 201 ordered tensors
of the two installed GGUF files match byte-for-byte, including names/types/shapes;
interpreted metadata and effective alignment also match. Whole-file SHA256 values
differ, so container-byte identity is not claimed. Filenames alone do not establish
independent native/export-command provenance. Complete tensor hashes and saved PNG
workflows are in `data/snofs-q2-user-render-comparison.json`.

The PNGs use seeds 312380090853788 (original), 312380090853789 (Q2_K) and
312380090853790 (Q2_K-Python). Other saved sampling settings agree: 5-step euler/simple
base plus 1.5x bicubic latent upscale and 5-step refinement at denoise 0.39, CFG 1,
512x960 base/768x1440 output, same prompts/VAE/text encoder and disabled LoRA.
Both Q2 graphs actively select UnetLoaderGGUF through switch B. All PNG chunk CRCs
validate and decoded pixel hashes differ. The images show coherent portraits
without gross image collapse, but face/hand/composition differences cannot be
attributed to quantization or backend with these different seeds. Repeat with
seed increment/randomization disabled to evaluate image parity. No new render
was started and the original files remain unchanged.


### Same-seed follow-up

The replacement PNG triplet uses seed **312380090853791** for both sampling
passes in all three variants. All other saved sampling, prompt, VAE, encoder
and disabled-LoRA settings agree. The original switch actively selects UNETLoader
(branch A); both Q2 variants select their respective UnetLoaderGGUF (branch B).
Direct comparison of decoded 768x1440 RGB bytes proves the two Q2 renders are
**pixel-identical**. Both installed model SHA256 values still match the earlier
201-tensor comparison. The original render differs visibly in face, hand pose,
clothing/jewelry details and composition, while remaining a coherent portrait.

This resolves the earlier seed mismatch for this supplied follow-up and supports
no additional export/backend drift between these two Q2 files in this setup.
The change from the FP8-source render is shared by both Q2 exports; precision
and loader change together, so a pure quantization-only causal claim remains
out of scope. Full PNG prompts/workflows, pixel hashes and active-model checks
are retained in `data/snofs-q2-same-seed-render-comparison.json`. Earlier evidence
retains its original hashes even though the supplied folders now contain the
replacement files. No new render was started by the agent.


## Test-artifact cleanup (2026-10-18)

At the user's explicit request, 600 generated GGUF/raw Q6 artifacts on `I:` were
deleted, reclaiming 77.73 GiB, including the prepared F16 intermediate. All 357
JSON/log evidence files were retained and verified unchanged by SHA256. Model
artifact paths on `I:` in earlier reports are historical and no longer resolve;
the original FLUX.2 source on `G:` remains present. Future performance work must
regenerate float preparation or use another explicitly verified input. The exact
deletion audit is `data/i-drive-cleanup-2026-10-18.json`.


## Bounded parallel writer and phase timings (2026-10-18)

The explicit compiled path now runs up to `min(8, os.cpu_count())` independent
65,536-value C calls concurrently. Only one batch is submitted at a time (at
most eight 256 KiB F32 inputs); results are consumed in input order. Small
single-window tensors avoid thread dispatch. Every callback result retains its
dtype/length/contiguity validation; exceptions/cancellation still abort atomic
publication and all workers are joined before cleanup. Raw research callbacks
must tolerate concurrent calls. The NumPy path remains serial. No DLL, native
math, preset rules, intermediate precision, backend default or UI option changed.

Application logs now separate kernel verification/self-tests, F16 preparation
and writing, encoding/output writing, tensor readback, and whole-file hashing.
Encoding and output writing remain one combined measurement; these logs do not
claim per-system-call profiling. At this benchmark stage the roughly 19.9 GB
full-model intermediate and its spool copy still existed; the direct-spool
follow-up below removes them when Keep intermediate is disabled.

`data/compiled-kernel-performance-validation.json` records one 1024x4096 real
FLUX.2 klein Base 9B matrix slice plus a tiny protected input matrix. On this
32-logical-CPU Windows host, eight workers improve the complete **slice writer**
(encode/write/readback/hash) as follows; imports, source-slice reading, float
preparation and kernel self-tests are outside these writer medians:

| Preset | Serial median (ms) | Parallel median (ms) | Ratio |
| --- | ---: | ---: | ---: |
| Q2_K | 182.05 | 67.66 | 2.69x |
| Q3_K_M | 65.14 | 42.46 | 1.53x |
| Q4_K_S | 207.39 | 72.64 | 2.86x |
| Q4_K_M | 204.01 | 74.56 | 2.74x |
| Q5_K_M | 177.58 | 69.53 | 2.55x |
| Q6_K | 143.18 | 62.36 | 2.30x |

Each mode has one discarded warmup and three measured alternating repetitions.
Serial and parallel GGUF files are byte-identical for every preset; their tensor
payloads/layouts and interpreted metadata match the independently verified
patched native converter. Application phase logs are also retained. The earlier
raw C-call worker-count pilot is `data/compiled-kernel-parallel-slice.json`; its
kernel-only timings are a different scope. Neither measurement predicts total
full-checkpoint time, peak model memory or cold-drive performance. No large
conversion was repeated for this slice benchmark and no files were created on
the cleaned `I:` drive during it. The later full conversion is recorded below.

Reproduce with fresh work/report destinations and a verified native manifest:

```text
uv run --offline python -m scripts.benchmark_compiled_writer --source ORIGINAL_FLUX_SAFETENSORS --kernel-package third_party/cpu-kernels/win32-x86_64 --references VERIFIED_REFERENCES.json --work NEW_WORK_DIRECTORY --output NEW_REPORT.json
```

The benchmark reads a small source slice and writes only its schematic files;
these are not inference models. It does not compile, download, or render anything.


## Direct final-payload spooling (2026-10-18)

With **Keep intermediate** disabled, compiled FLUX jobs now reuse `handle_tensors`
float preparation through `CompiledFluxWriter` and encode each prepared tensor
before spooling. FP8/scale reconstruction, clamping, F16/F32 protection, K-preset
rules and bounded C calls remain shared with the prior paths. No complete F16
GGUF is written and read again. Only final quantized/protected payloads are
spooled and copied into the atomic output; this is not a zero-copy writer.
Keeping an intermediate explicitly preserves the previous two-stage route.
Native-default and NumPy routes remain unchanged.

Complete tensor readback and file hashing run in the writer's `close()` before
atomic publication. Cancellation, worker/write/readback failures and source
changes preserve the prior destination and clean the owned spool. Prepared arrays
and one encoded tensor can coexist; full-model peak memory remains unmeasured.

`data/compiled-kernel-direct-preparation.json` records six real-weight slice
comparisons (one warmup and three alternating measurements per route). Direct
and old-path files are byte-identical and match independent native payloads and
metadata. Logical payload writes decrease 65.9-84.8% on this slice, excluding
small headers/padding; this is not measured physical disk traffic. Automated
comparisons cover F32, F16, BF16, raw FP8 and scaled FP8 for all six presets.

```text
uv run --offline --frozen python -m scripts.benchmark_compiled_preparation --source ORIGINAL_FLUX_SAFETENSORS --references VERIFIED_REFERENCES.json --work NEW_WORK_DIRECTORY --output NEW_REPORT.json
```

### Full FLUX.2 klein Base 9B Q2_K follow-up

One complete application conversion took **115.863 s**, including kernel checks,
shared preparation, parallel encoding, final spooling/copy and readback/hash.
Original-source hashing before the pipeline and an additional output hash after
it are outside this timing. The earlier 662.291 s run is historical, not a
controlled alternating benchmark; host load/cache state were not fixed.

The source and entire output SHA256 exactly match the earlier conversion.
The 3,356,285,248-byte output contains the same 115 Q2_K and 86 F32 tensors.
No new ComfyUI render is needed to establish unchanged file bytes. Evidence and
phase logs are retained in `data/flux2-q2-direct-conversion.json`; the new model,
log and status are on `I:\safetensors2GGUF-flux2-q2-direct-2026-10-18`.
No F16 intermediate was created. This new output is separate from the artifacts
removed in the earlier approved cleanup.
