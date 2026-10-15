# Offline research CPU kernel packages

Date: 2026-10-16. Explicit research tooling; no Workbench backend/default changes.

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

Before publication the builder executes ten C boundary checks and **435 native
golden comparisons**: 145 synthetic vectors for each of Q4_K/Q5_K/Q6_K. No
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
No selection still uses NumPy; no build/search/fallback occurs during conversion.

## Actual validation and limits

`data/compiled-kernel-package-validation.json` records a fresh Windows x86_64
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
but untested. System C/C++ runtime dependencies and clean-machine installation
are not qualified; no release binary is shipped or downloaded. Repeating the
build is supported, but bit-identical binaries are not promised (compiler/linker
timestamps and paths may differ). Build and package files remain in ignored
local directories; tracked source archives and build instructions remain the
project-controlled source of truth. Workbench integration and distribution of
validated platform binaries remain open.
