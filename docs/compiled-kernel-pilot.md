# Direct compiled CPU kernel pilot

Date: 2026-10-16. Research only; normal conversion and backend defaults are unchanged.

## Result

Python can keep orchestration while invoking the preserved b3962 Q4_K/Q5_K/Q6_K
reference kernels directly through a small C ABI and standard-library `ctypes`.
This is compiled upstream math, not faster Python arithmetic or a new algorithm.
It avoids per-chunk executable launches and introduces no Python dependency.
The application does not load this library yet.

`data/compiled-kernel-pilot.json` retains the library/source hashes, original
archive/commit, build commands, compiler/cache, ten real C boundary rejection
checks and 166 golden inputs per format. All **498 golden comparisons** pass,
including 21 small read-only samples from seven installed model families.
The benchmark slice also matches the independent frozen b3962 executable bytes.

One 1,048,576-value prepared FLUX slice, seven alternating repetitions, one CPU
worker, generic build without AVX/FMA or GPU acceleration:

| Format | Current NumPy | Python + bounded C calls | Same C kernel, one direct call |
|---|---:|---:|---:|
| Q4_K | 136.44 ms | 43.01 ms | 42.64 ms |
| Q5_K | 111.67 ms | 37.19 ms | 36.59 ms |
| Q6_K | 82.14 ms | 26.40 ms | 26.26 ms |

The bounded path uses 256 blocks/call and includes Python finite-input checks,
output allocation and payload hashing. The direct baseline uses one call with
preallocated output and omits the additional Python input scan; both perform
the C finite-input scan and payload hashing. Median bounded-call overhead is
about 0.6-1.6% over this direct baseline, and throughput is about 3.0-3.2 times
current NumPy on this slice. These are kernel measurements, not a full-model
conversion, file preparation, disk I/O, readback or peak-RAM benchmark.

## Reproduce the build and checks

The later [offline package builder](compiled-kernel-package.md) automates source
verification, compilation, manifest creation and native golden self-tests. It
also provides explicit package selection in the research writer. The manual
build recipe below remains the original pilot reproducer.

Use the existing pinned archive, CMake and a C/C++ compiler. No network fetch is
required. From the repository root, run this Python snippet with the project
environment; choose a fresh build directory if the example already exists:

```python
import hashlib, json, shutil, subprocess
from pathlib import Path
from scripts.inventory_converter_sources import verify_references
from scripts.compare_native_reference_vectors import extract_source

work = Path('build/kernel-reproduction').resolve()
work.mkdir(exist_ok=False)
source = next(e for e in verify_references()['sources'] if e['id'] == 'llama-b3962')
extract_source(source, work / 'source')
cmake = shutil.which('cmake')  # Or use the installed Visual Studio CMake executable.
assert cmake, 'Install/provide CMake and a C/C++ compiler'
commands = [
    [cmake, '-S', str(Path('tools/converter-kernel').resolve()), '-B', str(work / 'build'),
     '-DREFERENCE_SOURCE=' + str(work / 'source'), '-DCMAKE_BUILD_TYPE=Release'],
    [cmake, '--build', str(work / 'build'), '--config', 'Release',
     '--target', 'converter-kernel', '-j', '2'],
]
for command in commands:
    subprocess.run(command, check=True,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
library = next(p for p in (work / 'build').rglob('*converter-kernel*')
               if p.suffix in ('.dll', '.so', '.dylib'))
digest = hashlib.sha256(library.read_bytes()).hexdigest()
record = dict(commit=source['commit'], archive_sha256=source['sha256'],
              library_sha256=digest, commands=commands,
              compiler=next((work / 'build/CMakeFiles').glob('*/CMakeCCompiler.cmake')).read_text(),
              cache=(work / 'build/CMakeCache.txt').read_text())
(work / 'provenance.json').write_text(json.dumps(record, indent=2), encoding='utf8')
print(library, digest, work / 'provenance.json')
```

Pass those three values explicitly (the source is an existing prepared float
GGUF, not the original safetensors checkpoint):

```text
python -m scripts.check_compiled_kernel --library LIBRARY --sha256 SHA256 --build-record PROVENANCE_JSON --source PREPARED_FLOAT_GGUF --models-root COMFY_MODELS --output FRESH_REPORT_JSON
```

The checker also needs the existing independently verified native references
described in `docs/converter-native-reference-comparison.md`; it does not build/fetch them.
`--sha256` identifies a library you trust; matching a hash is not a sandbox for
arbitrary native code. Original models are sampled read-only.

## Boundaries and next gate

The C function accepts Q4/Q5/Q6, finite nonempty F32 input in multiples of 256,
at most 1,048,576 values/call, and exact caller-owned output capacity. It rejects
invalid arguments before writing. Callers must supply valid, nonoverlapping
allocations and immutable input. The Python checker copies strided inputs to
contiguous bounded arrays. Extreme finite F32 arithmetic is not qualified.

Only Windows/MSVC has been built and executed here. Linux/macOS build support is
written but untested. There is no automatic compilation, DLL search, silent
fallback, new application setting or binary distribution. No model dispatch or
loader behavior has changed. This reuses archived b3962 kernels rather than
merging the three converters or choosing a final implementation language.

The explicit trusted-kernel writer adapter below preserves the existing bounded
workflow. The [full Q4_K_M follow-up](flux-compiled-kernel-full-model-validation.md)
now records 201 native-equal tensor payloads, a 461.685 s write/readback/hash run,
32 actual DLL fixture comparisons and handled-error/atomic-output checks.
Its Q5_K_M/Q6_K follow-up also matches all 201 native payloads per format and
earlier Python whole-file hashes, with 410.520 s and 315.509 s write/readback/hash
runs. Other architectures, full Q4_K_S and reproducible platform packaging remain
gates before Workbench exposure.
Parallel CPU execution and GPU/Tensor Core acceleration remain separate work.

## Explicit research writer integration

The research FLUX writer can now use the library through an explicit selection:

```text
python -m scripts.requantize_flux_q6 --source PREPARED_FLOAT_GGUF --output NEW_GGUF --preset Q4_K_M --kernel-library LIBRARY --kernel-sha256 SHA256
```

Both library arguments are mandatory together. The checksum is verified before
loading; the output cannot alias the library. Without these arguments the writer
keeps its NumPy behavior. The Workbench/API still uses its existing backends.
There is no automatic build, library discovery or fallback on a native error.

The existing writer owns metadata, mixed-preset rules, protected float tensors,
source-change detection, per-tensor readback hashes and atomic publication. Only
the block encoding is replaced. A call receives at most 65,536 F32 values; the
writer checks cancellation before and after it, and validates returned packed
dtype, contiguity and length before writing. Kernel errors abort rather than
publishing a partial model. Full-model evidence is recorded separately in the
follow-up above; the slice measurements remain distinct from end-to-end timing.
