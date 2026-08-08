# Building a City96-patched `llama-quantize`

This tool never bundles a `llama-quantize` binary — Windows users typically get
one for free via **ComfyUI-Easy-Install** (a third-party bundle, not an
official ComfyUI or llama.cpp artifact), but Linux and macOS users must build
their own. This is also the correct path on Windows if you don't use
Easy-Install.

## Why a patched binary is required

Upstream `llama.cpp` release binaries only recognize LLM/text architectures.
K-quant quantization of diffusion-model GGUFs (Flux, SDXL, HunyuanVideo, Wan,
Lumina 2, …) requires `city96/ComfyUI-GGUF`'s `tools/lcpp.patch`, which adds
image-architecture registration, diffusion-aware tensor classification (which
weights may be K-quantized vs. which must stay high precision), and metadata
bypasses so `llama-quantize` doesn't reject a GGUF for missing LLM-only fields
(vocabulary, tokenizer, rope, attention config).

**This patched binary is only for diffusion-model GGUFs.** Do not use it to
quantize LLM/text-encoder GGUFs — use a plain upstream `llama-quantize`
release for those.

## Text-encoder K-quants build automatically — you don't need this guide for them

If you only want K-quants (`Q6_K`…`Q2_K`) for **text-encoder** conversion (the
**Convert Text Encoder → GGUF** tab), you don't need to follow this guide at
all: `text_encoder_convert.ensure_plain_llama_quantize()` builds a **plain,
unpatched** `llama-quantize` automatically the first time it's needed —
from the same `llama.cpp` checkout already auto-cloned for
`convert_hf_to_gguf.py` (see
[Text-Encoder Conversion](../README.md#text-encoder-conversion)), cached
under `.llama.cpp/build-quantize/` afterward. No manual clone, no patch, no
`cmake` commands to type — the tool runs them for you via `cmake -B` /
`cmake --build`.

The **only** thing you need to provide yourself is `cmake` and a C++
compiler on `PATH` — the exact same toolchain as the "Prerequisites" and
per-OS install commands below (Linux `apt install build-essential cmake`,
macOS `xcode-select --install` + `brew install cmake`, Windows MSYS2
`pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake`, or a Visual Studio
"Desktop development with C++" install). If either is missing, the tool
raises a clear error naming what's missing rather than silently failing —
install it and retry; no manual build step is otherwise required. Skip the
"Clone llama.cpp"/"Get and apply lcpp.patch"/"Build llama-quantize" steps
below entirely — those only apply to the diffusion-model patched binary.

## Prerequisites (all platforms)

- `git`
- `cmake` (3.14+)
- A C++17-capable compiler toolchain (see per-OS section below)

You do **not** need this tool's own Python environment for the build step —
only for running `convert.py`/the GUI afterward.

## 1. Clone llama.cpp and check out the matching tag

The patch is pinned to a specific `llama.cpp` tag. Using a different tag will
likely fail to apply or build.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout tags/b3962
```

## 2. Get and apply `lcpp.patch`

Download the patch from `city96/ComfyUI-GGUF`:

```bash
curl -LO https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/tools/lcpp.patch
```

If you cloned llama.cpp on Windows with `core.autocrlf` enabled, normalize the
patch's line endings first (`git apply` fails on CRLF patches applied to a LF
repo) — a plain Python one-liner does the same thing as city96's
`fix_lines_ending.py`:

```bash
python -c "open('lcpp.patch','wb').write(open('lcpp.patch','rb').read().replace(b'\r\n', b'\n'))"
```

Then apply it from inside the `llama.cpp` checkout:

```bash
git apply ../lcpp.patch
```

## 3. Build `llama-quantize`

### Linux

```bash
sudo apt install build-essential cmake git   # Debian/Ubuntu; use your distro's equivalent
mkdir build
cmake -B build
cmake --build build -j"$(nproc)" --target llama-quantize
```

Binary ends up at `build/bin/llama-quantize`.

### macOS

```bash
xcode-select --install     # Xcode Command Line Tools, if not already present
brew install cmake         # via Homebrew, https://brew.sh
mkdir build
cmake -B build
cmake --build build -j"$(sysctl -n hw.ncpu)" --target llama-quantize
```

Binary ends up at `build/bin/llama-quantize`.

### Windows — Visual Studio (matches city96's documented path)

Requires Visual Studio 2019 or 2022 with the "Desktop development with C++"
workload (Build Tools alone are sufficient — the full IDE is not required).

**VS2019 and other single-toolchain setups:**

```bat
mkdir build
cmake -B build
cmake --build build --config Debug -j10 --target llama-quantize
```

**VS2022 needs two extra steps** (a C++17 standard flag, and a header include
that VS2022's stricter `<chrono>` deprecation check requires):

```bat
mkdir build
cmake -B build -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_FLAGS="-std=c++17"
```

Edit `llama.cpp\common\log.cpp` and insert two lines right after the existing
first line (`#include "log.h"`):

```cpp
#include "log.h"

#define _SILENCE_CXX23_CHRONO_DEPRECATION_WARNING
#include <chrono>
```

Then build:

```bat
cmake --build build --config Debug -j10 --target llama-quantize
```

Binary ends up at `build\bin\Debug\llama-quantize.exe`.

### Windows — MSYS2/MinGW-w64 (no Visual Studio install required)

If you don't want to install Visual Studio Build Tools, `llama.cpp` also
builds under `MSYS2`'s `MinGW-w64` toolchain via CMake's MinGW generator. This
path isn't covered by city96's own instructions above (which assume MSVC) —
if it fails to apply/build against tag `b3962`, fall back to the MSVC path.

1. Install [MSYS2](https://www.msys2.org/), then from an **MSYS2 MinGW64**
   shell, install just the compiler (not the `-toolchain` meta-package, which
   also pulls in gdb, extra binutils, and other tools you don't need here)
   via `pacman` — MSYS2's package manager, adopted from Arch Linux but
   running natively on Windows here, not a sign you're in the wrong OS:
   ```bash
   pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake git
   ```
2. Clone, check out the tag, and apply the patch exactly as in steps 1-2
   above, from the MinGW64 shell.
3. Build:
   ```bash
   mkdir build
   cmake -B build -G "MinGW Makefiles"
   cmake --build build -j"$(nproc)" --target llama-quantize
   ```

Binary ends up at `build\bin\llama-quantize.exe`.

## 4. Point this tool at the binary

The GUI's **Advanced** section has a Browse button for the `llama-quantize`
path. For the CLI, or to make the GUI auto-detect it, set:

```bash
export LLAMA_QUANTIZE_PATH=/path/to/build/bin/llama-quantize   # Linux/macOS
set LLAMA_QUANTIZE_PATH=C:\path\to\build\bin\Debug\llama-quantize.exe  # Windows
```

Placing the binary on your system `PATH` also works — `find_exe()` in
`quantize.py` checks `PATH` as a fallback (see
[llama-quantize Sources](../README.md#llama-quantize-sources) in the main
README for the full discovery order).

## Caveats (from city96's own notes)

- **Never use this patched binary on LLM/text GGUFs.** It's tuned for
  diffusion-model tensor layouts.
- **Never quantize SDXL / SD1.x / other Conv2D-heavy checkpoints directly.**
  Extract the UNet first (this tool's `component_extract.py` / the **Extract
  Components** GUI tab does this).
- HunyuanVideo/Wan 5D tensors still need the separate
  `fix_5d_tensors.py` pass after quantization — the Web UI chains this
  automatically; the CLI workflow is documented under
  [5D Tensor Post-processing](../README.md#5d-tensor-post-processing-hunyuanvideo--wan)
  in the main README.
