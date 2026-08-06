# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Fix
- `safetensors_quant_nvfp4.py`: Guard zero/near-zero blocks against NaN division. Original code relied on implicit `argmin(NaN)→0, _KVALUES[0]==0` coincidence; replaced with explicit `torch.where` guard detecting post-cast zero block scales and deliberately zeroing normalized values for affected blocks. Prevents silent correctness bugs if PyTorch's NaN-handling behavior changes across versions/backends. Added `test_zero_block_no_nan_and_reconstructs_to_zero` to cover all-zero and near-zero tensors.

### Change
- `convert.py`: Stream safetensors tensors one at a time instead of loading the full state-dict into RAM. Replaces `safetensors.torch.load_file()` with a new `_LazyStateDict` wrapper around `safe_open()` that materializes tensors on demand via `get_tensor()`. Required to convert checkpoints larger than physical RAM (e.g. the 19 GiB FP8 `Qwen-Edit-abliterated-4step-v1.safetensors` on a 16 GiB system, which previously crashed mid-conversion with `KERNEL_DATA_INPAGE_ERROR` and segfaults caused by excessive Windows paging). Source dtype detection now reads dtypes from the safetensors header instead of iterating `.values()`, so dominant-dtype detection also stays out of RAM. `.ckpt` / `.pt` / `.bin` / `.pth` sources keep the existing eager path because `torch.load` has no streaming API.
- `convert.py`: `strip_prefix` refactored to share its prefix rules with the lazy path via a new `_build_key_map(keys)` helper. Public API and behaviour for eager dicts unchanged.
- `convert.py`: `GGUFWriter` now runs with `use_temp_file=True` so converted tensors spill to a `SpooledTemporaryFile` (256 MiB RAM buffer, remainder on disk in `%TEMP%`) instead of accumulating every F16/BF16 tensor in `self.tensors`. Without this, lazy-streaming on the read side only deferred the OOM: the Qwen-Edit 19 GiB FP8 conversion progressed to tensor 1384/1933 (71%) before crashing because the writer had buffered roughly 27 GiB of F16-equivalent tensor data in RAM. Pairs with `_LazyStateDict` to keep peak RAM bounded end-to-end.

### Add
- `safetensors_quant.py`: Quantized Safetensors → Safetensors output backend with dtype registry (F16/F16-mixed variants) and per-tensor quantization. Reuses `convert._quant_type_for`'s high-precision-tensor rule so mixed-precision safetensors output exactly matches existing GGUF mixed-precision behavior. Foundation for Task 2–3 (FP8/NVFP4 variants) and Tasks 4–5 (convert_safetensors.py and GUI integration).
- `tests/test_safetensors_quant.py`: Coverage for safetensors dtype registry, high-precision-tensor detection, and F16/F16-mixed quantization
- `safetensors_quant_nvfp4.py`: NVFP4 block-scaled 4-bit quantization for safetensors output (uint8-packed with per-16-element block scaling and global scale). Reuses `gguf.quants.NVFP4`'s E2M1 codebook table for bit-compatibility with ComfyUI's native NVFP4 loader (format reference: city96/ComfyUI-GGUF `comfy/quant_ops.py` QUANT_ALGOS["nvfp4"]).
- `tests/test_safetensors_quant_nvfp4.py`: Coverage for NVFP4 quantization (packing, scaling, dtype outputs, error cases, and round-trip reconstruction accuracy)
- `models/architectures.py`: `ModelQwenImage` — Qwen-Image and Qwen-Image-Edit DiT support (incl. 2509 multi-image edit variant and abliterated forks like `jiangchengchengNLP/Qwen-Edit-2509-abliterated`). Placed before `ModelFlux` / `ModelSD3` in `arch_list` because Qwen-Image shares `transformer_blocks.0.attn.norm_added_k.weight` and `add_q_proj.weight` with those architectures, which would otherwise trigger their reference-format `keys_banned` guard and abort conversion with "Model architecture not allowed for conversion. Use diffusers format, not reference format." Detection keys mirror upstream ComfyUI-GGUF `tools/convert.py` (`time_text_embed.timestep_embedder.linear_2.weight` + `transformer_blocks.0.attn.norm_added_q.weight` + `transformer_blocks.0.img_mlp.net.0.proj.weight`)
- `component_extract.py`: Analyze embedded SDXL VAE / CLIP-L / CLIP-G components, compare them against local ComfyUI standard files, and export selected components
- `gui.py`: **Extract Components** tab for SDXL component analysis and optional VAE / CLIP-L / CLIP-G export
- `tests/test_component_extract.py`: Coverage for SDXL component analysis, CLIP-G Q/K/V splitting, and `text_projection` transposition
- `README.md` / `docs/architecture.md`: Document the possible future Text Encoder to GGUF workflow, planned encoder-family coverage, and primary SDXL / OpenCLIP / CLIP / Qwen2.5-VL / Qwen3 / T5 / Mistral source links
- `README.md` / `docs/architecture.md`: Record a possible future checkpoint-level GGUF workflow covering component splitting, per-component quantization, mixed safetensors/GGUF layouts, and a ComfyUI loader-node direction
- `README.md` / `docs/architecture.md`: Record City96's `lcpp.patch` as a primary implementation reference for image GGUF architecture registration, tensor quantization policy, metadata bypasses, and future llama.cpp forward-porting
- `convert_safetensors.py`: End-to-end converter from model checkpoint to quantized `.safetensors` file. Sibling to `convert.py`'s GGUF writer: reuses architecture detection and state-dict loading, writes plain quantized safetensors output (no GGUF backend, no 5D side-car export, no shape_fix, no 1D-pad-token unsqueeze). Supports F16/F16-mixed, FP8/FP8-mixed, NVFP4/NVFP4-mixed quantization via `safetensors_quant.py`; includes callbacks for progress reporting, logging, and cancellation.
- `tests/test_convert_safetensors.py`: Coverage for safetensors quantized output (file I/O, dtype matching, FP8 scale tensors, quantization metadata, overwrite protection, and float8-input dtype coercion regression test)
- `gui.py`: **Convert → Safetensors** tab — GUI front-end for `convert_safetensors.convert_to_safetensors`, with source/output path pickers, an `SAFETENSORS_DTYPE_CHOICES`-backed format dropdown, and overwrite protection. The existing GGUF conversion tab is renamed **Convert → GGUF** to disambiguate the two output pipelines.
- `tests/test_gui.py`: Coverage asserting the renamed **Convert → GGUF** tab and new **Convert → Safetensors** tab (with its `SAFETENSORS_DTYPE_CHOICES`-driven "Output format" dropdown) are present in `build_app()`'s Blocks graph

## [0.1.0] - 2026-05-16

### Add
- `.github/workflows/ci.yml`: GitHub Actions CI for Windows-based `uv` dependency sync, pytest, and Ruff lint checks
- `tools/benchmark_llama_quantize.py`: Compare one or more `llama-quantize` binaries on the same GGUF source and report timing/output size
- `gui.py` / `quantize.py`: Conservative `llama-quantize` discovery for ComfyUI Easy-Install / City96-compatible binaries, plus a native file selector for choosing the executable when it is not found automatically
- `fix_pad_tokens.py`: Repair Lumina 2 GGUFs with 1D pad token shapes (`[D]` → `[1, D]`) for ComfyUI compatibility; returns output path for consistent streaming integration
- `gui.py`: **Fix Pad Tokens** tab — GUI front-end for `fix_pad_tokens.py`
- `gui.py`: K-quant pipeline auto-applies `fix_pad_tokens` after `llama-quantize` for architectures with `keys_unsqueeze` — step count and progress bar updated accordingly
- `models/architectures.py`: `ModelTemplate.keys_unsqueeze` — 1D tensor names to unsqueeze before writing; set to `["x_pad_token", "cap_pad_token"]` on `ModelLumina2`
- `models/architectures.py`: `ModelLumina2.keys_hiprec = ["x_pad_token", "cap_pad_token"]` — forces F32 storage for these BF16 pad tokens; prevents the size-doubling bug where a BF16 2D tensor `[1, D]` was stored as BF16 instead of F32 and loaded as `[2D]` (city96 issue #419)
- `convert.py`: Unsqueeze step in `handle_tensors` — applies `unsqueeze(0)` for keys listed in `model_arch.keys_unsqueeze` after dtype coercion
- `quantize.py`: `SIZE_RATIOS` dict — approximate output-size ratios relative to F16 source, used for the live size estimate in the UI
- `start_gui.bat`: one-click entry point
- `tests/test_quantize.py`: coverage for type registry, SIZE_RATIOS, Easy-Install discovery, subprocess interaction, and benchmark helpers
- `tests/test_gui.py`: coverage for `llama-quantize` Browse/Detect helper behavior
- `tests/test_convert.py`: float8 regression tests
- `pyproject.toml` / `uv.lock`: Add `ruff` as a dev dependency

### Change
- `gui.py`: Convert tab — add a description Markdown header so all three tabs share the same first-child structure; Gradio 6 sizes a tab by its first child, so the Convert tab now renders at the same width as the Fix Pad Tokens and Fix 5D Tensors tabs
- `tests/test_gui.py`: `TestLayoutParity` — assert `build_app()` returns without error and every tab has its description Markdown header
- `gui.py`: Add an Advanced thread-count control for `llama-quantize` and throttle GUI tensor log lines while keeping per-tensor progress updates
- `gui.py`: Make the `llama-quantize` path read-only; users select the executable with Browse instead of typing paths manually
- `AGENTS.md` / `README.md`: Clarify that Claude commit hooks live in `.claude/settings.json`, local permissions live in `.claude/settings.local.json`, tests run through `uv`, and `pyproject.toml`/`uv.lock` are the canonical dependency files
- `gui.py`: Page width increased from 860 px to 1100 px
- `gui.py`: Layout cleanup — all three tabs now use the same structure (card for inputs, action button below card, then status bar + log); removed `.card-actions` flex hack and `#size-info` padding tricks
- `gui.py`: Compact layout — inline Browse buttons, Quantization + size estimate side by side, Overwrite checkbox moved to Advanced, log reduced to 10/8 lines
- `gui.py`: Cancel button label shortened to `✕`; button heights reduced to 44 px
- `gui.py`: `gr.Progress()` removed from streaming handlers — it injected a UI element that called `scrollIntoView` and caused the page to jump during conversion; progress percentage is now embedded in the status text instead
- `gui.py`: `autoscroll=False` on log textboxes; `max_lines` fixed to prevent height-growth reflows
- `gui.py`: `show_progress="hidden"` on the convert event
- `gui.py`: Web UI — file browse buttons (source + output) using native tkinter dialogs
- `gui.py`: Output path auto-resolution (`_resolve_dst`) — handles directory paths, `{ftype}` placeholder, and full file paths
- `gui.py`: Quantization dropdown with 10 curated types (F32/F16/BF16/Q8_0 Python-native; Q6_K/Q5_K_M/Q4_K_M/Q4_K_S/Q3_K_M/Q2_K via llama-quantize)
- `gui.py`: Live size estimate — updates when source file or quantization changes
- `gui.py`: Cancel button (red) — signals `ConversionCancelled`, terminates llama-quantize subprocess, triggers `gc.collect()` to free RAM immediately
- `gui.py`: Scroll fix — `head=` injects a `<script>` that blocks all programmatic scroll APIs (`scrollIntoView`, `scrollTo`, `scroll`, `scrollBy`) so streaming updates never hijack the viewport
- `gui.py`: `_pipeline` step labels — dynamic total step count based on whether 5D fix and/or pad token fix are needed
- `gui.py`: Browse output folder composes full path with `{ftype}` placeholder so changing quantization after browsing always produces the correct filename
- `convert.py`: `ConversionCancelled` exception; `cancel_event` parameter on `handle_tensors` and `convert_file`; explicit `del writer; del state_dict` on cancel to release RAM
- `convert.py`: Float8 dtype coercion before `nan_to_num` (regression fix — `nan_to_num` does not support `float8_e4m3fn`)
- `quantize.py`: `cancel_event` parameter on `run_quantize`; `proc.terminate()` on cancel
- `fix_5d_tensors.py`: `on_progress`, `on_log` callback parameters

### Fix
- `tests/test_gui.py`: Make the non-Windows `llama-quantize` guidance test portable on Windows CI by avoiding patched `os.name` path construction
- `gui.py` / `convert.py`: Keep Lumina 2 pad tokens 1D in K-quant intermediates for compatibility with the patched Easy-Install `llama-quantize`, then re-apply the pad-token shape fix after quantization
- `.claude/settings.json`: Use a repo-local pytest temp directory and disable pytest cache in the commit test hook to avoid Windows temp/cache permission failures
- `gui.py`: K-quant Lumina 2 conversions — `llama-quantize` collapsed `[1, D]` pad token shapes back to `[D]`; pipeline now re-applies `fix_pad_tokens` automatically after quantization
- `convert.py`: `nan_to_num` crash on `float8_e4m3fn` tensors (Lumina2 models) — coerce to float16 before clamping
- `gui.py`: Output filename used stale quantization key when user changed quant after clicking Browse — now uses `{ftype}` placeholder resolved at conversion time
- `gui.py`: Step counter showed `[2/?]` — now shows correct total based on which post-processing steps are needed

---

<!-- Format:
## [x.y.z] - YYYY-MM-DD
### Add
### Change
### Fix
### Remove
-->
