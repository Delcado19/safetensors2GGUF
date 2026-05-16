# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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
