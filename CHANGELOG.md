# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Add
- `quantize.py`: `SIZE_RATIOS` dict — approximate output-size ratios relative to F16 source, used for the live size estimate in the UI
- `gui.py`: Web UI — file browse buttons (source + output) using native tkinter dialogs
- `gui.py`: Output path auto-resolution (`_resolve_dst`) — handles directory paths, `{ftype}` placeholder, and full file paths
- `gui.py`: Quantization dropdown with 10 curated types (F32/F16/BF16/Q8_0 Python-native; Q6_K/Q5_K_M/Q4_K_M/Q4_K_S/Q3_K_M/Q2_K via llama-quantize)
- `gui.py`: Live size estimate — updates when source file or quantization changes
- `gui.py`: Cancel button (red) — signals `ConversionCancelled`, terminates llama-quantize subprocess, triggers `gc.collect()` to free RAM immediately
- `gui.py`: Scroll fix — `head=` injects a `<script>` that blocks all programmatic scroll APIs (`scrollIntoView`, `scrollTo`, `scroll`, `scrollBy`) so streaming updates never hijack the viewport
- `gui.py`: `_pipeline` step labels — `[1/2+]`, `[2/2]`/`[2/3]`, `[3/3]` with correct total computed after step 1
- `gui.py`: Browse output folder composes full path with `{ftype}` placeholder so changing quantization after browsing always produces the correct filename
- `convert.py`: `ConversionCancelled` exception; `cancel_event` parameter on `handle_tensors` and `convert_file`; explicit `del writer; del state_dict` on cancel to release RAM
- `convert.py`: Float8 dtype coercion before `nan_to_num` (regression fix — `nan_to_num` does not support `float8_e4m3fn`)
- `quantize.py`: `cancel_event` parameter on `run_quantize`; `proc.terminate()` on cancel
- `fix_5d_tensors.py`: `on_progress`, `on_log` callback parameters
- `start_gui.bat`: one-click entry point
- `tests/test_quantize.py`: 24 tests covering type registry, SIZE_RATIOS, find_exe, subprocess interaction
- `tests/test_convert.py`: float8 regression tests

### Change
- `gui.py`: `gr.Progress()` removed from streaming handlers — it injected a UI element that called `scrollIntoView` and caused the page to jump during conversion; progress percentage is now embedded in the status text instead
- `gui.py`: `autoscroll=False` on log textboxes; `max_lines` fixed to prevent height-growth reflows
- `gui.py`: `show_progress="hidden"` on the convert event

### Fix
- `convert.py`: `nan_to_num` crash on `float8_e4m3fn` tensors (Lumina2 models) — coerce to float16 before clamping
- `gui.py`: Output filename used stale quantization key when user changed quant after clicking Browse — now uses `{ftype}` placeholder resolved at conversion time
- `gui.py`: Step counter showed `[2/?]` — now shows `[2/2]` or `[2/3]` based on whether a 5D side-car exists

---

<!-- Format:
## [x.y.z] - YYYY-MM-DD
### Add
### Change
### Fix
### Remove
-->
