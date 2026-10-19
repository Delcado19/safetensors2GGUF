# Complete-checkpoint companion-loader validation

## Scope

On 2026-10-19, the actual companion text-loader node was executed against the
installed ComfyUI on CPU, using complete locally installed checkpoint components.
No mock native model constructor, model download, node installation, server change
or image render was used. These checks establish export/readback and native model
construction, not inference quality, VRAM savings or low-bit inference.

| Source | Export | Retained tensors | Real CPU construction |
| --- | --- | --- | --- |
| RealVisXL V50 Lightning SDXL CLIP-L | Q4_K_M | 196 | Matches native source parameter shapes; both report the same absent optional text projection |
| RealVisXL V50 Lightning SDXL OpenCLIP-bigG | Q4_K_M | 518 | Matches native source parameter shapes, no missing weights |
| Huihui Qwen3-VL-4B, Krea2 type | Q4_K_M | 713, including 315 vision and 18 DeepStack | Matches native source parameter shapes, no missing weights |

Reports are `data/companion-clip-l-full.json`,
`data/companion-clip-bigg-full.json` and `data/companion-qwen3vl4b-full.json`.
Sources are read-only. SDXL encoders were extracted into isolated repository-local
`build/` directories; bigG's old OpenCLIP namespace was converted with the installed
ComfyUI helper before export. Model files remain local and untracked.
Source and output SHA256 hashes were recorded after construction with unchanged
source size/mtime. The final full suite passes 1082 tests in 244.67 s; Ruff and
the TypeScript/Vite build pass.

The companion nodes reconstruct complete CPU weights before native construction;
they do not provide low-bit execution. The support table remains **unknown** for
inference and now explicitly exposes the narrower CPU construction evidence.
The 4B result does not qualify Qwen3-VL-8B or other source layouts.

## Full VL versus text-only

`Q8_CR_TEXT_ONLY` was separately exercised on the complete Huihui BF16 4B source.
The output contains 650 tensors (including quantization sidecars), has the explicit
`comfy.gguf.qwen3vl.text_only` marker and retains no vision/DeepStack tensors.
See `data/qwen3vl4b-text-only-validation.json`. It requires the matching Molbal
loader; successful conversion is not native inference qualification.

The installed INT8 Qwen3-VL-8B repack uses flat `model.layers.*` language keys,
which previously collided with plain Qwen3-8B detection despite retaining vision.
Detection now checks the Qwen3-VL vision merger/DeepStack before plain text
signatures. Regression tests exercise detection and the real public full-VL
export path without invoking the language-only quantizer.

The ordinary one-layer Qwen3 text pipeline was rerun after that routing fix and
still matches all historical tensor bytes and metadata except its temporary
display name: `data/bundled-text-vl-routing-requalification-2026-10-19.json`.
Historical reports remain unchanged.

## Repeat checks

Run the full-model validator using the installed ComfyUI Python and fresh work
and report paths:

```powershell
G:/ComfyUI-Easy-Install/python_embeded/python.exe scripts/validate_companion_model.py --comfy-root G:/ComfyUI-Easy-Install/ComfyUI --source <source.safetensors> --family qwen3-vl --clip-type krea2 --work build/new-vl-check --output data/new-vl-check.json
uv run python -m scripts.check_qwen3vl_text_only --source <source.safetensors> --destination build/new-text-only.gguf --report data/new-text-only.json
```

For SDXL extraction, use `--prefix conditioner.embedders.0.transformer.` for
CLIP-L and `--prefix conditioner.embedders.1.model.` for bigG, with the corresponding
`--family`. Reports record the installed `comfy/sd.py` hash to identify the tested
native implementation. Do not interpret a newly installed ComfyUI as revalidated.

## ERNIE

The installed native ComfyUI contains ERNIE model detection and construction.
No complete ERNIE diffusion checkpoint was found in the local model directories,
including a header-based architecture scan rather than filenames alone.
A local source path or authorization to obtain the Comfy-Org checkpoint is
required before the complete-checkpoint diffusion-loader check can run.
Schematic tests are not substituted for that missing runtime evidence.
