# Complete-checkpoint companion-loader validation

## Scope

On 2026-10-19, the actual companion text/diffusion loader nodes were executed against the
installed ComfyUI on CPU, using complete locally installed checkpoint components.
No mock native model constructor, model download, node installation, server change
or image render was used. These checks establish export/readback and native model
construction, not inference quality, VRAM savings or low-bit inference.

| Source | Export | Retained tensors | Real CPU construction |
| --- | --- | --- | --- |
| RealVisXL V50 Lightning SDXL CLIP-L | Q4_K_M | 196 | Matches native source parameter shapes; both report the same absent optional text projection |
| RealVisXL V50 Lightning SDXL OpenCLIP-bigG | Q4_K_M | 518 | Matches native source parameter shapes, no missing weights |
| Huihui Qwen3-VL-4B, Krea2 type | Q4_K_M | 713, including 315 vision and 18 DeepStack | Matches native source parameter shapes, no missing weights |
| BigLoveEryn1 BF16, ERNIE-Image | Q4_K_M | 409 | Matches native source parameter shapes on CPU with BF16 weights, no missing/unexpected weights |

Reports are `data/companion-clip-l-full.json`,
`data/companion-clip-bigg-full.json`, `data/companion-qwen3vl4b-full.json` and
`data/companion-ernie-full.json`.
Sources are read-only. SDXL encoders were extracted into isolated repository-local
`build/` directories; bigG's old OpenCLIP namespace was converted with the installed
ComfyUI helper before export. Model files remain local and untracked.
Source and output SHA256 hashes were recorded after construction with unchanged
source size/mtime. The preceding CLIP/VL checkpoint passed 1082 tests in 244.67 s;
the final ERNIE regression gate is recorded in `CONTINUE.md`. Reports retain their
original source hashes; the later ERNIE-specific log/validator changes do not
rewrite earlier measurements.

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

The user supplied `diffusion_models/Ernie/BigLoveEryn1_bf16.safetensors`, resolving
the earlier missing-source blocker without a download. All 409 source tensors
survive Q4_K_M export/readback. The 5,299,410,112-byte GGUF constructs the same
410-entry native model state as the original through `S2GGUFDiffusionLoader`.
Both native source and companion construction report no missing/unexpected
weights, and actual model parameters remain BF16 on CPU. Export including internal
readback/hashing takes 77.144 s; complete reconstruction/construction and later
source hashing are outside that export timing.

The validator sets the native CPU BF16 weight option for ERNIE so simultaneous
source/model copies fit the host's RAM. This is not a GPU, inference, performance
or memory-saving claim. Missing/unexpected native diffusion-weight warnings now
fail validation rather than being silently ignored by the text-only warning
check. ERNIE export logs name the diffusion loader instead of the text loader.

Repeat the complete diffusion check using fresh paths and `--family ernie_image`:

```powershell
G:/ComfyUI-Easy-Install/python_embeded/python.exe scripts/validate_companion_model.py --comfy-root G:/ComfyUI-Easy-Install/ComfyUI --source <ernie.safetensors> --family ernie_image --work build/new-ernie-check --output data/new-ernie-check.json
```
