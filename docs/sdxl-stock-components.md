# Verified SDXL stock component fingerprints

Date: 2026-10-06. Scope: official SDXL Base 1.0 components from Stability AI,
plus direct FP16 and BF16 casts of those FP32 weights. This is exact weight
identity, not a quality assessment or identification of every SDXL-derived model.

## Provenance

The user provided the downloaded originals on I:. Whole-file SHA256 matched the
published Hugging Face hashes before normalized tensor fingerprints were computed.
The files remain untouched and are not shipped with this project.

| Component | Official source | Verified file SHA256 | Tensors |
| --- | --- | --- | ---: |
| VAE | [vae/diffusion_pytorch_model.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/vae/diffusion_pytorch_model.safetensors) | `1598f3d24932bcfe6634e8b618ea1e30ab1d57f5aad13a6d2de446d2199f2341` | 248 |
| CLIP-L | [text_encoder/model.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/text_encoder/model.safetensors) | `5c3d6454dd2d23414b56aa1b5858a72487a656937847b6fea8d0606d7a42cdbc` | 196 |
| CLIP-G | [text_encoder_2/model.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/text_encoder_2/model.safetensors) | `3a6032f63d37ae02bbc74ccd6a27440578cd71701f96532229d0154f55a8d3ff` | 517 |

The source URLs use main for navigation; the recorded file SHA256 pins the actual
content. Future upstream changes must not silently replace these fingerprints.
The nine normalized hashes are stored in `data/sdxl_stock_components.json`.

## Reproduction and runtime behavior

```bash
uv run python scripts/check_stock_components.py I:/ --verify
```

This verifies original file bytes first, then normalizes names/shapes and computes
FP32 fingerprints plus direct FP16/BF16 casts. It neither downloads nor writes
model files. A wrong original-file checksum or different registry fails the check.
Without `--verify` it prints the generated registry for maintainer review.

VAE Diffusers names normalize to original SDXL names, with reversed decoder block
indices and 2D attention Linear weights reshaped to 1x1 Conv weights. All 248
names, shapes, dtypes and values were independently checked against the installed
ComfyUI `diffusers_convert.convert_vae_state_dict` using the official VAE.
Reference: [ComfyUI normalization](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/diffusers_convert.py).
Embedded CLIP-G uses the existing QKV split and text-projection transpose.

`normalized-tensors-v1` hashes each tensor's dtype, shape and raw logical bytes,
then hashes the sorted normalized name/tensor-hash pairs. File metadata, padding
and tensor ordering are irrelevant; precision, values and the complete tensor
set must match. Fingerprints are not the whole-file hashes published by HF.

Recognition occurs during component extraction/diagnostics, even without local
references. Results identify the precise stock/cast variant. If no variant
matches, identity remains unknown; it does not prove modified or inferior weights.
RealVisXL V50 Lightning was checked read-only and did not match these registered
variants; no image generation or model conversion was performed.

A stock match does not create a usable external file. Extraction still writes
selected components unless an identical local reference is present and reuse
is enabled. Those reference files are never overwritten by reuse.

This registry does not cover alternate VAE fixes, refiner-specific components,
quantized encoders, mixed-dtype components or different casting histories.
Only register new fingerprints with verified provenance and reproducible checks.
