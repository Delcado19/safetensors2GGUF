# Extended adapter validation

Implementation resumed at the user's request on 2026-10-19. The 2026-10-18
paused snapshot in `CONTINUE.md` remains historical evidence.

## Export contract

The isolated frozen Molbal worker exports Ideogram, LTXV Upscaler, MiniMax H3,
MiniMax H3 VAE, MiniMax Music 3 and Qwen Image 2.1. Krea's additional K/ConvRot
formats use the same worker; its original non-K adapter remains separate.
The worker hashes every emitted tensor before writing and independently reads
back the final GGUF tensor names, count, types, shapes and complete payloads.
Architecture mismatches, duplicate tensors, missing tensors and altered bytes
fail before the parent atomically publishes the destination. Existing outputs
remain intact on failure. Shape metadata and ConvRot scale sidecars are retained
by the pinned upstream writer. This checks what the converter emitted; it does
not prove that upstream architecture filtering or every model variant is correct.

The six extended architectures have schematic export tests for every advertised
format: F16, BF16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0, Q2_K, Q3_K_M, Q4_K_S,
Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_CR, Q4_CR and Q4_CR_W4A4.
Q2/Q3 selection remains project policy, not an independently established native
preset for these new architectures. The verified bundled CPU kernel replaces
the frozen upstream K encoders with known positive-ramp clipping.

## Original-key companion layout

CLIP-L/bigG, complete Qwen2.5-VL/Qwen3-VL, MiniMax Music text and ERNIE use a
separate original-key layout. Text encoders require `S2GGUFTextEncoderLoader`;
ERNIE requires `S2GGUFDiffusionLoader`. Both reconstruct weights in CPU RAM
before native ComfyUI construction. These are compatibility loaders, not low-bit
inference kernels; reduced inference RAM/VRAM is not established. The nodes are
not installed into the user's ComfyUI by these changes.

Empty floating buffers retain their shape and are stored as F32 when BF16 or
Q8_0 codecs cannot encode empty rows. Integer buffers and scalars remain intact.
The independent `Q8_CR_TEXT_ONLY` Qwen3-VL route deliberately drops vision and
must not be confused with a complete VL export.

## Requalification

The final full suite passes **1078 tests** (245.80 s); Ruff and the frontend
TypeScript/Vite build pass. `data/extended-adapter-validation-2026-10-19.json`
records the 102 export cases, negative checks and validated source hashes.

`data/compiled-diffusion-requalification-2026-10-19.json` reruns all 66 schematic
native comparisons for eleven previously supported diffusion families.
`data/bundled-text-requalification-2026-10-19.json` reruns the real pinned Qwen3
export and Q4_K_M quantizer on the existing one-layer synthetic fixture: 14 tensor
payloads/types/shapes match the historical export. Model metadata matches except
for the display name derived from a random temporary directory. Neither report
rewrites historical hashes, renews historical performance numbers or proves
full-checkpoint inference. No new render is part of these checks.

Run regression checks with the repository environment:

```powershell
uv --cache-dir build/uv-cache run --offline --frozen --no-sync python -m pytest --tb=short -q --basetemp build/pytest-validation -p no:cacheprovider
uv --cache-dir build/uv-cache run --offline --frozen --no-sync ruff check .
# In frontend:
npm run build
```

Complete-checkpoint native model construction, inference, clean-machine packaging
and non-Windows execution remain open. Support-table cells for new routes remain
unknown; historical stock-loader failures do not establish that the new loader
works or fails.
