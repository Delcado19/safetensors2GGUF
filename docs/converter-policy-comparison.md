# Quantizer policy comparison and installed model header audit

Date: 2026-10-15. Branch: `research/unified-converter-inventory`.
Production conversion, format allowlists and installed models are unchanged.

## Evidence and boundaries

The [policy report](../data/converter-policy-comparison.json) records 84 successful
real `llama-quantize` executions and 2,502 output tensor-type decisions. These
small schematic GGUF fixtures exercise quantizer-stage policy, not inference or
the application's complete source-decoding/Python preprocessing pipeline.
Diffusion fixtures cover FLUX, SD3, SDXL and Wan; text fixtures cover Llama,
encoder/decoder T5 and Qwen3. All use F16 input and the application's six offered
K presets: Q2_K, Q3_K_M, Q4_K_S, Q4_K_M, Q5_K_M and Q6_K.

| Executable | Successful cases | Scope |
| --- | --- | --- |
| Frozen b3962 + frozen City96 patch | 48 | Diffusion scenarios and Llama/T5 |
| Frozen installed text llama.cpp | 18 | Llama/T5/Qwen3 |
| Newer frozen text llama.cpp | 18 | Llama/T5/Qwen3 |

Unsupported combinations are listed under `skipped`: unpatched text references
do not accept these diffusion architectures, and b3962 has no Qwen3 architecture.
Molbal's frozen Python K planner is executed separately on the same names/shapes
without allocating model-size tensors. This is a planning comparison; its GGUF
writer, source reconstruction and runtime are not executed here. It has no
equivalent text route for these fixtures, and Q2_K/Q3_K_M are not implemented by
that planner.

Sources remain pinned to the revisions documented in the
[native block comparison](converter-native-reference-comparison.md). The policy
report retains executable/source/patch hashes, compiler versions, cache flags,
commands, input/output hashes and exact output types. Tiny output GGUFs were
inspected with gguf-py. Native process logs and all final synthetic GGUF files
are under `I:\safetensors2GGUF-policy-final-2026-10-15`: 93 GGUFs (9 inputs and
84 outputs), approximately 5.19 MiB including logs. Earlier pilot folders remain
separate; their failed fixtures are not included in this result set.

## Measured differences

| Decision | Patched b3962 image quantizer | Frozen Molbal K planner |
| --- | --- | --- |
| FLUX Q4_K_S attention V | First four processed V tensors become Q5_K; remaining four stay Q4_K | All eight become Q5_K, independent of insertion order |
| FLUX `img_in.weight` | F16 architecture exclusion | Q4_K in this planner fixture |
| FLUX attention output, Q4_K_S | Q4_K | Q5_K |
| Norm and small matrix (1,024 elements) | Norm F16; small matrix Q4_K at this native stage | F32 protections |
| Non-convolution 4D fixture | F16 (image quantizer's 2D gate) | Planned Q4_K; writer/runtime eligibility is not established |
| Linear row width 160, Q4_K_M | Successful output with F16 fallback | Planner rejects the job |

Reversing the FLUX tensor order moves the four Q5_K selections from blocks 0-3
to blocks 7-4. Molbal's complete plan stays identical. Consequently, sorting
tensors for a new shared implementation can change City96-compatible mixed
output even if every mathematical encoder is byte-compatible.

The installed and newer frozen text quantizers select identical output types
for all 18 matching scenarios. Patched b3962 agrees with them on all six Llama
presets. T5 demonstrates an important exception:

| T5 preset | Number of tensor types differing between b3962 and newer text reference |
| --- | --- |
| Q2_K | 0 |
| Q3_K_M | 2 |
| Q4_K_S | 6 |
| Q4_K_M | 18 |
| Q5_K_M | 18 |
| Q6_K | 0 |

The T5 fixture uses encoder/decoder self-attention plus decoder cross-attention;
it is not a single-stack Llama fixture with a different architecture label.
Its small tensors and metadata are schematic, not a loadable T5 checkpoint.
These counts establish concrete policy differences on this input, not a claim
about every real T5/UMT5 checkpoint or relative model quality.

## Installed models: headers only

The [header audit](../data/installed-model-policy-headers.json) inspected all 41
safetensors files found in `diffusion_models`, `text_encoders` and `checkpoints`
under `G:\ComfyUI-Easy-Install\ComfyUI\models`. It reads roughly 6.7 MB of headers,
not the weight payloads. LoRAs, VAEs and existing GGUF files are outside this
audit. No installed file was changed and no real checkpoint was converted.

Each entry records a canonical JSON header checksum (not a full-file checksum),
stored dtype/shape counts, project architecture/family detection and examples
of row widths not divisible by 256. **Stored shapes are not dequantized shapes.**
Thirteen files contain raw-byte tensors and nine contain `.comfy_quant` tensors;
markers can overlap and do not alone identify the complete packing scheme.
These entries must pass the existing source reconstruction and precision guards
before any future conversion; header alignment is not export eligibility.

The audit includes six detected Krea files, seven Lumina2/Z-Image files, three
FLUX files, two Wan files, six SDXL checkpoints and several text families.
The current project detector does not classify the packed Qwen Image 2.1 source;
the Krea Qwen3-VL text encoder is explicitly rejected by the current text-family
guard. These are recorded planning boundaries, not proof the files are damaged.

## Reproduce

The scripts need the existing locked Python environment; the policy CLI also
needs Git (for the preserved City96 patch), CMake and a local C/C++ compiler.
They use preserved source archives and
do not fetch moving upstream revisions.

```bash
python -m scripts.inventory_installed_model_headers --models-root "G:/ComfyUI-Easy-Install/ComfyUI/models" --output data/installed-model-policy-headers.json
python -m scripts.compare_converter_policies --artifacts "I:/safetensors2GGUF-policy-new-run" --output data/converter-policy-comparison.json
```

If CMake is not on PATH, pass `--cmake /path/to/cmake`. The artifact directory
must not already exist. A fresh ignored build directory is created under
`build/converter-policy`; build logs and a build manifest are retained there.
The existing model source tree is never an output destination. For this local
setup the user designated `I:\` for test conversion artifacts; check free space
again before any large full-model run (232.5 GiB was observed before these tests).

Two build safeguards were required and are preserved in the research script:

- b3962 needs an explicit `<chrono>` include with MSVC 19.51. Its build forwards
  C++ flags to C files as well, so `compat-chrono.h` guards the include with
  `__cplusplus`. No quantization math is changed. The header hash is recorded.
- `git apply` inside a nested source directory can silently skip paths because
  it discovers the parent project repository. The script sets
  `GIT_CEILING_DIRECTORIES`, checks the materialized image selector and records
  all three patched file hashes. A zero exit code alone is insufficient.

Linux native execution, BF16 source policy, source reconstruction and full-model
quality remain untested in this phase. No ComfyUI render was started.

## Next implementation gate

Keep the shared mathematical encoder separate from architecture eligibility,
source precision, ordered/mixed preset decisions and shape fallback. Preserve
the tested City96 and text policies explicitly instead of silently adopting
Molbal's planner as one universal policy. A first common implementation should
reproduce both the block vectors and these recorded decisions before it can
replace a production path. Language selection remains open.
The [first shared Q6_K math pilot](shared-q6-pilot.md) now passes its block-byte
gate; model/preset integration remains a separate open gate.
