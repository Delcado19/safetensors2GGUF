# FLUX Q4_K_S/Q4_K_M research file export

Date: 2026-10-15. Branch: `research/unified-converter-inventory`.

## Result

Both research presets reproduce the pinned City96-patched b3962 output on eight
schematic files each: **16 native conversions and 400 exact tensor comparisons**.
Names, types, shapes, complete payload bytes, typed metadata and effective
alignment match. Evidence:

- [`data/flux-q4s-path-comparison.json`](../data/flux-q4s-path-comparison.json)
- [`data/flux-q4m-path-comparison.json`](../data/flux-q4m-path-comparison.json)

Each preset covers F16/F32 input, normal/reversed tensor order and aligned versus
incompatible row widths. Odd-row cases prepend an additional attention-V matrix
with width 160 to exercise counter behavior before aligned V matrices. Outputs
and native logs are in `I:/safetensors2GGUF-flux-q4s-final-2026-10-15` and
`I:/safetensors2GGUF-flux-q4m-final-2026-10-15`. Free space was checked before
writing; no installed model changed. No full checkpoint or render was started.

## Frozen image policy

| Eligible role | Q4_K_S | Q4_K_M |
| --- | --- | --- |
| Default matrix | Q4_K | Q4_K |
| Attention V | First four processed V tensors Q5_K, subsequent V Q4_K | Q6_K |
| Fused QKV | Q4_K | Q5_K |
| Matching feedforward-down role | Q5_K | Q6_K |
| Row width not divisible by 256 | F16 fallback | F16 fallback |
| Architecture/general exclusion or non-2D tensor | Preserve source float type/bytes | Preserve source float type/bytes |

Name aliases follow the frozen `img_tensor_get_type` patch. Text-model layer
heuristics are not substituted. For Q4_K_S, an eligible V tensor increments the
counter **before** row-width fallback. Thus a leading incompatible V consumes one
of the first four slots, leaving only three Q5_K aligned V matrices. Protected or
rank-excluded tensors do not consume a slot. Reversing eligible V order moves
which block names receive the extra bits. Preserve source tensor order throughout
planning and writing; sorting names would silently change this mixed policy.

The existing `scripts/requantize_flux_q6.py` name and Q6 default are retained for
compatibility. Its explicit `--preset Q4_K_S` and `--preset Q4_K_M` modes use the
shared bounded Q4/Q5/Q6 encoders, unchanged metadata copying, per-tensor readback,
source-stability checks, cancellation and atomic overwrite protection. Reports
retain all three codec source hashes plus writer and pinned native provenance.
Regression tests verify retained native order/fallback outcomes and real Q4
writes, file-type metadata, source preservation, cancellation and late invalid
data with an existing output. Earlier Q5/Q6 tests remain in place.

```bash
python -m scripts.requantize_flux_q6 --source prepared-F16.gguf --output model-Q4_K_S.gguf --preset Q4_K_S
python -m scripts.requantize_flux_q6 --source prepared-F16.gguf --output model-Q4_K_M.gguf --preset Q4_K_M
python -m scripts.check_flux_q6_path --preset Q4_K_S --artifacts /fresh/q4s-directory --output /fresh/q4s-report.json
python -m scripts.check_flux_q6_path --preset Q4_K_M --artifacts /fresh/q4m-directory --output /fresh/q4m-report.json
```

## Limits and next gate

These are tiny policy/math fixtures, not inference-capable models. No full Q4
checkpoint, loading/image quality, LoRA, offload, Linux native execution or
performance/peak-memory claim is established. Q4 is still a research file path;
the experimental Workbench/API backend remains limited to Q6_K/Q5_K_M, native
Q4 defaults and the Krea allowlist are unchanged. Next compare a complete prepared
FLUX checkpoint against native before considering Q4 app exposure. The original
Q4_K block pilot and shared-code regression evidence remain documented in
[the Q4 pilot](shared-q4-pilot.md).
