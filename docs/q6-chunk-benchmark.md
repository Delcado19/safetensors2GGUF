# Q6_K chunk-size measurement

Date: 2026-10-15. Windows CPU, Python 3.13.14 / NumPy 2.4.4.
Evidence: [`data/q6-chunk-benchmark-native.json`](../data/q6-chunk-benchmark-native.json).

## Method

Five isolated sequential worker processes encode the same first 1,048,576 values
of `double_blocks.0.img_attn.proj.weight` from the retained prepared FLUX.2 klein
Base 9B F16 GGUF. Each expands only that slice to F32 and performs three sequential
encodes. Timings include chunk iteration and payload SHA256, excluding imports,
source reading/expansion, output file writing and full-model preparation/readback.
There is no separate discarded warmup. No other conversion was started by this
experiment, but the host was not locked to a fixed frequency or exclusive load.

Every trial has the same payload SHA256. The matching 860,160-byte tensor prefix
from the retained native Q6_K file is independently read and matches it as well.
Sources remain unchanged by size/mtime; input F32 slice SHA256 is recorded.
The reference check hashes only the corresponding prefix, not both entire files.
The earlier full-file numeric validation remains separate evidence.

Process peak RSS comes from Windows `GetProcessMemoryInfo`, with the input-loaded
baseline peak recorded as well. It includes interpreter/imports/input and any
allocator/page-cache effects. It is neither a precise kernel allocation count
nor an estimate of full-model conversion peak RAM.

## Results

| Blocks per chunk | Median seconds | Process peak RSS (MiB) |
|---:|---:|---:|
| 16 | 0.35286 | 43.6 |
| 64 | 0.15102 | 43.8 |
| 256 (current default) | 0.09245 | 43.9 |
| 1024 | 0.12708 | 47.8 |
| 4096 | 0.12431 | 63.2 |

The current 256-block default is the fastest of these candidates on this sample.
Increasing to 1024 or 4096 does not justify a change here: both take longer and
increase the observed process peak. Keep the encoder and application's current
65,536-value windows unchanged. This is not a native-versus-Python throughput
comparison, an exhaustive tuning search, or a hardware-independent optimum.

## Reproduce

```bash
python -m scripts.benchmark_q6_chunks --source "I:/safetensors2GGUF-flux-q6-full-2026-10-15/prepared-F16.gguf" --reference "I:/safetensors2GGUF-flux-q6-full-2026-10-15/native-Q6_K.gguf" --output fresh-chunk-benchmark.json
```

The output report must be new. Workers run without visible subprocess consoles.
No weights are written, no models are downloaded, and no inference is started.
Linux measurement code exists but has not been executed in this experiment.
