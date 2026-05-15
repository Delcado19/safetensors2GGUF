"""Benchmark one or more llama-quantize binaries on the same GGUF input."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantize import benchmark_quantize_binaries  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark llama-quantize binaries on one source GGUF."
    )
    parser.add_argument("--src", required=True, help="Source GGUF, ideally F16/BF16/F32/Q8_0")
    parser.add_argument(
        "--exe",
        action="append",
        required=True,
        help="Path to a llama-quantize binary. Repeat to compare multiple binaries.",
    )
    parser.add_argument("--quant", default="Q4_K_M", help="Quantization type to benchmark")
    parser.add_argument("--threads", type=int, default=0, help="Thread count; 0 lets llama-quantize choose")
    parser.add_argument("--work-dir", help="Directory for temporary benchmark outputs")
    parser.add_argument("--quiet", action="store_true", help="Suppress llama-quantize subprocess output")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = benchmark_quantize_binaries(
        args.exe,
        args.src,
        args.quant,
        nthreads=args.threads or None,
        work_dir=args.work_dir,
        on_log=None if args.quiet else print,
    )

    print("llama-quantize benchmark")
    print(f"source: {args.src}")
    print(f"quant:  {args.quant}")
    print(f"threads: {args.threads or 'auto'}")
    print()
    successful = [r for r in results if r.error is None]
    fastest = min((r.seconds for r in successful), default=0)
    for result in results:
        ratio = result.seconds / fastest if fastest and result.error is None else 0.0
        size_mb = result.output_size / 1_000_000
        status = f"x{ratio:4.2f}" if result.error is None else "FAILED"
        print(f"{result.seconds:8.2f}s  {status:>6}  {size_mb:8.1f} MB  {result.exe}")
        if result.error is not None:
            print(f"          error: {result.error}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
