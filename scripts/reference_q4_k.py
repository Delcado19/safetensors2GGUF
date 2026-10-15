# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2024 The ggml authors
"""Research Q4_K blocks from frozen llama.cpp b3962 quantize_row_q4_K_ref.

MIT notice: third_party/converter_references/llama-b3962-LICENSE.
Shared affine fitting/packing lives in reference_q5_k; Q4 selects the native
15-level, -1.0/0.1, 21-candidate grid and omits Q5's high-bit plane.
This does not implement Q4_K_S/Q4_K_M model policies or application dispatch.
"""
import numpy as np

from scripts.reference_q5_k import _encode as _encode_affine, _iter_affine


def _encode(data):
    return _encode_affine(data, bits=4)


def iter_q4_k(data, *, chunk_blocks=256):
    """Yield owned Q4_K chunks: 256 finite F32 values / 144 packed bytes per block.

    Preserve strided logical C order with bounded working copies. Validation is
    incremental; consumers must publish atomically and close between chunks to
    cancel. Input must remain unchanged; extreme arithmetic overflow is rejected.
    """
    yield from _iter_affine(data, chunk_blocks, _encode, 'Q4_K')


def quantize_q4_k(data, *, chunk_blocks=256):
    """Return complete Q4_K bytes, preserving leading dimensions and input data.

    Use iter_q4_k when allocating a complete packed output would be too large.
    """
    data = np.asarray(data)
    if data.dtype != np.float32 or data.ndim == 0 or not data.size or data.shape[-1] % 256:
        raise ValueError('Q4_K requires nonempty float32 rows divisible by 256')
    result = np.empty(data.size // 256 * 144, dtype=np.uint8)
    offset = 0
    for chunk in iter_q4_k(data, chunk_blocks=chunk_blocks):
        result[offset:offset + chunk.size] = chunk.ravel()
        offset += chunk.size
    return result.reshape(*data.shape[:-1], data.shape[-1] // 256 * 144)
