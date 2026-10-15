# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2024 The ggml authors
"""Research NumPy port of unweighted GGML Q5_K reference blocks.

Based on llama.cpp c8c07d658a6cefc5a50cfdf6be7d726503612303,
ggml/src/ggml-quants.c: make_qkx2_quants and quantize_row_q5_K_ref.
MIT notice: third_party/converter_references/llama-b3962-LICENSE.
This encodes blocks only, not Q5_K_M model policies or GGUF files.
"""
import operator

import numpy as np


def _error(x, weights, codes, scale, minimum):
    error = np.zeros(len(x), dtype=np.float32)
    for i in range(32):
        difference = (scale * codes[:, i] + minimum) - x[:, i]
        error += weights[:, i] * (difference * difference)
    return error


def _fit(x):
    squares = np.zeros(len(x), dtype=np.float32)
    for i in range(32):
        squares += x[:, i] * x[:, i]
    weights = np.sqrt(squares / np.float32(32))[:, None] + np.abs(x)
    minimum = np.minimum(np.min(x, axis=1), np.float32(0))
    maximum = np.max(x, axis=1)
    active = maximum != minimum
    sum_w, sum_x = weights[:, 0].copy(), weights[:, 0] * x[:, 0]
    for i in range(1, 32):
        sum_w += weights[:, i]
        sum_x += weights[:, i] * x[:, i]
    inverse = np.float32(31) / np.where(active, maximum - minimum, np.float32(1))
    scale = np.float32(1) / inverse
    codes = np.clip(np.rint(inverse[:, None] * (x - minimum[:, None])), 0, 31).astype(np.uint8)
    best = _error(x, weights, codes, scale, minimum)
    for step in range(16):
        # Native fitting updates minimum after each winning candidate, affecting later steps.
        inverse = (np.float32(-0.5) + np.float32(0.1) * np.float32(step) + np.float32(31)) / np.where(
            active, maximum - minimum, np.float32(1))
        candidate = np.clip(np.rint(inverse[:, None] * (x - minimum[:, None])), 0, 31).astype(np.uint8)
        sum_l, sum_l2, sum_xl = (np.zeros(len(x), dtype=np.float32) for _ in range(3))
        for i in range(32):
            weighted = weights[:, i] * candidate[:, i]
            sum_l += weighted
            sum_l2 += weighted * candidate[:, i]
            sum_xl += weighted * x[:, i]
        determinant = sum_w * sum_l2 - sum_l * sum_l
        valid = active & (determinant > 0)
        this_scale = np.divide(sum_w * sum_xl - sum_x * sum_l, determinant,
                               out=np.zeros_like(scale), where=valid)
        this_min = np.divide(sum_l2 * sum_x - sum_l * sum_xl, determinant,
                             out=np.zeros_like(minimum), where=valid)
        positive = this_min > 0
        this_min[positive] = 0
        this_scale[positive] = np.divide(sum_xl, sum_l2, out=np.zeros_like(scale), where=sum_l2 != 0)[positive]
        error = _error(x, weights, candidate, this_scale, this_min)
        better = valid & (error < best)
        codes[better] = candidate[better]
        best[better], scale[better], minimum[better] = error[better], this_scale[better], this_min[better]
    scale[~active] = 0
    codes[~active] = 0
    return scale, -minimum, codes


def _encode(data):
    blocks = data.reshape(-1, 256)
    scales, minima, codes = _fit(blocks.reshape(-1, 32))
    scales, minima = scales.reshape(-1, 8), minima.reshape(-1, 8)
    maximum = np.max(scales, axis=1)
    # Native max_min starts at positive zero and only accepts strictly greater minima.
    max_min = np.maximum(np.max(minima, axis=1), np.float32(0))
    inv_scale = np.divide(np.float32(63), maximum, out=np.zeros_like(maximum), where=maximum > 0)
    inv_min = np.divide(np.float32(63), max_min, out=np.zeros_like(max_min), where=max_min > 0)
    sc = np.minimum(63, np.rint(inv_scale[:, None] * scales)).astype(np.uint8)
    mn = np.minimum(63, np.rint(inv_min[:, None] * minima)).astype(np.uint8)
    stored = (maximum / np.float32(63)).astype('<f2')
    stored_min = (max_min / np.float32(63)).astype('<f2')
    d = stored.astype(np.float32)[:, None] * sc
    dm = stored_min.astype(np.float32)[:, None] * mn
    recoded = np.divide(blocks.reshape(-1, 8, 32) + dm[:, :, None], d[:, :, None],
                        out=np.zeros_like(codes.reshape(-1, 8, 32), dtype=np.float32), where=d[:, :, None] != 0)
    recoded = np.clip(np.rint(recoded), 0, 31).astype(np.uint8)
    # FP16 scale underflow keeps the initial fitted codes, as in native Q5_K.
    codes = np.where(d[:, :, None] != 0, recoded, codes.reshape(-1, 8, 32))
    result = np.zeros((len(blocks), 176), dtype=np.uint8)
    result[:, :2] = stored[:, None].view(np.uint8)
    result[:, 2:4] = stored_min[:, None].view(np.uint8)
    result[:, 4:8] = sc[:, :4] | ((sc[:, 4:] >> 4) << 6)
    result[:, 8:12] = mn[:, :4] | ((mn[:, 4:] >> 4) << 6)
    result[:, 12:16] = (sc[:, 4:] & 15) | ((mn[:, 4:] & 15) << 4)
    for group in range(8):
        result[:, 16:48] |= (codes[:, group] >> 4) << group
    result[:, 48:] = ((codes[:, ::2] & 15) | ((codes[:, 1::2] & 15) << 4)).reshape(-1, 128)
    return result


def iter_q5_k(data, *, chunk_blocks=256):
    """Yield independent Q5_K chunks (256 finite F32 values / 176 bytes per block).

    Preserve logical C order for strided input; only each chunk is copied.
    Validation is incremental, so consumers must publish atomically and close or
    stop the iterator between chunks to cancel. Input must remain unchanged.
    Overflow is rejected; the complete finite float32 range is not guaranteed.
    """
    data = np.asarray(data)
    if data.dtype != np.float32 or data.ndim == 0 or not data.size or data.shape[-1] % 256:
        raise ValueError('Q5_K requires nonempty float32 rows divisible by 256')
    if isinstance(chunk_blocks, (bool, np.bool_)):
        raise ValueError('chunk_blocks must be a positive integer')
    try:
        chunk_blocks = operator.index(chunk_blocks)
    except TypeError as error:
        raise ValueError('chunk_blocks must be a positive integer') from error
    if chunk_blocks <= 0:
        raise ValueError('chunk_blocks must be a positive integer')
    for start in range(0, data.size, chunk_blocks * 256):
        chunk = data.flat[start:start + chunk_blocks * 256].reshape(-1, 256)
        if not np.isfinite(chunk).all():
            raise ValueError('Q5_K requires finite input')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                encoded = _encode(chunk)
        except FloatingPointError as error:
            raise ValueError('Q5_K input exceeds the reference prototype numerical range') from error
        yield encoded


def quantize_q5_k(data, *, chunk_blocks=256):
    """Return complete Q5_K bytes with leading input dimensions preserved.

    Working arrays are chunk-bounded; use iter_q5_k to avoid a complete output
    allocation. Reject invalid dtype/row size and never mutate the input.
    """
    data = np.asarray(data)
    if data.dtype != np.float32 or data.ndim == 0 or not data.size or data.shape[-1] % 256:
        raise ValueError('Q5_K requires nonempty float32 rows divisible by 256')
    result = np.empty(data.size // 256 * 176, dtype=np.uint8)
    offset = 0
    for chunk in iter_q5_k(data, chunk_blocks=chunk_blocks):
        result[offset:offset + chunk.size] = chunk.ravel()
        offset += chunk.size
    return result.reshape(*data.shape[:-1], data.shape[-1] // 256 * 176)
