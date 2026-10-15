# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2024 The ggml authors
"""Research-only NumPy port of GGML's unweighted Q6_K reference encoder.

Based on llama.cpp c8c07d658a6cefc5a50cfdf6be7d726503612303,
ggml/src/ggml-quants.c: nearest_int, make_qx_quants(rmse_type=1),
quantize_row_q6_K_ref. Upstream MIT notice is preserved in
third_party/converter_references/llama-b3962-LICENSE.
Model eligibility, mixed presets and file writing are separate responsibilities.
"""
import numpy as np


def _metrics(groups, weights, codes):
    # Keep scalar C accumulation order: NumPy's pairwise sum can change bytes.
    sum_lx = np.zeros(len(groups), dtype=np.float32)
    sum_l2 = np.zeros_like(sum_lx)
    for index in range(16):
        value, weight, code = groups[:, index], weights[:, index], codes[:, index]
        sum_lx += (weight * value) * code
        sum_l2 += (weight * code) * code
    return sum_lx, sum_l2


def _fit_groups(groups):
    maximum = np.take_along_axis(groups, np.argmax(np.abs(groups), axis=1)[:, None], axis=1)[:, 0]
    active = np.abs(maximum) >= np.float32(1e-15)
    maximum = np.where(active, maximum, np.float32(1))
    inverse = np.float32(-32) / maximum
    codes = np.clip(np.rint(inverse[:, None] * groups), -32, 31).astype(np.int8)
    weights = groups * groups
    sum_lx, sum_l2 = _metrics(groups, weights, codes)
    scales = np.divide(sum_lx, sum_l2, out=np.zeros_like(sum_lx), where=sum_l2 != 0)
    best = scales * sum_lx
    for step in range(-9, 10):
        if step == 0:
            continue
        inverse = -(np.float32(32) + np.float32(0.1) * np.float32(step)) / maximum
        candidate = np.clip(np.rint(inverse[:, None] * groups), -32, 31).astype(np.int8)
        sum_lx, sum_l2 = _metrics(groups, weights, candidate)
        better = (sum_l2 > 0) & (sum_lx * sum_lx > best * sum_l2)
        scales[better] = sum_lx[better] / sum_l2[better]
        best[better] = scales[better] * sum_lx[better]
        codes[better] = candidate[better]
    scales[~active] = 0
    encoded = (codes.astype(np.int16) + 32).astype(np.uint8)
    encoded[~active] = 0
    return scales, encoded


def quantize_q6_k(data):
    """Encode finite float32 rows in GGUF Q6_K layout (256 values / 210 bytes).

    Returns uint8 with the last dimension replaced by its packed byte length.
    Rejects empty/misaligned input and numerical overflow instead of changing
    dtype or silently choosing a different precision. Does not mutate input.
    This is a correctness prototype, not a production or performance contract.
    """
    data = np.asarray(data)
    if data.dtype != np.float32 or data.ndim == 0 or data.size == 0 or data.shape[-1] % 256:
        raise ValueError('Q6_K requires nonempty float32 rows divisible by 256')
    if not np.isfinite(data).all():
        raise ValueError('Q6_K requires finite input')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return _encode(data)
    except FloatingPointError as error:
        raise ValueError('Q6_K input exceeds the reference prototype numerical range') from error


def _encode(data):
    # shortcut: fits all groups in memory; chunk before using on full-model tensors.
    blocks = data.reshape(-1, 256)
    scales, codes = _fit_groups(blocks.reshape(-1, 16))
    scales = scales.reshape(-1, 16)
    codes = codes.reshape(-1, 16, 16)
    largest = np.take_along_axis(scales, np.argmax(np.abs(scales), axis=1)[:, None], axis=1)[:, 0]
    active = np.abs(largest) >= np.float32(1e-15)
    inverse = np.float32(-128) / np.where(active, largest, np.float32(1))
    stored_d = (np.float32(1) / inverse).astype('<f2')
    signed_scales = np.minimum(127, np.rint(inverse[:, None] * scales)).astype(np.int8)
    d = stored_d.astype(np.float32)[:, None] * signed_scales
    # Native Q6 retains the initial fitted codes where the stored FP16 scale is zero.
    adjusted = np.divide(blocks.reshape(-1, 16, 16), d[:, :, None],
                         out=np.zeros_like(codes, dtype=np.float32), where=d[:, :, None] != 0)
    recoded = (np.clip(np.rint(adjusted), -32, 31).astype(np.int16) + 32).astype(np.uint8)
    codes = np.where(d[:, :, None] != 0, recoded, codes).reshape(-1, 2, 4, 32)
    result = np.empty((len(blocks), 210), dtype=np.uint8)
    low = result[:, :128].reshape(-1, 2, 64)
    low[:, :, :32] = (codes[:, :, 0] & 15) | ((codes[:, :, 2] & 15) << 4)
    low[:, :, 32:] = (codes[:, :, 1] & 15) | ((codes[:, :, 3] & 15) << 4)
    result[:, 128:192] = ((codes[:, :, 0] >> 4) | ((codes[:, :, 1] >> 4) << 2)
                         | ((codes[:, :, 2] >> 4) << 4) | ((codes[:, :, 3] >> 4) << 6)).reshape(-1, 64)
    result[:, 192:208] = signed_scales.view(np.uint8)
    result[:, 208:] = stored_d[:, None].view(np.uint8)
    result[~active] = 0
    return result.reshape(*data.shape[:-1], data.shape[-1] // 256 * 210)
