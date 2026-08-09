"""Tests for safetensors_quant_nvfp4.py — Nvidia NVFP4 block-scaled quantization."""

from __future__ import annotations

import torch

from safetensors_quant_nvfp4 import _KVALUES, _swizzle_block_scale, quantize_nvfp4


def _vllm_swizzle_blockscale_reference(scale: torch.Tensor) -> torch.Tensor:
    """Literal transcription of vLLM's swizzle_blockscale()
    (vllm/model_executor/layers/quantization/utils/nvfp4_utils.py), minus the
    .cuda() call. Used only to independently verify _swizzle_block_scale
    against the reference implementation — a self-consistent round-trip test
    (quantize -> dequantize) can't catch a swizzle/unswizzle direction bug
    that's self-inverting, so this is the one check that actually
    discriminates a correct layout from a plausible-looking wrong one."""
    def round_up(x, m):
        return ((x + m - 1) // m) * m

    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)  # (1, M, K)
    B, M, K = scale.shape
    M_padded, K_padded = round_up(M, 128), round_up(K, 4)
    padded = torch.zeros((B, M_padded, K_padded), dtype=scale.dtype, device=scale.device)
    padded[:B, :M, :K] = scale
    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous()
    if scale_ndim == 2:
        return swizzled.reshape(M_padded, K_padded)
    return swizzled.reshape(B, M_padded, K_padded)


class TestQuantizeNvfp4:
    def test_returns_three_tensors(self):
        data = torch.randn(32, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        assert set(out.keys()) == {
            "block.weight", "block.weight_scale", "block.weight_scale_2",
        }

    def test_weight_is_packed_uint8_half_last_dim(self):
        data = torch.randn(4, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        w = out["block.weight"]
        assert w.dtype == torch.uint8
        assert w.shape == (4, 16)  # 32 elems / 2 per byte

    def test_scale_is_fp8_e4m3fn_per_16_block(self):
        data = torch.randn(4, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        scale = out["block.weight_scale"]
        assert scale.dtype == torch.float8_e4m3fn
        # ComfyUI's cuBLAS NVFP4 kernel needs the scale padded/swizzled to
        # (roundup(rows,128), roundup(n_blocks,4)) — see _swizzle_block_scale.
        assert scale.shape == (128, 4)

    def test_scale_swizzled_shape_matches_comfy_kitchen_kernel_formula(self):
        # 4 rows -> roundup(4,128)=128; 96 elems -> 6 blocks of 16 -> roundup(6,4)=8
        data = torch.randn(4, 96, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        assert out["block.weight_scale"].shape == (128, 8)

        # rows already a multiple of 128, blocks already a multiple of 4 -> no padding
        data2 = torch.randn(128, 64, dtype=torch.float32)
        out2 = quantize_nvfp4(data2, "block.weight")
        assert out2["block.weight_scale"].shape == (128, 4)

    def test_swizzle_matches_vllm_reference_implementation(self):
        torch.manual_seed(0)
        for m, k in [(4, 2), (128, 4), (37, 11), (200, 9)]:
            scale = (torch.rand(m, k) * 400 - 200).to(torch.float8_e4m3fn)
            ours = _swizzle_block_scale(scale)
            reference = _vllm_swizzle_blockscale_reference(scale)
            assert ours.shape == reference.shape
            assert torch.equal(ours.to(torch.float32), reference.to(torch.float32)), (
                f"swizzle mismatch for shape ({m}, {k})"
            )

    def test_kvalues_are_standard_undoubled_e2m1_magnitudes(self):
        # Regression guard for the "doubled table paired with plain-e4m3fn
        # scale" bug (docs/issues_analysis.md #11): every weight decoded at
        # exactly half its intended magnitude in ComfyUI. gguf.quants.NVFP4's
        # table is doubled to compensate for ITS OWN custom ue4m3 scale
        # encoding (a *0.5 baked into decode) — we don't use that encoding
        # (plain torch.float8_e4m3fn), so our table must be the standard,
        # undoubled OCP E2M1 magnitudes with max 6.0, not 12.0.
        assert _KVALUES.abs().max().item() == 6.0
        assert sorted(_KVALUES[:8].tolist()) == [0, 0.5, 1, 1.5, 2, 3, 4, 6]

    def test_block_of_exact_table_values_round_trips_exactly(self):
        # Every element in each block sits exactly on an E2M1 grid point, so
        # (unlike the generic tolerance test) reconstruction should be near
        # loss-free — this is what actually caught the doubled-table bug
        # (that bug reconstructed every value at exactly half magnitude).
        from safetensors_quant_nvfp4 import dequantize_nvfp4  # test-only helper

        data = torch.zeros(4, 16, dtype=torch.float32)
        data[0, :] = 6.0
        data[1, :] = 3.0
        data[2, :] = -4.0
        data[3, :] = 1.5
        out = quantize_nvfp4(data, "block.weight")
        recon = dequantize_nvfp4(out, "block.weight")
        # fp8_e4m3fn-rounding the per-block scale itself contributes a few
        # percent of noise even for on-grid values; the doubled-table bug
        # this test guards against was off by exactly 2x, not a few percent.
        assert torch.allclose(recon, data, atol=data.abs().max().item() * 0.05)

    def test_packs_even_index_in_high_nibble_hi_first_true(self):
        # Regression guard for the "nibble-swapped" bug (docs/issues_analysis.md
        # #13): comfy_kitchen's real inference-path matmul kernel
        # (scaled_mm_nvfp4) has no nibble-order parameter at all — it's
        # hardcoded to hi_first=True (even-indexed element in the HIGH
        # nibble). A self-consistent round-trip test can't catch the wrong
        # convention since encode/decode agreeing with each other proves
        # nothing about matching the kernel's fixed convention — this test
        # checks the packed byte value directly against a hand-computed
        # expectation instead.
        data = torch.zeros(1, 16, dtype=torch.float32)
        data[0, 0] = 6.0   # table index 7 (max positive)
        data[0, 1] = 0.5   # table index 1
        # block_scale calibrates to amax=6.0 -> block_scale*scale_2 == 1.0,
        # so values map directly to table indices without rescaling noise.
        out = quantize_nvfp4(data, "block.weight")
        packed_byte = out["block.weight"][0, 0].item()
        # index 0 (value 6.0 -> table idx 7) in HIGH nibble, index 1 (value
        # 0.5 -> table idx 1) in LOW nibble: (7 << 4) | 1 == 0x71 == 113.
        assert packed_byte == 0x71, f"expected 0x71 (hi_first=True), got {hex(packed_byte)}"

    def test_scale_2_is_global_float32_scalar(self):
        data = torch.randn(4, 32, dtype=torch.float32)
        out = quantize_nvfp4(data, "block.weight")
        s2 = out["block.weight_scale_2"]
        assert s2.dtype == torch.float32
        assert s2.numel() == 1

    def test_raises_on_non_multiple_of_16_last_dim(self):
        data = torch.randn(4, 17, dtype=torch.float32)
        try:
            quantize_nvfp4(data, "block.weight")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_dequant_reconstructs_within_nvfp4_tolerance(self):
        torch.manual_seed(0)
        data = torch.randn(8, 64, dtype=torch.float32) * 3
        out = quantize_nvfp4(data, "block.weight")
        from safetensors_quant_nvfp4 import dequantize_nvfp4  # test-only helper
        recon = dequantize_nvfp4(out, "block.weight")
        assert recon.shape == data.shape
        # 4-bit float has coarse steps; allow generous relative tolerance
        assert torch.allclose(recon, data, atol=data.abs().max().item() * 0.35)

    def test_zero_block_no_nan_and_reconstructs_to_zero(self):
        # Verify that zero/near-zero blocks don't produce NaN through division by zero.
        # This guards against: block_scale_fp8 → 0.0 after cast, then normalized = blocks / 0.0 → NaN.
        from safetensors_quant_nvfp4 import dequantize_nvfp4  # test-only helper

        # All-zero tensor: one large zero block
        data_zero = torch.zeros(2, 32, dtype=torch.float32)
        out_zero = quantize_nvfp4(data_zero, "block.weight")
        recon_zero = dequantize_nvfp4(out_zero, "block.weight")
        assert torch.isfinite(recon_zero).all(), "reconstruction contains NaN or Inf"
        assert torch.allclose(recon_zero, data_zero, atol=1e-6)

        # Near-zero tensor (values below FP8 representability): should quantize without NaN
        data_tiny = torch.full((2, 32), 1e-9, dtype=torch.float32)
        out_tiny = quantize_nvfp4(data_tiny, "block.weight")
        recon_tiny = dequantize_nvfp4(out_tiny, "block.weight")
        assert torch.isfinite(recon_tiny).all(), "reconstruction of tiny values contains NaN or Inf"
        # Tiny values should reconstruct to near-zero (they underflow to 0 in FP8)
        assert recon_tiny.abs().max() < 1e-6
