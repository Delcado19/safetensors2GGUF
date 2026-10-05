"""Source-aware planning and Kitchen-compatible rounding regression checks."""
import json

import pytest
import torch
from safetensors.torch import save_file, load_file

from convert import _read_safetensors_header
from convert_safetensors import convert_to_safetensors
from models.architectures import ModelQwenImage, ModelLumina2, select_precision_profile
from safetensors_quant import estimate_safetensors_output_size
from safetensors_quant_fp8 import quantize_fp8_scaled
from safetensors_quant_int8 import quantize_int8_tensorwise
from safetensors_quant_nvfp4 import _nearest_e2m1_index


@pytest.mark.parametrize("target", ["FP8_MIXED", "INT8_MIXED", "NVFP4_MIXED"])
def test_reconstructed_protected_weights_are_bf16_and_estimate_matches(tmp_path, target):
    """Verify real streaming payloads for prefixed, scaled FP8 mixed sources."""
    key = "transformer_blocks.0.img_mod.1.weight"
    tensors = quantize_fp8_scaled(torch.randn(128, 256), key)
    other_key = "transformer_blocks.1.img_mod.1.weight"
    tensors.update(quantize_fp8_scaled(torch.randn(128, 256), other_key))
    tensors["__index_timestep_zero__"] = torch.empty(0)
    tensors["norm_out.weight"] = torch.ones(32, dtype=torch.float32)
    source, output = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": {
        name.removesuffix(".weight"): {"format": "float8_e4m3fn"}
        for name in (key, other_key)}})}
    # Wrapped names exercise prefix normalization as well as source sidecar removal.
    save_file({"model.diffusion_model." + k: v for k, v in tensors.items()}, str(source), metadata=metadata)
    estimate = estimate_safetensors_output_size(str(source), target, ModelQwenImage())
    convert_to_safetensors(str(source), str(output), target_key=target,
                           model_arch=ModelQwenImage(), on_log=lambda _: None)
    header = _read_safetensors_header(str(output))
    actual = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    assert actual == estimate
    assert header[key]["dtype"] == "BF16"
    assert header[other_key]["dtype"] == {"FP8_MIXED": "F8_E4M3", "INT8_MIXED": "I8", "NVFP4_MIXED": "U8"}[target]
    assert header["norm_out.weight"]["dtype"] == "F32"
    assert load_file(str(output))["__index_timestep_zero__"].numel() == 0
    assert key.removesuffix(".weight") + ".weight_scale" not in header


def test_profiles_do_not_mutate_architecture_or_guess_turbo():
    """Keep family defaults independent and require explicit ambiguous variants."""
    arch = ModelQwenImage()
    chosen = select_precision_profile({"__index_timestep_zero__": None}, arch)
    assert "img_mod.1" in arch.keys_hiprec
    assert "img_mod.1" not in chosen.keys_hiprec
    assert "transformer_blocks.0.img_mod.1" in chosen.keys_hiprec
    assert select_precision_profile({}, arch).precision_profile == "conservative"
    turbo = select_precision_profile({}, ModelLumina2(), "z_image_turbo")
    assert "attention" not in turbo.keys_hiprec
    assert "attention" in select_precision_profile({}, ModelLumina2()).keys_hiprec
    with pytest.raises(ValueError):
        select_precision_profile({}, arch, "qwen_edit_2511")
    with pytest.raises(ValueError):
        select_precision_profile({}, arch, "z_image_turbo")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_int8_divides_in_source_dtype_with_underflow_guard(dtype):
    """Compare against typed division and cover FP16 scale underflow on zeros."""
    data = torch.linspace(-1, 1, 4096).to(dtype)
    result = quantize_int8_tensorwise(data, "x.weight")
    scale = result["x.weight_scale"].to(dtype)
    expected = (data / scale).round().clamp(-128, 127).to(torch.int8)
    assert torch.equal(expected, result["x.weight"])
    zeros = quantize_int8_tensorwise(torch.zeros(32, dtype=dtype), "x.weight")
    assert not zeros["x.weight"].any()


def test_e2m1_nearest_even_and_signed_zero():
    """Check every positive E2M1 boundary and representative negative encodings."""
    values = torch.tensor([0., -0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 7., -.75, -5.])
    assert _nearest_e2m1_index(values).tolist() == [0, 8, 0, 2, 2, 4, 4, 6, 6, 7, 10, 14]


def test_atomic_output_retries_windows_sharing_violation(tmp_path, monkeypatch):
    """Preserve published output while retrying a transient temporary-file lock."""
    from pathlib import Path
    import conversion_io

    unlink = Path.unlink
    calls = []

    def locked_once(path, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            exc = PermissionError("temporary sharing violation")
            exc.winerror = 32
            raise exc
        return unlink(path, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_once)
    monkeypatch.setattr(conversion_io.time, "sleep", lambda _: None)
    output = tmp_path / "checkpoint.safetensors"
    with conversion_io.atomic_output(output) as temporary:
        Path(temporary).write_bytes(b"complete checkpoint")
    assert output.read_bytes() == b"complete checkpoint"
    assert len(calls) == 2
    assert not Path(temporary).exists()
