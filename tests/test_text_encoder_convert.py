"""Tests for text_encoder_convert.py — auto-cloning llama.cpp and running convert_hf_to_gguf.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from text_encoder_convert import (
    TEXT_ENCODER_FORMAT_CHOICES,
    TEXT_ENCODER_OUTTYPES,
    TEXT_ENCODER_SAFETENSORS_FORMATS,
    ensure_llama_cpp,
    find_convert_script,
)


class _FakeTensor:
    """Shape-only stand-in for a torch tensor -- avoids allocating real GB-scale
    tensors just to test shape-signature detection."""
    def __init__(self, shape):
        self.shape = shape


class TestOuttypes:
    def test_is_list_of_tuples(self):
        assert isinstance(TEXT_ENCODER_OUTTYPES, list)
        for label, value in TEXT_ENCODER_OUTTYPES:
            assert isinstance(label, str) and label
            assert isinstance(value, str) and value


class TestFormatChoices:
    def test_is_list_of_tuples(self):
        assert isinstance(TEXT_ENCODER_FORMAT_CHOICES, list)
        for label, key in TEXT_ENCODER_FORMAT_CHOICES:
            assert isinstance(label, str) and label
            assert isinstance(key, str) and key

    def test_covers_gguf_direct_kquant_and_safetensors_formats(self):
        keys = {key for _, key in TEXT_ENCODER_FORMAT_CHOICES}
        assert {"F32", "F16", "BF16", "Q8_0"} <= keys
        assert {"Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q2_K"} <= keys
        assert TEXT_ENCODER_SAFETENSORS_FORMATS <= keys

    def test_size_savings_percentages_match_their_source_ratio_tables(self):
        # Hand-written literals here mirror the two other dropdowns' own
        # source-of-truth ratio tables -- quantize.py's SIZE_RATIOS for the
        # GGUF entries (this tab reuses the diffusion-model tab's own
        # percentages verbatim), safetensors_quant.py's _SIZE_RATIOS for the
        # safetensors entries -- keeping all three dropdowns from silently
        # drifting apart.
        import re

        from quantize import SIZE_RATIOS
        from safetensors_quant import _SIZE_RATIOS

        for label, key in TEXT_ENCODER_FORMAT_CHOICES:
            match = re.search(r"(\d+)% smaller than F16", label)
            if not match:
                continue
            ratios = SIZE_RATIOS if key in SIZE_RATIOS else _SIZE_RATIOS
            expected = round((1 - ratios[key]) * 100)
            assert int(match.group(1)) == expected, (
                f"{key!r} label says {match.group(1)}% but its ratio table gives {expected}%"
            )


class TestEnsureLlamaCpp:
    def test_returns_existing_dir_without_cloning(self, tmp_path):
        (tmp_path / "convert_hf_to_gguf.py").write_text("# stub")
        with patch("text_encoder_convert._llama_cpp_dir", return_value=tmp_path), \
             patch("text_encoder_convert.subprocess.run") as mock_run:
            found = ensure_llama_cpp()
            assert found == tmp_path
            mock_run.assert_not_called()

    def test_clones_when_missing(self, tmp_path):
        clone_dir = tmp_path / "llama.cpp"

        def _fake_clone(cmd, **kwargs):
            clone_dir.mkdir(parents=True)
            (clone_dir / "convert_hf_to_gguf.py").write_text("# stub")
            return subprocess.CompletedProcess(cmd, 0)

        with patch("text_encoder_convert._llama_cpp_dir", return_value=clone_dir), \
             patch("text_encoder_convert.subprocess.run", side_effect=_fake_clone) as mock_run:
            found = ensure_llama_cpp()
            assert found == clone_dir
            assert (clone_dir / "convert_hf_to_gguf.py").is_file()
            cmd = mock_run.call_args[0][0]
            assert cmd[:2] == ["git", "clone"]
            assert "--depth" in cmd
            assert str(clone_dir) in cmd
            assert "https://github.com/ggml-org/llama.cpp" in cmd

    def test_raises_runtime_error_when_git_not_found(self, tmp_path):
        with patch("text_encoder_convert._llama_cpp_dir", return_value=tmp_path / "llama.cpp"), \
             patch("text_encoder_convert.subprocess.run", side_effect=FileNotFoundError()):
            try:
                ensure_llama_cpp()
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "git" in str(exc).lower()

    def test_raises_runtime_error_when_clone_fails(self, tmp_path):
        err = subprocess.CalledProcessError(1, ["git", "clone"], stderr="fatal: network error")
        with patch("text_encoder_convert._llama_cpp_dir", return_value=tmp_path / "llama.cpp"), \
             patch("text_encoder_convert.subprocess.run", side_effect=err):
            try:
                ensure_llama_cpp()
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "network error" in str(exc)


class TestFindConvertScript:
    def test_returns_script_path_under_llama_cpp_dir(self, tmp_path):
        with patch("text_encoder_convert.ensure_llama_cpp", return_value=tmp_path):
            assert find_convert_script() == tmp_path / "convert_hf_to_gguf.py"


class TestFetchBaseConfigFiles:
    def test_downloads_config_and_available_tokenizer_files(self, tmp_path):
        from text_encoder_convert import fetch_base_config_files

        calls = []

        def _fake_download(repo_id, filename, local_dir):
            calls.append(filename)
            if filename not in ("config.json", "tokenizer.json"):
                raise Exception("not found")  # simulate missing optional file
            out = Path(local_dir) / filename
            out.write_text("{}")
            return str(out)

        with patch("text_encoder_convert.hf_hub_download", side_effect=_fake_download):
            downloaded = fetch_base_config_files("Qwen/Qwen3-8B", tmp_path)

        assert "config.json" in downloaded
        assert "tokenizer.json" in downloaded
        assert (tmp_path / "config.json").is_file()

    def test_raises_if_config_json_missing(self, tmp_path):
        from text_encoder_convert import fetch_base_config_files

        def _always_fail(repo_id, filename, local_dir):
            raise Exception("404")

        with patch("text_encoder_convert.hf_hub_download", side_effect=_always_fail):
            try:
                fetch_base_config_files("bad/repo", tmp_path)
                assert False, "expected RuntimeError"
            except RuntimeError:
                pass


class TestTextEncoderShapeCriticalKeys:
    def test_position_embedding_protected_under_every_lossy_format(self):
        # Real bug found 2026-08-14 testing SDXL's CLIP-L/bigG text encoders:
        # comfy/clip_model.py reads position_embedding.weight via a bare
        # attribute access (comfy.ops.cast_to(...)), bypassing dequant
        # entirely. NVFP4 crashed outright (halves the on-disk last dim --
        # "size of tensor a (768) must match tensor b (384)"); FP8/INT8
        # quantized silently and produced corrupted (black-image) output.
        import torch
        from safetensors_quant import quantize_tensor_st
        from text_encoder_convert import _TEXT_ENCODER_MODEL_ARCH

        data = torch.randn(77, 768, dtype=torch.float32)
        key = "text_model.embeddings.position_embedding.weight"
        for target_key in ("FP8", "INT8", "NVFP4"):
            out = quantize_tensor_st(data, key, _TEXT_ENCODER_MODEL_ARCH, target_key)
            assert set(out.keys()) == {key}, target_key
            assert out[key].dtype == torch.float16, target_key
            assert out[key].shape == data.shape, target_key

    def test_text_projection_protected_under_every_lossy_format(self):
        # Real bug found 2026-08-14: clip_g FP8 still rendered solid black
        # after the position_embedding fix above. text_projection.weight is
        # the ONLY weight feeding SDXL's global pooled "y" conditioning
        # vector (CLIP-L contributes no projected output at all) -- a single
        # global vector has no per-token error-averaging, so it's uniquely
        # exposed to quantization noise skewing the whole image.
        import torch
        from safetensors_quant import quantize_tensor_st
        from text_encoder_convert import _TEXT_ENCODER_MODEL_ARCH

        data = torch.randn(1280, 1280, dtype=torch.float32)
        key = "text_projection.weight"
        for target_key in ("FP8", "INT8", "NVFP4"):
            out = quantize_tensor_st(data, key, _TEXT_ENCODER_MODEL_ARCH, target_key)
            assert set(out.keys()) == {key}, target_key
            assert out[key].dtype == torch.float16, target_key
            assert out[key].shape == data.shape, target_key

    def test_relative_attention_bias_protected_under_every_lossy_format(self):
        # Real bug found 2026-08-18 testing HiDream's T5-XXL text encoder in
        # ComfyUI: relative_attention_bias.weight is a small [num_buckets,
        # num_heads] table read via a bare nn.Embedding lookup (comfy/
        # text_encoders/t5.py's T5Attention.forward), same unpacking gap as
        # position_embedding above. NVFP4/NVFP4_MIXED crashed loading with
        # "size mismatch ... torch.Size([32, 32]) ... current model is
        # torch.Size([32, 64])" (NVFP4 halves the on-disk last dim).
        import torch
        from safetensors_quant import quantize_tensor_st
        from text_encoder_convert import _TEXT_ENCODER_MODEL_ARCH

        data = torch.randn(32, 64, dtype=torch.float32)
        key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        for target_key in ("FP8", "INT8", "NVFP4"):
            out = quantize_tensor_st(data, key, _TEXT_ENCODER_MODEL_ARCH, target_key)
            assert set(out.keys()) == {key}, target_key
            assert out[key].dtype == torch.float16, target_key
            assert out[key].shape == data.shape, target_key


class TestGgufUnsupportedFamilies:
    def test_rejects_clip_family_via_base_repo_id_before_any_subprocess(self, tmp_path):
        # llama.cpp's convert_hf_to_gguf.py has no CLIPModel converter --
        # must fail fast, before fetch_base_config_files or Popen ever run.
        from text_encoder_convert import convert_text_encoder

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stub")
        with patch("text_encoder_convert.fetch_base_config_files") as mock_fetch, \
             patch("text_encoder_convert.subprocess.Popen") as mock_popen:
            try:
                convert_text_encoder(str(weights), "openai/clip-vit-large-patch14")
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "clip-l" in str(exc)
                assert "CLIPModel" in str(exc)
        mock_fetch.assert_not_called()
        mock_popen.assert_not_called()

    def test_rejects_auto_detected_clip_family(self, tmp_path):
        from text_encoder_convert import convert_text_encoder

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stub")
        with patch("text_encoder_convert.load_state_dict", return_value={}), \
             patch("text_encoder_convert.detect_text_encoder_family", return_value="clip-bigg") as _, \
             patch("text_encoder_convert.subprocess.Popen") as mock_popen:
            try:
                convert_text_encoder(str(weights))
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "clip-bigg" in str(exc)
        mock_popen.assert_not_called()


class TestConvertTextEncoder:
    def test_propagates_error_when_llama_cpp_unavailable(self, tmp_path):
        from text_encoder_convert import convert_text_encoder

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stub")
        with patch("text_encoder_convert.find_convert_script", side_effect=RuntimeError("git not found")):
            try:
                convert_text_encoder(str(weights), "Qwen/Qwen3-8B")
                assert False, "expected RuntimeError"
            except RuntimeError:
                pass

    def test_runs_subprocess_with_expected_args(self, tmp_path):
        from text_encoder_convert import convert_text_encoder

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stub")
        script = tmp_path / "convert_hf_to_gguf.py"
        script.write_text("# stub")

        with patch("text_encoder_convert.find_convert_script", return_value=script), \
             patch("text_encoder_convert.fetch_base_config_files", return_value=["config.json"]), \
             patch("text_encoder_convert.subprocess.Popen") as mock_popen:
            mock_proc = mock_popen.return_value
            mock_proc.stdout = iter(["INFO: done\n"])
            mock_proc.wait.return_value = 0
            mock_proc.returncode = 0

            out = convert_text_encoder(
                str(weights), "Qwen/Qwen3-8B", dst_path=str(tmp_path / "out.gguf"),
                outtype="f16",
            )

        assert out == str(tmp_path / "out.gguf")
        called_cmd = mock_popen.call_args[0][0]
        assert sys.executable == called_cmd[0]
        assert str(script) == called_cmd[1]
        assert "--outtype" in called_cmd
        assert "f16" in called_cmd

    def test_auto_detects_family_and_skips_repo_id_when_blank(self, tmp_path):
        # base_repo_id omitted entirely -- must fingerprint the weights and use the
        # matching vendored family instead of requiring a manual HF repo ID.
        from text_encoder_convert import convert_text_encoder

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stub")
        script = tmp_path / "convert_hf_to_gguf.py"
        script.write_text("# stub")

        qwen3_4b_shapes = {"model.embed_tokens.weight": _FakeTensor((151936, 2560))}
        qwen3_4b_shapes.update({
            f"model.layers.{i}.self_attn.q_proj.weight": _FakeTensor((2560, 2560))
            for i in range(36)
        })

        with patch("text_encoder_convert.find_convert_script", return_value=script), \
             patch("text_encoder_convert.load_state_dict", return_value=qwen3_4b_shapes), \
             patch("text_encoder_convert._copy_vendored_family") as mock_copy, \
             patch("text_encoder_convert.fetch_base_config_files") as mock_fetch, \
             patch("text_encoder_convert.subprocess.Popen") as mock_popen:
            mock_proc = mock_popen.return_value
            mock_proc.stdout = iter(["INFO: done\n"])
            mock_proc.wait.return_value = 0
            mock_proc.returncode = 0

            convert_text_encoder(str(weights), dst_path=str(tmp_path / "out.gguf"))

        mock_copy.assert_called_once()
        assert mock_copy.call_args[0][0] == "qwen3-4b"
        mock_fetch.assert_not_called()

    def test_raises_when_family_undetectable_and_no_repo_id_given(self, tmp_path):
        from text_encoder_convert import convert_text_encoder

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stub")
        script = tmp_path / "convert_hf_to_gguf.py"
        script.write_text("# stub")

        with patch("text_encoder_convert.find_convert_script", return_value=script), \
             patch("text_encoder_convert.load_state_dict", return_value={"some.unrelated.key": _FakeTensor((1, 1))}):
            try:
                convert_text_encoder(str(weights))
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "auto-detect" in str(exc)


class TestDetectTextEncoderFamily:
    def test_detects_qwen3_4b_from_shape_signature(self):
        from text_encoder_convert import detect_text_encoder_family

        state_dict = {"model.embed_tokens.weight": _FakeTensor((151936, 2560))}
        state_dict.update({
            f"model.layers.{i}.self_attn.q_proj.weight": _FakeTensor((2560, 2560))
            for i in range(36)
        })
        assert detect_text_encoder_family(state_dict) == "qwen3-4b"

    def test_distinguishes_ministral3_pe_from_mistral_small_24b(self):
        # These were the two families whose base repo ID ambiguity (both live
        # under different subfolders of baidu/ERNIE-Image / share Llama-style
        # key names) motivated shape-signature detection over key-name matching.
        from text_encoder_convert import detect_text_encoder_family

        ministral3_pe = {"model.embed_tokens.weight": _FakeTensor((131072, 3072))}
        ministral3_pe.update({
            f"model.layers.{i}.self_attn.q_proj.weight": _FakeTensor((3072, 3072))
            for i in range(26)
        })
        assert detect_text_encoder_family(ministral3_pe) == "ernie-image-pe"

        mistral_small = {"model.embed_tokens.weight": _FakeTensor((131072, 5120))}
        mistral_small.update({
            f"model.layers.{i}.self_attn.q_proj.weight": _FakeTensor((5120, 5120))
            for i in range(40)
        })
        assert detect_text_encoder_family(mistral_small) == "mistral-small-3.2-24b"

    def test_detects_llama_3_1_8b_from_shape_signature(self):
        # HiDream-I1's 4th text encoder (Llama-3.1-8B-Instruct); added
        # 2026-08-18 after this family had no signature entry at all.
        from text_encoder_convert import detect_text_encoder_family

        state_dict = {"model.embed_tokens.weight": _FakeTensor((128256, 4096))}
        state_dict.update({
            f"model.layers.{i}.self_attn.q_proj.weight": _FakeTensor((4096, 4096))
            for i in range(32)
        })
        assert detect_text_encoder_family(state_dict) == "llama-3.1-8b"

    def test_detects_umt5_xxl_from_shape_signature(self):
        # Wan 2.1/2.2's text encoder; added 2026-08-19 after this family had
        # no signature entry at all. Same hidden_size/layer count as t5-xxl,
        # distinguished only by vocab_size (256384 vs. 32128).
        from text_encoder_convert import detect_text_encoder_family

        state_dict = {"shared.weight": _FakeTensor((256384, 4096))}
        state_dict.update({
            f"encoder.block.{i}.layer.0.SelfAttention.q.weight": _FakeTensor((4096, 4096))
            for i in range(24)
        })
        assert detect_text_encoder_family(state_dict) == "umt5-xxl"

    def test_returns_none_for_unknown_shape(self):
        from text_encoder_convert import detect_text_encoder_family

        state_dict = {"model.embed_tokens.weight": _FakeTensor((999, 999))}
        assert detect_text_encoder_family(state_dict) is None

    def test_returns_none_without_embedding_key(self):
        from text_encoder_convert import detect_text_encoder_family

        assert detect_text_encoder_family({"some.other.weight": _FakeTensor((1, 1))}) is None


class TestLocateMsvcBuildEnv:
    def test_returns_none_when_not_windows(self):
        from text_encoder_convert import _locate_msvc_build_env

        with patch("text_encoder_convert.sys.platform", "linux"):
            assert _locate_msvc_build_env() is None

    def test_returns_none_when_cmake_already_on_path(self):
        from text_encoder_convert import _locate_msvc_build_env

        with patch("text_encoder_convert.sys.platform", "win32"), \
             patch("text_encoder_convert.shutil.which", return_value=r"C:\cmake\cmake.exe"):
            assert _locate_msvc_build_env() is None

    def test_returns_none_when_vswhere_missing(self):
        from text_encoder_convert import _locate_msvc_build_env

        with patch("text_encoder_convert.sys.platform", "win32"), \
             patch("text_encoder_convert.shutil.which", return_value=None), \
             patch("text_encoder_convert.Path.is_file", return_value=False):
            assert _locate_msvc_build_env() is None

    def test_parses_vcvarsall_environment_when_found(self, tmp_path):
        from text_encoder_convert import _locate_msvc_build_env

        vs_install = tmp_path / "VS"
        vcvarsall = vs_install / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        vcvarsall.parent.mkdir(parents=True)
        vcvarsall.write_text("@echo off")

        def _fake_run(cmd, **kwargs):
            if cmd[0].endswith("vswhere.exe"):
                return subprocess.CompletedProcess(cmd, 0, stdout=str(vs_install) + "\n")
            # simulates `cmd /c "call vcvarsall.bat x64 && set"` output
            return subprocess.CompletedProcess(
                cmd, 0, stdout="PATH=C:\\VS\\bin;C:\\Windows\nINCLUDE=C:\\VS\\include\n"
            )

        with patch("text_encoder_convert.sys.platform", "win32"), \
             patch("text_encoder_convert.shutil.which", return_value=None), \
             patch("text_encoder_convert.Path.is_file", return_value=True), \
             patch("text_encoder_convert.subprocess.run", side_effect=_fake_run):
            env = _locate_msvc_build_env()

        assert env is not None
        assert env["PATH"] == "C:\\VS\\bin;C:\\Windows"
        assert env["INCLUDE"] == "C:\\VS\\include"


class TestEnsurePlainLlamaQuantize:
    def test_returns_existing_built_binary_without_building(self, tmp_path):
        from text_encoder_convert import ensure_plain_llama_quantize

        build_dir = tmp_path / "build-quantize"
        (build_dir / "bin").mkdir(parents=True)
        binary = build_dir / "bin" / "llama-quantize"
        binary.write_bytes(b"stub")

        with patch("text_encoder_convert._quantize_build_dir", return_value=build_dir), \
             patch("text_encoder_convert.subprocess.run") as mock_run, \
             patch("text_encoder_convert.subprocess.Popen") as mock_popen:
            found = ensure_plain_llama_quantize()
            assert found == binary
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_builds_when_missing(self, tmp_path):
        from text_encoder_convert import ensure_plain_llama_quantize

        build_dir = tmp_path / "build-quantize"
        llama_cpp_dir = tmp_path / "llama.cpp"

        def _fake_build(*args, **kwargs):
            (build_dir / "bin").mkdir(parents=True)
            (build_dir / "bin" / "llama-quantize").write_bytes(b"stub")
            proc = type("P", (), {})()
            proc.stdout = iter(["[100%] Built target llama-quantize\n"])
            proc.wait = lambda: 0
            proc.returncode = 0
            return proc

        with patch("text_encoder_convert._quantize_build_dir", return_value=build_dir), \
             patch("text_encoder_convert.ensure_llama_cpp", return_value=llama_cpp_dir), \
             patch("text_encoder_convert._locate_msvc_build_env", return_value=None), \
             patch("text_encoder_convert.subprocess.run") as mock_run, \
             patch("text_encoder_convert.subprocess.Popen", side_effect=_fake_build):
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            found = ensure_plain_llama_quantize()

        assert found == build_dir / "bin" / "llama-quantize"
        configure_cmd = mock_run.call_args[0][0]
        assert configure_cmd[0] == "cmake"
        assert "-B" in configure_cmd and str(build_dir) in configure_cmd

    def test_resolves_cmake_absolute_path_from_located_msvc_env(self, tmp_path):
        # Regression: on Windows, subprocess's env= only sets the CHILD
        # process's environment block — CreateProcess still resolves a bare
        # "cmake" against the CALLING process's own PATH, not env's, so a
        # cmake found only via _locate_msvc_build_env's PATH would otherwise
        # raise FileNotFoundError despite being right there in that PATH.
        # ensure_plain_llama_quantize must resolve the absolute path itself.
        from text_encoder_convert import ensure_plain_llama_quantize

        build_dir = tmp_path / "build-quantize"
        llama_cpp_dir = tmp_path / "llama.cpp"
        vs_cmake_dir = tmp_path / "vs-cmake-bin"
        vs_cmake_dir.mkdir()
        resolved_cmake = str(vs_cmake_dir / "cmake.exe")
        msvc_env = {"PATH": str(vs_cmake_dir)}

        def _fake_build(*args, **kwargs):
            (build_dir / "bin").mkdir(parents=True)
            (build_dir / "bin" / "llama-quantize").write_bytes(b"stub")
            proc = type("P", (), {})()
            proc.stdout = iter([])
            proc.wait = lambda: 0
            proc.returncode = 0
            return proc

        with patch("text_encoder_convert._quantize_build_dir", return_value=build_dir), \
             patch("text_encoder_convert.ensure_llama_cpp", return_value=llama_cpp_dir), \
             patch("text_encoder_convert._locate_msvc_build_env", return_value=msvc_env), \
             patch("text_encoder_convert.shutil.which", return_value=resolved_cmake) as mock_which, \
             patch("text_encoder_convert.subprocess.run") as mock_run, \
             patch("text_encoder_convert.subprocess.Popen", side_effect=_fake_build) as mock_popen:
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            ensure_plain_llama_quantize()

        mock_which.assert_called_once_with("cmake", path=str(vs_cmake_dir))
        assert mock_run.call_args[0][0][0] == resolved_cmake
        assert mock_popen.call_args[0][0][0] == resolved_cmake

    def test_raises_runtime_error_when_cmake_not_found(self, tmp_path):
        from text_encoder_convert import ensure_plain_llama_quantize

        build_dir = tmp_path / "build-quantize"
        with patch("text_encoder_convert._quantize_build_dir", return_value=build_dir), \
             patch("text_encoder_convert.ensure_llama_cpp", return_value=tmp_path / "llama.cpp"), \
             patch("text_encoder_convert._locate_msvc_build_env", return_value=None), \
             patch("text_encoder_convert.subprocess.run", side_effect=FileNotFoundError()):
            try:
                ensure_plain_llama_quantize()
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "cmake" in str(exc).lower()


class TestConvertTextEncoderKquant:
    def test_runs_gguf_pass_then_quantize(self, tmp_path):
        from text_encoder_convert import convert_text_encoder_kquant

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"stub")
        dst = str(tmp_path / "out-Q4_K_M.gguf")
        exe = tmp_path / "llama-quantize"
        exe.write_bytes(b"stub")

        with patch("text_encoder_convert.convert_text_encoder") as mock_convert, \
             patch("text_encoder_convert.ensure_plain_llama_quantize", return_value=exe), \
             patch("text_encoder_convert.run_quantize") as mock_quantize:
            out = convert_text_encoder_kquant(
                str(weights), "Qwen/Qwen3-8B", dst_path=dst, quant_key="Q4_K_M",
            )

        assert out == dst
        mock_convert.assert_called_once()
        assert mock_convert.call_args.kwargs["outtype"] == "f16"
        mock_quantize.assert_called_once()
        quantize_args = mock_quantize.call_args
        assert quantize_args.args[1] == dst
        assert quantize_args.args[2] == "Q4_K_M"
        assert quantize_args.kwargs["exe"] == exe


class TestConvertTextEncoderToSafetensors:
    def test_quantizes_without_hf_download(self, tmp_path):
        import torch
        from safetensors.torch import save_file, load_file
        from text_encoder_convert import convert_text_encoder_to_safetensors

        src = tmp_path / "model.safetensors"
        save_file(
            {
                # 64x64 (4096 elems) — well above QUANTIZATION_THRESHOLD (1024),
                # so it doesn't get treated as a small hiprec tensor in *_MIXED mode.
                "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64, dtype=torch.float32),
                "model.layers.0.self_attn.q_proj.bias": torch.randn(64, dtype=torch.float32),
            },
            str(src),
        )

        with patch("text_encoder_convert.hf_hub_download") as mock_download:
            out = convert_text_encoder_to_safetensors(str(src), target_key="FP8_MIXED")
            mock_download.assert_not_called()

        result = load_file(out)
        # The "model." prefix must survive intact -- it's the tensor's own
        # genuine HF module path here (e.g. Qwen3's "model.layers.0...."),
        # not a diffusion-checkpoint wrapper artifact. Stripping it breaks
        # ComfyUI's own text-encoder architecture detection (comfy/sd.py's
        # detect_te_model looks for literal "model.layers.0...." keys) on
        # the output file, silently falling back to the wrong text-encoder
        # class at load time -- a real bug found via a live ComfyUI test.
        assert "model.layers.0.self_attn.q_proj.weight" in result
        assert "layers.0.self_attn.q_proj.weight" not in result
        assert result["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn
        assert "model.layers.0.self_attn.q_proj.weight_scale" in result
        # 1D bias always stays plain/unscaled
        assert "model.layers.0.self_attn.q_proj.bias_scale" not in result

    def test_int8_quantizes_without_hf_download(self, tmp_path):
        # INT8/INT8_MIXED were added to TEXT_ENCODER_FORMAT_CHOICES
        # 2026-08-13 -- convert_text_encoder_to_safetensors() already
        # supported them via the shared convert_to_safetensors() pipeline,
        # the dropdown just never offered them. Same "model." prefix
        # preservation and int8_tensorwise dtype as the FP8_MIXED test above.
        import torch
        from safetensors.torch import save_file, load_file
        from text_encoder_convert import convert_text_encoder_to_safetensors

        src = tmp_path / "model.safetensors"
        save_file(
            {
                "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64, dtype=torch.float32),
                "model.layers.0.self_attn.q_proj.bias": torch.randn(64, dtype=torch.float32),
            },
            str(src),
        )

        with patch("text_encoder_convert.hf_hub_download") as mock_download:
            out = convert_text_encoder_to_safetensors(str(src), target_key="INT8_MIXED")
            mock_download.assert_not_called()

        result = load_file(out)
        assert "model.layers.0.self_attn.q_proj.weight" in result
        assert "layers.0.self_attn.q_proj.weight" not in result
        assert result["model.layers.0.self_attn.q_proj.weight"].dtype == torch.int8
        assert "model.layers.0.self_attn.q_proj.weight_scale" in result

    def test_embedding_table_protected_from_nvfp4_packing(self, tmp_path):
        # NVFP4 halves a tensor's on-disk last dimension (2 values/byte) --
        # fine for weights ComfyUI dequantizes through its MixedPrecisionOps
        # Linear wrapper, but an nn.Embedding is loaded via a plain
        # load_state_dict with no such unpacking. Confirmed against a live
        # ComfyUI install: Qwen3-4B's NVFP4 output raised "size mismatch for
        # model.embed_tokens.weight: ... torch.Size([151936, 1280]) ... the
        # shape in current model is torch.Size([151936, 2560])".
        import torch
        from safetensors.torch import save_file, load_file
        from text_encoder_convert import convert_text_encoder_to_safetensors

        src = tmp_path / "model.safetensors"
        save_file(
            {"model.embed_tokens.weight": torch.randn(32, 64, dtype=torch.float32)},
            str(src),
        )

        with patch("text_encoder_convert.hf_hub_download") as mock_download:
            out = convert_text_encoder_to_safetensors(str(src), target_key="NVFP4")
            mock_download.assert_not_called()

        result = load_file(out)
        # Shape-critical fallback: stays F16, on-disk shape untouched -- no
        # NVFP4 packing, no .weight_scale sidecar.
        assert result["model.embed_tokens.weight"].dtype == torch.float16
        assert result["model.embed_tokens.weight"].shape == (32, 64)
        assert "model.embed_tokens.weight_scale" not in result


class TestConvertTextEncoderAny:
    def test_dispatches_safetensors_formats_without_repo_id(self, tmp_path):
        from text_encoder_convert import convert_text_encoder_any

        with patch("text_encoder_convert.convert_text_encoder_to_safetensors", return_value="out.safetensors") as mock_st:
            out = convert_text_encoder_any(str(tmp_path / "m.safetensors"), "", None, "NVFP4_MIXED")
        assert out == "out.safetensors"
        mock_st.assert_called_once()

    def test_f16_st_dispatches_to_safetensors_with_translated_target_key(self, tmp_path):
        # Regression: F16 alone is a GGUF outtype (test_dispatches_direct_
        # gguf_formats below) -- F16_ST is the distinct GUI-facing key that
        # produces a real .safetensors file instead, added after a user
        # selected "F16" expecting safetensors output and got a GGUF. F16_ST
        # itself isn't a real safetensors_quant target_key (it would log
        # "-> F16_ST" and name the file "...-F16_ST.safetensors") -- it must
        # be translated to the real "F16" key before reaching
        # convert_text_encoder_to_safetensors.
        from text_encoder_convert import convert_text_encoder_any

        with patch("text_encoder_convert.convert_text_encoder_to_safetensors", return_value="out.safetensors") as mock_st:
            out = convert_text_encoder_any(str(tmp_path / "m.safetensors"), "", None, "F16_ST")
        assert out == "out.safetensors"
        assert mock_st.call_args.kwargs["target_key"] == "F16"

    def test_dispatches_kquant_formats(self, tmp_path):
        from text_encoder_convert import convert_text_encoder_any

        with patch("text_encoder_convert.convert_text_encoder_kquant", return_value="out.gguf") as mock_kq:
            out = convert_text_encoder_any(str(tmp_path / "m.safetensors"), "Qwen/Qwen3-8B", None, "Q4_K_M")
        assert out == "out.gguf"
        mock_kq.assert_called_once()

    def test_dispatches_direct_gguf_formats(self, tmp_path):
        from text_encoder_convert import convert_text_encoder_any

        with patch("text_encoder_convert.convert_text_encoder", return_value="out.gguf") as mock_gguf:
            out = convert_text_encoder_any(str(tmp_path / "m.safetensors"), "Qwen/Qwen3-8B", None, "F16")
        assert out == "out.gguf"
        assert mock_gguf.call_args.kwargs["outtype"] == "f16"

    def test_unknown_format_raises(self, tmp_path):
        from text_encoder_convert import convert_text_encoder_any

        try:
            convert_text_encoder_any(str(tmp_path / "m.safetensors"), "repo", None, "BOGUS")
            assert False, "expected ValueError"
        except ValueError:
            pass
