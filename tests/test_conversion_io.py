"""Regression checks for output preservation, job isolation and silent cancellation."""

from contextlib import closing
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import gguf
import pytest
import torch
from safetensors.torch import load_file, save_file

from conversion_io import atomic_output, iter_process_output, validate_output
from convert import ConversionCancelled, convert_file
from convert_safetensors import convert_to_safetensors
from models.architectures import ModelTemplate, ModelWan
import gui
from quantize import run_quantize
from text_encoder_convert import convert_text_encoder
from fix_5d_tensors import fix_5d_tensors
from fix_pad_tokens import fix_pad_tokens


@pytest.mark.parametrize("failure", [RuntimeError("cancelled"), OSError("disk full")])
def test_atomic_output_preserves_existing_destination(tmp_path, failure):
    destination = tmp_path / "model.gguf"
    destination.write_bytes(b"original")
    with pytest.raises(type(failure), match=str(failure)):
        with atomic_output(destination, overwrite=True) as temporary:
            Path(temporary).write_bytes(b"partial")
            raise failure
    assert destination.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_output_does_not_replace_concurrently_created_destination(tmp_path):
    destination = tmp_path / "model.gguf"
    with pytest.raises(FileExistsError):
        with atomic_output(destination) as temporary:
            Path(temporary).write_bytes(b"new")
            destination.write_bytes(b"other job")
    assert destination.read_bytes() == b"other job"
    assert list(tmp_path.iterdir()) == [destination]


def test_empty_success_cannot_replace_existing_output(tmp_path):
    destination = tmp_path / "output.gguf"
    destination.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="empty output"):
        with atomic_output(destination, overwrite=True):
            pass
    assert destination.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [destination]


def test_output_validation_rejects_hard_link_alias(tmp_path):
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"source")
    alias = tmp_path / "alias.safetensors"
    os.link(source, alias)
    with pytest.raises(ValueError, match="different files"):
        validate_output(source, alias, overwrite=True)


@pytest.mark.parametrize("during_write", [False, True])
def test_safetensors_cancellation_preserves_existing_output(tmp_path, during_write):
    source, destination = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    save_file({"a.weight": torch.ones(2, 2), "b.weight": torch.ones(2, 2)}, str(source))
    destination.write_bytes(b"original")
    event = threading.Event()
    if not during_write:
        event.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        convert_to_safetensors(
            str(source), str(destination), target_key="F16", overwrite=True,
            model_arch=ModelTemplate(), cancel_event=event, on_log=lambda _: None,
            on_progress=lambda *_: event.set(),
        )
    assert destination.read_bytes() == b"original"
    assert set(load_file(str(source))) == {"a.weight", "b.weight"}
    assert len(list(tmp_path.iterdir())) == 2


def test_safetensors_rejects_source_as_destination(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file({"weight": torch.ones(2, 2)}, str(source))
    original = source.read_bytes()
    with pytest.raises(ValueError, match="different files"):
        convert_to_safetensors(
            str(source), str(source), target_key="F16", overwrite=True,
            model_arch=ModelTemplate(), on_log=lambda _: None,
        )
    assert source.read_bytes() == original


def test_gguf_write_failure_preserves_existing_output(tmp_path):
    source, destination = tmp_path / "source.safetensors", tmp_path / "output.gguf"
    save_file({"double_blocks.0.img_attn.proj.weight": torch.ones(2, 2)}, str(source))
    destination.write_bytes(b"original")
    with patch("convert.gguf.GGUFWriter.write_tensors_to_file", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            convert_file(str(source), str(destination), interact=False, overwrite=True, on_log=lambda _: None)
    assert destination.read_bytes() == b"original"
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("quant_key", ["F16", "Q4_K_M"])
def test_gui_refuses_existing_final_output_before_conversion(tmp_path, quant_key):
    destination = tmp_path / "output.gguf"
    destination.write_bytes(b"original")
    with patch("gui.convert_file") as convert:
        with pytest.raises(OSError, match="overwrite is disabled"):
            gui._pipeline("source.safetensors", str(destination), quant_key, "", 0, False, False, queue.Queue())
    convert.assert_not_called()
    assert destination.read_bytes() == b"original"


@pytest.mark.parametrize("overwrite", [False, True])
def test_gui_forwards_overwrite_for_direct_gguf(tmp_path, overwrite):
    destination = str(tmp_path / "output.gguf")
    arch = SimpleNamespace(arch="flux", fix_path=destination + ".5d.safetensors")
    with patch("gui.convert_file", return_value=(destination, arch)) as convert:
        assert gui._pipeline("source.safetensors", destination, "F16", "", 0, False, overwrite, queue.Queue()) == destination
    assert convert.call_args.kwargs["overwrite"] is overwrite


def test_5d_sidecars_isolate_interleaved_jobs_and_discard_stale_keys(tmp_path):
    first, second = ModelWan(), ModelWan()
    first.fix_path = str(tmp_path / "first.gguf.5d.safetensors")
    second.fix_path = str(tmp_path / "second.gguf.5d.safetensors")
    save_file({"stale": torch.zeros(1)}, first.fix_path)
    data = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    first.handle_nd_tensor("first", data)
    second.handle_nd_tensor("second", data * 2)
    first.handle_nd_tensor("extra", data * 3)
    assert set(load_file(first.fix_path)) == {"first", "extra"}
    assert set(load_file(second.fix_path)) == {"second"}


@pytest.mark.parametrize("close_stdout", [False, True])
def test_silent_child_is_cancelled_and_reaped(tmp_path, close_stdout):
    # Closing stdout covers the separate wait-for-exit path as well as pipe reads.
    script = "import os,time; print('ready',flush=True); "
    if close_stdout:
        script += "os.close(1); "
    script += "time.sleep(60)"
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    event = threading.Event()
    with closing(iter_process_output(proc, event, ConversionCancelled)) as lines:
        assert next(lines).strip() == "ready"
        timer = threading.Timer(0.2, event.set)
        timer.start()
        try:
            with pytest.raises(ConversionCancelled):
                list(lines)
        finally:
            timer.cancel()
            timer.join()
    assert proc.poll() is not None


def test_quantize_failure_preserves_existing_output(tmp_path):
    source, destination, exe = tmp_path / "source.gguf", tmp_path / "output.gguf", tmp_path / "quantize.exe"
    source.write_bytes(b"source")
    destination.write_bytes(b"original")
    exe.touch()
    proc = SimpleNamespace(stdout=iter([]), returncode=1, wait=lambda **_: None)
    with patch("quantize.subprocess.Popen", return_value=proc):
        with pytest.raises(RuntimeError, match="code 1"):
            run_quantize(source, destination, "Q4_K_M", exe=exe, overwrite=True)
    assert destination.read_bytes() == b"original"


def test_text_encoder_refuses_existing_output_before_network_work(tmp_path):
    destination = tmp_path / "output.gguf"
    destination.write_bytes(b"original")
    with patch("text_encoder_convert.find_convert_script") as find_script:
        with pytest.raises(OSError, match="overwrite is disabled"):
            convert_text_encoder("source.safetensors", "Qwen/Qwen3-8B", str(destination))
    find_script.assert_not_called()
    assert destination.read_bytes() == b"original"


def test_text_encoder_process_failure_preserves_existing_output(tmp_path):
    source, destination = tmp_path / "source.safetensors", tmp_path / "output.gguf"
    save_file({"weight": torch.ones(2, 2)}, str(source))
    destination.write_bytes(b"original")
    proc = SimpleNamespace(stdout=iter([]), returncode=1, wait=lambda **_: None)
    with (
        patch("text_encoder_convert.find_convert_script", return_value=tmp_path / "converter.py"),
        patch("text_encoder_convert.fetch_base_config_files"),
        patch("text_encoder_convert.subprocess.Popen", return_value=proc),
    ):
        with pytest.raises(RuntimeError, match="code 1"):
            convert_text_encoder(str(source), "Qwen/Qwen3-8B", str(destination), overwrite=True)
    assert destination.read_bytes() == b"original"
    assert len(list(tmp_path.iterdir())) == 2


def test_log_callback_failure_reaps_child():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; print('ready',flush=True); time.sleep(60)"],
        stdout=subprocess.PIPE, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    with pytest.raises(RuntimeError, match="callback"):
        with closing(iter_process_output(proc)) as lines:
            assert next(lines).strip() == "ready"
            raise RuntimeError("callback")
    assert proc.poll() is not None


@pytest.mark.parametrize("repair", [fix_pad_tokens, fix_5d_tensors])
def test_gguf_repair_write_failure_preserves_existing_output(tmp_path, repair):
    source, destination = tmp_path / "source.gguf", tmp_path / "output.gguf"
    writer = gguf.GGUFWriter(str(source), "lumina2")
    writer.add_tensor("x_pad_token", np.zeros(16, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    save_file({"extra.weight": torch.ones(1, 1, 1, 1, 1)}, str(source) + ".5d.safetensors")
    destination.write_bytes(b"original")
    with patch("gguf.GGUFWriter.write_tensors_to_file", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            repair(str(source), str(destination), overwrite=True, on_log=lambda _: None)
    assert destination.read_bytes() == b"original"
    assert not list(tmp_path.glob(".output.gguf.*"))


def test_wan_pipeline_uses_only_current_jobs_5d_tensors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source, destination = tmp_path / "source.safetensors", tmp_path / "output.gguf"
    save_file({
        "blocks.0.self_attn.norm_q.weight": torch.ones(8),
        "text_embedding.2.weight": torch.ones(2, 2),
        "head.modulation": torch.ones(1, 2, 2),
        "current.weight": torch.ones(1, 1, 1, 1, 1),
    }, str(source))
    save_file({"stale.weight": torch.zeros(1, 1, 1, 1, 1)}, "fix_5d_tensors_wan.safetensors")

    def fake_quantize(src, dst, *args, **kwargs):
        shutil.copyfile(src, dst)

    with patch("gui.run_quantize", side_effect=fake_quantize):
        result = gui._pipeline(
            str(source), str(destination), "Q4_K_M", "", 0, False, False, queue.Queue(),
        )
    reader = gguf.GGUFReader(result)
    names = {tensor.name for tensor in reader.tensors}
    assert "current.weight" in names
    assert "stale.weight" not in names
    assert result == str(tmp_path / "output-fixed.gguf")


@pytest.mark.parametrize("failure_point", ["write", "publish"])
def test_5d_replacement_failure_preserves_previous_companion(tmp_path, failure_point):
    source, destination = tmp_path / "source.safetensors", tmp_path / "output.gguf"
    save_file({"current.weight": torch.ones(1, 1, 1, 1, 1), "norm": torch.ones(2)}, str(source))
    with patch("convert.detect_arch", side_effect=lambda _: ModelWan()):
        _, arch = convert_file(str(source), str(destination), interact=False, on_log=lambda _: None)
        original = destination.read_bytes()
        companion = Path(arch.fix_path)
        original_companion = companion.read_bytes()
        target = "convert.gguf.GGUFWriter.write_tensors_to_file" if failure_point == "write" else "conversion_io.os.replace"
        with patch(target, side_effect=OSError("disk failure")):
            with pytest.raises(OSError, match="disk failure"):
                convert_file(str(source), str(destination), interact=False, overwrite=True, on_log=lambda _: None)
    assert destination.read_bytes() == original
    assert companion.read_bytes() == original_companion
    assert list(tmp_path.glob("*.5d.safetensors")) == [companion]


def test_5d_metadata_discovers_job_specific_companion(tmp_path):
    source, destination = tmp_path / "source.safetensors", tmp_path / "output.gguf"
    save_file({"current.weight": torch.ones(1, 1, 1, 1, 1), "norm": torch.ones(2)}, str(source))
    with patch("convert.detect_arch", side_effect=lambda _: ModelWan()):
        convert_file(str(source), str(destination), interact=False, on_log=lambda _: None)
    fixed = tmp_path / "fixed.gguf"
    inserted = fix_5d_tensors(str(destination), str(fixed), on_log=lambda _: None)
    assert inserted == ["current.weight"]
    assert "current.weight" in {tensor.name for tensor in gguf.GGUFReader(str(fixed)).tensors}
