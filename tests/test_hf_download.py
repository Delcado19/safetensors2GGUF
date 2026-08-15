import os
import shutil

import torch
from safetensors.torch import save_file

import hf_download


def _fake_hf_hub_download(repo_id, filename, local_dir):
    """Copy a pre-made shard from the fake 'hub' into local_dir, like the real download would."""
    src = _FAKE_HUB[repo_id][filename]
    dst = local_dir + "/" + filename
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(src, dst)
    return dst


_FAKE_HUB: dict = {}


class _FakeHfApi:
    def __init__(self, files):
        self._files = files

    def list_repo_files(self, repo_id):
        return self._files


def test_download_repo_as_single_safetensors_merges_shards(tmp_path, monkeypatch):
    # Build two fake shards on disk, register them as what hf_hub_download "downloads".
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard_a = shard_dir / "model-00001-of-00002.safetensors"
    shard_b = shard_dir / "model-00002-of-00002.safetensors"
    save_file({"layer.a.weight": torch.zeros(2, 2)}, str(shard_a))
    save_file({"layer.b.weight": torch.ones(3)}, str(shard_b))

    repo_id = "someorg/some-model"
    _FAKE_HUB[repo_id] = {
        "model-00001-of-00002.safetensors": str(shard_a),
        "model-00002-of-00002.safetensors": str(shard_b),
    }

    monkeypatch.setattr(hf_download, "HfApi", lambda: _FakeHfApi(list(_FAKE_HUB[repo_id])))
    monkeypatch.setattr(hf_download, "hf_hub_download", _fake_hf_hub_download)

    dest = tmp_path / "out"
    logs = []
    out_path = hf_download.download_repo_as_single_safetensors(repo_id, dest, on_log=logs.append)

    assert out_path == str(dest / "some-model.safetensors")
    from safetensors import safe_open
    with safe_open(out_path, framework="pt") as f:
        keys = set(f.keys())
        assert keys == {"layer.a.weight", "layer.b.weight"}
        assert torch.equal(f.get_tensor("layer.b.weight"), torch.ones(3))

    # temp shard folder must be cleaned up -- callers only ever see the merged file
    assert not (dest / f".hf_download_{repo_id.replace('/', '_')}").exists()


def test_cancelled_download_leaves_partial_shards_for_resume(tmp_path, monkeypatch):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard_a = shard_dir / "model-00001-of-00002.safetensors"
    shard_b = shard_dir / "model-00002-of-00002.safetensors"
    save_file({"layer.a.weight": torch.zeros(2, 2)}, str(shard_a))
    save_file({"layer.b.weight": torch.ones(3)}, str(shard_b))

    repo_id = "someorg/cancel-me"
    _FAKE_HUB[repo_id] = {
        "model-00001-of-00002.safetensors": str(shard_a),
        "model-00002-of-00002.safetensors": str(shard_b),
    }
    monkeypatch.setattr(hf_download, "HfApi", lambda: _FakeHfApi(list(_FAKE_HUB[repo_id])))
    monkeypatch.setattr(hf_download, "hf_hub_download", _fake_hf_hub_download)

    dest = tmp_path / "out"

    class _AlreadyCancelled:
        def is_set(self):
            return True

    try:
        hf_download.download_repo_as_single_safetensors(
            repo_id, dest, cancel_event=_AlreadyCancelled()
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc) == "cancelled"

    # unlike a successful run, a cancelled run must NOT delete the temp
    # folder -- that's what lets a retry resume instead of starting at 0
    assert (dest / f".hf_download_{repo_id.replace('/', '_')}").exists()


def test_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    dest = tmp_path / "out"
    dest.mkdir()
    existing = dest / "some-model.safetensors"
    existing.write_bytes(b"placeholder")

    monkeypatch.setattr(hf_download, "HfApi", lambda: _FakeHfApi(["model.safetensors"]))

    try:
        hf_download.download_repo_as_single_safetensors("someorg/some-model", dest, overwrite=False)
        assert False, "expected OSError"
    except OSError:
        pass


def test_raises_when_repo_has_no_safetensors_files(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_download, "HfApi", lambda: _FakeHfApi(["README.md", "config.json"]))
    try:
        hf_download.download_repo_as_single_safetensors("someorg/no-weights", tmp_path)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def _setup_two_variant_repo(tmp_path, monkeypatch):
    # Mirrors Lockout/qwen3-4b-heretic-zimage: two complete checkpoints, each in
    # its own subfolder, sharing tensor key names -- a real repo layout that
    # broke the original "grab every .safetensors in the repo" approach with a
    # spurious "duplicate tensor key" merge failure.
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    v1 = shard_dir / "v1.safetensors"
    v2 = shard_dir / "v2.safetensors"
    save_file({"model.embed_tokens.weight": torch.zeros(2, 2)}, str(v1))
    save_file({"model.embed_tokens.weight": torch.ones(2, 2)}, str(v2))

    repo_id = "someorg/two-variants"
    _FAKE_HUB[repo_id] = {
        "variant-a/model.safetensors": str(v1),
        "variant-b/model.safetensors": str(v2),
    }
    monkeypatch.setattr(hf_download, "HfApi", lambda: _FakeHfApi(list(_FAKE_HUB[repo_id])))
    monkeypatch.setattr(hf_download, "hf_hub_download", _fake_hf_hub_download)
    return repo_id


def test_raises_on_ambiguous_repo_with_multiple_checkpoint_variants(tmp_path, monkeypatch):
    repo_id = _setup_two_variant_repo(tmp_path, monkeypatch)
    try:
        hf_download.download_repo_as_single_safetensors(repo_id, tmp_path / "out")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "variant-a" in str(exc) and "variant-b" in str(exc)


def test_subfolder_scopes_download_to_one_variant(tmp_path, monkeypatch):
    repo_id = _setup_two_variant_repo(tmp_path, monkeypatch)
    dest = tmp_path / "out"

    out_path = hf_download.download_repo_as_single_safetensors(repo_id, dest, subfolder="variant-b")

    assert out_path == str(dest / "variant-b.safetensors")
    from safetensors import safe_open
    with safe_open(out_path, framework="pt") as f:
        assert torch.equal(f.get_tensor("model.embed_tokens.weight"), torch.ones(2, 2))
