"""Download a HuggingFace model repo and merge its safetensors shards into one file.

HuggingFace repos commonly split large checkpoints across multiple
model-NNNNN-of-NNNNN.safetensors shards. This tool's conversion pipeline
(convert_safetensors.py, convert.py) expects a single-file checkpoint, so this
module downloads every shard into a throwaway temp folder, streams tensors
into a single .safetensors file, then deletes the shards --
callers only ever see the merged file.

Some repos also ship more than one complete checkpoint side by side, each in
its own subfolder (e.g. an old and a "V2" variant) -- grabbing every
.safetensors file in the whole repo would merge tensors from two unrelated
checkpoints that happen to share key names, and neither one is a valid model
on its own. _find_shard_group() scopes the file list to a single subfolder
(explicit or auto-picked when there's only one) so this can't happen.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from conversion_io import atomic_output
from convert import _read_safetensors_header
from convert_safetensors import _build_header, _tensor_bytes, _write_header


def _find_shard_group(all_files: list[str], subfolder: str | None) -> tuple[str, list[str]]:
    """Return (folder, shard filenames) for the single model variant to download.

    Groups every .safetensors file in the repo by its containing folder ('' for
    repo root). If subfolder is given, that folder's shards are used directly
    (RuntimeError if it has none). Otherwise, auto-picks when exactly one folder
    has .safetensors files; if more than one does (a repo shipping several
    complete checkpoints side by side), raises RuntimeError listing them so the
    caller can retry with an explicit subfolder instead of silently merging
    tensors from two unrelated checkpoints.
    """
    groups: dict[str, list[str]] = {}
    for f in all_files:
        if not f.endswith(".safetensors"):
            continue
        folder = f.rsplit("/", 1)[0] if "/" in f else ""
        groups.setdefault(folder, []).append(f)

    if subfolder is not None:
        subfolder = subfolder.strip().strip("/")
        if subfolder not in groups:
            raise RuntimeError(
                f"Subfolder {subfolder!r} has no .safetensors files. "
                f"Available: {sorted(groups) or '(none)'}"
            )
        return subfolder, sorted(groups[subfolder])

    if not groups:
        raise RuntimeError("Repo has no .safetensors files.")
    if len(groups) > 1:
        raise RuntimeError(
            "Repo ships more than one checkpoint -- pick one with the subfolder "
            f"field. Available: {sorted(groups)}"
        )
    ((folder, shards),) = groups.items()
    return folder, sorted(shards)


def download_repo_as_single_safetensors(
    repo_id: str,
    dest_dir: str | Path,
    subfolder: str | None = None,
    overwrite: bool = False,
    on_log=None,
    on_progress=None,
    cancel_event=None,
) -> str:
    """Download one model variant's .safetensors shards and merge them into one file.

    subfolder scopes the download to one checkpoint when the repo ships more
    than one side by side (see _find_shard_group); leave it None for repos with
    a single checkpoint. Plans the output from shard headers and streams one
    tensor at a time into a temporary file; publishes the output only after
    success, preserving any existing output on errors or cancellation.
    Returns the path to the merged file. Raises
    RuntimeError("cancelled") if cancel_event is set mid-download or mid-merge,
    RuntimeError if the repo/subfolder has no .safetensors files, is ambiguous,
    or two shards define the same tensor key, and OSError if the output already
    exists and overwrite is False. On any of these, the partially-downloaded
    shards are left on disk (not cleaned up) so re-running with the same
    repo_id/subfolder resumes instead of starting over -- see the tmp_dir
    comment below.
    """
    def _log(msg: str) -> None:
        if on_log:
            on_log(msg)

    repo_id = repo_id.strip()
    if not repo_id:
        raise RuntimeError("No repo ID given.")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    _log(f"INFO:  Listing files in {repo_id} ...")
    all_files = HfApi().list_repo_files(repo_id)
    folder, shard_files = _find_shard_group(all_files, subfolder)
    _log(f"INFO:  Found {len(shard_files)} shard(s) in {folder or '(repo root)'}.")

    out_name = folder.rsplit("/", 1)[-1] if folder else repo_id.split("/")[-1]
    out_path = dest_dir / f"{out_name}.safetensors"
    if out_path.is_file() and not overwrite:
        raise OSError(f"Output exists and overwrite is disabled: {out_path}")

    # Deliberately NOT a try/finally cleanup: on cancel or error, the partial
    # shards are left in tmp_dir instead of being wiped, so re-running with the
    # same repo_id/subfolder resumes via hf_hub_download's own .incomplete-file
    # resume logic (and skips shards that already finished) instead of
    # re-downloading everything from zero. Only a clean, full success cleans up.
    tmp_dir = dest_dir / f".hf_download_{repo_id.replace('/', '_')}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    local_paths: list[Path] = []
    for idx, filename in enumerate(shard_files):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")
        if on_progress:
            on_progress(idx, len(shard_files), f"downloading {filename}")
        _log(f"INFO:  Downloading {filename} ...")
        local_paths.append(Path(hf_hub_download(repo_id, filename, local_dir=str(tmp_dir))))

    # Plan from headers only, then stream one tensor at a time. Retaining every
    # shard's tensors in a merged dict defeats >RAM checkpoint conversion.
    entries = []
    seen = set()
    for shard_path in local_paths:
        header = _read_safetensors_header(str(shard_path))
        for key, info in header.items():
            if key == "__metadata__":
                continue
            if key in seen:
                raise RuntimeError(f"Duplicate tensor key {key!r} across shards -- refusing to merge.")
            seen.add(key)
            entries.append((key, info["dtype"], tuple(info["shape"])))
    header, _ = _build_header(entries, {})
    _log(f"INFO:  Writing {len(entries)} tensors -> {out_path}")
    with atomic_output(out_path, overwrite) as temporary:
        with open(temporary, "wb") as output:
            _write_header(output, header)
            for idx, shard_path in enumerate(local_paths):
                if on_progress:
                    on_progress(idx, len(local_paths), f"merging {shard_path.name}")
                _log(f"INFO:  Merging {shard_path.name} ...")
                with safe_open(str(shard_path), framework="pt") as shard:
                    # Preserve planning order: safetensors keys() may sort keys
                    # differently from the original JSON header's insertion order.
                    for key in _read_safetensors_header(str(shard_path)):
                        if key == "__metadata__":
                            continue
                        if cancel_event is not None and cancel_event.is_set():
                            raise RuntimeError("cancelled")
                        output.write(_tensor_bytes(shard.get_tensor(key)))
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    _log(f"INFO:  Done -> {out_path}")
    return str(out_path)
