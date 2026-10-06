# Pinned molbal source snapshot

This folder preserves the **complete, unmodified** tracked source tree from
`molbal/ComfyUI-GGUF` commit `5a0a3ffa0e3eae5c6af8b0b981b660a24d5fbc04`.
The archive and license are committed in this project, not just referenced by an
upstream URL or held in an ignored cache. `source.json` records provenance,
integrity and preparation status. There are no git submodules in this snapshot.

Verify without network access or executing third-party source:

```bash
uv run python scripts/verify_vendor_sources.py
```

The archive contains 52 regular files under `molbal-ComfyUI-GGUF-5a0a3ff/`.
`LICENSE` is an exact copy of the Apache-2.0 license inside the archive.
Upstream source comments/copyright notices remain untouched in the snapshot.
Any future modified extraction/adapter must identify our modifications separately.

This preserves molbal's source only. It does **not** yet bundle Python wheels,
ComfyUI/Kitchen, the existing two llama.cpp toolchains or platform binaries, and
does not claim a fully offline installation or verified Krea render.

See [Krea integration preparation](../../docs/krea2-backend-preparation.md).
