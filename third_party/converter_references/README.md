# Frozen converter analysis references

These complete upstream tracked-source archives are **analysis references**.
They do not replace or update any production backend. `sources.json` records
exact revisions, upstream download URLs, file counts, archive sizes and SHA-256
checksums. Root licenses are copied beside each archive; bundled license and
copyright notices remain inside the unmodified upstream trees.

| Reference | Role |
| --- | --- |
| `llama-b3962` | Original diffusion quantizer base |
| `city96-patch` | Complete ComfyUI-GGUF tools tree containing `tools/lcpp.patch` |
| `llama-text-installed` | Existing local text-encoder checkout, 2026-08-09 |
| `llama-current` | Upstream text-encoder comparison, frozen on 2026-10-12 |
| `molbal-current` | Molbal comparison, 2026-10-11; production molbal stays pinned separately |
| `kompute-b3962` | Full source for b3962's `ggml/src/kompute` git submodule |

To reconstruct b3962, unpack its archive and place the Kompute archive's contents
at `ggml/src/kompute`. Apply `tools/lcpp.patch` from the City96 archive to the
b3962 tree. A read-only `git apply --check` against the archived files passed.
See [the existing build recipe](../../docs/building-llama-quantize.md).

Verify and regenerate the static inventory without network access or importing
third-party code:

```bash
uv run python scripts/inventory_converter_sources.py --output data/converter-source-inventory.json
```

The archives total approximately 104 MiB. There are no additional gitlinks in
the saved upstream trees, including the Kompute source. Snapshotting tracked
source does not bundle Python wheels, SDKs, compiler tools, or optional build-time
downloads. A verified source snapshot is not proof of a fully offline build.
New builds, output parity and runtime compatibility remain pending.

See [the consolidation inventory](../../docs/converter-consolidation-inventory.md).
