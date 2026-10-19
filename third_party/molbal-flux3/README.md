# Pinned FLUX.3 adapter source

This complete Apache-2.0 source archive comes from
https://github.com/molbal/ComfyUI-GGUF at commit
`b07c7e0437eeb4b2dab6e96509cbb8efdabf1b01`.
The source archive and unchanged upstream license are distributed here;
`source.json` records its exact size, SHA256, root and file count.
`scripts.verify_vendor_sources.verify_snapshot` verifies it offline.

Only the FLUX.3 extended adapter selects this snapshot. Existing Molbal adapters
retain their independently pinned sources. The isolated worker replaces upstream
K-block math with the project's verified owned CPU kernels and uses its existing
payload manifest/readback guard. No source download occurs during conversion.

Upstream source notices and bundled dependencies remain inside the full archive.
No new loader is installed automatically. Use a compatible FLUX.3-capable Molbal
loader for this GGUF architecture; export success does not establish inference
quality or compatibility with older City96 loaders.
