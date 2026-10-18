# Pinned text-conversion binary

`win32-x86_64` ships an unpatched MIT-licensed llama.cpp CPU quantizer, built
from the complete `llama-text-installed` archive already preserved in
`../converter_references`. It is separate from the City96 image toolchain.

`tools.json` records the source identity, exact file hashes and build commands.
The Windows C runtime is statically linked; only Windows system DLLs are
imported. Normal application use verifies these files and never builds them.

See [package documentation](../../docs/compiled-kernel-package.md#pinned-text-tools)
for qualification, maintenance rebuild instructions and remaining limits.
