#!/bin/sh
# Launch from the repository even when invoked by an absolute path with spaces.
set -eu
case "$0" in
    */*) cd -- "${0%/*}" ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' 'ERROR: uv is not installed or not on PATH.' \
        'Install it from https://docs.astral.sh/uv/ and restart your terminal.' >&2
    exit 1
fi

if [ ! -f frontend/dist/index.html ]; then
    if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
        printf '%s\n' 'ERROR: Install Node.js/npm to build the frontend on first launch.' >&2
        exit 1
    fi
    if ! (cd frontend && npm ci && npm run build); then
        printf '%s\n' 'ERROR: Frontend build failed. See the output above.' >&2
        exit 1
    fi
fi

printf '%s\n' 'Starting safetensors2GGUF Workbench at http://127.0.0.1:8765' \
    'Open that URL in your browser. Press Ctrl+C to stop the server.'
# Replace the shell so terminal signals and the server exit status propagate directly.
exec uv run --frozen python web_api.py
