"""Transactional output files and cancellable subprocess output for converters."""

from contextlib import contextmanager
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading


def validate_output(src, dst, overwrite=False):
    """Reject source aliases and existing destinations unless overwrite is enabled."""
    source, destination = Path(src), Path(dst)
    if source.resolve() == destination.resolve() or (
        source.exists() and destination.exists() and os.path.samefile(source, destination)
    ):
        raise ValueError("Source and destination must be different files.")
    if destination.exists() and not overwrite:
        raise OSError(f"Output exists and overwrite is disabled: {dst}")


@contextmanager
def atomic_output(dst, overwrite=False):
    """Yield a sibling temporary path; publish it only after successful completion.

    A hard link publishes without replacing a concurrently created destination
    when overwrite is disabled. Keeping the temporary file on the destination
    filesystem makes replacement atomic and avoids cross-volume moves.
    """
    destination = Path(dst)
    if destination.exists() and not overwrite:
        raise OSError(f"Output exists and overwrite is disabled: {dst}")
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=destination.suffix
    )
    os.close(fd)
    try:
        yield temporary
        # A zero-exit subprocess may still fail to write its promised output.
        if Path(temporary).stat().st_size == 0:
            raise RuntimeError(f"Converter produced an empty output: {dst}")
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def iter_process_output(proc, cancel_event=None, cancel_error=None):
    """Yield process lines while polling cancellation even when stdout is silent.

    Always reap the child, including when a log callback fails. The reader thread
    keeps blocking pipe reads away from cancellation checks on Windows.
    """
    messages = queue.Queue()

    def read_stdout():
        try:
            for line in proc.stdout:
                messages.put(line)
        except Exception as exc:
            messages.put(exc)
        finally:
            messages.put(None)

    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise cancel_error() if cancel_error else RuntimeError("cancelled")

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    try:
        while True:
            check_cancelled()
            try:
                message = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if message is None:
                break
            if isinstance(message, Exception):
                raise message
            yield message
        while True:
            check_cancelled()
            try:
                proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
        check_cancelled()
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    finally:
        reader.join(timeout=1)
        if hasattr(proc.stdout, "close"):
            proc.stdout.close()
