"""Offline source preservation and tamper detection for the pinned molbal backend."""
import shutil
from pathlib import Path

import pytest
from scripts.verify_vendor_sources import verify_snapshot


def test_snapshot_is_complete_and_detects_corruption(tmp_path):
    """The tracked snapshot verifies locally; altered payloads fail before extraction/import."""
    assert verify_snapshot()['files'] == 52
    original = Path(__file__).resolve().parents[1] / 'third_party' / 'molbal'
    for filename in ('source.json', 'source.tar.gz', 'LICENSE'):
        shutil.copyfile(original / filename, tmp_path / filename)
    archive = tmp_path / 'source.tar.gz'
    with archive.open('ab') as output:
        output.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        verify_snapshot(tmp_path)


def test_flux3_snapshot_is_complete_and_separately_pinned():
    folder = Path(__file__).resolve().parents[1] / 'third_party/molbal-flux3'
    result = verify_snapshot(folder)
    assert result['files'] == 54
    assert result['commit'] == 'b07c7e0437eeb4b2dab6e96509cbb8efdabf1b01'
    assert result['commit'] != verify_snapshot()['commit']
