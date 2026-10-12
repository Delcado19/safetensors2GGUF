"""Portable launcher tests with stub tools; no installs, builds or servers run."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TestLinuxLauncher(unittest.TestCase):
    """Exercise actual POSIX shell branching and process exit propagation."""

    def setUp(self):
        """Create an isolated repository and tool directory for each launcher check."""
        self.shell = shutil.which('sh')
        if not self.shell:
            self.skipTest('POSIX shell is unavailable')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo with spaces'
        (self.root / 'frontend' / 'dist').mkdir(parents=True)
        self.bin = Path(self.temp.name) / 'tools'
        self.bin.mkdir()
        self.log = Path(self.temp.name) / 'calls.log'
        self.script = self.root / 'start_gui.sh'
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'start_gui.sh', self.script)

    def _tool(self, name, body):
        path = self.bin / name
        path.write_text('#!/bin/sh\n' + body, encoding='utf-8', newline='\n')
        path.chmod(0o755)

    def _run(self, **environment):
        result = subprocess.run([self.shell, self.script.as_posix()], cwd=self.temp.name,
            env={**os.environ, 'PATH': self.bin.as_posix(), 'FAKE_LOG': self.log.as_posix(), **environment},
            text=True, capture_output=True, timeout=10)
        calls = self.log.read_text().splitlines() if self.log.exists() else []
        return result, calls

    def test_missing_uv_explains_requirement(self):
        """Fail before doing work and explain how to install the missing launcher tool."""
        result, calls = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn('uv is not installed', result.stderr)
        self.assertIn('https://docs.astral.sh/uv/', result.stderr)
        self.assertEqual(calls, [])

    def test_missing_node_does_not_start_server(self):
        """A missing frontend requires build tools before the server may run."""
        self._tool('uv', 'printf "uv\n" >> "$FAKE_LOG"\n')
        result, calls = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn('Install Node.js/npm', result.stderr)
        self.assertEqual(calls, [])

    def test_existing_frontend_skips_npm_and_propagates_exit(self):
        """An existing build needs only uv, and invocation from another cwd remains safe."""
        (self.root / 'frontend' / 'dist' / 'index.html').write_text('built')
        self._tool('uv', 'printf "uv %s\ncwd=%s\n" "$*" "$PWD" >> "$FAKE_LOG"\nexit 7\n')
        result, calls = self._run()
        self.assertEqual(result.returncode, 7)
        self.assertEqual(calls[0], 'uv run --frozen python web_api.py')
        self.assertTrue(calls[1].replace('\\', '/').endswith('/repo with spaces'))

    def test_first_launch_builds_before_start(self):
        """Locked frontend dependencies and build complete before executing the server."""
        self._tool('node', 'exit 0\n')
        self._tool('npm', 'printf "npm %s\n" "$*" >> "$FAKE_LOG"\n')
        self._tool('uv', 'printf "uv %s\n" "$*" >> "$FAKE_LOG"\n')
        result, calls = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ['npm ci', 'npm run build', 'uv run --frozen python web_api.py'])

    def test_failed_dependency_install_stops_launch(self):
        """Never attempt the build or server after a failed npm ci."""
        self._tool('node', 'exit 0\n')
        self._tool('npm', 'printf "npm %s\n" "$*" >> "$FAKE_LOG"\nexit 4\n')
        self._tool('uv', 'printf "uv\n" >> "$FAKE_LOG"\n')
        result, calls = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn('Frontend build failed', result.stderr)
        self.assertEqual(calls, ['npm ci'])

    def test_failed_frontend_build_stops_launch(self):
        """A build error preserves its diagnostic and never launches the server."""
        self._tool('node', 'exit 0\n')
        self._tool('npm', 'printf "npm %s\n" "$*" >> "$FAKE_LOG"\nif [ "$1" = run ]; then exit 6; fi\n')
        self._tool('uv', 'printf "uv\n" >> "$FAKE_LOG"\n')
        result, calls = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn('Frontend build failed', result.stderr)
        self.assertEqual(calls, ['npm ci', 'npm run build'])


if __name__ == '__main__':
    unittest.main()
