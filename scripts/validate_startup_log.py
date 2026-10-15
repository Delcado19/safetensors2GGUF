"""Headless browser checks on an isolated server; no model jobs are started.

Run with: uv run --offline --no-project --with playwright python scripts/validate_startup_log.py
Requires an existing frontend build and installed Microsoft Edge.
"""
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    """Verify backend selection, first load and log following/reading/rollover in the built application."""
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as address:
        address.bind(('127.0.0.1', 0))
        port = address.getsockname()[1]
    url = f'http://127.0.0.1:{port}/'
    # Own this isolated process only; never restart or stop the user's running app.
    python = root / '.venv' / 'Scripts' / 'python.exe'
    if not python.is_file():
        python = root / '.venv' / 'bin' / 'python'
    server = subprocess.Popen([str(python), '-X', 'utf8',
                               str(root / 'web_api.py'), '--port', str(port)], cwd=root,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        deadline = time.monotonic() + 60
        while True:
            if server.poll() is not None:
                raise RuntimeError('Isolated server exited during startup')
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    assert response.headers['Cache-Control'] == 'no-store'
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Isolated server did not become ready')
                time.sleep(0.1)
        state = {'jobs': []}
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='msedge', headless=True)
            try:
                page = browser.new_page(viewport={'width': 1440, 'height': 1100})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.route('**/api/jobs', lambda route: route.fulfill(json=state['jobs']))
                page.goto(url)
                expect(page.get_by_text('Local engine connected', exact=True)).to_be_visible()
                assert not errors, errors
                page.locator('#format').select_option('Q5_K_M')
                page.get_by_text('Advanced settings', exact=True).click()
                backend = page.get_by_label('GGUF converter', exact=True)
                expect(backend).to_have_value('native')
                backend.select_option('python_flux')
                expect(page.locator('#executable')).to_be_disabled()
                expect(page.locator('#threads')).to_be_disabled()
                expect(page.locator('#gguf-backend-hint')).to_contain_text('visible pose/composition drift')
                page.locator('#format').select_option('Q6_K')
                expect(backend).to_have_value('python_flux')
                expect(page.locator('#executable')).to_be_disabled()
                page.locator('#format').select_option('Q4_K_M')
                expect(backend).to_have_count(0)
                expect(page.locator('#executable')).to_be_enabled()
                page.locator('#format').select_option('Q5_K_M')
                backend.select_option('native')
                expect(page.locator('#executable')).to_be_enabled()
                job = {'id': 'log-test', 'source': 'simulated.safetensors', 'format': 'Q6_K',
                       'status': 'running', 'progress': .5, 'phase': 'Simulated log', 'output': '',
                       'error': '', 'logs': [f'line {i}' for i in range(400)], 'started': time.time(),
                       'finished': None, 'model_kind': 'diffusion', 'indeterminate': False,
                       'operation': 'convert', 'outputs': [], 'result': None}
                state['jobs'] = [job]
                page.get_by_text('View technical log', exact=True).click()
                output = page.locator('.job-log pre')
                at_bottom = '(e) => e.scrollHeight - e.clientHeight - e.scrollTop < 2'
                expect(output).to_contain_text('line 399')
                assert output.evaluate(at_bottom)
                # Keep length fixed at the server's 400-line cap: text, not count, must drive following.
                job['logs'] = job['logs'][1:] + ['line 400']
                expect(output).to_contain_text('line 400')
                assert output.evaluate(at_bottom)
                output.evaluate("e => { e.scrollTop = 0; e.dispatchEvent(new Event('scroll')); }")
                job['logs'] = job['logs'][1:] + ['line 401']
                expect(output).to_contain_text('line 401')
                assert output.evaluate('e => e.scrollTop < 30'), 'New output interrupted reading'
                output.evaluate("e => { e.scrollTop = e.scrollHeight; e.dispatchEvent(new Event('scroll')); }")
                job['logs'] = job['logs'][1:] + ['line 402']
                expect(output).to_contain_text('line 402')
                assert output.evaluate(at_bottom), 'Following did not resume'
                page.get_by_text('View technical log', exact=True).click()
                job['logs'] = job['logs'][1:] + ['line 403']
                expect(output).to_contain_text('line 403')
                page.get_by_text('View technical log', exact=True).click()
                assert output.evaluate(at_bottom), 'Opening the log did not reveal the latest line'
                assert not errors, errors
            finally:
                browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    print('PASS: Q5/Q6 backend controls and native fallback selection; first load without reload; log follows at 400-line cap, pauses while reading, resumes and reopens at latest output')


if __name__ == '__main__':
    main()
