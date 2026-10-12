"""Browser regression checks for audited UX fixes; model jobs/keys are simulated."""

import argparse
import time

from playwright.sync_api import sync_playwright, expect


def main():
    """Exercise preparation, worker visibility and connection recovery without real writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        config = page.request.get(args.url + '/api/config').json()
        state = {'jobs': [], 'offline': False, 'credential_error': True, 'configured': False, 'restart': False}

        def api(route):
            path = route.request.url.split('/api/', 1)[1]
            if path == 'config' and state['restart']:
                route.fulfill(json={**config, 'token': 'browser-test-new-engine'})
            elif path == 'jobs' and state['restart'] and route.request.headers.get('x-workbench-token') != 'browser-test-new-engine':
                route.fulfill(status=403, json={'detail': 'Missing or invalid workbench session'})
            elif path == 'jobs':
                route.fulfill(status=503 if state['offline'] else 200,
                              json={'detail': 'Simulated disconnect'} if state['offline'] else state['jobs'])
            elif path == 'civitai/credentials':
                if route.request.method == 'POST':
                    state['configured'] = True
                route.fulfill(status=503 if state['credential_error'] else 200,
                              json={'detail': 'Simulated credential-status failure'} if state['credential_error'] else
                              {'configured': state['configured'], 'source': 'session_environment' if state['configured'] else 'none',
                               'vault_available': True})
            elif path == 'civitai/inspect':
                route.fulfill(json={'version': 3276318, 'selected_file': 3160899,
                                    'files': [{'id': 3160899, 'name': 'test.safetensors', 'bytes': 16, 'sha256': '0' * 64}]})
            elif route.request.method != 'GET':
                # Auditing never submits a real job, cancels work, or writes credentials.
                route.abort()
            else:
                route.continue_()

        page.route('**/api/**', api)
        page.goto(args.url)
        expect(page.get_by_text('Local engine connected', exact=True)).to_be_visible()
        page.get_by_role('button', name='Downloads', exact=True).click()
        providers = page.get_by_role('group', name='Model source')
        page.get_by_label('Repository ID', exact=True).fill('example/prepared-model')
        page.get_by_label('Revision', exact=True).fill('my-branch')
        providers.get_by_role('button', name='Civitai', exact=True).click()
        page.get_by_label('Version ID or model/download link', exact=True).fill('3276318')
        providers.get_by_role('button', name='Hugging Face', exact=True).click()
        expect(page.get_by_label('Repository ID', exact=True)).to_have_value('example/prepared-model')
        expect(page.get_by_label('Revision', exact=True)).to_have_value('my-branch')
        providers.get_by_role('button', name='Civitai', exact=True).click()
        expect(page.get_by_label('Version ID or model/download link', exact=True)).to_have_value('3276318')
        page.get_by_role('button', name='Activity', exact=True).click()
        page.get_by_role('button', name='Downloads', exact=True).click()
        expect(page.get_by_label('Version ID or model/download link', exact=True)).to_have_value('3276318')
        assert page.locator('#hf-repo').is_hidden(), 'Hidden providers must leave keyboard navigation'

        job = {'format': 'FP8', 'progress': .2, 'phase': 'Converting', 'output': '', 'error': '',
               'logs': [], 'started': time.time(), 'finished': None, 'model_kind': 'diffusion',
               'indeterminate': False, 'operation': 'convert', 'outputs': [], 'result': None}
        state['jobs'] = [dict(job, id='test-active', source='running.safetensors', status='running'),
                         dict(job, id='test-old', source='old.safetensors', status='succeeded',
                              progress=1, finished=time.time(), output='old-output.safetensors')]
        page.get_by_role('button', name='Activity', exact=True).click()
        page.locator('.history-item').filter(has_text='old.safetensors').click()
        expect(page.get_by_role('button', name='Cancel job', exact=True)).to_be_visible()
        expect(page.get_by_role('region', name='Selected previous job')).to_contain_text('old-output.safetensors')
        state['offline'] = True
        expect(page.get_by_text('Engine disconnected · retrying', exact=True)).to_be_visible()
        expect(page.get_by_text('Local engine connected', exact=True)).to_have_count(0)
        state['offline'] = False
        expect(page.get_by_text('Local engine connected', exact=True)).to_be_visible()
        expect(page.get_by_text('Simulated disconnect', exact=True)).to_have_count(0)

        state['jobs'] = []
        page.get_by_role('button', name='Downloads', exact=True).click()
        # Completing the simulated worker clears stale active selection through its final snapshot.
        state['jobs'] = [dict(job, id='test-active', source='running.safetensors', status='succeeded', progress=1),
                         dict(job, id='test-old', source='old.safetensors', status='succeeded', progress=1)]
        expect(page.get_by_role('button', name='Retry credential status', exact=True)).to_be_enabled()
        page.get_by_role('button', name='Configure API key', exact=True).click()
        expect(page.get_by_role('button', name='Retry credential status', exact=True)).to_be_focused()
        state['credential_error'] = False
        page.get_by_role('button', name='Retry credential status', exact=True).click()
        expect(page.get_by_label('Credential source', exact=True)).to_be_enabled()
        expect(page.get_by_text('Configure a Civitai API key before downloading.', exact=True)).to_be_visible()
        page.get_by_role('button', name='Inspect model', exact=True).click()
        page.locator('#civitai-destination').fill('C:\\test-download')
        expect(page.get_by_role('button', name='Download / resume', exact=True)).to_be_disabled()
        page.get_by_role('button', name='Configure API key', exact=True).click()
        expect(page.get_by_label('Credential source', exact=True)).to_be_focused()
        page.get_by_label('Credential source', exact=True).select_option('environment')
        expect(page.get_by_label('Variable name', exact=True)).to_have_attribute('type', 'text')
        page.get_by_label('Variable name', exact=True).fill('BROWSER_FAKE_KEY')
        page.get_by_role('button', name='Use key', exact=True).click()
        expect(page.get_by_text('Environment key loaded for this session.', exact=True)).to_be_visible()
        expect(page.get_by_role('button', name='Download / resume', exact=True)).to_be_enabled()
        expect(page.get_by_label('Version ID or model/download link', exact=True)).to_have_value('3276318')

        page.get_by_label('Credential source', exact=True).select_option('windows_vault')
        expect(page.get_by_label('Credential name', exact=True)).to_have_attribute('type', 'text')
        page.get_by_label('Credential source', exact=True).select_option('direct')
        expect(page.locator('#civitai-key')).to_have_attribute('type', 'password')

        page.get_by_role('button', name='Format guide', exact=True).click()
        page.locator('.support-cell').first.focus()
        page.keyboard.press('Enter')
        detail = page.get_by_role('region', name='Selected compatibility details')
        expect(detail).to_be_focused()
        bounds = detail.bounding_box()
        assert 0 <= bounds['y'] < 1100, bounds
        page.keyboard.press('Tab')
        assert page.get_by_role('button', name='Use format', exact=True).evaluate(
            'e => e.disabled || e === document.activeElement')

        page.locator('html').evaluate("e => e.dataset.theme = 'light'")
        focus = page.get_by_role('button', name='Downloads', exact=True)
        page.keyboard.press('Tab')
        focus.focus()
        assert focus.evaluate("e => getComputedStyle(e).outlineColor") == 'rgb(53, 96, 32)'
        page.emulate_media(reduced_motion='reduce')
        assert focus.evaluate("e => getComputedStyle(e).transform") == 'none'
        page.set_viewport_size({'width': 390, 'height': 844})
        for label in ('Convert model', 'Activity', 'Extract & repair', 'Format guide', 'Downloads'):
            page.get_by_role('button', name=label, exact=True).first.click()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), label

        # Restart rotates the API session; drafts survive while lost jobs become terminal.
        state['jobs'] = [dict(job, id='test-active', source='running.safetensors', status='running')]
        page.get_by_role('button', name='Activity', exact=True).click()
        page.locator('.history-item').filter(has_text='running.safetensors').click()
        expect(page.get_by_role('button', name='Cancel job', exact=True)).to_be_visible()
        state['restart'] = True
        state['jobs'] = []
        expect(page.get_by_text('Engine restarted. This job is no longer available.', exact=True)).to_be_visible()
        expect(page.get_by_text('Local engine connected', exact=True)).to_be_visible()
        expect(page.get_by_role('button', name='Cancel job', exact=True)).to_have_count(0)
        page.get_by_role('button', name='Downloads', exact=True).click()
        expect(page.get_by_label('Version ID or model/download link', exact=True)).to_have_value('3276318')
        browser.close()
    print('PASS: drafts, active worker/history, connection recovery, focus, support details, credential recovery/gating and responsive layouts')


if __name__ == '__main__':
    main()
