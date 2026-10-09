"""Run the user's saved Z-Image Base workflow against existing I: conversions.

Preserves inference widgets, adds stage PNGs, freezes saved seeds, and exports
through ComfyUI's own frontend so custom widgets keep their serialization.
No conversion, downloads, source-workflow edits or automatic shutdown.
"""

import argparse
import hashlib
import html
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

WORKFLOW = Path('G:/ComfyUI-Easy-Install/ComfyUI/user/default/workflows/Z-Image Base/Z-Image Base - FameGrid Revolution - with formula.json')
RUN = Path(__file__).resolve().parents[1] / 'runtime-results/zimage-base-2026-10-09'
PROMPT = ('A gorgeous adult woman leaning against a motel doorway at night, full body, '
          'both hands visible, wearing a short black dress and heels, neon sign glowing '
          'behind her, wet pavement reflections, smoky cinematic atmosphere, sultry '
          'expression, dark noir sensuality, realistic skin, premium cover art, masterpiece.')
URL = 'http://127.0.0.1:8189'


def gallery():
    """Retain a local clickable comparison page; every PNG stays at full resolution."""
    sections = []
    for stage in ('base', 'refined', 'final'):
        cards = []
        for path in sorted((RUN / 'images').glob(f'*__{stage}_*.png'),
                           key=lambda p: (not p.name.startswith('SOURCE_BF16__'), p.name)):
            label = path.name.split('__')[0]
            cards.append(f'<figure><a href="images/{html.escape(path.name)}">'
                         f'<img loading="lazy" src="images/{html.escape(path.name)}"></a>'
                         f'<figcaption>{html.escape(label)}</figcaption></figure>')
        sections.append(f'<h2>{stage.title()}</h2><div class="grid">' + ''.join(cards) + '</div>')
    body = '<!doctype html><meta charset="utf-8"><title>Z-Image Base comparison</title>'
    body += '<style>body{background:#14171c;color:#eee;font:16px system-ui;margin:30px auto;max-width:1400px;padding:20px}a{color:#9cf}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}figure{margin:0}img{width:100%;height:auto}figcaption{padding:10px 0;font-weight:600}</style>'
    body += '<h1>Z-Image Base · Motel portrait comparison</h1><p>Click any image for its original resolution. Base is the raw Base-model decode; refined includes the stored Turbo pass; final includes the stored upscaling and postprocessing.</p>'
    (RUN / 'index.html').write_text(body + ''.join(sections), encoding='utf-8')


def request(route, payload=None):
    """JSON requests with bounded connection waits; preserve backend validation errors."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(URL + route, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def prepare(label, filename):
    """Only change positive prompt, Base loader, saved RNG mode, links and PNG saving."""
    workflow = json.loads(WORKFLOW.read_text(encoding='utf-8'))
    nodes = {str(node['id']): node for node in workflow['nodes']}
    nodes['6']['widgets_values'] = [PROMPT]
    nodes['407']['widgets_values'][0] = filename
    if filename.endswith('.gguf'):
        nodes['407']['type'] = 'UnetLoaderGGUF'
        nodes['407']['widgets_values'] = [filename]
    nodes['438']['widgets_values'][0]['mode'] = 'fixed'
    nodes['438']['properties']['seedState'] = json.dumps(nodes['438']['widgets_values'][0])
    for node_id in ('265', '344'):
        nodes[node_id]['widgets_values'][8] = 'fixed'
    # Saved Use Everywhere edges are actual dependencies; materialize them.
    for edge in workflow['extra'].get('ue_links', []):
        workflow['last_link_id'] += 1
        link_id = workflow['last_link_id']
        source, target = nodes[edge['upstream']], nodes[edge['downstream']]
        target['inputs'][edge['downstream_slot']]['link'] = link_id
        source['outputs'][edge['upstream_slot']].setdefault('links', [])
        source['outputs'][edge['upstream_slot']]['links'].append(link_id)
        workflow['links'].append([link_id, source['id'], edge['upstream_slot'],
                                  target['id'], edge['downstream_slot'], edge['type']])
    # Redirect the two original save nodes, preserving their image inputs.
    for node_id, stage in (('426', 'refined'), ('393', 'final')):
        node = nodes[node_id]
        node['type'] = 'SaveImage'
        node['widgets_values'] = [f'{label}__{stage}']
        node['inputs'] = [node['inputs'][0]]
        node['inputs'][0]['name'] = 'images'
        node['properties'] = {}
    # Also retain the raw Base decode before sharpening/Turbo/upscalers.
    save_id = workflow['last_node_id'] + 1
    workflow['last_node_id'] = save_id
    workflow['last_link_id'] += 1
    link_id = workflow['last_link_id']
    nodes['359']['outputs'][0]['links'].append(link_id)
    workflow['links'].append([link_id, 359, 0, save_id, 0, 'IMAGE'])
    workflow['nodes'].append({'id': save_id, 'type': 'SaveImage', 'pos': [4000, 0],
        'size': [300, 100], 'flags': {}, 'order': 100, 'mode': 0, 'properties': {},
        'inputs': [{'name': 'images', 'type': 'IMAGE', 'link': link_id}],
        'outputs': [], 'widgets_values': [f'{label}__base']})
    return workflow


def export(page, workflow):
    """Let the installed frontend serialize dynamic combos and custom state widgets."""
    exported = page.evaluate('''async (workflow) => {
      const {app} = await import('/scripts/app.js');
      await app.loadGraphData(workflow);
      await new Promise(resolve => setTimeout(resolve, 2000));
      return await app.graphToPrompt();
    }''', workflow)
    # Pixaroma's frontend currently replaces legacy saved random/fixed state.
    # Pin its backend runSeed explicitly instead of accepting a fresh UI RNG.
    seed = next(n for n in workflow['nodes'] if n['id'] == 438)['widgets_values'][0]['seed']
    exported['output']['438']['inputs'] = {'SeedState': json.dumps({'runSeed': seed}), 'seed': seed}
    assert exported['output']['417']['inputs']['value'] == 23
    assert exported['output']['424']['inputs']['value'] == 3.5
    assert json.loads(exported['output']['396']['inputs']['ResolutionState'])['w'] == 1280
    assert json.loads(exported['output']['396']['inputs']['ResolutionState'])['h'] == 1568
    assert all(exported['output'][key]['class_type'] == 'SaveImage' for key in ('426', '393', '441'))
    info = request('/object_info')
    # Resolve old filenames only when the same punctuation-insensitive name exists.
    for node_id, field in (('407', 'unet_name'), ('383', 'model_name'), ('391', 'model_name')):
        node = exported['output'][node_id]
        if node['class_type'] == 'UnetLoaderGGUF':
            continue
        spec = info[node['class_type']]['input']['required'][field]
        choices = spec[0] if isinstance(spec[0], list) else spec[1]['options']
        wanted = node['inputs'][field]
        def normalized(value):
            return ''.join(c for c in value.lower() if c.isalnum())
        matches = [value for value in choices if normalized(value) == normalized(wanted)]
        if len(matches) != 1:
            raise ValueError(f'Cannot resolve saved model filename: {wanted}')
        node['inputs'][field] = matches[0]
    return exported


def main():
    """Save reproducible graphs, submit sequentially and retain history for every model."""
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--only', help='Limit to a named prepared test for a diagnostic retry')
    parser.add_argument('--resume', action='store_true', help='Keep already completed three-stage tests')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    plans = [('SOURCE_BF16', 'Z-Image Base/zImageBase.safetensors')]
    plans += [(path.stem.removeprefix('zImageBase-'), path.name)
              for path in sorted(Path('I:/').glob('zImageBase*'))
              if path.suffix in ('.safetensors', '.gguf')]
    plans += [('SOURCE_REPEAT', 'Z-Image Base/zImageBase.safetensors')]
    if args.only:
        plans = [plan for plan in plans if plan[0] == args.only]
        if not plans:
            raise ValueError('Unknown test label')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(URL)
        page.wait_for_function('!!window.comfyAPI?.app?.app?.graph && !!window.comfyAPI?.app?.app?.canvas', timeout=120000)
        page.wait_for_timeout(5000)
        for label, filename in plans:
            result_path = RUN / f'{label}.result.json'
            if args.resume and result_path.exists():
                previous = json.loads(result_path.read_text(encoding='utf-8'))
                history = previous.get('history', {})
                if history.get('status', {}).get('status_str') == 'success' and all(
                    history.get('outputs', {}).get(node, {}).get('images') for node in ('426', '393', '441')):
                    print('KEPT', label, flush=True)
                    continue
            workflow = prepare(label, filename)
            exported = export(page, workflow)
            if not exported['output']:
                page.screenshot(path=str(RUN / 'export-error.png'))
                raise RuntimeError('Empty prompt export: ' + page.locator('body').inner_text()[-4000:])
            (RUN / f'{label}.workflow.json').write_text(json.dumps(workflow, indent=2), encoding='utf-8')
            (RUN / f'{label}.api.json').write_text(json.dumps(exported, indent=2), encoding='utf-8')
            print('PREPARED', label, len(exported['output']), flush=True)
            if args.prepare_only:
                continue
            queue = request('/queue')
            if queue['queue_running'] or queue['queue_pending']:
                raise RuntimeError('Test server is not idle; refusing to overlap another job')
            start = time.monotonic()
            try:
                if label == 'SOURCE_REPEAT':
                    request('/free', {'free_memory': True, 'unload_models': True})
                    time.sleep(2)
                response = request('/prompt', {'prompt': exported['output'],
                    'extra_data': {'extra_pnginfo': {'workflow': exported['workflow']}}})
                if response.get('node_errors'):
                    raise RuntimeError('Workflow validation failed: ' + json.dumps(response['node_errors']))
                prompt_id = response['prompt_id']
                print('SUBMITTED', label, prompt_id, flush=True)
                while True:
                    history = request('/history/' + prompt_id)
                    if prompt_id in history:
                        break
                    if time.monotonic() - start > 7200:
                        raise TimeoutError('Per-model 2-hour limit exceeded; investigate queue before continuing')
                    time.sleep(5)
                history = history[prompt_id]
                if label == 'SOURCE_REPEAT':
                    cached = {str(node) for message, details in history['status']['messages']
                              if message == 'execution_cached' for node in details.get('nodes', [])}
                    assert not cached.intersection({'407', '265', '359'}), 'Reference repeat reused the Base render cache'
                if history['status']['status_str'] == 'success' and not all(
                    node in history['outputs'] and history['outputs'][node].get('images')
                    for node in ('426', '393', '441')):
                    raise RuntimeError('Incomplete workflow execution: stage images are missing')
                result = {'label': label, 'file': filename, 'prompt_id': prompt_id,
                          'seconds': time.monotonic() - start, 'history': history}
            except Exception as error:
                result = {'label': label, 'file': filename, 'seconds': time.monotonic() - start,
                          'error': str(error)}
            (RUN / f'{label}.result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            gallery()
            print('FINISHED', label, result.get('error') or result['history']['status'], flush=True)
            if label == 'SOURCE_BF16' and (result.get('error') or result['history']['status']['status_str'] != 'success'):
                raise RuntimeError('Reference workflow failed; fix the harness before testing conversions')
        browser.close()
    (RUN / 'workflow-source.sha256').write_text(hashlib.sha256(WORKFLOW.read_bytes()).hexdigest(), encoding='ascii')


if __name__ == '__main__':
    main()
