"""Browser smoke test against a running workbench (uv run --with playwright).

Writes small synthetic models and screenshots under the ignored
.pytest-workbench-browser-tmp directory; never accesses installed user models.
"""
import argparse
import json
import uuid
from pathlib import Path

import torch
import gguf
import numpy as np
from playwright.sync_api import sync_playwright, expect
from safetensors.torch import save_file


def main():
    """Exercise file selection, real conversion, themes and responsive layouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--hf-live', action='store_true', help='Download a public 520 KB Hub test checkpoint through the UI')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / '.pytest-workbench-browser-tmp'
    root.mkdir(exist_ok=True)
    source = root / 'qwen-ui.safetensors'
    save_file({'time_text_embed.timestep_embedder.linear_2.weight': torch.ones(2, 2),
               'transformer_blocks.0.attn.norm_added_q.weight': torch.ones(2),
               'transformer_blocks.0.img_mlp.net.0.proj.weight': torch.randn(128, 256),
               '__index_timestep_zero__': torch.empty(0)}, str(source))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.get_by_text('Local engine connected').wait_for()
        page.screenshot(path=str(root / 'desktop-dark.png'), full_page=True)
        page.get_by_role('button', name='Browse files').click()
        expect(page.get_by_role('button', name='Go', exact=True)).to_be_enabled()
        page.get_by_label('Folder path').fill(str(root))
        page.get_by_role('button', name='Go', exact=True).click()
        page.locator('.file-entry').filter(has_text='qwen-ui.safetensors').click()
        assert page.locator('#source').input_value() == str(source)
        page.get_by_role('button', name='Safetensors Native ComfyUI loading').click()
        # Test reruns explicitly authorize replacement of this owned artifact.
        output = root / f'qwen-ui-output-{uuid.uuid4().hex}.safetensors'
        page.locator('#destination').fill(str(output))
        page.get_by_text('Advanced settings', exact=True).click()
        page.get_by_label('Replace an existing output file').check()
        page.get_by_role('button', name='Inspect & estimate').click()
        page.get_by_text('qwen_image', exact=True).wait_for()
        page.get_by_role('button', name='Convert model', exact=True).last.click()
        page.get_by_role('heading', name='Your model is ready').wait_for(timeout=30000)
        assert output.is_file()
        page.screenshot(path=str(root / 'desktop-success.png'), full_page=True)
        page.get_by_role('button', name='Light appearance').click()
        page.evaluate('scrollTo(0, 0)')
        page.screenshot(path=str(root / 'desktop-light.png'), full_page=True)
        page.get_by_role('button', name='Activity', exact=True).click()
        page.get_by_text('succeeded', exact=True).first.wait_for()
        page.get_by_role('button', name='Format guide', exact=True).click()
        page.get_by_role('heading', name='Compatibility is model-specific.').wait_for()
        page.get_by_role('button', name='Convert model', exact=True).first.click()
        page.emulate_media(reduced_motion='reduce')
        for width, height, label in [(390, 844, 'mobile-light'), (768, 1024, 'tablet-light')]:
            page.set_viewport_size({'width': width, 'height': height})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(root / f'{label}.png'), full_page=True)
        page.get_by_role('button', name='Browse files').click()
        page.keyboard.press('Escape')
        expect(page.locator('dialog')).not_to_be_visible()
        expect(page.get_by_role('button', name='Browse files')).to_be_focused()
        # 200% text scaling should preserve controls without horizontal scrolling.
        page.set_viewport_size({'width': 1440, 'height': 1100})
        original_size = page.locator('#source').evaluate('(node) => parseFloat(getComputedStyle(node).fontSize)')
        page.add_style_tag(content=':root { font-size: 28px !important; }')
        enlarged_size = page.locator('#source').evaluate('(node) => parseFloat(getComputedStyle(node).fontSize)')
        assert enlarged_size >= original_size * 1.99
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')

        # Text encoders keep genuine model.* paths instead of diffusion stripping.
        encoder = root / 'text-encoder-ui.safetensors'
        save_file({'model.embed_tokens.weight': torch.ones(32, 128, dtype=torch.bfloat16),
                   'model.layers.0.self_attn.q_proj.weight': torch.randn(128, 128, dtype=torch.bfloat16)}, str(encoder))
        page.goto(args.url)
        page.get_by_text('Local engine connected').wait_for()
        page.get_by_label('Model type').select_option('text_encoder')
        assert page.locator('#format').input_value() == 'F16'
        page.get_by_label('Source path').fill(str(encoder))
        page.get_by_role('button', name='Safetensors Native CLIPLoader').click()
        page.get_by_label('Quantization').select_option('F16_ST')
        encoder_output = root / f'text-encoder-f16-{uuid.uuid4().hex}.safetensors'
        page.locator('#destination').fill(str(encoder_output))
        page.get_by_role('button', name='Inspect & estimate').click()
        page.get_by_text('Unknown family', exact=True).wait_for()
        with page.expect_response(lambda response: response.url.endswith('/api/jobs') and response.request.method == 'POST') as started:
            page.get_by_role('button', name='Convert model', exact=True).last.click()
        assert started.value.status == 202
        page.get_by_role('heading', name='Your model is ready').wait_for(timeout=30000)
        assert encoder_output.is_file()
        page.evaluate('scrollTo(0, 0)')
        page.screenshot(path=str(root / 'text-encoder-desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(root / 'text-encoder-mobile.png'), full_page=True)

        checkpoint = root / f'embedded-{uuid.uuid4().hex}.safetensors'
        save_file({'first_stage_model.decoder.weight': torch.ones(2, 2),
                   'conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight': torch.ones(2, 2),
                   'conditioner.embedders.1.model.token_embedding.weight': torch.ones(2, 2)}, str(checkpoint))
        tools_root = root / f'models-{uuid.uuid4().hex}'
        page.set_viewport_size({'width': 1440, 'height': 1100})
        page.get_by_role('button', name='Extract & repair', exact=True).click()
        page.get_by_label('Source checkpoint').fill(str(checkpoint))
        page.get_by_label('ComfyUI models root').fill(str(tools_root))
        page.get_by_label('VAE', exact=True).check()
        page.get_by_role('button', name='Extract components', exact=True).click()
        expect(page.locator('.job-panel .output-path')).to_contain_text(checkpoint.stem + '-clip_g', timeout=30000)
        assert (tools_root / 'vae' / f'{checkpoint.stem}-vae.safetensors').is_file()
        page.get_by_label('Operation').select_option('analyze')
        page.get_by_label('ComfyUI models root').fill(str(tools_root))
        page.get_by_role('button', name='Compare components', exact=True).click()
        page.get_by_role('heading', name='Comparison complete').wait_for(timeout=30000)
        expect(page.locator('.component-results')).to_contain_text('no local reference')
        page.screenshot(path=str(root / 'tools-desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(root / 'tools-mobile.png'), full_page=True)

        repair_source = root / f'pad-tokens-{uuid.uuid4().hex}.gguf'
        writer = gguf.GGUFWriter(path=None, arch='lumina2')
        writer.add_tensor('x_pad_token', np.arange(4, dtype=np.float32))
        writer.write_header_to_file(path=str(repair_source))
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        page.get_by_label('Operation').select_option('pad_tokens')
        page.get_by_role('button', name='Browse tool source').click()
        page.get_by_label('Folder path').fill(str(root))
        page.get_by_role('button', name='Go', exact=True).click()
        page.locator('.file-entry').filter(has_text=repair_source.name).click()
        page.get_by_role('button', name='Repair pad tokens', exact=True).click()
        expect(page.locator('.job-panel .output-path')).to_contain_text(repair_source.stem + '-fixed.gguf', timeout=30000)
        repaired = root / f'{repair_source.stem}-fixed.gguf'
        assert gguf.GGUFReader(str(repaired)).tensors[0].data.shape == (1, 4)
        page.screenshot(path=str(root / 'repair-mobile.png'), full_page=True)
        sidecar = root / f'{repair_source.name}.5d.safetensors'
        save_file({'restored.weight': torch.ones(1, 2, 3, 4, 5)}, str(sidecar))
        page.get_by_label('Operation').select_option('restore_5d')
        restored = root / f'restored-{uuid.uuid4().hex}.gguf'
        page.get_by_label('Output GGUF or folder').fill(str(restored))
        page.get_by_role('button', name='Restore 5D tensors', exact=True).click()
        expect(page.locator('.job-panel .output-path')).to_contain_text(restored.name, timeout=30000)
        assert any(len(tensor.shape) == 5 for tensor in gguf.GGUFReader(str(restored)).tensors)

        if args.hf_live:
            download_root = root / f'hf-live-{uuid.uuid4().hex}'
            download_root.mkdir()
            page.set_viewport_size({'width': 1440, 'height': 1100})
            page.get_by_role('button', name='Hugging Face', exact=True).click()
            expect(page.get_by_role('button', name='Download checkpoint')).to_be_disabled()
            # Native variant selection must remain explicit for ambiguous repos.
            page.route('**/api/hf/inspect', lambda route: route.fulfill(json={
                'repo_id': 'ui-test/variants', 'revision': 'a' * 40,
                'groups': [{'subfolder': '', 'files': 1, 'bytes': 100},
                           {'subfolder': 'variant', 'files': 2, 'bytes': 200}]}))
            page.get_by_label('Repository ID').fill('ui-test/variants')
            page.get_by_role('button', name='Inspect repository').click()
            expect(page.get_by_label('Checkpoint folder')).to_have_value('__choose__')
            page.get_by_label('Download folder', exact=True).fill(str(download_root))
            expect(page.get_by_role('button', name='Download checkpoint')).to_be_disabled()
            page.get_by_label('Checkpoint folder').select_option('variant')
            expect(page.get_by_role('button', name='Download checkpoint')).to_be_enabled()
            page.unroute('**/api/hf/inspect')
            page.get_by_label('Repository ID').fill('hf-internal-testing/tiny-random-bert')
            page.get_by_role('button', name='Inspect repository').click()
            expect(page.get_by_label('Checkpoint folder')).to_have_value('', timeout=30000)
            # Changing a revision invalidates the previous pinned plan.
            page.get_by_label('Revision', exact=True).fill('invalidated')
            expect(page.get_by_label('Checkpoint folder')).not_to_be_visible()
            page.get_by_label('Revision', exact=True).fill('main')
            page.get_by_role('button', name='Inspect repository').click()
            expect(page.get_by_label('Checkpoint folder')).to_have_value('', timeout=30000)
            page.get_by_role('button', name='Browse download folder').click()
            page.get_by_label('Folder path').fill(str(download_root))
            page.get_by_role('button', name='Go', exact=True).click()
            page.get_by_role('button', name='Use this folder').click()
            expect(page.get_by_label('Download folder', exact=True)).to_have_value(str(download_root) + '/')
            page.evaluate('scrollTo(0, 0)')
            page.screenshot(path=str(root / 'huggingface-desktop.png'), full_page=True)
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(root / 'huggingface-mobile.png'), full_page=True)
            page.get_by_role('button', name='Download checkpoint').click()
            expect(page.locator('.job-panel .output-path')).to_contain_text('tiny-random-bert.safetensors', timeout=120000)
            downloaded = download_root / 'tiny-random-bert.safetensors'
            from safetensors import safe_open
            with safe_open(str(downloaded), framework='pt') as checkpoint:
                assert len(checkpoint.keys()) > 0
            assert not (download_root / '.hf_download_hf-internal-testing_tiny-random-bert').exists()
            page.screenshot(path=str(root / 'huggingface-success-mobile.png'), full_page=True)
        assert not errors, errors
        print(json.dumps({'browser': 'Chromium', 'conversion': 'FP8_MIXED succeeded',
                          'text_encoder': 'F16_ST succeeded',
                          'tools': 'components, comparison, pad repair and 5D restoration succeeded',
                          'huggingface': 'live tiny-random-bert download succeeded' if args.hf_live else 'not requested',
                          'viewport_checks': [1440, 768, 390], 'page_errors': errors,
                          'artifacts': str(root)}))
        browser.close()


if __name__ == '__main__':
    main()
