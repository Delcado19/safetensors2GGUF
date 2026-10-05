"""Browser smoke test against a running workbench (uv run --with playwright).

Writes small synthetic models and screenshots under the ignored
.pytest-workbench-browser-tmp directory; never accesses installed user models.
"""
import argparse
import json
from pathlib import Path

import torch
from playwright.sync_api import sync_playwright, expect
from safetensors.torch import save_file


def main():
    """Exercise file selection, real conversion, themes and responsive layouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
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
        output = root / 'qwen-ui-output.safetensors'
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
        assert not errors, errors
        print(json.dumps({'browser': 'Chromium', 'conversion': 'FP8_MIXED succeeded',
                          'viewport_checks': [1440, 768, 390], 'page_errors': errors,
                          'artifacts': str(root)}))
        browser.close()


if __name__ == '__main__':
    main()
