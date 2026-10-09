"""Read retained Z-Image validation PNGs; write descriptive metrics and a contact sheet.

Metrics are image differences, not perceptual quality scores. Model/workflow files
and original PNGs are never changed. Use after or during the retained render batch.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RUN = Path(__file__).resolve().parents[1] / 'runtime-results/zimage-base-2026-10-09'


def main():
    """Summarize all available stages and retain a labeled Base-stage overview."""
    metrics = []
    for stage in ('base', 'refined', 'final'):
        reference = RUN / 'images' / f'SOURCE_BF16__{stage}_00001_.png'
        if not reference.exists():
            continue
        with Image.open(reference) as image:
            reference_pixels = np.asarray(image.convert('RGB'), dtype=np.int16)
        for path in sorted((RUN / 'images').glob(f'*__{stage}_*.png')):
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB'), dtype=np.int16)
            assert pixels.shape == reference_pixels.shape, f'Stage dimensions differ: {path.name}'
            metrics.append({'file': path.name, 'stage': stage, 'shape': list(pixels.shape),
                'pixel_sha256': hashlib.sha256(pixels.astype(np.uint8).tobytes()).hexdigest(),
                'mean_abs_difference': float(np.abs(pixels - reference_pixels).mean()),
                'identical_to_source': bool(np.array_equal(pixels, reference_pixels)),
                'mean': float(pixels.mean()), 'std': float(pixels.std())})
    (RUN / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    for stage in ('base', 'refined', 'final'):
        paths = sorted((RUN / 'images').glob(f'*__{stage}_*.png'),
                       key=lambda p: (not p.name.startswith('SOURCE_BF16__'), p.name))
        if not paths:
            continue
        width, height = 320, 430
        sheet = Image.new('RGB', (width * 4, height * ((len(paths) + 3) // 4)), '#171b22')
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(paths):
            x, y = index % 4 * width, index // 4 * height
            with Image.open(path) as original:
                image = original.convert('RGB')
                image.thumbnail((width, height - 32))
                sheet.paste(image, (x + (width - image.width) // 2, y))
            draw.text((x + 8, y + height - 25), path.name.split('__')[0], fill='white')
        sheet.save(RUN / f'{stage}-contact-sheet.jpg', quality=95)
    print(f'{len(metrics)} stage comparisons and three contact sheets retained.')


if __name__ == '__main__':
    main()
