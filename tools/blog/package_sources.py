"""Package the current HTML figures, vector fallbacks and numeric checks."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
for topic in ['transformer-arithmetic','kl-divergence']:
    if not (ROOT/f'tools/blog/{topic}').exists():
        continue
    out=ROOT/f'source/downloads/{topic}/sources.zip'
    paths=[ROOT/'tools/blog/README.md',ROOT/'tools/blog/package_sources.py',
           ROOT/'tools/blog/build-qkv-preview.py',ROOT/'tools/blog/build_html_figures.py',
           ROOT/'tools/blog/build_editorial_charts.py',ROOT/'source/figures/figure.css',
           ROOT/'source/figures/figure.js',ROOT/'source/figures/manifest.json']
    paths.extend((ROOT/f'source/figures/{topic}').rglob('*.html'))
    paths.extend((ROOT/f'source/images/{topic}').glob('*.svg'))
    paths.extend((ROOT/f'source/downloads/{topic}').glob('*.py'))
    with ZipFile(out,'w',ZIP_DEFLATED) as z:
        for p in sorted(paths):
            if p.is_file() and p.suffix in ['.md','.py','.html','.css','.js','.json','.svg']:
                z.write(p,p.relative_to(ROOT))
    print(out.relative_to(ROOT),out.stat().st_size)
