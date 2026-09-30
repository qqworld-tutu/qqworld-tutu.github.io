"""Compile a standalone TikZ source and export PDF, outlined SVG, and PNGs.

Usage: python render_tikz.py path/to/figure.tex
Outputs are placed alongside the input file. Requires XeLaTeX (ctex/Fandol, TikZ, standalone) and PyMuPDF.
"""
from pathlib import Path
import subprocess
import sys
import re
import tempfile
import shutil
import xml.etree.ElementTree as ET

tex_root = Path.home() / 'Library/TinyTeX'
sys.path.insert(0, str(tex_root / 'render-python'))
import pymupdf

if len(sys.argv) != 2:
    raise SystemExit('Usage: python tools/render_tikz.py path/to/figure.tex')
source = Path(sys.argv[1]).resolve(strict=True)
if source.suffix != '.tex':
    raise SystemExit('Expected a .tex source')
compiler = shutil.which('xelatex') or tex_root / 'bin/universal-darwin/xelatex'
build_dir = Path(tempfile.mkdtemp(prefix='arithmetic-tikz-'))
subprocess.run(
    [str(compiler), '-no-shell-escape', '-interaction=batchmode',
     '-halt-on-error', f'-output-directory={build_dir}', source.name],
    cwd=source.parent, check=True,
)
log = (build_dir / source.with_suffix('.log').name).read_text(errors='replace')
warnings = [line for line in log.splitlines() if 'Missing character' in line or 'Overfull' in line]
if warnings:
    raise SystemExit('Rendering warnings:\n' + '\n'.join(warnings))
pdf_path = source.with_suffix('.pdf')
shutil.copyfile(build_dir / pdf_path.name, pdf_path)
with pymupdf.open(pdf_path) as doc:
    if len(doc) != 1:
        raise SystemExit('Expected a one-page standalone figure')
    page = doc[0]
    # Outlines preserve Chinese and mathematical glyphs on other machines.
    # TeX math fonts can map glyphs to control-code metadata. The visible glyph
    # is already an outline; drop only XML-invalid references in that metadata.
    svg_text = page.get_svg_image(text_as_path=True)
    def valid_reference(match):
        token = match.group(1)
        code = int(token[1:], 16) if token.startswith('x') else int(token)
        valid = code in (9, 10, 13) or 0x20 <= code <= 0xD7FF or 0xE000 <= code <= 0xFFFD or 0x10000 <= code <= 0x10FFFF
        return match.group(0) if valid else ''
    svg_text = re.sub(r'&#(x[0-9a-fA-F]+|[0-9]+);', valid_reference, svg_text)
    svg = ET.fromstring(svg_text)
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    svg.insert(0, ET.Element('{http://www.w3.org/2000/svg}rect',
                            width='100%', height='100%', fill='white'))
    source.with_suffix('.svg').write_text(
        ET.tostring(svg, encoding='unicode'), encoding='utf-8')
    for transparent in (False, True):
        suffix = '-transparent.png' if transparent else '-white.png'
        page.get_pixmap(dpi=200, alpha=transparent).save(
            str(source.with_name(source.stem + suffix)))
    text = page.get_text()
    print(f'Exported {pdf_path.name}, outlined SVG, white/transparent PNG')
    print(f'PDF pages: {len(doc)}; extracted text characters: {len(text)}')
