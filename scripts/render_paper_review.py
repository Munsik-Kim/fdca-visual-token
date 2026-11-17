#!/usr/bin/env python3
"""Render every revision PDF page and numbered contact sheets for visual review.

Rendering is preparation for inspection, not automatic visual approval.
Outputs stay under the ignored build/ directory.
"""
from pathlib import Path
import hashlib
import json

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    pdf = ROOT/'paper/FDCA_v1.0.1.pdf'
    out = ROOT/'build/paper-review-v1.0.1'
    out.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf)
    pages = []
    for index, page in enumerate(document):
        target = out/f'page-{index+1:02d}.png'
        pixmap = page.get_pixmap(dpi=120, alpha=False)
        pixmap.save(target)
        pages.append({'page':index+1,'image':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    for start in range(0, len(pages), 4):
        sheet = Image.new('RGB', (1240, 1790), '#cccccc')
        draw = ImageDraw.Draw(sheet)
        for offset, record in enumerate(pages[start:start+4]):
            image = Image.open(out/record['image'])
            image.thumbnail((600, 850))
            x, y = (offset%2)*620+10, (offset//2)*895+30
            sheet.paste(image,(x,y))
            draw.text((x,y-22),f"Page {record['page']}",fill='black')
        sheet.save(out/f'contact-{start//4+1:02d}.png')
    report={'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pages':pages,'render_dpi':120,'renderer':f'PyMuPDF {pymupdf.VersionBind}','visual_approval':'requires inspection of every rendered page'}
    (out/'render_manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pass':True,'rendered_pages':len(pages),'directory':str(out.relative_to(ROOT))}))


if __name__=='__main__':main()
