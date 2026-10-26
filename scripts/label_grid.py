"""Render chips with a labelled coordinate grid (every 64 px) for point annotation by eye."""
import sys, json, numpy as np
from PIL import Image, ImageDraw
def render(files, out, cols=3, disp=448, start=0, marks=None):
    S = 512; k = disp / S; n = len(files); rows = (n + cols - 1) // cols
    G = Image.new("RGB", (cols * (disp + 6), rows * (disp + 6)), (255, 255, 255))
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB").resize((disp, disp)); d = ImageDraw.Draw(im)
        for v in range(64, S, 64):
            d.line([(v * k, 0), (v * k, disp)], fill=(255, 255, 0), width=1); d.line([(0, v * k), (disp, v * k)], fill=(255, 255, 0), width=1)
            d.text((v * k + 2, 2), str(v), fill=(255, 255, 0)); d.text((2, v * k + 2), str(v), fill=(255, 255, 0))
        if marks and i in marks:
            for (x, y, r) in marks[i]: d.ellipse([(x - r) * k, (y - r) * k, (x + r) * k, (y + r) * k], outline=(255, 0, 255), width=2)
        d.rectangle([0, 0, 40, 14], fill=(0, 0, 0)); d.text((2, 1), f"#{start + i}", fill=(255, 80, 80))
        G.paste(im, ((i % cols) * (disp + 6), (i // cols) * (disp + 6)))
    G.save(out)
