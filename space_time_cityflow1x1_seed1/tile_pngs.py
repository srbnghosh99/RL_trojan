#!/usr/bin/env python3
"""
Tile already-rendered per-lane PNGs into multi-panel pages.

A single contact sheet stops working past a few dozen lanes -- at 192 lanes every
panel is squeezed to a sliver. This instead pastes the individual full-resolution
PNGs into a grid, several per page, and emits page_01.png, page_02.png, ... plus
an optional single multi-page PDF.

Because it composites finished images rather than re-plotting, every panel keeps
the exact resolution and layout it was rendered at.

Usage
-----
  python3 tile_pngs.py                                  # space_time_all/ -> tiled/
  python3 tile_pngs.py --indir space_time_all --cols 2 --rows 4
  python3 tile_pngs.py --pdf all_lanes.pdf
  python3 tile_pngs.py --pattern "space_time__road_1_*.png"
  python3 tile_pngs.py --indir figures --cell-width 1100
"""
import argparse
import glob
import os
import re
import sys

from PIL import Image

BG = (252, 252, 251)      # matches the figures' surface
LABEL = (82, 81, 78)


def natural_key(p):
    """Sort road_1_2_3_0 before road_1_10_0_0 rather than lexicographically."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="space_time_all")
    ap.add_argument("--pattern", default="space_time__*.png")
    ap.add_argument("--exclude", default="ALL_LANES",
                    help="skip files whose name contains this (set '' to keep all)")
    ap.add_argument("--outdir", default=None, help="default: <indir>/tiled")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=4, help="rows per page")
    ap.add_argument("--cell-width", type=int, default=1400,
                    help="each panel is scaled to this width, aspect preserved")
    ap.add_argument("--gap", type=int, default=18, help="pixels between panels")
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--pdf", default=None, help="also write a single multi-page PDF here")
    ap.add_argument("--max-pages", type=int, default=0, help="0 = no limit")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.indir, a.pattern)), key=natural_key)
    if a.exclude:
        files = [f for f in files if a.exclude not in os.path.basename(f)]
    if not files:
        print(f"No PNGs matching {a.pattern!r} in {a.indir}/", file=sys.stderr)
        return 1

    outdir = a.outdir or os.path.join(a.indir, "tiled")
    os.makedirs(outdir, exist_ok=True)

    per_page = a.cols * a.rows
    n_pages = (len(files) + per_page - 1) // per_page
    if a.max_pages:
        n_pages = min(n_pages, a.max_pages)
    print(f"{len(files)} panels -> {n_pages} page(s) at {a.cols}x{a.rows}")

    # Scale every panel to the same width so the grid is regular; cell height is
    # the tallest scaled panel, so nothing is cropped.
    probe = Image.open(files[0])
    cell_w = a.cell_width
    cell_h = max(1, round(probe.height * cell_w / probe.width))
    probe.close()

    page_w = a.margin * 2 + a.cols * cell_w + (a.cols - 1) * a.gap
    page_h = a.margin * 2 + a.rows * cell_h + (a.rows - 1) * a.gap
    print(f"panel {cell_w}x{cell_h}px   page {page_w}x{page_h}px")

    pages = []
    for p in range(n_pages):
        chunk = files[p * per_page:(p + 1) * per_page]
        page = Image.new("RGB", (page_w, page_h), BG)
        for i, f in enumerate(chunk):
            r, c = divmod(i, a.cols)
            with Image.open(f) as im:
                im = im.convert("RGB")
                h = round(im.height * cell_w / im.width)
                im = im.resize((cell_w, h), Image.LANCZOS)
                x = a.margin + c * (cell_w + a.gap)
                y = a.margin + r * (cell_h + a.gap)
                page.paste(im, (x, y))
        out = os.path.join(outdir, f"page_{p + 1:02d}.png")
        page.save(out, optimize=True)
        pages.append(page)
        print(f"  {os.path.basename(out)}  ({len(chunk)} panels)")

    if a.pdf:
        pdf_path = a.pdf if os.path.isabs(a.pdf) else os.path.join(outdir, a.pdf)
        pages[0].save(pdf_path, "PDF", resolution=150.0,
                      save_all=True, append_images=pages[1:])
        print(f"\nPDF -> {pdf_path} ({len(pages)} pages)")

    print(f"\ndone -> {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
