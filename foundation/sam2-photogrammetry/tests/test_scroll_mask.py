"""
Synthetic checks for scroll-mask.py. Each case builds a small turntable sequence of
segment.py-style palette masks (wall touching the border, a static scroll on the axis,
markers orbiting it) and runs the script on it.

Run with:  python tests/test_scroll_mask.py   (or pytest)
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scroll-mask.py")
H, W, FRAMES = 300, 450, 24
CY, CX, R = 150, 225, 40
WALL, SCROLL, FIRST_MARKER = 9, 3, 4


def make_frame(i, variant):
    arr = np.zeros((H, W), np.uint8)
    arr[:60] = WALL
    yy, xx = np.mgrid[:H, :W]
    # Markers orbit the axis; drawn first so the scroll occludes them
    for k in range(4):
        a = 2 * np.pi * (i / FRAMES + k / 4)
        my, mx = int(CY + 0.55 * (R + 60) * np.sin(a)), int(CX + (R + 60) * np.cos(a))
        arr[my - 14:my + 14, mx - 14:mx + 14] = FIRST_MARKER + k
    disk = (yy - CY) ** 2 + (xx - CX) ** 2 <= R ** 2
    arr[disk] = SCROLL
    if variant == "halves":
        arr[disk & (xx > CX)] = 1
    if variant == "embedded":
        arr[(yy - CY) ** 2 + (xx - CX - 10) ** 2 <= 12 ** 2] = 2
    if variant == "occluded":
        # A marker parked right behind the scroll, touching it along one edge
        arr[CY - R - 25:CY - R + 3, CX - 15:CX + 15] = 2
        arr[disk] = SCROLL
    return arr


def run_case(variant):
    with tempfile.TemporaryDirectory() as root:
        mask_dir = os.path.join(root, "scroll", "orientation", "Masks")
        os.makedirs(mask_dir)
        for i in range(FRAMES):
            img = Image.frombytes("P", (W, H), make_frame(i, variant).tobytes())
            img.putpalette([g for g in range(256) for _ in range(3)])
            img.save(os.path.join(mask_dir, f"IMG{i:04d}_mask.png"))
        out = subprocess.run([sys.executable, SCRIPT, root, "--workers", "1", "--sample-step", "1"],
                             capture_output=True, text=True, check=True).stdout
        out_dir = os.path.join(root, "scroll", "orientation", "ScrollMasks")
        written = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
        first = np.array(Image.open(os.path.join(out_dir, written[0]))) if written else None
        return out, written, first


def test_clean_sequence_picks_scroll():
    out, written, first = run_case("clean")
    assert "scroll label 3" in out and "ok" in out, out
    assert len(written) == FRAMES
    expected = np.where(make_frame(0, "clean") == SCROLL, 255, 0)
    assert np.array_equal(first, expected)


def test_marker_behind_scroll_is_not_a_split():
    out, written, _ = run_case("occluded")
    assert "scroll label 3" in out and "ok" in out, out
    assert len(written) == FRAMES


def test_scroll_split_into_halves_fails_margin_guard():
    out, written, _ = run_case("halves")
    assert "FAIL" in out and "next candidate moves only" in out, out
    assert written == []


def test_embedded_fragment_fails_contact_guard():
    out, written, _ = run_case("embedded")
    assert "FAIL" in out and "of its outline on the scroll" in out, out
    assert written == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
