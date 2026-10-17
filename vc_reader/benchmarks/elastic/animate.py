"""Make enlarged, registered GIF comparisons from the measured elastic fields.

Intermediate frames resample the original 3D CT with a scaled displacement
field. They are not crossfades or repeated warps of an already warped image.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import compare as experiment

# Same parameters and seed as the static comparison. Include the broader,
# stronger B-spline example, rather than artificially magnifying its motion.
CASES = [('bspline', 8, 4.), ('bspline', 16, 4.), ('bspline', 32, 8.),
         ('normal_bspline', 16, 4.), ('coarse', 16, 4.), ('dense', 16, 4.)]
PALETTE = [channel for value in range(256) for channel in (value,)*3]
TILE, GAP, HEADER = 512, 16, 110


def frame(volume, name, method, scale, requested, actual, fraction):
    canvas = Image.new('L', (3*TILE+4*GAP, TILE+HEADER+GAP), 18)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=23)
    small = ImageFont.load_default(size=19)
    title = f'Paris 4 / {name} / {experiment.TITLES[method]} / scale {scale}'
    draw.text((GAP, 8), title, font=font, fill=245)
    state = 'ORIGINAL' if fraction == 0 else ('FULL DEFORMATION' if fraction == 1 else 'DEFORMING')
    draw.text((GAP, 39), f'{state}   {fraction:.0%}   |   RMS {fraction*actual:.2f} voxels'
              f'   |   requested endpoint {requested:g}, achieved {actual:.2f}', font=small, fill=230)
    for axis, (plane, label) in enumerate(zip(experiment.planes(volume), ['YX (Z=64)', 'ZX (Y=64)', 'ZY (X=64)'])):
        # Fixed contrast and a fixed grayscale palette across all frames.
        pixels = np.rint(np.clip(plane*(255.*255./190.), 0, 255)).astype(np.uint8)
        tile = Image.fromarray(pixels).resize((TILE, TILE), Image.Resampling.NEAREST)
        x = GAP+axis*(TILE+GAP)
        draw.text((x, 78), label+' / 4x display', font=small, fill=220)
        canvas.paste(tile, (x, HEADER))
    # Preserve grayscale exactly; no per-frame palette quantization or dithering.
    result = Image.frombytes('P', canvas.size, canvas.tobytes())
    result.putpalette(PALETTE)
    return result


def save_gif(path, frames, durations):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=2)
    with Image.open(path) as saved:
        assert saved.n_frames == len(frames), (path, saved.n_frames, len(frames))
        assert saved.info.get('loop') == 0
        for i, expected in enumerate(frames):
            saved.seek(i)
            np.testing.assert_array_equal(np.asarray(saved.convert('L')), np.asarray(expected.convert('L')))


def gallery(output, records):
    options = ''.join(f'<option value="{i}"'+(' selected' if i == 2 else '')+'>'+
                      html.escape(f'{r["chunk"]} — {experiment.TITLES[r["method"]]} — scale {r["scale"]}, '
                                  f'RMS {r["achieved_rms"]:.2f}')+'</option>' for i, r in enumerate(records))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paris 4 elastic animations</title>
<style>body{background:#151515;color:#eee;font:17px system-ui;margin:24px}
select,button,a{font:inherit;margin:6px;padding:8px}a{color:#9bd0ff}
img{display:block;width:100%;max-width:1600px;height:auto;margin-top:16px}
p{max-width:1050px;line-height:1.5}</style>
<h1>Paris 4: watch the deformation</h1>
<p>The same slice stays in the same place. Motion scales the actual displacement field
from zero to its measured endpoint and back, with a pause at each end.
Blink switches directly between original and endpoint. No crossfade, extra motion
amplification, or slice scrolling. The three panels are fixed orthogonal sections.</p>
<label>Example <select id="example">OPTIONS</select></label>
<label>View <select id="mode"><option value="motion">Smooth motion</option>
<option value="blink">Before / after blink</option>
<option value="original">Original (still)</option><option value="deformed">Deformed (still)</option></select></label>
<a id="download">Open selected file</a><img id="animation" alt="Registered CT deformation comparison">
<p>Start with cubic B-spline, scale 32 for the largest visible bend. Scale 16 is
the benchmark comparison, and scale 8 shows finer waviness. Endpoint strength
is capped by the same conservative fold-prevention policy as the static images.
These animations illustrate augmentation strength, not physical motion over time.</p>
<script>const records=RECORDS;
const example=document.getElementById('example'),mode=document.getElementById('mode');
function show(){const record=records[Number(example.value)];
const path=record[mode.value];document.getElementById('animation').src=path;
document.getElementById('download').href=path;}
example.onchange=show;mode.onchange=show;show();</script></html>'''
    (output/'index.html').write_text(page.replace('OPTIONS', options).replace('RECORDS', json.dumps(records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=experiment.ROOT/'vc_reader/elastic_comparison')
    args = parser.parse_args()
    reference = json.loads((args.comparison/'results.json').read_text())
    output = args.comparison/'animations'
    output.mkdir(parents=True, exist_ok=True)
    assert reference['extension_sha256'] == hashlib.sha256(Path(experiment.lab.__file__).read_bytes()).hexdigest()
    records = []
    # Eased 0 -> 1 -> 0 cycle; exact endpoints pause for 900 ms.
    fractions = (.5-.5*np.cos(np.linspace(0, np.pi, 13))).tolist()
    fractions[0], fractions[-1] = 0., 1.
    order = list(range(13))+list(range(11, 0, -1))
    durations = [900 if i in (0, 12) else 70 for i in order]
    for name, chunk in experiment.CHUNKS.items():
        raw, origin = experiment.read_crop(Path(reference['array']), chunk)
        prior = next(v for v in reference['inputs'] if v['name'] == name)
        assert hashlib.sha256(raw).hexdigest() == prior['raw_sha256']
        image = raw.astype(np.float32)/255
        original = image[64:192, 64:192, 64:192]
        normals = experiment.normal_lattice(raw, origin, 16)['normals']
        for method, scale, requested in CASES:
            print(f'{name}: {method}, scale {scale}, requested RMS {requested:g}', flush=True)
            base = experiment.lab.run(image, method, 128, 64., float(scale), .01,
                       experiment.BOUND, 123, normals if method == 'normal_bspline' else experiment.EMPTY)['field']
            field = experiment.lab.normalize(base, requested, experiment.BOUND)
            actual = float(np.sqrt(np.mean(np.sum(field*field, axis=-1))))
            endpoint = experiment.lab.warp(image, field, 64.)
            previous = next(v for v in reference['cases'] if v['chunk'] == name and v['method'] == method
                            and v['scale'] == scale and v['strength'] == requested)
            assert hashlib.sha256(endpoint).hexdigest() == previous['image_sha256'], 'Endpoint differs from benchmark'
            np.testing.assert_array_equal(experiment.lab.warp(image, np.zeros_like(field), 64.), original)
            frames = []
            for fraction in fractions:
                volume = original if fraction == 0 else (endpoint if fraction == 1 else
                         experiment.lab.warp(image, field*fraction, 64.))
                frames.append(frame(volume, name, method, scale, requested, actual, fraction))
            stem = f'{name}_{method}_s{scale}_r{requested:g}'
            motion, blink = stem+'_motion.gif', stem+'_blink.gif'
            save_gif(output/motion, [frames[i] for i in order], durations)
            save_gif(output/blink, [frames[0], frames[-1]], [1000, 1000])
            frames[0].save(output/(stem+'_original.png'))
            frames[-1].save(output/(stem+'_deformed.png'))
            records.append(dict(chunk=name, method=method, scale=scale, requested_rms=requested,
                                achieved_rms=actual, motion=motion, blink=blink,
                                original=stem+'_original.png', deformed=stem+'_deformed.png',
                                endpoint_sha256=previous['image_sha256']))
    (output/'manifest.json').write_text(json.dumps(dict(seed=123, fractions=fractions, frame_order=order,
        durations_ms=durations, contrast_uint8=[0, 190], records=records), indent=2)+'\n')
    gallery(output, records)
    print(f'Saved and decoded/verified {2*len(records)} GIFs in {output}', flush=True)


if __name__ == '__main__':
    main()
