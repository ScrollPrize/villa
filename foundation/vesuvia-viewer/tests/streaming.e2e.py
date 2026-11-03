import json
from pathlib import Path

import numpy as np

from page import PIXEL_TOLERANCE, SETTLED, centre_pixel, gl_errors, settle

REPO = Path(__file__).resolve().parent.parent
out = REPO / "outputs"
out.mkdir(exist_ok=True)

large = out / "e2e-large.zarr"
large.mkdir(exist_ok=True)
(large / ".zarray").write_text(json.dumps({
    "zarr_format": 2, "shape": [256, 256, 128], "chunks": [256, 256, 128], "dtype": "|u1", "order": "C",
    "filters": None, "compressor": None, "fill_value": 0,
}))
(large / "0.0.0").write_bytes(bytes([173]) * (256 * 256 * 128))
fill = out / "e2e-fill.zarr"
fill.mkdir(exist_ok=True)
(fill / ".zarray").write_text(json.dumps({"zarr_format": 2, "shape": [16,16,16], "chunks": [16,16,16],
                                       "dtype": "|u1", "order": "C", "filters": None, "compressor": None, "fill_value": 93}))
bricks = out / "e2e-bricks.zarr"
bricks.mkdir(exist_ok=True)
(bricks / ".zarray").write_text(json.dumps({
    "zarr_format": 2, "shape": [256, 256, 256], "chunks": [256, 256, 256], "dtype": "<u2", "order": "C",
    "filters": None, "compressor": None, "fill_value": 0,
}))
axis = np.arange(256, dtype=np.uint8)
values_ZYX = (axis[:, None, None] + 3 * axis[None, :, None] + 5 * axis[None, None, :]).astype('<u2') * 257
(bricks / "0.0.0").write_bytes(values_ZYX.tobytes())
ramp = out / "e2e-ramp.zarr"
ramp.mkdir(exist_ok=True)
origin_zyx, scale_zyx = [100, 200, 300], [8, 8, 4]
(ramp / ".zattrs").write_text(json.dumps({"multiscales": [{"datasets": [
    {"path": str(k), "coordinateTransformations": [
        {"type": "scale", "scale": [d * scale for scale in scale_zyx]},
        {"type": "translation", "translation": [origin + (d - 1) / 2 * scale for origin, scale in zip(origin_zyx, scale_zyx)]},
    ]} for k, d in enumerate([1, 4])
]}]}))
for k, d in enumerate([1, 4]):
    folder = ramp / str(k)
    folder.mkdir(exist_ok=True)
    size = 64 // d
    (folder / ".zarray").write_text(json.dumps({"zarr_format": 2, "shape": [size] * 3, "chunks": [size] * 3,
        "dtype": "|u1", "order": "C", "filters": None, "compressor": None, "fill_value": 0}))
    (folder / "0.0.0").write_bytes(bytes(4 * x * d + 2 * (d - 1) for z in range(size) for y in range(size) for x in range(size)))

HASH = "#ct=outputs/e2e-large.zarr&overlays=0,0"


async def check(browser):
    failures = []
    await browser.wait_for(SETTLED, 90)
    await browser.evaluate("(async () => { const { readHash } = await import('./ui/hash.js');"
                           f" await Promise.all([vesuvia.load(readHash('#ct=outputs/e2e-fill.zarr&layer=outputs/e2e-fill.zarr')), vesuvia.load(readHash('#ct=outputs/e2e-large.zarr&overlays=0,0'))]) }})()")
    await settle(browser)
    published = await browser.evaluate("({ url: vesuvia.scene.ct.volume.url, layer: vesuvia.scene.layers.length === 0 })")
    if not (published["url"].endswith("outputs/e2e-large.zarr") and published["layer"]):
        failures.append(f"overlapping reloads published {published}, expected the large CT alone")
    pixel = await browser.evaluate(centre_pixel("xy"))
    if pixel != [173, 173, 173, 255]:
        failures.append(f"large chunk: drew {pixel}, expected [173, 173, 173, 255]")

    await browser.evaluate("(async () => { const {readHash} = await import('./ui/hash.js');"
                           " await vesuvia.load(readHash('#ct=outputs/e2e-bricks.zarr&layer=outputs/e2e-bricks.zarr&zoom=1&black=0&white=65535&overlays=0,0'));"
                           " vesuvia.scene.layers[0].style = {visible:true,mode:'probability',opacity:1,threshold:0,color:[1,0,1]};"
                           " vesuvia.onChange() })()")
    for zyx in ([1, 2, 3], [127, 127, 127], [128, 128, 128], [254, 254, 254], [160, 80, 40]):
        await browser.evaluate(f"vesuvia.scene.focus_zyx = {zyx}; vesuvia.onChange()")
        await settle(browser)
        value = (zyx[0] + 3 * zyx[1] + 5 * zyx[2]) % 256
        probability = value / 255
        alpha = probability ** 2 * (3 - 2 * probability)
        expected = [value * (1 - alpha) + 255 * alpha, value * (1 - alpha), value * (1 - alpha) + 255 * alpha]
        pixel = await browser.evaluate(centre_pixel('xy'))
        if any(abs(actual - want) > PIXEL_TOLERANCE for actual, want in zip(pixel, expected)):
            failures.append(f"brick at {zyx}: drew {pixel}, expected {expected}")
    brick_shape = await browser.evaluate("vesuvia.scene.layers[0].residency.levels[0].chunk_zyx")
    if brick_shape != [128, 128, 128]:
        failures.append(f"ink brick shape {brick_shape}, expected [128, 128, 128]")

    await browser.evaluate("(async () => { const {readHash} = await import('./ui/hash.js');"
                           " await vesuvia.load(readHash('#ct=outputs/e2e-fill.zarr&black=0&white=255&overlays=0,0')); })()")
    await settle(browser)
    if await browser.evaluate(centre_pixel('xy')) != [93,93,93,255]:
        failures.append("missing chunk did not render its fill value")

    await browser.evaluate("""(async () => {
      const { readHash } = await import('./ui/hash.js');
      await vesuvia.load(readHash('#ct=outputs/e2e-ramp.zarr&layer=outputs/e2e-ramp.zarr&atlas=8&focus=32,32,32&zoom=1&black=0&white=255'));
      vesuvia.scene.overlays = {outline:false,crosshair:false};
      vesuvia.scene.layers[0].style.visible = false; vesuvia.onChange();
    })()""")
    for level in (0, 1):
        await browser.evaluate(f"vesuvia.scene.levelCap={level}; vesuvia.onChange()")
        await settle(browser)
        pixel = await browser.evaluate(centre_pixel('xy'))
        if pixel != [128, 128, 128, 255]:
            failures.append(f"NGFF ramp level {level}: {pixel}, expected grey128")
    return failures + await gl_errors(browser)
