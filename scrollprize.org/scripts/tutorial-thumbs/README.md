# Tutorial-strip thumbnail generators

Procedural generators (numpy + PIL + ffmpeg) for the pipeline thumbnails in
`static/img/tutorial-thumbs/`, rendered as chained stages of one story:

| Script | Output | Story |
| --- | --- | --- |
| `scan_thumb.py` | `top-scanning-small.webm` | charcoal scroll spins; x-ray point source converts it to red voxels bottom-up (75 frames) |
| `unwrap_thumb.py` | `top-representation-small.webm` | voxels fade; crumpled triangular mesh grows outward from the umbilicus; voxels dissolve; camera zooms to the top half (150 frames) |
| `flatten_thumb.py` | `top-segmentation-small.webm` | mesh towel-unrolls into a tilted flat sheet; solid red voxelization sweeps back over it (150 frames) |
| `ink_thumb.py` | `top-prediction-small.webm` | opens on flatten's exact framing, then the red sheet LIFTS straight up to a hover as the rig rotates to a flat working tilt and the camera eases back (the stack reads as a right prism, not oblique — both sheets almost fully in frame: kernel, gap, and the rows being painted below) and the scan starts: a tiny glowing inference kernel visibly sweeps the hovering intact input, its reading projected DOWN across the gap to a synced write head that rasters the ink prediction line by line onto the exact plane the sheet lay on; the camera dwells a long beat on the projection, then one slow direct zoom dives to a letter-by-letter cruise (the input rises clear of the frame as the dive starts) (the kernel's screen speed stays steady — the zoom absorbs the slow-down; the raster front rides just below the letter tops), and one last zoom pulls straight out to the completed reconstruction: the text finishes filling in as it lands, the rig untilts to a flat dead-on view, the red input rises away and fades like a retracting scanner, and the finished prediction (a real PHerc. Paris 4 prediction) holds padded for the last second (300 frames) |

Each gif opens on the previous one's final frame (pose, spin phase, camera).
`ink_thumb.py` duplicates `flatten_thumb.py`'s geometry prelude verbatim
(same seeds, same rng consumption order) so its frame 0 is pixel-identical
to flatten's final frame. All render at 600px and downsample to 300px;
encode at **24 fps** with durations that keep the strip in sync
(3.125 / 6.25 / 6.25 / 12.5 s — everything re-syncs every 12.5 s).

Usage (no host installs — run via uv):
    uv run --with pillow,numpy python scan_thumb.py loop   # frames -> scanloop/
    ffmpeg -framerate 24 -i scanloop/f%03d.png -c:v libvpx-vp9 -b:v 0 -crf 38 \
        -pix_fmt yuv420p -cues_to_front 1 top-scanning-small.webm
    ffmpeg -i top-scanning-small.webm -frames:v 1 -c:v libwebp -quality 82 \
        top-scanning-small.webp      # poster

If ffmpeg is not installed, use the static binary bundled with the
`imageio-ffmpeg` wheel:
    uv run --with imageio-ffmpeg python -c \
        "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"

Notes:
- Scripts currently write frames to a hardcoded SCRATCH dir near the top of
  each file — point it somewhere local before running.
- `ink_thumb.py` reads a real ink-prediction jpg from a hardcoded local path
  (PRED constant: PHerc. Paris 4, 2 µm, hann-blended tile inference); point
  it at any grayscale prediction — the image is flipped upright and
  recolored to the strip palette at load.
- The `?v=N` query on the component's URLs (`src/components/TutorialsTop.js`)
  must be bumped whenever these files are replaced, to bust browser caches.
- `scan_thumb.py` contains a vestigial `build_floor_tex()` from earlier
  tabletop iterations (unused since the scene went floorless).
- The pre-2026 `top-prediction-small.webm` (Blender-style render of a red
  voxel sheet scanned by a cyan cube, in the repo since the initial import;
  no generation code survives) is in git history before this replacement.
