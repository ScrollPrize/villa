# Tutorial-strip thumbnail generators

Procedural generators (numpy + PIL + ffmpeg) for the pipeline thumbnails in
`static/img/tutorial-thumbs/`, rendered as chained stages of one story:

| Script | Output | Story |
| --- | --- | --- |
| `scan_thumb.py` | `top-scanning-small.webm` | charcoal scroll spins; x-ray point source converts it to red voxels bottom-up (75 frames) |
| `unwrap_thumb.py` | `top-representation-small.webm` | voxels fade; crumpled triangular mesh grows outward from the umbilicus; voxels dissolve; camera zooms to the top half (150 frames) |
| `flatten_thumb.py` | `top-segmentation-small.webm` | mesh towel-unrolls into a tilted flat sheet; solid red voxelization sweeps back over it (150 frames) |

Each gif opens on the previous one's final frame (pose, spin phase, camera).
All render at 600px and downsample to 300px; encode at **24 fps** so the strip
re-syncs with the untouched `top-prediction-small.webm` (6.25 s) every cycle.

Usage:
    python3 scan_thumb.py loop        # frames -> scanloop/
    ffmpeg -framerate 24 -i scanloop/f%03d.png -c:v libvpx-vp9 -b:v 0 -crf 38 \
        -pix_fmt yuv420p -cues_to_front 1 top-scanning-small.webm
    ffmpeg -i top-scanning-small.webm -frames:v 1 -c:v libwebp -quality 82 \
        top-scanning-small.webp      # poster

Notes:
- Scripts currently write frames to a hardcoded SCRATCH dir near the top of
  each file — point it somewhere local before running.
- The `?v=N` query on the component's URLs (`src/components/TutorialsTop.js`)
  must be bumped whenever these files are replaced, to bust browser caches.
- `scan_thumb.py` contains a vestigial `build_floor_tex()` from earlier
  tabletop iterations (unused since the scene went floorless).
