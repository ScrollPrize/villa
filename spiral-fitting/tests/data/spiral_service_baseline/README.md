# Pre-port Spiral service and workspace baselines

Captured for the object-store service port (OBJECT_STORE_SERVICE_PLAN.md,
stage 1 step 1) at `ea2e0bf05`, the last commit before `b795e458f` made the
object store mandatory. At that commit the service still read patch
directories, fiber JSONs and PCL documents from its dataset root. Stage 1
step 12 compares the store-backed service and workspace against these.

## Display geometry (`display_geometry_files.json`)

Viewer-frame (reference-volume voxel XYZ) coordinates of a fixed set of the
acceptance dataset's inputs, as the file-backed workspace displayed them:
three patches (one with `mask.tif`, one with invalid vertices), one PCL
collection of each role, and three fibers (one with a branch). From the
repository root, with `D=/mnt/bcache/projects/vesuvius-scrolls/spiral/dataset`:

```sh
python spiral-fitting/tests/display_geometry_dump.py files \
  --patch $D/verified_patches/auto_grown_20260420203759816_region_000 \
  --patch $D/verified_patches/0004_fill_sel_20260512_105437_26 \
  --patch $D/verified_patches/band-seed16000278-20260729-231955-336.tifxyz \
  --pcl $D/abs_winding.json#1 --pcl $D/relative_windings.json#196 \
  --pcl $D/same_windings.json#210 \
  --fiber $D/fibers/dj_20260717T165249423_000001.json \
  --fiber $D/fibers/kb_20260729T182039882_000211.json \
  --fiber $D/fibers/kb_20260708T210943392_000147.json \
  --fiber-scale 4 --out display_geometry_files.json
```

File mode reads only the files, so it gives the same result at either
checkout. Patches and PCL points are displayed as stored; the version-3
fibers declare no frame and are at 4x the reference volume, the scale their
store ingest declares. The store-backed workspace is compared with

```sh
python spiral-fitting/tests/display_geometry_dump.py store STORE.vcobj \
  --object UUID[@GEOMETRY_REVISION] ... --out store.json
python spiral-fitting/tests/display_geometry_dump.py compare \
  display_geometry_files.json store.json
```

Against the stage 0 acceptance store (`vc_object_ingest` with the README's
frames and repairs) every coordinate is identical. The one difference is
expected: `same_windings:210`'s point IDs are 7542-7544 in the store and 0-2
in the file, because the ingest's `--renumber-point-ids` repair renumbers
same-winding collections whose IDs repeat across collections.

## Service lifecycle (`lifecycle_transcript_preport.json`)

Initialize -> run -> stop -> preview -> save -> reconnect -> load checkpoint
over HTTP, with the resident runtime faked, recording each request's HTTP
status, response keys and lifecycle values and the published event kinds.
From the pre-port checkout's `spiral-fitting/`:

```sh
PYTHONPATH=. python <this branch>/spiral-fitting/tests/service_lifecycle_transcript.py \
  --out lifecycle_transcript_preport.json
```

Not captured: the real short session's preview manifest and checkpoint
provenance keys. It needs the GPU, which another job held throughout
(2026-10-27); a real session at the store-backed service is part of stage 1
step 14.
