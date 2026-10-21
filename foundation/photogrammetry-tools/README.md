# Photogrammetry review tools

Scripts for the stages of the [photogrammetry instructions](../photogrammetry-docs/INSTRUCTIONS.md): splitting a capture
into acquisition blocks, then one review script per stage, each printing the checks the instructions ask for and, where
useful, writing an image to inspect. Keep their output with the run.

| Stage | Script | Checks | Image |
|---|---|---|---|
| Acquisition blocks | `find_blocks.py scan RAW_DIR OUT_DIR`, then `find_blocks.py link OUT_DIR/blocks.tsv RAW_DIR ROOT` | groups a flat RAW folder into blocks from capture times and camera settings (for captures without annotation photographs); frame-number gaps; duplicate files | contact sheet of every burst |
| Masking | `mask_sheet.py ROOT sheet.jpg` | every JPEG has a same-sized, non-empty `ScrollMasks/` mask; mask area range and largest frame-to-frame change per directory | overlay contact sheet |
| Sparse SfM | `sfm_review.py SFM_DIR BLOCKS` | posed views and residuals per block; verified image pairs between orientations; camera circles and shared axes per orientation (needs `review_sfm_data.json`) | |
| Scale | `scale_review.py CANDIDATES` | per candidate: scale, triangulated corners, marker side lengths, off-plane residual, marker ring radius | |
| OpenMVS | `mesh_qa.py MESH.ply [...] --undistorted DIR --render views.png --texture TEXTURED.ply` | masks reached OpenMVS; finite, valid, watertight, manifold, one component; principal-axis extents | shaded and textured views along each principal axis |
| Marker scale | `marker_scale.py SCENE.json IMAGE_DIR keep-*.txt [--json OUT]` (needs OpenCV with aruco) | per block, lens distortion refitted from the markers and corners re-triangulated at 1.5 px: scale factor, side lengths, planarity, ring radius; the factor to use (median of the blocks with a consistent ring) | |
| Split orientations: join | `split_align.py align / silhouette / merge / check` (run scripts `split_recon.sh`, `merged_openmvs.sh`) | rim alignment of two separately reconstructed orientations, hypothesis ranking and rim scale; thickness against the photographs; merged scene for one OpenMVS run; merged reprojection, mesh vs both halves, mesh outline vs masks | outline contact sheet |
| PGS input | `prepare_input.py WORK ID` | builds `pgs-input/` (photographs and masks), the scale include lists and the per-orientation split inputs; refuses a photograph without a mask | |
| Unattended run | `sudo bash batch/batch_recon.sh WORK ID ...` | joint SfM for every object, then the joint route (marker scale, mm scene, OpenMVS) or the split route with the automatic join (`batch/auto_split.sh`, rules in `batch/split_gate.py`) | |
| Case | `scrollcase_qa.py OUTPUT_DIR ID --render kit.png` | three STLs and a CSV row; per-STL topology; split vs cavity direction; scroll-to-case clearance, nothing inside case material | sections and assembly views |

`ROOT` and `BLOCKS` are the acquisition tree (one `<orientation>_camera_<height>/` directory per block, as in the
instructions); `SFM_DIR` is the PGS-Recon `--stop-after sfm` output (it contains `mvg/`); `CANDIDATES` holds one directory
per scale candidate from the instructions' scaler loop. For the trajectory review, export the poses first, in the PGS-Recon
container:

```bash
openMVG_main_ConvertSfM_DataFormat -i <SFM_DIR>/mvg/recon_dir/sfm_data.bin \
  -o <SFM_DIR>/mvg/recon_dir/review_sfm_data.json -V -I -E
```

## Environment

All scripts run in the [scrollcase](../scrollcase/) virtual environment, which has numpy, scipy, Pillow, trimesh and
meshlib; `find_blocks.py` also needs the `exiftool` command, which the masking step already requires, and
`marker_scale.py` needs OpenCV with the aruco module (`opencv-contrib-python`), which that environment lacks:

```bash
../scrollcase/.venv/bin/python mesh_qa.py ...
```

Rendering samples a few million surface points; cap memory on shared machines, e.g.
`systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 ...`.

## Tests

```bash
../scrollcase/.venv/bin/python -m unittest discover -s tests
```
