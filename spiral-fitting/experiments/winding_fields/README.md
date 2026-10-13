# Winding-field experiments

Research code for a winding field built from H/V fiber alternation (each fiber layer is half a winding).
These are experiment scripts, not a package. Most of them hardcode the remote experiment root
`/ephemeral/sean/experiments/winding_fields` (`data/`, `runs/`) and run with the spiral-fitting venv
(`PYTHONPATH=pydeps` for skimage).

Main pipeline (track graph → coupled slab fields → tifxyz):

- `track_graph.py`: global L1 sheet graph on track segments. Edges come from along-track links, H×V crossings and,
  unless `--no_wi` is given, winding_inference.
- Inputs from automated fiber volumes:
  - `afv_to_tracks.py` and `afv_crossings.py` turn an `.afv` into the track format and build the H×V
    ordering crossings.
  - `merge_sources.py` puts surface tracks and afv fibers into one graph.
- `track_targets.py` turns solved labels into voxel fill targets.
- `solve3d.py --track_targets` fills one slab. `run_full.py --mode tracks` drives all slabs.
- `finish_rb_gen.sh G R NAME`: red-black coupled slab fills, then stitch, extract, diagnostics, combine and flatten.
  Set `NO_WI=1` to skip the winding_inference numbering shift in `stitch.py`.
- `run_merged.sh` runs the merged tracks+afv experiments, with and without winding_inference.

Evaluation only, never used as input: `eval_mesh.py`, `mesh_sheets.py`, `track_eval.py`, `tracks_vs_mesh.py`.
`solve3d_v*.py`, `run_full_v*.py` and `extract_tifxyz_v1.py` are earlier versions kept for reference.
`flatten_windings.py` uses `spiral-fitting/tifxyz.py`.
