# Continuation to sweep 20

- Independent review approved uv_continue_plan.md; no solver changes.
- Submitted approved runner with run_uv_categorical_ccw_continue.json and
  run_uv_categorical_cw_continue.json. New roots data/uv_categorical_v1/{ccw,cw}_sweep20.
- Ten additional updates, unchanged parameters; local0..10=global10..20.
- Shared 12GiB GPU0/1. Initial manager status waiting_for_release.
- Existing warm-start checks hashes, evidence, IDs and shapes; old outputs intact.
- Both grants acquired and local sweep0 verified: CCW110/CW104 violations and
  retained arc62.4747946%/62.3828339%, exactly matching previous final states.
  Jobs actively computing additional sweeps; previous duration ~13min per10.
- Visualization follow-up (saved10-sweep states): equal U/V scale, no tooltips;
  LZW tiled BigTIFF replaces PNG. User reduced requested height4000 to1000px
  total including annotations/histograms. CPU-only rerender via approved runner.
  Independent review approved. Five tests pass incl exact tile-vs-full drawing
  across seams, full SVG fiber counts/no titles, TIFF1000px height/LZW metadata.
  TIFFs approximately5.5MiB each. No optimizer/inference changes.
- Both20-sweep solutions finished and saved. Additional10sweeps took496.09s
  CCW/523.70sCW. Sign violations110->28CCW and104->22CW; retained arc
  62.4748->64.8926%CCW and62.3828->64.7460%CW. Rejected pieces
  22.7383->20.5711%CCW and22.7887->20.6593%CW. Final-state cost not yet
  evaluated separately: last pre_update_energy belongs to sweep19, not20.
- User requested another80sweeps, total100. Added *_continue100.json configs:
  only warm_start, output_root and categorical_sweeps changed. Independent review
  confirmed saved20-sweep solutions and unchanged numerical/resource settings.
  Submitted both via approved evaluation runner; outputs ccw_sweep100/cw_sweep100.
  Prior additional10sweeps took8m16s/8m44s: extrapolated80sweeps~66–70min.
  Surface-construction implementation not yet specified; leave solves running.
