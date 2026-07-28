# 2026-07-28

- Native 3D Trace2CP live inference now uses the shared fiber 3D inference
  adapter plus shared Lasagna normal encoding and raw fiber prediction decode
  helpers.
- Native 3D Trace2CP now defaults to metric-only output; `--vis` explicitly
  enables JPG rendering and partial image updates.
