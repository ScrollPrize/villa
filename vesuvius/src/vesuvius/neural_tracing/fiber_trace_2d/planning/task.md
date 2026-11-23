# Task: finish selected vc_reader migrations

Address audit items 1, 2, 3, 7 and 8: delete prepare_scroll.py; replace winding interpolation and surface/CT sampling with vc_reader; use shared rendering and surface transfer; delete obsolete reader/writer modules. Leave Zarr metadata/chunk interpretation, tifxyz metadata parsing, and projection (items 4–6) unchanged.
