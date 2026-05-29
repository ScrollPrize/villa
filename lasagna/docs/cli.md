# CLI structure

[`fit.py`](../fit.py) is the CLI entrypoint for 3D surface fitting. It assembles arguments from independent parts, loads data/model, runs optimizer stages, and exports results.

## Argument parts

Each part provides `add_args(parser)` + `from_args(args)`:

- **data**: [`cli_data.py`](../cli_data.py)
  - `--input` path to `.lasagna.json` volume manifest
  - `--device` (default `cuda`)
  - `--seed cx cy cz` seed point in fullres voxels
  - `--model-w`, `--model-h` model extent in fullres voxels
  - `--winding-volume` optional winding volume zarr path
  - `--cuda-gridsample` enable custom CUDA uint8 sampling kernel
  - `--erode-valid-mask` erode valid mask by N pixels
  - `--sparse-prefetch-backend` sparse streaming prefetcher (`tensorstore` default; `python-zarr` keeps the zarr fallback path)

- **model**: [`cli_model.py`](../cli_model.py)
  - `--mesh-step` (default 100) height step in fullres voxels
  - `--winding-step` (default 25) radial step per winding
  - `--mesh-h`, `--mesh-w` mesh grid dimensions
  - `--depth` number of windings / model depth layers
  - `--windings` legacy alias for `--depth`
  - `--subsample-mesh`, `--subsample-winding` HR subsampling factors (default 4)
  - `--model-input`, `--model-output` checkpoint paths
  - `--init-mode` (`arc` or `straight`)
  - `--pyramid-d` enable depth-dimension pyramid

- **opt**: [`cli_opt.py`](../cli_opt.py)
  - `--snapshot-interval` save model snapshots every N steps
  - `--corr-snap` correction point snap mode
  - `--normal-mask-zero` mask zero-normal vertices in direction loss

## fit.py-specific arguments

- `--out-dir` output directory for snapshots and debug
- `--model-init` (`seed`, `ext`, `model`, or `flatten`; default `seed`)
- `--flatten-solver` (`torch`, `inverse`, or `forward`; default `torch`) selects the flattening variant when `--model-init flatten`
- `--tifxyz-init` tifxyz directory used when `--model-init ext`
- `--progress` print machine-readable `PROGRESS` lines to stdout

## JSON config

All arguments can be set via JSON config files (merged left-to-right). The `args` key in the JSON maps to CLI flags:

```json
{
  "args": {
    "input": "path/to/vol.lasagna.json",
    "mesh-step": 100
  },
  "stage_1": { "steps": 200, "global_opt": { ... } },
  "stage_2": { "steps": 100, "global_opt": { ... } }
}
```

Top-level keys (other than `args`) are parsed as optimizer stages by `optimizer.load_stages_cfg()`.

Special config keys consumed by fit.py/fit_service.py before stage parsing:
- `external_surfaces`: list of `{"path": "...", "offset": 1.0}` for offset mode, or one `{"path": "..."}` tifxyz source for flatten mode
- `corr_points`: correction point collections from VC3D
- `voxel_size_um`: voxel size for area calculations

`args.model-init` selects the initial mesh source:
- `seed` creates a fresh model from `args.seed`, `args.model-w`, `args.model-h`, and `args.depth`.
- `ext` initializes from the selected tifxyz mesh sent by VC3D.
- `model` initializes from the selected segment's `model.pt`.
- `flatten` optimizes a 2D flattening over one external tifxyz source. The default [`configs/flatten.json`](../configs/flatten.json) keeps the existing inverse-map Adam path. [`configs/flatten_forward.json`](../configs/flatten_forward.json) sets `flatten_solver: "forward"` to optimize source-vertex UVs with the same pyramid/Adam stages, then invert that UV map at export. Use exactly one `external_surfaces` entry supplied by VC3D or the calling config.

For VC3D integration, VC3D is transport only: it sends the selected tifxyz/model data and UI state it has available. `fit_service.py` / `fit.py` decide whether those fields are consumed as `tifxyz-init`, `external_surfaces`, approval-inpaint input, model checkpoint input, or ignored as surplus transport data.
