\# CLI structure

[`fit.py`](fit.py) is the CLI entrypoint. It assembles arguments from independent parts and then performs the top-level orchestration.

Parts (current placeholders):

- data: [`cli_data.py`](cli_data.py)
	- loading args
		- `--input` currently expects a directory containing `*_cos.tif`, `*_mag.tif`, `*_dir0.tif`, `*_dir1.tif`
		- `--crop x y w h` crops all channels
		- `--downscale` (default 4.0) downsamples all channels equally
	- UNet inference will move here later
- model: [`cli_model.py`](cli_model.py)
	- step-size args (mesh/winding step in pixels)
	- target subsampling args (`--subsample-mesh`, `--subsample-winding`)
- opt: [`cli_opt.py`](cli_opt.py)
	- stages / device args
	- `--snapshot-interval` (default 1000) saves a `vis.save(...)` snapshot

- vis: [`cli_vis.py`](cli_vis.py)
	- saves `res_grid_<postfix>.jpg` and `res_tgt_<postfix>.tif`
	- during `fit.py` execution we currently save `res_*_init` and `res_*_final`

Each part provides:

- `add_args(parser)` to register its arguments
- `from_args(args)` to convert `argparse` namespace into a typed config object
