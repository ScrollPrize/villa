# teça

Use teça to view Herculaneum scroll CT volumes in a browser, such as the Vesuvius Challenge's open data. It reads one
OME-Zarr CT volume and its tifxyz surfaces directly from their HTTP(S) host. It draws three slice panes at any rotation,
a 3D pane of the slice planes, and a Surface pane that shows a surface flat or through the volume along its normals.

Extensions add prediction layers, ink images, paint, point annotations, a 3D view of the CT, the open data's catalog,
CT filters, mesh edits, surface meshes and bookmarks. To remove one, delete its line in `src/main.js`.

## Start

```sh
python3 serve.py --open   # http://127.0.0.1:8767/
```

Add `--port` to choose another port. The help button in the top bar lists every key.

The address says what to open:

- `?volume=URL` opens a Zarr volume at its centre.
- `?overlay=URL` adds a tifxyz surface. Repeat it to add more.
- `#session=` holds the whole view. The viewer keeps it up to date, so you can copy the address to share the view.

This link opens PHerc. Paris 4 from S3 with its CT and one segment:

```
http://127.0.0.1:8767/?volume=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/&overlay=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/segments/20231221180251/mesh/20231221180251-on-20260411134726-2.4um.tifxyz/
```

To add the 3D ink prediction and the segment's 2D ink image, use the **Open data** choices in the layers and ink panels.

## Host it

Serve only `index.html`, `styles.css`, `src/` and `vendor/` from an HTTPS static host, with JavaScript and WebAssembly
content types. The browser requires HTTPS or localhost. A data host must allow CORS and byte-range requests. `serve.py`
is for local development.

Everything the viewer keeps, such as settings, downloaded chunks, paint and points, stays in the browser.

## Extensions

[docs/extensions.md](docs/extensions.md) is the contract with extensions.

## Develop

```sh
npm ci                                                        # Biome, the only development dependency
npm test                                                      # unit tests and the rules
npm run lint                                                  # npm run format writes the formatting
uv run tools/e2e.py                                           # browser checks on the real scroll, on a GPU
uv run tools/compare.py --a TREE --b TREE --runs 5 --out DIR  # two trees' timings and pixels
uv run tools/map.py                                           # keys, controls and panes of the running page
git config core.hooksPath "$PWD/tools/hooks"                  # from this folder: run the checks before each commit
```

The vendored code is three.js r164 and zarrita 0.7.4.
