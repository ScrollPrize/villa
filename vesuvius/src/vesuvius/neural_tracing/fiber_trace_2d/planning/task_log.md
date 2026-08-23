# Task log: catalogue-backed Fiberlet normal inputs

- Inspected the current manager: `fiberlet run` requires two completed local
  manager runs, while catalogue indexing exposes only CT volumes.
- Inspected native VC Lasagna access: direct remote manifests already support
  persistent read-through caching via `--remote-cache-dir`; this will be reused
  rather than duplicated.
- The cached Paris4 catalogue currently exposes the published Fiber Lasagna
  entry but no regular Lasagna-normal entry. A real catalogue refresh is still
  required before final availability reporting.
- Independent review corrected the normal contract to require
  `grad_mag`/`nx`/`ny`, preserved differing published Fiber/normal source
  levels, required the existing one-voxel base-shape tolerance, and identified
  remote provenance/resume and manager-owned cache-option requirements.
- Refreshed public catalogue SHA-256
  `992b52e239f4af7156fd168ab4b5f4411caf99f2a37884490d57843a8db903a2`.
  It still has no Paris4 normal entry. Direct anonymous S3 listing also found
  zero objects below `PHercParis4/representations/predictions/lasagna/`; this
  is missing source data, not a stale manager index.
- VC3D contains similar open-data Lasagna prefix preparation in a Qt GUI
  translation unit. Sharing that implementation with Python is not feasible
  without a cross-language library/CLI extraction beyond this manager fix.
  The Python manager will reuse the downloader's existing paginated S3 helper
  rather than copy listing logic within Python; cross-language duplication is
  recorded as an explicit deviation.
- Added exact-model Lasagna prediction indexing, role classification, stable
  `atlas:<model>@L<level>` selectors, cached root-manifest discovery, anonymous
  HTTPS locators, and the existing `lasagna-remote.json` persistent lazy cache
  layout.
- Made the normal input optional, retained explicit local-run overrides, added
  remote dependency identity and exact-hash resume, and corrected local normal
  validation to require native-compatible `grad_mag`/`nx`/`ny`.
- Added one-voxel-per-axis base-shape tolerance while preserving distinct Fiber
  and normal source levels. Updated cache-only completion, CLI help, manager
  docs, specs, and changelog.
- Validation:
  - `python -m pytest -q lasagna/tests/test_manager.py
    lasagna/tests/test_manager_open_data.py
    lasagna/tests/test_inference_provenance.py
    lasagna/tests/test_bootstrap_venv.py
    volume-cartographer/python/test_fiberlets_cli.py`: 87 passed.
  - `python -m pytest -q lasagna/tests/test_download_omezarr.py
    lasagna/tests/test_download_volume_list.py`: 25 passed.
  - `volume-cartographer/build/dev-quickbuild-gcc/bin/test_fiberlet_storage`:
    26 test cases passed.
  - Real cached completion exposes `atlas:20260419180421@L2` for PHerc0332 and
    PHerc1299 Fiber runs.
- Real public PHerc0332 resolution cached and validated manifest SHA-256
  `201c45c6e54a2ffd77b58fcf541322e4fc5cb031313326dc8d5254d19e5f2521`
  with `cos/grad_mag/nx/ny`, without launching preprocessing.
- Corrected the initial manager-only manifest-SHA cache directory after review.
  Published normals now use VC3D's exact canonical URL conversion,
  `open_data/lasagna/<sample>/<volume>/<identity>` directory calculation, and
  full `lasagna-remote.json` fields. Manifest SHA remains integrity/provenance
  only and does not address the cache or synthesize an Atlas run UUID; the
  exact remote artifact/manifest URL is the published source locator.
- Tightened resolution to the first `public-read` catalogue origin and made
  malformed/mismatched markers, invalid catalogue coordinate identity,
  outer/inner manifest disagreement, and missing/invalid remote Zarr
  descriptors hard errors. No alternate prediction or cache layout is used as
  a fallback.
- Real PHerc0332 preparation now resolves to
  `open_data/lasagna/PHerc0332/20251211183505/078820a387cf852f`, validates the
  manifest and all four group descriptors, and retains manifest SHA-256 only as
  dependency integrity metadata.
- Revalidation after the cache correction: 112 focused Python tests and 26
  native Fiberlet storage cases passed; `git diff --check` passed.
