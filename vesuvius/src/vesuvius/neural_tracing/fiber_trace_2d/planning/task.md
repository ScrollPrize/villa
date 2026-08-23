# Task: manage whole-volume Fiberlet preprocessing and publication

Integrate the existing native whole-volume command

```text
vc_fiberlets preprocess-volume FIBER_MANIFEST OUTPUT_ZARR \
  --normal-manifest NORMAL_MANIFEST --threads 32
```

into `las_manager` on the Fiberlet development branch.

- Add a `fiberlet` command group that launches preprocessing from completed,
  local manager-produced Fiber 3D and regular Lasagna prediction runs.
- Run the native process as a durable manager job in tmux with the same run
  listing, attachment, logging, status reconciliation, and command recording
  used by inference.
- Keep the resumable float anchor cache local while making the final combined
  Fiberlet Zarr a portable artifact.
- Make the final Zarr self-describing: persist every effective processing,
  coordinate, layout, storage, and codec setting required to construct a
  compatible reader, together with stable source/model/scale identities.
- Derive dataset identity from that canonical structured metadata. Global
  processing settings belong in the identity; runtime directories, manifest
  paths, cache paths, and output paths do not.
- Reuse `las_manager open-data validate` and `open-data upload` for completed
  Fiberlet jobs.
- Add the minimal Atlas data type, ingestion, copy-first publication, and
  public catalogue support needed to publish Fiberlet Zarrs. Fiberlets are a
  derived representation of an existing volume, not a new trained model.
- Keep existing Fiber/Lasagna inference and upload behavior compatible.

This task begins with planning only. Do not implement until the plan is
approved.
