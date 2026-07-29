# Remote Lasagna Manifest Support For Native Fiber Metric

Add low-level support for opening Lasagna-style manifests directly from remote
HTTP/S3 URLs, and use that path in the native 3D fiber metric command-line
runner.

Current behavior must continue to work:

- local `.lasagna.json` manifests;
- local manifests with adjacent `lasagna-remote.json` read-through cache
  markers;
- explicit `--normal-manifest` for normals.

New behavior:

- the fiber inference manifest positional argument may be a remote
  `s3://`, `s3+REGION://`, `http://`, or `https://` manifest URL;
- the normal manifest may use the same remote forms;
- remote manifests are fetched on demand for the current run, parsed as
  manifests, and not persisted as durable cache state;
- relative `groups.*.zarr` paths in the manifest are resolved against the
  remote manifest's parent URL, so the referenced Zarr groups stream from the
  same artifact location;
- absolute `groups.*.zarr` paths are also supported: local absolute paths are
  opened as local Zarr groups, and absolute remote `s3://`, `s3+REGION://`,
  `http://`, or `https://` paths stream through the same read-through cache;
- remote Zarr objects are persisted in the user-supplied local remote cache
  directory through the existing object-for-object read-through store;
- if a remote manifest is requested without an explicit cache directory, fail
  with a clear error;
- no VC3D project JSON / volpkg support is required in this task.
