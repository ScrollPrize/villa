# Graph object identity and external files

The graph identifies objects by UUID. Patch import preserves the UUID in
`meta.json`; fiber import preserves the UUID in the fiber JSON. Edges reference
those UUIDs. `name` is a display label, and `storage_ref` is a location hint,
normally relative to the graph file's directory. The graph stores metadata and
relationships; geometry remains in the external files.

Use `GraphDb::resolveStoragePath` in C++, or `resolve_storage_path` in Python,
to locate a file-backed patch or fiber:

```python
from vc.graph import GraphDb

with GraphDb.open("/data/graph.vcgr", read_only=True) as graph:
    patch_uuid = "a8fc18bb-7710-4143-908f-6cc6d3e03ecc"
    patch_dir = graph.resolve_storage_path(patch_uuid)
    # If geometry lives elsewhere, supply its search root:
    patch_dir = graph.resolve_storage_path(patch_uuid, dataset_root="/datasets/spiral")
```

Resolution first verifies the source UUID at the saved path. If that path is
missing, unreadable, or belongs to a different UUID, it searches the dataset
root recursively. By default the root is the graph file's directory; an
explicit root also rebases relative path hints. Absolute hints remain absolute.
The result is an absolute path (`pathlib.Path` in Python): the patch directory
or fiber JSON file. Missing or unsupported objects return `None`.

Discovery reads JSON metadata without loading TIFF geometry. It skips hidden
directories (including `.strip-cache`) and `.zarr` stores, and does not recurse
through directory symlinks. It raises an error on filesystem traversal failures
or multiple matching locations discovered during fallback, rather than choosing
an arbitrary copy. A valid saved hint is returned immediately without searching
for duplicates elsewhere. Fallback discovery is performed afresh on each call;
for bulk geometry loading, retain resolved paths for the duration of the load.

Renaming files or moving a dataset does not change object or edge UUIDs.
Resolution works with existing graphs and read-only handles and does not rewrite
stored hints, names, generations, or approval state. Callers loading geometry
should use this resolver instead of concatenating `storage_ref` themselves.
`sweep_external_data()` remains an audit of the stored path hints, so it can
report stale paths even when UUID discovery succeeds.

Track records use their existing dataset reference and external track key;
they are not individual UUID-bearing patch/fiber files and are outside this
resolver's scope.
