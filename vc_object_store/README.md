# vc_object_store

A SQLite object database for geometry, revisioned metadata, and relationships.
Includes native ingestion/export, spatial queries, relationship detectors, C++
command-line tools, and a Python package. Storage and detectors build without
volume-cartographer or Qt. CT-based detectors link vc_reader's C++ reader and
structure-tensor normal implementation.

Shared C++ and Python geometry types represent point collections, fibers,
tracks and patches independently of files and object identity. Snapshot views
provide typed, bounded reads and checked sparse edits. See the
[geometry API](docs/typed-geometry.md) for examples and lifetime rules.

## Build

Build from a villa checkout, with `vc_reader` beside this directory:

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja -C build
```

Dependencies: C++20 (C++23 for Python bindings), SQLite with R-tree support,
OpenSSL Crypto, zstd, nlohmann-json, OpenCV core,
libtiff and OpenMP. The native reader additionally needs libcurl, c-blosc and
VC-Delta3D; its CMake build fetches the pinned codec if no installed package is
available. `VC_OBJECT_STORE_READER_SOURCE` selects a different reader checkout.

For the storage library alone, disable `VC_OBJECT_STORE_BUILD_ADAPTERS` and
`VC_OBJECT_STORE_BUILD_TOOLS`. This does not configure the reader or require
OpenCV, TIFF, OpenMP, Python, or Qt.

For Python, configure with `-DVC_OBJECT_STORE_BUILD_PYTHON=ON` and
`-DPython_EXECUTABLE=/path/to/python`. The selected environment needs nanobind,
NumPy, and pytest for the Python tests. Python 3.14+ is supported.

```sh
ninja -C build
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python \
  /path/to/python -m pytest -q tests/python
```

Python packaging uses scikit-build-core. Build wheels from this monorepo checkout;
the sibling native reader sources are required at build time. The installed wheel
contains the native reader library, and does not require VC or a Python reader
installation. CMake installation installs the CLIs, headers, and native libraries.

## Use

```python
from vc_object_store import ObjectStore

with ObjectStore.create("objects.vcobj", author="reviewer") as store:
    report = store.ingest_geometry_source("patch", "/data/patch")
    assert not report["problems"], report["problems"]
```

```sh
build/bin/vc_object_ingest --db objects.vcobj --author "$USER" --patch /data/patch
build/bin/vc_object_links --db objects.vcobj --author "$USER" --link-patch-overlaps
build/bin/vc_object_links --db objects.vcobj --author "$USER" \
  --link-fiber-patches --ct /data/ct.zarr
```

See the [API guide](docs/README.md) for checked revisions, geometry reads and
edits, graph snapshots, spatial queries, and synchronization. Detector distances
and scale options use working voxels. CT normals use uint8 Zarr data with complete
109³ support. Native CT opening supports local and anonymous HTTP/S3 arrays and
pyramid levels; credentials and disk-cache configuration are not exposed by this
adapter. Codec support follows vc_reader.

Surface proximity queries use the store's resident bounds index and exact
triangles. Bounds are not quantized.

Tests are self-contained and do not require the volume-cartographer checkout or
its fixture data.
