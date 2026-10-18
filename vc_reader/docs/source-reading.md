# Reading local label sources

`Source.open()` currently reads local PNG, JPEG, TIFF, and explicitly ordered
stacks of 2D image files. It returns spatial crops with stored values or a
caller-declared integer/color encoding. This is the first implementation
increment of the label-reader plan; remote file caching and Zarr source
resolution remain outstanding. Existing `open_zarr()` is unchanged.

Use the optional `labels` extra (NumPy, Pillow, tifffile). Local image reads do
not import the native reader, PyTorch, or CUDA. Decoder imports happen only for
the selected file format. Additional TIFF codecs can require `imagecodecs`;
errors identify the compression and missing decoder. No installation happens
at runtime.

```python
from vc_reader import Encoding, Source, Storage
from vc_reader.reader import DecodedCache

source = Source(
    'annotation', Storage('/data/annotation.png', 'png'),
    role='label', axes=('y', 'x'),
    encoding=Encoding('integer', ((0, 0), (255, 1)), 'uint8'),
)
cache = DecodedCache(max_bytes=256 << 20, max_decode_bytes=256 << 20)
with source.open(decoded_cache=cache) as reader:
    labels = reader.read(start=(100, 200), size=(64, 96))
```

Starts/sizes contain only spatial coordinates, in YX or ZYX order. Output uses
storage channel positions: CYX stays CYX, YXC stays YXC. Select channels/time
through `source.with_selection(channels=(2, 0), time=0)`; a selected time axis
is removed, and singleton channels remain. TIFF series and pyramid levels must
be selected when ambiguous. Anonymous TIFF dimensions require declared axes;
channels are never guessed to be depth.

Reader `shape`, `dtype`, and `axes` describe the selected, encoded output.
Color encoding removes the color-channel axis. Palette PNGs expose their stored
indices and `reader.palette` (palette values, mode, and transparency); they are
not implicitly expanded to RGB. PNG alpha and grayscale bit depth are preserved.
Integer IDs, including uint64 values above float32 precision, never pass through
floating-point lookup arithmetic. Stored float32 TIFF samples stay float32.

Out-of-bounds crops retain the requested size and use `Source.fill_value`, or
zero for these image formats. The explicit fill belongs to the output dtype
and bypasses the encoding lookup. Unknown in-bounds encoding values raise unless
an encoding fallback was supplied. Failed/missing files raise rather than being
filled. `read_many()` returns an ordered list of arrays. Results remain usable
after subsequent reads, eviction, or reader close.

## Memory and workers

`decoded_cache_bytes` and `max_decode_bytes` default to 256 MiB per reader.
Alternatively, pass one `DecodedCache` to multiple readers to share both budgets
within each process. Do not also pass byte limits to those readers: configure
them on the cache object. Closing a reader clears its owned cache but leaves an
explicitly shared cache available to its other users.

PNG/JPEG decode whole images. TIFF reads selected pages and memory-maps
uncompressed pages when supported. Compressed TIFF currently decodes a selected
page within the explicit limit; direct tile/strip decoding remains outstanding.
Preflight estimates include decoder buffers, compressed input, decoded segments,
and selection/encoding scratch. These are conservative allocation estimates,
not an operating-system RSS limit on opaque codec internals. Cache retention
and caller-owned output arrays are accounted separately. A small returned crop
does not retain a whole decoded image. `cache.stats()` exposes retained bytes,
hits/misses, and the peak estimated working set.

Work on a shared cache is serialized; tifffile decode uses one worker. Separate
cache owners can run concurrently. Pickling stores descriptors/configuration,
not cached arrays or file handles. Spawn/fork workers reopen lazily and start
with empty caches and new locks. `close()` is idempotent, and reads after close
raise. Local-file validators detect changes during an open reader's lifetime.
Ordered stacks inspect the first image at open, validate other planes as they
are requested, and never sort filenames or probe missing unused planes.

## Remaining plan work

- HTTP/S3 file materialization, authentication reuse, offline file hits/misses,
  bounded encoded disk caching, and concurrent atomic publication.
- `Source.open()` over Zarr arrays/groups, storage transforms and units,
  corresponding levels, changed-grid registration, and companion validation.
- TIFF tile/strip decoding, color mappings across separate channel pages, and
  additional codec/layout coverage. Unsupported cases fail explicitly.
- Complete real remote-asset acceptance, decoder memory/I/O instrumentation,
  and macOS/arm64 runtime checks.

Raw reads accept unregistered sources; they do not establish alignment.
Aligned sampling and CPU/CUDA categorical equality belong to plan 03.

## Run and validate

With the package and `labels` dependencies available, no native rebuild is
needed for this increment:

```bash
AGENTS_AGENT_MODE=1 python -m unittest discover -s tests -p test_source_readers.py -v
AGENTS_AGENT_MODE=1 python benchmarks/bench_source_readers.py --output /tmp/source-reading.json
```

The isolated worktree was validated using dependencies from the existing local
environment without installation. These exact commands keep the worktree ahead
of the installed editable package and the standard library ahead of installed
backports, including in subprocesses:

```bash
cd /tmp/vc-reader-plan02-worktree/vc_reader
export AGENTS_AGENT_MODE=1
export PYTHONPATH="src:/usr/lib/python3.14:/home/sean/Documents/villa2/villa/vc_reader/.venv/lib/python3.14/site-packages"
/usr/bin/python3.14 -S -m unittest discover -s tests -p test_source_readers.py -v
VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda /usr/bin/python3.14 -S -m unittest discover -s tests -v
/usr/bin/python3.14 -S benchmarks/bench_source_readers.py --output /tmp/vc-reader-plan02-local-benchmark.json
```

The 16 new tests cover exact decoding/encoding, padding, TIFF selection,
ordered stacks, metadata disagreement, pre-decode limits, shared caches,
concurrency/eviction, buffer ownership, lazy imports, and spawn/fork reopening.
CUDA sampling is not implemented in this increment.

The full configured suite completed with 52 passing tests and 20 CUDA-dependent
skips (72 total); the sandbox exposed no usable CUDA device. All public reader
exports and an existing uint16 Zarr crop were also smoke-tested against the
cached native build. No dependencies were installed or extensions rebuilt.

The real-data smoke benchmark uses the bundled PHercParis4 geometry mask
`benchmarks/fixtures/fiber_132487/mask.tif`, shape `(25, 4800)`, uint8,
uncompressed TIFF. Its SHA-256 is
`bcc69634665c8bcde3e0736232eea41e00c039f83ce52ce60244d041cc16166f`.
It is a surface geometry mask, not an ink-label dataset. Six fixed `(16, 512)`
crops are checked against tifffile, with two warmups and ten measured
repetitions (60 measured crops per method). Linux amd64, Python 3.14.4,
NumPy 2.5.4, tifffile 2026.10.15; the benchmark loads no native extension.

| Method | Mean ms/crop | p50 | p95 |
| --- | ---: | ---: | ---: |
| Direct tifffile page read/copy | 0.0949 | 0.0944 | 0.0999 |
| Source reader | 0.1161 | 0.1159 | 0.1229 |

Opening took 1.041 ms and the first read took 0.318 ms. The source reader's
warm traced Python allocation peak was 191,028 bytes; whole-process peak RSS
was 50,089,984 bytes. Mapped pages retained zero decoded-cache bytes. Physical
I/O bytes were not measured. The OS file cache was not cleared, so the first
read is not a disk-cold measurement. This is initial wrapper-overhead evidence,
not completion of the plan's remote/codec acceptance protocol.
