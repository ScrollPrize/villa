# Task: Lasagna inference manager CLI

Add an installed `las_manager` command under `lasagna/` that manages Fiber 3D
inference, Lasagna inference, and Vesuvius Atlas upload in ordered implementation
phases over one shared persisted run/catalog model.

Inspect `/home/hendrik/vesuvius-atlas` and make each Fiber inference bundle emit
portable, schema-versioned provenance sufficient to construct the Atlas
`DataEntry`. Follow Atlas's copy-first publication model: the upload phase
puts the bundle at a configured source S3 origin, ingests it on the source
volume, and lets Atlas data-sync publish eligible data. Fiber and direct
Lasagna predictions use Atlas's existing `lasagna` artifact/model task; their
portable provenance retains the producing backend and product layout.

The manager must provide global configuration, cached open-data volume
discovery, snapshot discovery and metadata, volume prefetch, tmux-backed
inference launches, inference/run status, tmux integration, shell completion,
and reproducibility records. Fiber and Lasagna inference outputs must default
to Zstd compression. It must reuse the existing downloader and inference
entrypoints rather than copy their behavior.

Required command families are `config`, `snapshot`, `volume`, `inference`,
`run`, and `tmux`, with unambiguous command-prefix abbreviations. Open-data
metadata comes from
`https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json`,
is explicitly refreshable, and refreshes automatically when at least one hour
old. Inference launches run in well-named tmux sessions and produce one run
directory containing top-level logs/metadata plus an `artifacts/` subdirectory.
