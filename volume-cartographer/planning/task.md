# Task: separate persistent-cache probe, download, and decode scheduling

Refactor regular `ChunkCache` work into three independent shared queues:

1. A high-concurrency local persistent-cache probe queue with 32 workers.
2. A remote download queue retaining the configured download concurrency.
3. A CPU decode queue independent of both local probing and downloading.

The local probe must classify persistent data, persistent empty markers, and
cache misses without decoding payloads. Cached data proceeds to decode while a
cache miss proceeds immediately to download, so slow cached decodes cannot
delay admission of known remote work. Successful remote reads must likewise
hand encoded payloads to the decode queue instead of decoding on download
workers.

All three queues must retain the existing interactive/background and
view-relative priority model, atomic demand publication, invalidation, source
sharing, rendered values, persistent formats, and diagnostics semantics.
