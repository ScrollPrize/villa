#!/usr/bin/env python3
"""Run the continuous CPU tracer using this checkout's native build."""
from run_fiber_chunk_volume import load_runtime

if __name__ == '__main__':
    load_runtime()
    from vc.fiber_stream_volume import main
    raise SystemExit(main())
