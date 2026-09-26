"""Decode zarr array levels in place so loader workers memory-map raw chunks.

  python scripts/decode_store.py /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr/[0-5] ...
"""
from vesuvius.neural_tracing.fiber_follow.decode_store import main

if __name__ == '__main__':
    main()
