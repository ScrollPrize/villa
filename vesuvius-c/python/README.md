# Vesuvius-C Python Bindings

Highly optimized Python bindings for the official `ScrollPrize/villa/vesuvius-c` library.

## Features
- **Zero-copy** data transfer from C to NumPy.
- **Remote Fetching**: Automatically download and cache Zarr chunks from `dl.ash2txt.org`.
- **High Performance**: Bypasses Python's memory management for massive volume processing.

## Installation

### Prerequisites
```bash
sudo apt-get install -y libcurl4-openssl-dev libblosc2-dev libjson-c-dev
```

### Build and Install
```bash
cd villa/vesuvius-c/python
pip install .
```

## Usage
```python
from vesuvius_c import VesuviusVolume

# Initialize with a local cache directory and an optional remote URL
vol = VesuviusVolume(
    cache_dir="./cache", 
    url="https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/"
)

# Fetch an arbitrary chunk by voxel coordinates (z, y, x)
chunk = vol.get_chunk(z=1000, y=2000, x=3000, depth=64, height=256, width=256)
print(chunk.shape) # (64, 256, 256)
```
