import ctypes
import os
import sys
import numpy as np
import subprocess

# --- Library Loading Logic ---
_lib = None

def _load_library():
    global _lib
    if _lib is not None:
        return _lib
        
    base_dir = os.path.dirname(__file__)
    lib_path = os.path.join(base_dir, 'libvesuvius.so')
    
    if not os.path.exists(lib_path):
        # Try to build it if it doesn't exist
        print(f"libvesuvius.so not found at {lib_path}. Attempting to build...")
        try:
            # Simple build command
            cmd = [
                "gcc", "-shared", "-fPIC", "-O3",
                "-I..",
                "-o", lib_path,
                os.path.join(base_dir, "vesuvius_c_impl.c"),
                "-lcurl", "-lblosc2", "-ljson-c", "-lm"
            ]
            subprocess.run(cmd, check=True, cwd=base_dir)
            print("Successfully built libvesuvius.so")
        except Exception as e:
            print(f"Failed to build libvesuvius.so: {e}")
            print("Please ensure you have gcc, libcurl-dev, libblosc2-dev, and libjson-c-dev installed.")
            sys.exit(1)
            
    _lib = ctypes.CDLL(lib_path)
    return _lib

# Initialize lib
_lib = _load_library()

# --- Struct Definitions ---
class ZarrCompressorSettings(ctypes.Structure):
    _fields_ = [
        ("blocksize", ctypes.c_int32),
        ("clevel", ctypes.c_int32),
        ("cname", ctypes.c_char * 32),
        ("id", ctypes.c_char * 32),
        ("shuffle", ctypes.c_int32),
    ]

class ZarrMetadata(ctypes.Structure):
    _fields_ = [
        ("shape", ctypes.c_int32 * 3),
        ("chunks", ctypes.c_int32 * 3),
        ("compressor", ZarrCompressorSettings),
        ("dtype", ctypes.c_char * 8),
        ("fill_value", ctypes.c_int32),
        ("order", ctypes.c_char),
        ("zarr_format", ctypes.c_int32),
        ("dimension_separator", ctypes.c_char),
    ]

class Volume(ctypes.Structure):
    _fields_ = [
        ("cache_dir", ctypes.c_char * 1024),
        ("url", ctypes.c_char * 1024),
        ("metadata", ZarrMetadata),
    ]

class Chunk(ctypes.Structure):
    _fields_ = [
        ("dims", ctypes.c_int * 3),
        # data follows: float data[]
    ]

# --- Function Signatures ---
_lib.vs_zarr_parse_zarray.argtypes = [ctypes.c_char_p]
_lib.vs_zarr_parse_zarray.restype = ZarrMetadata

_lib.vs_zarr_read_chunk.argtypes = [ctypes.c_char_p, ZarrMetadata]
_lib.vs_zarr_read_chunk.restype = ctypes.POINTER(Chunk)

_lib.vs_chunk_free.argtypes = [ctypes.POINTER(Chunk)]
_lib.vs_chunk_free.restype = None

_lib.vs_vol_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.vs_vol_new.restype = ctypes.POINTER(Volume)

_lib.vs_vol_get_chunk.argtypes = [ctypes.POINTER(Volume), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
_lib.vs_vol_get_chunk.restype = ctypes.POINTER(Chunk)

_lib.vs_vol_free.argtypes = [ctypes.POINTER(Volume)]
_lib.vs_vol_free.restype = None

# --- High-Level API ---

class VesuviusVolume:
    """Modern wrapper supporting remote data fetching and on-disk caching."""
    def __init__(self, cache_dir, url=None):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.url = url or ""
        self._vol_ptr = _lib.vs_vol_new(self.cache_dir.encode('utf-8'), self.url.encode('utf-8'))
        if not self._vol_ptr:
            raise RuntimeError("Failed to initialize VesuviusVolume")
        
        self.metadata = self._vol_ptr.contents.metadata
        self.shape = tuple(self.metadata.shape)
        self.chunks = tuple(self.metadata.chunks)

    def get_chunk(self, z, y, x, depth=None, height=None, width=None):
        """Fetch an arbitrary volume chunk by voxel coordinates."""
        start = (ctypes.c_int32 * 3)(z, y, x)
        dims = (ctypes.c_int32 * 3)(
            depth or self.chunks[0], 
            height or self.chunks[1], 
            width or self.chunks[2]
        )
        
        chunk_ptr = _lib.vs_vol_get_chunk(self._vol_ptr, start, dims)
        if not chunk_ptr:
            raise RuntimeError(f"Failed to fetch chunk at ({z}, {y}, {x})")
            
        try:
            actual_dims = tuple(chunk_ptr.contents.dims)
            size = actual_dims[0] * actual_dims[1] * actual_dims[2]
            data_addr = ctypes.addressof(chunk_ptr.contents) + ctypes.sizeof(ctypes.c_int * 3)
            float_array = (ctypes.c_float * size).from_address(data_addr)
            return np.ctypeslib.as_array(float_array).copy().reshape(actual_dims)
        finally:
            _lib.vs_chunk_free(chunk_ptr)

    def __del__(self):
        if hasattr(self, '_vol_ptr') and self._vol_ptr:
            _lib.vs_vol_free(self._vol_ptr)
