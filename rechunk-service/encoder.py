"""H264 encoding/decoding for 3D volumetric chunks via ffmpeg subprocess.

Produces VC3D-format output compatible with volume-cartographer's VcDecompressor.

Header format (20 bytes, little-endian):
  [0:4]   magic   "VC3D"
  [4:6]   codec   uint16 (0=H264, 1=H265, 2=AV1)
  [6:8]   qp      uint16
  [8:12]  depth   uint32 (Z)
  [12:16] height  uint32 (Y)
  [16:20] width   uint32 (X)
"""

import struct
import subprocess
import numpy as np

MAGIC = b"VC3D"
HEADER_SIZE = 20
CODEC_H264 = 0
CODEC_H265 = 1
CODEC_AV1 = 2


def write_header(codec: int, qp: int, z: int, y: int, x: int) -> bytes:
    return struct.pack("<4sHHIII", MAGIC, codec, qp, z, y, x)


def read_header(data: bytes) -> dict:
    if len(data) < HEADER_SIZE:
        raise ValueError("data too small for VC3D header")
    magic, codec, qp, z, y, x = struct.unpack("<4sHHIII", data[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(f"invalid magic: {magic!r}")
    return {"codec": codec, "qp": qp, "depth": z, "height": y, "width": x}


def encode_h264(volume: np.ndarray, qp: int = 26) -> bytes:
    """Encode a (Z, Y, X) uint8 volume as VC3D H264.

    Each Z slice becomes one grayscale frame. ffmpeg encodes as YUV420p H264
    with constant QP for deterministic quality.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected 3D array, got {volume.ndim}D")
    volume = volume.astype(np.uint8, copy=False)
    z, y, x = volume.shape

    # ffmpeg needs even dimensions for YUV420p
    pad_y = (y + 1) & ~1
    pad_x = (x + 1) & ~1
    if pad_y != y or pad_x != x:
        padded = np.zeros((z, pad_y, pad_x), dtype=np.uint8)
        padded[:, :y, :x] = volume
        volume = padded

    raw_bytes = volume.tobytes()

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{pad_x}x{pad_y}",
        "-r", "1",
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-qp", str(qp),
        "-f", "h264",
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd, input=raw_bytes, capture_output=True, timeout=300
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode()}")

    header = write_header(CODEC_H264, qp, z, y, x)
    return header + proc.stdout


def decode_h264(data: bytes) -> np.ndarray:
    """Decode VC3D H264 data back to a (Z, Y, X) uint8 volume."""
    hdr = read_header(data)
    z, y, x = hdr["depth"], hdr["height"], hdr["width"]
    bitstream = data[HEADER_SIZE:]

    pad_y = (y + 1) & ~1
    pad_x = (x + 1) & ~1

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "h264",
        "-i", "pipe:0",
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd, input=bitstream, capture_output=True, timeout=300
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode()}")

    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    expected = z * pad_y * pad_x
    if raw.size < expected:
        raise RuntimeError(
            f"decoded size {raw.size} < expected {expected} "
            f"({z}x{pad_y}x{pad_x})"
        )
    volume = raw[:expected].reshape(z, pad_y, pad_x)
    return volume[:, :y, :x].copy()
