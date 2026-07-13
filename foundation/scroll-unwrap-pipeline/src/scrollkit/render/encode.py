"""M4 encode path: raw-RGB24 piping into ffmpeg, 1080p derivatives, rawcache,
and ffprobe self-checks.

Bound by the docs/RENDER-STYLE.md encode block (system ffmpeg 6.1):
  - masters 4K: libx264 -preset slow -crf 19 -pix_fmt yuv420p -profile high
    -level 5.1, bt709 tags, +faststart, 30 fps; frames arrive on stdin as
    rawvideo rgb24 (no intermediate PNGs);
  - 1080p derivatives are DOWNSCALED from the 4K master with lanczos
    (-crf 18 -level 4.2) — never re-rendered;
  - the RGB->YUV conversion is forced through bt709/tv-range in swscale so the
    stream tags match the actual matrix (ffmpeg would otherwise default to 601).

The rawcache stores the exact frames piped for the PLAIN variant (post-vignette)
so the reveal variant can re-stream them without touching the GPU; it is a flat
rgb24 file + sidecar JSON, deleted once a mesh's variants are done.
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import numpy as np

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

#: bt709 stream tags (masters and derivatives)
COLOR_TAGS = [
    "-colorspace", "bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
]

#: rgb24 -> yuv420p inside swscale with the bt709 matrix + tv range, so the
#: bt709 tags above describe what was actually encoded
RGB_TO_709_VF = "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p"

__all__ = [
    "FFmpegWriter",
    "RawCache",
    "check_faststart",
    "ffprobe_check",
    "ffprobe_info",
    "h264_args",
    "make_derivative",
]


def h264_args(crf: int, level: str, preset: str = "slow") -> list[str]:
    return [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level:v", str(level),
        *COLOR_TAGS,
        "-movflags", "+faststart",
        "-r", "30",
    ]


class FFmpegWriter:
    """Pipe (H, W, 3) uint8 frames straight into one ffmpeg encode process."""

    def __init__(self, path, width: int, height: int, *, fps: int = 30, crf: int = 19,
                 level: str = "5.1", preset: str = "slow", log_path=None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = int(width), int(height)
        self.frames_written = 0
        self.log_path = Path(log_path) if log_path else self.path.with_suffix(".ffmpeg.log")
        cmd = [
            FFMPEG, "-y", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}", "-r", str(fps), "-i", "-",
            "-an", "-vf", RGB_TO_709_VF,
            *h264_args(crf, level, preset),
            str(self.path),
        ]
        self._log = open(self.log_path, "wb")
        self._log.write((" ".join(cmd) + "\n").encode())
        self._log.flush()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=self._log, stderr=self._log)

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"frame must be uint8 ({self.height},{self.width},3), got {frame.dtype} {frame.shape}"
            )
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames_written += 1

    def close(self) -> int:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        rc = self.proc.wait()
        self._log.close()
        if rc != 0:
            tail = Path(self.log_path).read_text(errors="replace")[-2000:]
            raise RuntimeError(f"ffmpeg failed (rc={rc}) for {self.path}:\n{tail}")
        return rc

    def __enter__(self) -> "FFmpegWriter":
        return self

    def __exit__(self, exc_type, *exc) -> None:
        if exc_type is None:
            self.close()
        else:  # error path: kill the encoder, keep the original exception
            self.proc.kill()
            self.proc.wait()
            self._log.close()


def make_derivative(master, out, width: int, height: int, *, crf: int = 18,
                    level: str = "4.2", preset: str = "slow") -> Path:
    """1080p derivative DOWNSCALED from the encoded master (lanczos) — no re-render."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".ffmpeg.log")
    cmd = [
        FFMPEG, "-y", "-loglevel", "warning",
        "-i", str(master), "-an",
        "-vf", f"scale={int(width)}:{int(height)}:flags=lanczos",
        *h264_args(crf, level, preset),
        str(out),
    ]
    with open(log, "wb") as lf:
        lf.write((" ".join(cmd) + "\n").encode())
        lf.flush()
        rc = subprocess.run(cmd, stdout=lf, stderr=lf).returncode
    if rc != 0:
        raise RuntimeError(f"ffmpeg derivative failed (rc={rc}) for {out}: see {log}")
    return out


# --------------------------------------------------------------------------------------
# ffprobe self-checks
# --------------------------------------------------------------------------------------
def check_faststart(path) -> bool:
    """True iff the moov atom precedes mdat (web-ready / +faststart applied)."""
    order = []
    with open(path, "rb") as fh:
        offset = 0
        fh.seek(0, 2)
        end = fh.tell()
        fh.seek(0)
        while offset < end:
            header = fh.read(8)
            if len(header) < 8:
                break
            size, btype = struct.unpack(">I4s", header)
            if size == 1:  # 64-bit box
                size = struct.unpack(">Q", fh.read(8))[0]
            elif size == 0:  # to EOF
                size = end - offset
            order.append(btype.decode("latin1"))
            offset += size
            fh.seek(offset)
    if "moov" not in order or "mdat" not in order:
        return False
    return order.index("moov") < order.index("mdat")


def ffprobe_info(path) -> dict:
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries",
        "stream=codec_name,profile,level,width,height,pix_fmt,r_frame_rate,"
        "avg_frame_rate,nb_frames,nb_read_packets,color_space,color_transfer,"
        "color_primaries:format=duration,size",
        "-of", "json", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {res.stderr[-500:]}")
    info = json.loads(res.stdout)
    st = info["streams"][0]
    st["faststart"] = check_faststart(path)
    st["file_size"] = int(info.get("format", {}).get("size", 0))
    st["duration"] = float(info.get("format", {}).get("duration", 0.0))
    return st


def ffprobe_check(path, *, width: int, height: int, frames: int, fps: str = "30/1",
                  pix_fmt: str = "yuv420p", require_faststart: bool = True) -> dict:
    """Validate an encode; raise RuntimeError listing every violation (fail loudly)."""
    st = ffprobe_info(path)
    problems = []
    if int(st["width"]) != int(width) or int(st["height"]) != int(height):
        problems.append(f"resolution {st['width']}x{st['height']} != {width}x{height}")
    if st["r_frame_rate"] != fps:
        problems.append(f"r_frame_rate {st['r_frame_rate']} != {fps}")
    if st.get("avg_frame_rate") not in (fps, "0/0"):
        problems.append(f"avg_frame_rate {st.get('avg_frame_rate')} != {fps}")
    n = int(st.get("nb_frames") or st.get("nb_read_packets") or -1)
    npkt = int(st.get("nb_read_packets") or -1)
    if n != int(frames) or npkt != int(frames):
        problems.append(f"frame count nb_frames={n} packets={npkt} != {frames}")
    if st["pix_fmt"] != pix_fmt:
        problems.append(f"pix_fmt {st['pix_fmt']} != {pix_fmt}")
    if st.get("color_space") != "bt709":
        problems.append(f"color_space {st.get('color_space')} != bt709")
    if require_faststart and not st["faststart"]:
        problems.append("moov after mdat (faststart missing)")
    if problems:
        raise RuntimeError(f"ffprobe check FAILED for {path}: " + "; ".join(problems))
    return st


# --------------------------------------------------------------------------------------
# raw frame cache (plain variant -> reveal variant, no re-render)
# --------------------------------------------------------------------------------------
class RawCache:
    """Flat rgb24 frame cache + sidecar JSON meta. ~6 GB at 4K/240 frames;
    deleted (delete()) once the mesh's variants are done."""

    def __init__(self, path, width: int, height: int, mode: str = "w") -> None:
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(".json")
        self.width, self.height = int(width), int(height)
        self.frame_bytes = self.width * self.height * 3
        self.n_frames = 0
        self._fh = None
        if mode == "w":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "wb")
        elif mode != "r":
            raise ValueError("mode must be 'w' or 'r'")

    @classmethod
    def open(cls, path) -> "RawCache":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        rc = cls(path, meta["width"], meta["height"], mode="r")
        rc.n_frames = int(meta["n_frames"])
        expected = rc.n_frames * rc.frame_bytes
        actual = path.stat().st_size
        if actual != expected:
            raise RuntimeError(f"rawcache {path}: size {actual} != meta {expected}")
        return rc

    def append(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError(f"rawcache frame mismatch: {frame.dtype} {frame.shape}")
        self._fh.write(np.ascontiguousarray(frame).tobytes())
        self.n_frames += 1

    def finalize(self, extra_meta: dict | None = None) -> None:
        self._fh.close()
        self._fh = None
        meta = {"width": self.width, "height": self.height, "n_frames": self.n_frames,
                "pix_fmt": "rgb24", **(extra_meta or {})}
        self.meta_path.write_text(json.dumps(meta, indent=1))

    def read(self, idx: int) -> np.ndarray:
        if not 0 <= idx < self.n_frames:
            raise IndexError(idx)
        with open(self.path, "rb") as fh:
            fh.seek(idx * self.frame_bytes)
            buf = fh.read(self.frame_bytes)
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)

    def iter_frames(self):
        with open(self.path, "rb") as fh:
            for _ in range(self.n_frames):
                buf = fh.read(self.frame_bytes)
                yield np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)

    def delete(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self.path.unlink(missing_ok=True)
        self.meta_path.unlink(missing_ok=True)
