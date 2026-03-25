from __future__ import annotations

from pathlib import Path

import numpy as np


def _to_u8(img_float: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(img_float, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


def downsample_preview_for_media(
    img: np.ndarray,
    *,
    source_downsample: int,
    media_downsample: int,
) -> np.ndarray:
    """Match the legacy stitched-media downsample behavior exactly."""
    source_downsample = int(source_downsample)
    media_downsample = int(media_downsample)
    if source_downsample < 1:
        raise ValueError(f"source_downsample must be >= 1, got {source_downsample}")
    if media_downsample < 1:
        raise ValueError(f"media_downsample must be >= 1, got {media_downsample}")
    if source_downsample > 1 or media_downsample == 1:
        return np.ascontiguousarray(img)

    in_h, in_w = img.shape[:2]
    factor = int(media_downsample)
    if (in_h % factor) == 0 and (in_w % factor) == 0:
        out_h = in_h // factor
        out_w = in_w // factor
        if img.ndim == 2:
            if img.dtype == np.uint8:
                reduced = img.reshape(out_h, factor, out_w, factor).mean(axis=(1, 3), dtype=np.float32)
            else:
                reduced = img.reshape(out_h, factor, out_w, factor).mean(axis=(1, 3))
        elif img.ndim == 3:
            channels = int(img.shape[2])
            if img.dtype == np.uint8:
                reduced = img.reshape(out_h, factor, out_w, factor, channels).mean(
                    axis=(1, 3),
                    dtype=np.float32,
                )
            else:
                reduced = img.reshape(out_h, factor, out_w, factor, channels).mean(axis=(1, 3))
        else:
            raise ValueError(f"unsupported image ndim for stitched preview downsample: {img.ndim}")
        if img.dtype == np.uint8:
            reduced = np.clip(np.rint(reduced), 0.0, 255.0).astype(np.uint8, copy=False)
        else:
            reduced = reduced.astype(img.dtype, copy=False)
        return np.ascontiguousarray(reduced)

    out_h = max(1, (int(in_h) + factor - 1) // factor)
    out_w = max(1, (int(in_w) + factor - 1) // factor)
    if out_h == in_h and out_w == in_w:
        return np.ascontiguousarray(img)

    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stitched preview downsample fallback requires OpenCV (cv2)"
        ) from exc

    resized = cv2.resize(
        img,
        (out_w, out_h),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized)


def probs_to_preview_u8(
    probs: np.ndarray,
    *,
    source_downsample: int,
    media_downsample: int,
) -> np.ndarray:
    return downsample_preview_for_media(
        _to_u8(np.asarray(probs, dtype=np.float32)),
        source_downsample=int(source_downsample),
        media_downsample=int(media_downsample),
    )


def write_preview_png(*, out_path: str | Path, image_u8: np.ndarray) -> str:
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_u8 = np.ascontiguousarray(np.asarray(image_u8, dtype=np.uint8))
    if image_u8.ndim not in {2, 3}:
        raise ValueError(f"preview image must be 2D or 3D uint8 array, got shape {tuple(image_u8.shape)}")

    try:
        from PIL import Image  # type: ignore

        Image.fromarray(image_u8).save(out_path, format="PNG")
        return str(out_path)
    except ModuleNotFoundError:
        pass

    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "writing stitched preview PNGs requires Pillow or OpenCV (cv2)"
        ) from exc

    to_write = image_u8
    if image_u8.ndim == 3 and int(image_u8.shape[2]) == 3:
        to_write = image_u8[..., ::-1]
    if not bool(cv2.imwrite(str(out_path), to_write)):
        raise RuntimeError(f"failed to write preview PNG to {str(out_path)!r}")
    return str(out_path)
