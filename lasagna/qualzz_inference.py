"""AFV V2 inference using the shared Lasagna tile engine and plain Zarr v2."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import zarr

from lasagna.compact_zarr import _open_group, _create_array, atomic_chunk, chunk_file
from lasagna.cooperative_inference import cooperative_signals
from lasagna.tiled_predict3d import (
    OutputProductSpec, _omezarr_compressor, resolve_inference_devices,
    run_tiled_inference_3d,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _NormalizedModel(torch.nn.Module):
    def __init__(self, model, normalize):
        super().__init__()
        self.model, self.normalize = model, normalize

    def forward(self, tile):
        # The shared reader supplies uint8-scale CT divided by 255.
        # Delegate normalization to the checkpoint's own inference helper.
        return self.model(self.normalize((tile.float() * 255.0).round()))


class QualzzModelAdapter:
    """Use reviewed local HF model.py, not a duplicated architecture/normalizer."""
    native_output_stride = 1

    def __init__(self, model_dir, chunk_size=128, *, model_hashes=None):
        self.model_dir = str(Path(model_dir).resolve())
        self.model_hashes = model_hashes
        self.output_products = (OutputProductSpec(
            name="semantic", level=0, scaledown=1, inference_scaledown=1,
            channels=("vertical", "horizontal"), chunk_size=chunk_size,
        ),)

    def load_model(self, *, device):
        for name, expected in (self.model_hashes or {}).items():
            if _sha256(Path(self.model_dir) / name) != expected:
                raise ValueError(f"model file changed during inference: {name}")
        path = Path(self.model_dir) / "model.py"
        spec = importlib.util.spec_from_file_location("qualzz_afv_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return _NormalizedModel(module.load_model(self.model_dir, device=device),
                                module.normalize_ct).eval()

    def run_tile_inference(self, model, tile, *, device):
        # FP32, matching the published helper; no implicit precision change.
        return model(tile)

    def product_tensors_from_output(self, output):
        logits = output["seg"]
        if logits.ndim != 5 or logits.shape[1] != 4:
            raise ValueError("AFV V2 requires B,4,Z,Y,X segmentation logits (BG,V,H,I)")
        return {"semantic": logits.float().softmax(dim=1)[:, 1:3]}

    def finalize_product_slab(self, product, raw_slab):
        # Keep averaged probabilities floating point until BOTH output scales
        # have been formed. Quantizing here would bias the tracing downsample.
        return dict(zip(product.channel_names, raw_slab))


def _quantize(probabilities):
    if not np.isfinite(probabilities).all():
        raise ValueError("non-finite AFV probabilities")
    return np.rint(np.clip(probabilities, 0, 1) * 255).astype(np.uint8)


def _pool2(probabilities):
    """Match the generator's ordered eight additions, dropping odd end planes."""
    shape = tuple(n // 2 for n in probabilities.shape[1:])
    pooled = np.zeros((2, *shape), np.float32)
    for z, y, x in itertools.product((0, 1), repeat=3):
        pooled += probabilities[:, z:shape[0]*2:2, y:shape[1]*2:2, x:shape[2]*2:2]
    return pooled / 8


class ProbabilityZarrOutputAdapter:
    """Atomic paired-scale publication without any Lasagna schema metadata."""
    def __init__(self, root, product, shape, *, voxel_size_um, provenance):
        self.root = str(root)
        self.product = product
        self.shape = tuple(shape)
        self.voxel_size_um = float(voxel_size_um)
        self.provenance = provenance
        if product.chunk_size < 2 or product.chunk_size % 2 or min(shape) < 2:
            raise ValueError("paired output needs even chunk size >=2 and spatial shape >=2")

    def prepare(self):
        root = Path(self.root)
        metadata = dict(channel_order="VH", axes=["c", "z", "y", "x"],
                        voxel_size_um=self.voxel_size_um, inference=self.provenance)
        if root.exists() and any(root.iterdir()):
            attrs = root / ".zattrs"
            if not attrs.exists() or json.loads(attrs.read_text()) != metadata:
                raise ValueError("output provenance differs; use a new output directory")
        group = _open_group(root)
        codec = _omezarr_compressor("blosc-zstd")
        for name, factor in (("semantic", 1), ("tracing", 2)):
            shape = tuple(n // factor for n in self.shape)
            chunk = self.product.chunk_size // factor
            _create_array(group, name, (2, *shape), (2, chunk, chunk, chunk),
                          np.uint8, codec, array_path=root / name)
        group.attrs.update(metadata)

    def product_chunk_complete(self, product, *, chunk_origin_zyx):
        origin = tuple(chunk_origin_zyx)
        native = chunk_file(Path(self.root) / "semantic", origin).is_file()
        if any(o // 2 >= n // 2 for o, n in zip(origin, self.shape)):
            return native  # Last unpaired plane has no tracing output.
        return native and chunk_file(Path(self.root) / "tracing", tuple(o//2 for o in origin)).is_file()

    def write_product_chunk(self, product, *, chunk_origin_zyx, data):
        values = np.stack([data[name] for name in product.channel_names]).astype(np.float32)
        coarse = _pool2(values)
        # Native chunk is published last. An interruption between writes leaves
        # the pair incomplete and the common resume scheduler recomputes it.
        if all(n > 0 for n in coarse.shape[1:]):
            atomic_chunk(Path(self.root) / "tracing", tuple(o//2 for o in chunk_origin_zyx),
                         _quantize(coarse))
        atomic_chunk(Path(self.root) / "semantic", chunk_origin_zyx, _quantize(values))

    def update_metadata(self, products):
        pass


@cooperative_signals
def run_inference(*, model_dir, input_path, output, voxel_size_um, devices,
                  tile_size=128, overlap=32, border=16, chunk_size=128,
                  flush_workers=8, accumulator_workers=8):
    if not np.isfinite(voxel_size_um) or not 8.64 <= voxel_size_um <= 9.6:
        raise ValueError("AFV V2 expects input voxels between 8.64 and 9.6 micrometres")
    if tile_size < 128 or tile_size % 64 or not 0 <= 2*border <= overlap < tile_size:
        raise ValueError("tile size must be >=128 and divisible by64; require 0<=2*border<=overlap<tile")
    if flush_workers < 0 or accumulator_workers < 0:
        raise ValueError("worker counts must be nonnegative")
    model_dir, input_path = Path(model_dir).resolve(), Path(input_path).resolve()
    hashes = {name: _sha256(model_dir / name)
              for name in ("model.py", "plans.json", "checkpoint_final.pth")}
    source = zarr.open_array(str(input_path), mode="r")
    if source.ndim != 3 or np.dtype(source.dtype) != np.uint8:
        raise ValueError("AFV inference expects a 3D uint8 CT Zarr array")
    shape = tuple(int(n) for n in source.shape)
    adapter = QualzzModelAdapter(model_dir, chunk_size, model_hashes=hashes)
    product, = adapter.output_products
    output = Path(output).resolve()
    if output == input_path or input_path in output.parents or output in input_path.parents:
        raise ValueError("output and input stores must not contain one another")
    writer = ProbabilityZarrOutputAdapter(output, product, shape, voxel_size_um=voxel_size_um,
        provenance=dict(model_sha256=hashes, input=str(input_path), input_shape=list(shape),
                        tile_size=tile_size, overlap=overlap, border=border,
                        precision="fp32", accumulator_dtype="float32",
                        normalization="model.py:normalize_ct per tile", version=1))
    writer.prepare()
    if any(d.type == "cuda" for d in devices) and os.environ.get("LASAGNA_COOPERATIVE") != "1":
        raise ValueError("CUDA inference requires GPU-manager handling (LASAGNA_COOPERATIVE=1)")
    model = adapter.load_model(device=devices[0]) if devices[0].type == "cpu" else None
    nz, ny, nx = shape
    run_tiled_inference_3d(model, source, crop_slices=(0,nz,0,ny,0,nx), device=devices[0],
        model_adapter=adapter, output_adapter=writer, products=(product,),
        output_regions_zyx={product.name: (0,0,0,nz,ny,nx)},
        full_output_shapes_zyx={product.name: shape}, input_zarr_path=str(input_path),
        output_scaledown_base=1, tile_size=tile_size, overlap=overlap, border=border,
        tmp_dir=str(output.parent), temp_prefix=output.stem + "_", devices=devices,
        progress=dict(t0=time.time(), finalized_base_z=0, finalized_base_z_total=nz),
        flush_workers=flush_workers, accumulator_workers=accumulator_workers,
        product_accumulator_dtype="float32", input_reader="tensorstore")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Reviewed HF snapshot folder; executes its model.py")
    parser.add_argument("--input", required=True, help="Local uint8 CT Zarr array at model resolution")
    parser.add_argument("--output", required=True, help="Standard Zarr v2 output group")
    parser.add_argument("--voxel-size-um", type=float, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--device", default=None)
    selection.add_argument("--devices", default=None)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--border", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--flush-workers", type=int, default=8)
    parser.add_argument("--accumulator-workers", type=int, default=8)
    args = vars(parser.parse_args(argv))
    args["devices"] = resolve_inference_devices(device=args.pop("device"), devices=args["devices"])
    args["input_path"] = args.pop("input")
    os.environ["LASAGNA_COOPERATIVE"] = "1"
    run_inference(**args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
