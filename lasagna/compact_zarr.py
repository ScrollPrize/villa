"""Schema-v3 packed prediction storage, shared by full and demand inference."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import threading
import uuid

import numcodecs
import numpy as np
import zarr

from lasagna.lasagna_volume import ChannelGroup, LasagnaVolume


def resolve_output(path, output_format="compact"):
    """Existing sidecars retain their layout; new outputs default to one store."""
    path = Path(path)
    if output_format not in ("compact", "legacy"):
        raise ValueError("output_format must be compact or legacy")
    if path.is_file() and path.name.endswith(".lasagna.json"):
        return path, "legacy"
    if path.name.endswith(".lasagna.json") and output_format == "compact":
        path = path.with_name(path.name.removesuffix(".lasagna.json") + ".zarr")
    if output_format == "legacy" and not path.name.endswith(".lasagna.json"):
        raise ValueError("legacy output requires a .lasagna.json path")
    if output_format == "compact" and path.exists() and not (path / ".zattrs").exists():
        raise ValueError(f"unidentified compact output: {path}")
    return path, output_format


def _open_group(path):
    try:
        return zarr.open_group(str(path), mode="a", zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode="a")


def blank_fiber_channels(values):
    """Use stored/quantized presence, not unquantized model probabilities."""
    presences = {k: v for k, v in values.items() if k == "presence" or k.endswith("_presence")}
    if not presences:
        return values
    any_valid = np.maximum.reduce(tuple(presences.values())) != 0
    result = dict(values)
    for name in values:
        if name in presences:
            continue
        own = next((p for p in presences if p != "presence" and name.startswith(p.removesuffix("presence"))), None)
        result[name] = np.where(presences[own] != 0 if own else any_valid, values[name], 0)
    return result


def _create_array(group, name, shape, chunks, dtype, compressor, *, array_path):
    if name in group:
        array = group[name]
        if tuple(array.shape) != tuple(shape) or tuple(array.chunks) != tuple(chunks) or np.dtype(array.dtype) != np.dtype(dtype):
            raise ValueError(f"existing packed array has incompatible layout: {group.path}/{name}")
        metadata = json.loads((Path(array_path) / ".zarray").read_text())
        expected_codec = None if compressor is None else compressor.get_config()
        if (metadata.get("order", "C") != "C" or metadata.get("filters")
                or metadata.get("compressor") != expected_codec
                or metadata.get("dimension_separator", ".") != "/"
                or metadata.get("fill_value") != 0):
            raise ValueError(f"existing packed array has incompatible encoding: {group.path}/{name}")
        return
    try:
        group.create_array(name, shape=shape, chunks=chunks, dtype=dtype, fill_value=0,
                           compressor=compressor, chunk_key_encoding={"name": "v2", "separator": "/"})
    except (AttributeError, TypeError):
        group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype, fill_value=0,
                             compressor=compressor, dimension_separator="/")


@lru_cache(maxsize=512)
def _array_metadata(array_path: str):
    # Array metadata is immutable for a validated inference/resume layout.
    return json.loads((Path(array_path) / ".zarray").read_text())


def chunk_file(array_path, origin):
    metadata = _array_metadata(str(array_path))
    indices = (0, *(int(v) // int(c) for v, c in zip(origin, metadata["chunks"][1:])))
    return Path(array_path) / metadata.get("dimension_separator", ".").join(map(str, indices))


def atomic_chunk(array_path, origin, values):
    """Publish a whole CZYX chunk, including explicit all-fill completion."""
    array_path = Path(array_path)
    metadata = _array_metadata(str(array_path))
    chunks, shape = tuple(metadata["chunks"]), tuple(metadata["shape"])
    expected = (shape[0], *(min(c, n-o) for c, n, o in zip(chunks[1:], shape[1:], origin)))
    if tuple(values.shape) != expected or any(o < 0 or o % c for o, c in zip(origin, chunks[1:])):
        raise ValueError("packed publication requires one complete aligned chunk")
    if expected == chunks:
        block = np.ascontiguousarray(values, dtype=np.dtype(metadata["dtype"]))
    else:
        block = np.zeros(chunks, dtype=np.dtype(metadata["dtype"]))
        block[tuple(slice(0, n) for n in expected)] = values
    codec = metadata.get("compressor")
    payload = block.tobytes(order="C") if codec is None else numcodecs.get_codec(codec).encode(block)
    destination = chunk_file(array_path, origin)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".tmp.{destination.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as file:
            file.write(payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class CompactOutputAdapter:
    """One spatial publication owns every logical product in a scale group."""
    def __init__(self, *, root, products, n_levels, validity_channel, blank_invalid=False):
        self.root = str(root)
        self.products = tuple(products)
        self.n_levels = int(n_levels)
        self.validity_channel = validity_channel
        self.blank_invalid = bool(blank_invalid)
        self.external_channels = {}
        self.by_level = {}
        for product in products:
            self.by_level.setdefault(product.level, []).append(product)
        names = [c.name for p in products for c in p.channels]
        if len(set(names)) != len(names) or validity_channel not in names:
            raise ValueError("packed channels must be unique and include validity_channel")

    def product_by_name(self, name):
        return next(p for p in self.products if p.name == name)

    def array_path(self, product, level=None):
        return Path(self.root) / f"sd{product.level}" / str(product.level if level is None else level)

    def channel_path(self, channel):
        product = next(p for p in self.products if channel.name in p.channel_names)
        return str(Path(self.root) / f"sd{product.level}")

    def chunk_paths(self, origin):
        return tuple(chunk_file(self.array_path(products[0]), origin) for products in self.by_level.values())

    def product_chunk_complete(self, product, *, chunk_origin_zyx):
        return chunk_file(self.array_path(product), chunk_origin_zyx).is_file()

    def channel_chunk_exists(self, product, channel_name, *, chunk_origin_zyx):
        return self.product_chunk_complete(product, chunk_origin_zyx=chunk_origin_zyx)

    def update_metadata(self, products):
        pass

    def prepare(self, *, base_shape_zyx, ome_chunk, ome_compressor="blosc-zstd",
                source_to_base=1., grad_mag_factor=1., crop=None, provenance="inference.json", extra_fields=None):
        from lasagna.tiled_predict3d import _omezarr_compressor, _omezarr_level_shape
        root = Path(self.root)
        if (root / ".zattrs").exists():
            attrs = json.loads((root / ".zattrs").read_text())
            if "lasagna" not in attrs:
                raise ValueError("refusing to adopt unidentified packed output")
            previous = LasagnaVolume.load(root)
            expected = {f"sd{n}": ChannelGroup(f"sd{n}/{n}", n, [c.name for p in ps for c in p.channels])
                        for n, ps in self.by_level.items()}
            if (previous.groups != expected or previous.base_shape_zyx != tuple(base_shape_zyx)
                    or previous.source_to_base != source_to_base or previous.validity_channel != self.validity_channel):
                raise ValueError("existing packed manifest differs from requested output")
            if previous.extra_fields.get("blank_invalid", False) != self.blank_invalid:
                raise ValueError("cannot change blank_invalid during resume")
            if previous.extra_fields.get("width_encoding") != (extra_fields or {}).get("width_encoding"):
                raise ValueError("cannot change width_encoding during resume")
            if previous.grad_mag_factor != grad_mag_factor:
                raise ValueError("cannot change grad_mag_factor during resume")
        else:
            previous = None
        _open_group(root)
        manifest = previous or LasagnaVolume(path=root, version=3, source_to_base=source_to_base,
            base_shape_zyx=tuple(base_shape_zyx), validity_channel=self.validity_channel,
            grad_mag_factor=grad_mag_factor)
        manifest.provenance_json = provenance
        manifest.extra_fields.update(extra_fields or {})
        manifest.extra_fields["blank_invalid"] = self.blank_invalid
        codec = _omezarr_compressor(ome_compressor)
        for n, products in self.by_level.items():
            channels = [c.name for p in products for c in p.channels]
            manifest.groups[f"sd{n}"] = ChannelGroup(f"sd{n}/{n}", n, channels)
            group = _open_group(root / f"sd{n}")
            dtype = np.result_type(*(p.dtype for p in products))
            datasets = []
            for level in range(n, self.n_levels):
                shape = _omezarr_level_shape(base_shape_zyx, level)
                # Match the coordinator's grid; source_to_base transforms
                # coordinates in metadata, not the array dimensions.
                _create_array(group, str(level), (len(channels), *shape),
                              (len(channels), *(min(v, ome_chunk) for v in shape)), dtype, codec,
                              array_path=root / f"sd{n}" / str(level))
                datasets.append({"path": str(level), "coordinateTransformations": [
                    {"type": "scale", "scale": [1, *([source_to_base * 2**level] * 3)]}]})
            group.attrs["multiscales"] = [{"version": "0.4", "name": f"sd{n}",
                "axes": [{"name": "c", "type": "channel"}, *({"name": a, "type": "space"} for a in "zyx")],
                "datasets": datasets}]
        if crop is not None:
            manifest.add_crop(tuple(crop))
        manifest.save()

    def write_products_chunk(self, items, *, chunk_origin_zyx):
        by_level = {}
        for product, values in items:
            by_level.setdefault(product.level, {}).update(values)
        for level, values in by_level.items():
            products = self.by_level[level]
            channels = [c.name for p in products for c in p.channels]
            for name in channels:
                if name not in values and name in self.external_channels:
                    source = zarr.open_array(self.external_channels[name], mode="r")
                    first = next(iter(values.values()))
                    values[name] = np.asarray(source[tuple(slice(o,o+n) for o,n in zip(chunk_origin_zyx, first.shape))])
            if set(values) != set(channels):
                raise ValueError(f"partial packed chunk publication: expected {channels}, got {list(values)}")
            if self.blank_invalid:
                values = blank_fiber_channels(values)
            array_path = self.array_path(products[0])
            # Invalidate derived outputs before replacing a finest chunk.
            origin = tuple(chunk_origin_zyx)
            for coarse in range(level + 1, self.n_levels):
                coarse_origin = tuple((o // (2**(coarse-level)) // products[0].chunk_size) * products[0].chunk_size for o in origin)
                chunk_file(self.array_path(products[0], coarse), coarse_origin).unlink(missing_ok=True)
            atomic_chunk(array_path, origin, np.stack([values[c] for c in channels]))

    def validate_chunk(self, origin, read_arrays=None):
        arrays = {} if read_arrays is None else read_arrays
        try:
            for products in self.by_level.values():
                path = str(self.array_path(products[0]))
                if not chunk_file(path, origin).is_file():
                    return False
                if path not in arrays:
                    arrays[path] = zarr.open_array(path, mode="r")
                a = arrays[path]
                values = np.asarray(a[(slice(None), *(slice(o, min(o+c, n)) for o,c,n in zip(origin, a.chunks[1:], a.shape[1:])))])
                if not np.isfinite(values).all():
                    return False
            return True
        except (OSError, ValueError, RuntimeError):
            return False

    def build_pyramids(self, workers=1):
        from lasagna.omezarr_pyramid import _mean_pool2x_u8, _mean_pool2x_f32, _moment_pool2x_normals
        from lasagna.tiled_predict3d import _product_channel_kind
        for native, products in self.by_level.items():
            names = [c.name for p in products for c in p.channels]
            channel_dtypes = {c.name: np.dtype(p.dtype) for p in products for c in p.channels}
            for level in range(native+1, self.n_levels):
                source = zarr.open_array(str(self.array_path(products[0], level-1)), mode="r")
                dest = zarr.open_array(str(self.array_path(products[0], level)), mode="r")
                # Discover only chunks with source data; never scan the full volume grid.
                origins = set()
                parent = self.array_path(products[0], level-1)
                for file in parent.glob("0/*/*/*"):
                    if file.is_file() and all(part.isdigit() for part in file.relative_to(parent).parts):
                        indices = tuple(map(int, file.relative_to(parent).parts[1:]))
                        origins.add(tuple((i*c//2//d)*d for i,c,d in zip(indices, source.chunks[1:], dest.chunks[1:])))
                def build(origin):
                    if chunk_file(self.array_path(products[0], level), origin).is_file():
                        return
                    slices = tuple(slice(o*2, min((o+c)*2, n)) for o,c,n in zip(origin, dest.chunks[1:], source.shape[1:]))
                    raw = np.asarray(source[(slice(None), *slices)])
                    output = [_mean_pool2x_u8(raw[i], zero_overrides=(_product_channel_kind(name) == "grad_mag"))
                              if channel_dtypes[name] == np.dtype("uint8") else _mean_pool2x_f32(raw[i])
                              for i, name in enumerate(names)]
                    for i, name in enumerate(names):
                        if _product_channel_kind(name) == "nx":
                            sibling = name[:-2] + "ny"
                            if sibling in names:
                                j = names.index(sibling)
                                presence = name[:-2] + "presence"
                                mask = None
                                if self.blank_invalid:
                                    ps = [raw[k] for k,c in enumerate(names) if c == "presence" or c.endswith("_presence")]
                                    if presence in names:
                                        mask = raw[names.index(presence)]
                                    elif ps:
                                        mask = np.maximum.reduce(ps)
                                output[i], output[j] = _moment_pool2x_normals(raw[i], raw[j], validity=mask)
                    if self.blank_invalid:
                        # Width/embeddings are undefined at zero presence, not
                        # physical zero measurements to average into valid data.
                        ps = [raw[i] for i,name in enumerate(names) if name == "presence" or name.endswith("_presence")]
                        if ps:
                            valid = (np.maximum.reduce(ps) != 0).astype(np.float32)
                            denominator = _mean_pool2x_f32(valid)
                            for i,name in enumerate(names):
                                if name == "width" or name.startswith("embedding_"):
                                    mean = _mean_pool2x_f32(raw[i] * valid) / np.maximum(denominator, 1e-12)
                                    output[i] = (np.rint(mean).clip(0,255).astype(np.uint8)
                                                 if channel_dtypes[name] == np.dtype("uint8") else mean)
                    values = dict(zip(names, output))
                    if self.blank_invalid:
                        values = blank_fiber_channels(values)
                    atomic_chunk(self.array_path(products[0], level), origin, np.stack([values[k] for k in names]))
                with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
                    for _ in pool.map(build, sorted(origins)):
                        pass
