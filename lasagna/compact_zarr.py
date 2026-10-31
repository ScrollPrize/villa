"""Schema-v3 packed prediction storage, shared by full and demand inference."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from vc_reader.reader import ZarrWriter, create_group, open_zarr

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


def _create_array(group, name, shape, chunks, dtype, codec):
    """``codec``: the compressor metadata, or None."""
    if name in group:
        try:
            array = open_zarr(Path(group.path) / name)
        except ValueError as error:  # such as Fortran order
            raise ValueError(f"existing packed array has incompatible encoding: {group.path}/{name}: {error}") from error
        if tuple(array.shape) != tuple(shape) or tuple(array.chunks) != tuple(chunks) or np.dtype(array.dtype) != np.dtype(dtype):
            raise ValueError(f"existing packed array has incompatible layout: {group.path}/{name}")
        if (array.filters or array.compressor != codec
                or array.meta.separator != "/" or array.fill_value != 0):
            raise ValueError(f"existing packed array has incompatible encoding: {group.path}/{name}")
        return
    from lasagna.tiled_predict3d import _writer_codec
    group.create_array(name, shape=shape, chunks=chunks, dtype=dtype, separator="/", **_writer_codec(codec))


@lru_cache(maxsize=512)
def _writer(array_path: str) -> ZarrWriter:
    # Array metadata is immutable for a validated inference/resume layout.
    return ZarrWriter.open(array_path)


def _chunk_index(array_path, origin) -> tuple[int, ...]:
    chunks = _writer(str(array_path)).chunks[1:]
    return (0, *(int(v) // int(c) for v, c in zip(origin, chunks)))


def atomic_chunk(array_path, origin, values):
    """Publish a whole CZYX chunk, including explicit all-fill completion."""
    writer = _writer(str(array_path))
    chunks, shape = writer.chunks, writer.shape
    expected = (shape[0], *(min(c, n-o) for c, n, o in zip(chunks[1:], shape[1:], origin)))
    if tuple(values.shape) != expected or any(o < 0 or o % c for o, c in zip(origin, chunks[1:])):
        raise ValueError("packed publication requires one complete aligned chunk")
    writer[(slice(None), *(slice(o, o+n) for o, n in zip(origin, expected[1:])))] = values


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
        # Packed arrays are always "/"-separated (see _create_array).
        return tuple(self.array_path(products[0]).joinpath(*map(str, _chunk_index(self.array_path(products[0]), origin)))
                     for products in self.by_level.values())

    def _chunk_stored(self, product, level, origin):
        path = self.array_path(product, level)
        return _writer(str(path)).chunk_exists(_chunk_index(path, origin))

    def product_chunk_complete(self, product, *, chunk_origin_zyx):
        return self._chunk_stored(product, None, chunk_origin_zyx)

    def channel_chunk_exists(self, product, channel_name, *, chunk_origin_zyx):
        return self.product_chunk_complete(product, chunk_origin_zyx=chunk_origin_zyx)

    def update_metadata(self, products):
        pass

    def prepare(self, *, base_shape_zyx, ome_chunk, ome_compressor="blosc-zstd",
                source_to_base=1., grad_mag_factor=1., crop=None, provenance="inference.json", extra_fields=None):
        from lasagna.tiled_predict3d import _omezarr_level_shape, omezarr_compressor_config
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
        create_group(root)
        manifest = previous or LasagnaVolume(path=root, version=3, source_to_base=source_to_base,
            base_shape_zyx=tuple(base_shape_zyx), validity_channel=self.validity_channel,
            grad_mag_factor=grad_mag_factor)
        manifest.provenance_json = provenance
        manifest.extra_fields.update(extra_fields or {})
        manifest.extra_fields["blank_invalid"] = self.blank_invalid
        codec = omezarr_compressor_config(ome_compressor)
        for n, products in self.by_level.items():
            channels = [c.name for p in products for c in p.channels]
            manifest.groups[f"sd{n}"] = ChannelGroup(f"sd{n}/{n}", n, channels)
            group = create_group(root / f"sd{n}")
            dtype = np.result_type(*(p.dtype for p in products))
            datasets = []
            for level in range(n, self.n_levels):
                shape = _omezarr_level_shape(base_shape_zyx, level)
                # Match the coordinator's grid; source_to_base transforms
                # coordinates in metadata, not the array dimensions.
                _create_array(group, str(level), (len(channels), *shape),
                              (len(channels), *(min(v, ome_chunk) for v in shape)), dtype, codec)
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
                    source = open_zarr(self.external_channels[name])
                    first = next(iter(values.values()))
                    values[name] = source[tuple(slice(o,o+n) for o,n in zip(chunk_origin_zyx, first.shape))]
            if set(values) != set(channels):
                raise ValueError(f"partial packed chunk publication: expected {channels}, got {list(values)}")
            if self.blank_invalid:
                values = blank_fiber_channels(values)
            array_path = self.array_path(products[0])
            # Invalidate derived outputs before replacing a finest chunk.
            origin = tuple(chunk_origin_zyx)
            for coarse in range(level + 1, self.n_levels):
                coarse_origin = tuple((o // (2**(coarse-level)) // products[0].chunk_size) * products[0].chunk_size for o in origin)
                coarse_path = self.array_path(products[0], coarse)
                _writer(str(coarse_path)).remove_chunk(_chunk_index(coarse_path, coarse_origin))
            atomic_chunk(array_path, origin, np.stack([values[c] for c in channels]))

    def validate_chunk(self, origin, read_arrays=None):
        arrays = {} if read_arrays is None else read_arrays
        try:
            for products in self.by_level.values():
                if not self._chunk_stored(products[0], None, origin):
                    return False
                path = str(self.array_path(products[0]))
                if path not in arrays:
                    arrays[path] = open_zarr(path, read_mode="slab")
                a = arrays[path]
                values = a[(slice(None), *(slice(o, min(o+c, n)) for o,c,n in zip(origin, a.chunks[1:], a.shape[1:])))]
                if not np.isfinite(values).all():
                    return False
            return True
        except (OSError, ValueError, RuntimeError):
            return False

    def build_pyramids(self, workers=1):
        from lasagna.omezarr_pyramid import _mean_pool2x_u8, _mean_pool2x_f32, _moment_pool2x_normals, stored_chunks
        from lasagna.tiled_predict3d import _product_channel_kind
        for native, products in self.by_level.items():
            names = [c.name for p in products for c in p.channels]
            channel_dtypes = {c.name: np.dtype(p.dtype) for p in products for c in p.channels}
            for level in range(native+1, self.n_levels):
                parent = self.array_path(products[0], level-1)
                source = open_zarr(parent, threads=1)
                dest = _writer(str(self.array_path(products[0], level)))
                # Discover only chunks with source data; never scan the full volume grid.
                origins = {tuple((i*c//2//d)*d for i,c,d in zip(index[1:], source.chunks[1:], dest.chunks[1:]))
                           for index in stored_chunks(parent)}
                def build(origin):
                    if self._chunk_stored(products[0], level, origin):
                        return
                    slices = tuple(slice(o*2, min((o+c)*2, n)) for o,c,n in zip(origin, dest.chunks[1:], source.shape[1:]))
                    raw = source[(slice(None), *slices)]
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
