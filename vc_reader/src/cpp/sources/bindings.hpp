#pragma once
// Python bindings of the source reader; descriptors are bound in _catalog.
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/ndarray.h>

#include "vc_reader/aligned_reader.hpp"
#include "vc_reader/source_reader.hpp"

namespace vcr::sources::python {
namespace nb = nanobind;
using namespace nb::literals;

inline nb::object toPython(const nlohmann::json& value)
{
    if (value.is_null()) return nb::none();
    if (value.is_number_float()) return nb::float_(value.get<double>());
    if (value.is_number_unsigned()) return nb::cast(value.get<std::uint64_t>());
    if (value.is_number_integer()) return nb::cast(value.get<std::int64_t>());
    return nb::module_::import_("json").attr("loads")(value.dump());
}
template <class T> nb::tuple tupleOf(const std::vector<T>& values)
{
    nb::list out;
    for (const auto& value : values) out.append(value);
    return nb::tuple(out);
}
// A block as a NumPy array owning it.
inline nb::object toArray(const Block& block)
{
    auto* holder = new std::shared_ptr<std::uint8_t[]>(block.data);
    nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::shared_ptr<std::uint8_t[]>*>(p); });
    const std::size_t bytes[1] = {block.nbytes};
    nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<1>> buffer(holder->get(), 1, bytes, owner);
    return nb::module_::import_("numpy").attr("ndarray")(tupleOf(block.shape), block.dtype, nb::cast(buffer));
}
using Regions = std::vector<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>>;

// Options through a Python callable, whose owner is released with the GIL held.
inline OptionsFor optionsFrom(nb::object callable)
{
    std::shared_ptr<PyObject> holder(callable.release().ptr(), [](PyObject* p) {
        nb::gil_scoped_acquire acquire;
        Py_DECREF(p);
    });
    return [holder](const std::string& location) {
        nb::gil_scoped_acquire acquire;
        return nb::cast<zarr::OpenOptions>(nb::handle(holder.get())(location));
    };
}

inline nb::dict statsOf(const SourceReader::Stats& s)
{
    nb::dict out, cache;
    cache["nbytes"] = s.decodedCache.nbytes;
    cache["entries"] = s.decodedCache.entries;
    cache["hits"] = s.decodedCache.hits;
    cache["misses"] = s.decodedCache.misses;
    cache["peak_working_bytes"] = s.decodedCache.peakWorkingBytes;
    out["decoded_cache"] = cache;
    out["remote_bytes"] = s.remoteBytes;
    out["encoded_cache_hits"] = s.encodedCacheHits;
    out["image_payload_bytes"] = s.imagePayloadBytes;
    if (!s.zarr.is_null()) {
        out["zarr"] = toPython(s.zarr);
        out["zarr_cache_bytes"] = s.zarrCacheBytes;
        out["zarr_chunk_bytes"] = s.zarrChunkBytes;
    }
    return out;
}

// ---- aligned reads ----
inline nb::object sampled(const SampledArray& array)
{
#ifdef VC_READER_CUDA
    if (array.onDevice()) {
        const auto code = array.dtype[1] == 'f' ? nb::dlpack::dtype_code::Float : array.dtype[1] == 'i' ? nb::dlpack::dtype_code::Int : array.dtype[1] == 'b' ? nb::dlpack::dtype_code::Bool : nb::dlpack::dtype_code::UInt;
        nb::object owned = reader_cuda::owned(std::static_pointer_cast<device::Buffer>(array.device), std::size_t(array.count()), {std::uint8_t(code), std::uint8_t((array.dtype[2] - '0') * 8), 1}, array.deviceId);
        return nb::module_::import_("vc_reader.cuda").attr("DeviceArray")(owned, tupleOf(array.shape), array.dtype, array.deviceId);
    }
#endif
    Block block;
    block.shape = array.shape;
    block.dtype = array.dtype;
    block.data = array.host;
    block.nbytes = std::size_t(array.count()) * std::size_t(array.dtype[2] - '0');
    return toArray(block);
}
// Metadata as Python values, sequences as tuples.
inline nb::object metadataValue(const nlohmann::json& value)
{
    if (value.is_array()) {
        nb::list out;
        for (const auto& item : value) out.append(metadataValue(item));
        return nb::tuple(out);
    }
    if (value.is_object()) {
        nb::dict out;
        for (const auto& item : value.items()) out[item.key().c_str()] = metadataValue(item.value());
        return out;
    }
    return toPython(value);
}
inline nb::object sampleResult(const SampleResult& result)
{
    nb::module_ sources = nb::module_::import_("vc_reader.sources");
    nb::dict targets;
    for (const auto& [name, target] : result.targets) {
        nb::dict masks;
        for (const auto& [key, mask] : target.masks) masks[key.c_str()] = sampled(mask);
        targets[name.c_str()] = sources.attr("TargetResult")(sampled(target.data), sampled(target.valid), masks);
    }
    return sources.attr("SampleResult")(targets, metadataValue(result.metadata));
}
inline nb::object alignedStats(const AlignedStats& s)
{
    nb::dict out, sources, masks;
    out["tile_cache_bytes"] = s.tileCacheBytes;
    out["tile_cache_hits"] = s.tileCacheHits;
    out["plan_cache_bytes"] = s.planCacheBytes;
    out["plan_cache_hits"] = s.planCacheHits;
    out["tile_reads"] = s.tileReads;
    out["uploaded_bytes"] = s.uploadedBytes;
    out["peak_tile_work_bytes"] = s.peakTileWorkBytes;
    for (const auto& [name, stats] : s.sources) sources[name.c_str()] = statsOf(stats);
    out["sources"] = sources;
    if (!s.masks.empty()) {
        for (const auto& [name, stats] : s.masks) masks[name.c_str()] = statsOf(stats);
        out["masks"] = masks;
    }
    if (s.hasSurface) {
        nb::dict surface;
        surface["geometry_read_bytes"] = s.geometryReadBytes;
        surface["candidate_upload_bytes"] = s.candidateUploadBytes;
        surface["largest_geometry_tile"] = s.largestGeometryCells;
        surface["preparation_seconds"] = s.preparationSeconds;
        surface["candidate_index_seconds"] = s.candidateIndexSeconds;
        surface["raster_call_seconds"] = s.rasterCallSeconds;
        surface["output_allocation_seconds"] = s.outputAllocationSeconds;
        out["surface"] = surface;
    }
    return out;
}
inline AlignedOptions alignedOptions(const std::string& backend, std::int64_t device, std::size_t cache, std::size_t work, std::size_t plan, std::size_t surface, nb::object options, std::shared_ptr<DecodedCache> decoded)
{
    AlignedOptions out;
    out.backend = backend;
    out.device = device;
    out.samplingCacheBytes = cache;
    out.samplingWorkBytes = work;
    out.planCacheBytes = plan;
    out.surfaceOutputBytes = surface;
    out.reader.options = optionsFrom(std::move(options));
    out.reader.decodedCache = std::move(decoded);
    return out;
}
inline std::vector<std::pair<std::string, Source>> namedSources(const nb::dict& targets)
{
    std::vector<std::pair<std::string, Source>> out;
    for (auto [key, value] : targets) {
        if (!nb::isinstance<nb::str>(key)) throw std::invalid_argument("target names must be nonempty strings; image is reserved");
        const auto name = nb::cast<std::string>(key);
        if (!nb::isinstance<Source>(value)) throw DeclarationTypeError(name + ": target must be a Source");
        out.emplace_back(name, nb::cast<Source>(value));
    }
    return out;
}
template <class T> void bindEngineCommon(nb::class_<T, AlignedEngine>& cls)
{
    cls.def("stats", [](T& r) { return alignedStats(r.stats()); });
}

inline void bind(nb::module_& m)
{
    nb::module_::import_("vc_reader._catalog");
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const DecodeBudgetError& e) {
            PyErr_SetString(PyExc_MemoryError, e.what());
        } catch (const ImageFormatError& e) {
            PyErr_SetString(PyExc_OSError, e.what());
        }
    });
    m.def("array_placement", &arrayPlacement, "registration"_a.none(), "A complete array-to-volume placement; ValueError when it is not one.");
    m.def("pixel_affine", &pixelAffine, "rows"_a, "An explicit pixel-to-UV transform in XYZ affine arithmetic.");
    nb::class_<ResolvedSource>(m, "ResolvedSource")
        .def_ro("location", &ResolvedSource::location).def_ro("selection", &ResolvedSource::selection)
        .def_prop_ro("axes", [](const ResolvedSource& r) { return tupleOf(r.axes); })
        .def_prop_ro("shape", [](const ResolvedSource& r) { return tupleOf(r.shape); })
        .def_ro("dtype", &ResolvedSource::dtype).def_ro("registration", &ResolvedSource::registration)
        .def_ro("storage_from_array", &ResolvedSource::storageFromArray).def_ro("storage_units", &ResolvedSource::storageUnits);
    nb::class_<DecodedCache::Lease>(m, "DecodedCacheLease")
        .def("__enter__", [](nb::handle self) { return nb::borrow(self); })
        .def("__exit__", [](DecodedCache::Lease& lease, nb::args) { lease.release(); });
    nb::class_<DecodedCache>(m, "DecodedCache", "One LRU and decode budget shared by all readers referencing this object. "
                                                "Decode operations on an owner are serialized, including conversion scratch. Independent owners may run "
                                                "concurrently. Fork and pickle start empty caches with fresh locks. Returned crops never alias these cached arrays.")
        .def(nb::init<std::size_t, std::size_t>(), "max_bytes"_a = kDefaultDecodeBytes, "max_decode_bytes"_a = kDefaultDecodeBytes)
        .def_prop_ro("max_bytes", &DecodedCache::maxBytes).def_prop_ro("max_decode_bytes", &DecodedCache::maxDecodeBytes)
        .def("stats", [](DecodedCache& c) {
            const auto s = c.stats();
            nb::dict out;
            out["nbytes"] = s.nbytes;
            out["entries"] = s.entries;
            out["hits"] = s.hits;
            out["misses"] = s.misses;
            out["peak_working_bytes"] = s.peakWorkingBytes;
            return out;
        })
        .def("clear", &DecodedCache::clear)
        .def("working", &DecodedCache::working, "nbytes"_a, "Reserve scratch for a read that retains nothing, as a context manager.")
        .def("__getstate__", [](const DecodedCache& c) { return nb::make_tuple(c.maxBytes(), c.maxDecodeBytes()); })
        .def("__setstate__", [](DecodedCache& c, const std::pair<std::size_t, std::size_t>& state) { new (&c) DecodedCache(state.first, state.second); });
    nb::class_<SourceReader>(m, "SourceReader")
        .def("__init__", [](SourceReader* self, const Source& source, nb::object options, std::shared_ptr<DecodedCache> cache) {
            SourceReader::Settings settings;
            settings.options = optionsFrom(std::move(options));
            settings.decodedCache = std::move(cache);
            nb::gil_scoped_release release;
            new (self) SourceReader(source, std::move(settings));
        }, "source"_a, "options"_a, "decoded_cache"_a)
        .def_prop_ro("source", &SourceReader::source)
        .def_prop_ro("axes", [](SourceReader& r) { return tupleOf(r.axes()); })
        .def_prop_ro("shape", [](SourceReader& r) { return tupleOf(r.shape()); })
        .def_prop_ro("dtype", &SourceReader::dtype)
        .def_prop_ro("spatial_axes", [](SourceReader& r) { return tupleOf(r.spatialAxes()); })
        .def_prop_ro("fill_value", [](SourceReader& r) { return toPython(r.fillValue()); })
        .def_prop_ro("palette", [](SourceReader& r) -> nb::object {
            const auto& palette = r.palette();
            if (palette.is_null()) return nb::none();
            nb::dict out;
            out["mode"] = palette.at("mode").get<std::string>();
            out["values"] = nb::tuple(toPython(palette.at("values")));
            const auto& alpha = palette.at("transparency");
            if (alpha.is_array()) {
                std::string bytes;
                for (const auto& v : alpha) bytes += char(v.get<int>());
                out["transparency"] = nb::bytes(bytes.data(), bytes.size());
            } else {
                out["transparency"] = toPython(alpha);
            }
            return out;
        })
        .def_prop_ro("registration", &SourceReader::registration)
        .def_prop_ro("resolved", &SourceReader::resolved)
        .def_prop_ro("companion_names", [](SourceReader& r) { return tupleOf(r.companionNames()); })
        .def("companion", &SourceReader::companion, "name"_a)
        .def("read", [](SourceReader& r, const std::vector<std::int64_t>& start, const std::vector<std::int64_t>& size) {
            Block block;
            {
                nb::gil_scoped_release release;
                block = r.read(start, size);
            }
            return toArray(block);
        }, "start"_a, "size"_a, "One owned crop, padded in the encoded output domain.")
        .def("read_many", [](SourceReader& r, const Regions& regions) {
            std::vector<Block> blocks;
            {
                nb::gil_scoped_release release;
                blocks = r.readMany(regions);
            }
            nb::list out;
            for (const auto& block : blocks) out.append(toArray(block));
            return out;
        }, "regions"_a)
        .def_prop_ro("is_zarr", &SourceReader::isZarr)
        .def_prop_ro("raw_axes", [](SourceReader& r) { return tupleOf(r.rawAxes()); })
        .def_prop_ro("is_direct", [](SourceReader& r) { return r.directArray() != nullptr; })
        .def("zarr_array", &SourceReader::zarrArray)
        .def_prop_ro("zarr_datasets", &SourceReader::zarrDatasets)
        .def("stats", [](SourceReader& r) { return statsOf(r.stats()); })
        .def_prop_ro("temporary_directory", &SourceReader::temporaryDirectory)
        .def("loaded", &SourceReader::loaded, "plane"_a)
        .def("close", &SourceReader::close, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("closed", &SourceReader::closed);

    nb::class_<AlignedEngine>(m, "AlignedEngine")
        .def_prop_ro("target_names", [](const AlignedEngine& e) { return tupleOf(e.targetNames()); })
        .def("target_channels", &AlignedEngine::targetChannels, "name"_a)
        .def("close", &AlignedEngine::close, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("metadata", [](const AlignedEngine& e) { return metadataValue(e.metadata()); })
        .def_prop_ro("closed", &AlignedEngine::closed);
    using Pair = std::array<std::int64_t, 2>;
    using Triple3 = std::array<std::int64_t, 3>;
    const auto results = [](const std::vector<SampleResult>& values) {
        nb::list out;
        for (const auto& value : values) out.append(sampleResult(value));
        return out;
    };
    const auto pairOf = [](std::pair<SampledArray, SampledArray> value) { return nb::make_tuple(sampled(value.first), sampled(value.second)); };
    nb::class_<SurfaceProjection>(m, "SurfaceProjection")
        .def_prop_ro("origin", &SurfaceProjection::origin).def_prop_ro("shape", &SurfaceProjection::shape)
        .def("regions", &SurfaceProjection::regions, "tile"_a)
        .def("read_crop", [](SurfaceProjection& p, const Triple3& origin, const Triple3& shape) {
            std::optional<SampleResult> result;
            {
                nb::gil_scoped_release release;
                result = p.readCrop(origin, shape);
            }
            return sampleResult(*result);
        }, "origin"_a, "shape"_a);
    auto sample = nb::class_<SampleReader, AlignedEngine>(m, "SampleReader");
    bindEngineCommon(sample);
    sample.def("__init__", [](SampleReader* self, nb::handle reference, nb::dict targets, std::optional<catalog::Catalog> catalog, const std::string& backend, std::int64_t device,
                              std::size_t cache, std::size_t work, std::size_t plan, std::size_t surface, nb::object options, std::shared_ptr<DecodedCache> decoded) {
            auto settings = alignedOptions(backend, device, cache, work, plan, surface, std::move(options), std::move(decoded));
            auto named = namedSources(targets);
            if (nb::isinstance<catalog::VolumeLevel>(reference)) {
                const auto level = nb::cast<catalog::VolumeLevel>(reference);
                nb::gil_scoped_release release;
                new (self) SampleReader(level, std::move(named), std::move(settings));
            } else if (nb::isinstance<Source>(reference)) {
                auto source = nb::cast<Source>(reference);
                nb::gil_scoped_release release;
                new (self) SampleReader(std::move(source), std::move(named), std::move(catalog), std::move(settings));
            } else {
                throw std::invalid_argument("reference must be a VolumeLevel or registered volumetric image Source");
            }
        })
        .def_prop_ro("shape", &SampleReader::shape)
        .def_prop_ro("reference_map", &SampleReader::referenceMap)
        .def("has_target", &SampleReader::hasTarget, "name"_a)
        .def("read_many", [results](SampleReader& r, const std::vector<std::pair<Triple3, Triple3>>& regions) {
            std::vector<SampleResult> values;
            {
                nb::gil_scoped_release release;
                values = r.readMany(regions);
            }
            return results(values);
        }, "regions"_a)
        .def("sample_target", [pairOf](SampleReader& r, const std::string& name, const Triple3& start, const Triple3& size) {
            std::optional<std::pair<SampledArray, SampledArray>> value;
            {
                nb::gil_scoped_release release;
                value = r.sampleTarget(name, start, size);
            }
            return pairOf(std::move(*value));
        }, "name"_a, "start"_a, "size"_a)
        .def("render_surface", [](SampleReader& r, const Surface& surface, const Pair& start, const Pair& size, const std::vector<double>& offsets) {
            std::optional<SampleResult> value;
            {
                nb::gil_scoped_release release;
                value = r.renderSurface(surface, start, size, offsets);
            }
            return sampleResult(*value);
        }, "surface"_a, "start"_a, "size"_a, "offsets_um"_a)
        .def("projection", [](SampleReader& r, const Surface& surface, const Pair& start, const Pair& size, const std::string& label, double thickness, double padding) {
            nb::gil_scoped_release release;
            return r.projection(surface, start, size, label, thickness, padding);
        }, "surface"_a, "start"_a, "size"_a, "label_target"_a, "thickness_um"_a, "padding_um"_a)
        .def("surface_named", &SampleReader::surfaceNamed, "id"_a);
    auto paired = nb::class_<PairedReader, AlignedEngine>(m, "PairedReader");
    bindEngineCommon(paired);
    paired.def("__init__", [](PairedReader* self, const Source& label, const Source& image, const catalog::Pairing& pairing, std::int64_t level, nb::object layers, const std::string& backend,
                              std::int64_t device, std::size_t cache, std::size_t work, std::size_t plan, nb::object options, std::shared_ptr<DecodedCache> decoded) {
            auto settings = alignedOptions(backend, device, cache, work, plan, std::size_t(512) << 20, std::move(options), std::move(decoded));
            std::shared_ptr<PyObject> choice(layers.release().ptr(), [](PyObject* p) { nb::gil_scoped_acquire acquire; Py_DECREF(p); });
            const auto chooser = [choice](std::int64_t depth) {
                nb::gil_scoped_acquire acquire;
                const auto [start, count, drop] = nb::cast<std::tuple<std::int64_t, std::int64_t, bool>>(nb::handle(choice.get())(depth));
                PairedReader::Layers out;
                out.start = start;
                out.count = count;
                out.drop = drop;
                return out;
            };
            nb::gil_scoped_release release;
            new (self) PairedReader(label, image, pairing, level, chooser, std::move(settings));
        })
        .def_prop_ro("shape", &PairedReader::shape).def_prop_ro("layer_start", &PairedReader::layerStart).def_prop_ro("layer_count", &PairedReader::layerCount)
        .def_prop_ro("drop_layer", &PairedReader::dropLayer)
        .def("has_target", &PairedReader::hasTarget, "name"_a)
        .def("read_many", [results](PairedReader& r, const std::vector<std::pair<Pair, Pair>>& regions) {
            std::vector<SampleResult> values;
            {
                nb::gil_scoped_release release;
                values = r.readMany(regions);
            }
            return results(values);
        }, "regions"_a)
        .def("sample_target", [pairOf](PairedReader& r, const std::string& name, const Triple3& start, const Triple3& size) {
            std::optional<std::pair<SampledArray, SampledArray>> value;
            {
                nb::gil_scoped_release release;
                value = r.sampleTarget(name, start, size);
            }
            return pairOf(std::move(*value));
        }, "name"_a, "start"_a, "size"_a);
    auto geometry = nb::class_<SurfaceGeometry, AlignedEngine>(m, "SurfaceGeometry");
    bindEngineCommon(geometry);
    geometry.def("__init__", [](SurfaceGeometry* self, const Surface& surface, std::size_t cache, std::size_t work, nb::object options, std::shared_ptr<DecodedCache> decoded) {
            auto settings = alignedOptions("cpu", 0, cache, work, 0, std::size_t(512) << 20, std::move(options), std::move(decoded));
            nb::gil_scoped_release release;
            new (self) SurfaceGeometry(surface, std::move(settings));
        })
        .def_prop_ro("shape", &SurfaceGeometry::shape)
        .def("read_points", [](SurfaceGeometry& g, const Pair& start, const Pair& size, const std::vector<double>& offsets) {
            std::optional<std::pair<Block, Block>> value;
            {
                nb::gil_scoped_release release;
                value = g.readPoints(start, size, offsets);
            }
            return nb::make_tuple(toArray(value->first), toArray(value->second));
        }, "start"_a, "size"_a, "offsets_um"_a);
}
} // namespace vcr::sources::python
