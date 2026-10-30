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
#include <nanobind/ndarray.h>

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
    nb::class_<EncodedFiles>(m, "EncodedFiles", "Seekable image files through the reader's transport and disk budget.")
        .def("__init__", [](EncodedFiles* self, nb::object options) { new (self) EncodedFiles(optionsFrom(std::move(options))); }, "options"_a)
        .def("read", [](EncodedFiles& files, const std::string& location) {
            std::string bytes;
            {
                nb::gil_scoped_release release;
                const auto opened = files.open(location);
                char block[1 << 16];
                ssize_t n;
                off_t at = 0;
                while ((n = ::pread(opened.fd, block, sizeof block, at)) > 0) {
                    bytes.append(block, std::size_t(n));
                    at += n;
                }
            }
            return nb::bytes(bytes.data(), bytes.size());
        }, "location"_a, "The whole file; FileNotFoundError or CacheMissError when absent.")
        .def_prop_ro("bytes_read", &EncodedFiles::bytesRead).def_prop_ro("cache_hits", &EncodedFiles::cacheHits)
        .def("close", &EncodedFiles::close);
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
        .def("stats", [](SourceReader& r) {
            const auto s = r.stats();
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
            out["zarr_cache_bytes"] = s.zarrCacheBytes;
            out["zarr_chunk_bytes"] = s.zarrChunkBytes;
            return out;
        })
        .def_prop_ro("temporary_directory", &SourceReader::temporaryDirectory)
        .def("loaded", &SourceReader::loaded, "plane"_a)
        .def("close", &SourceReader::close, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("closed", &SourceReader::closed);
}
} // namespace vcr::sources::python
