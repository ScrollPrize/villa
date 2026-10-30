// Python bindings for the C++ read path (module vc_reader._reader).
#include "array.hpp"
#include "cuda_available.hpp"
#include "vc_reader/reader.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_metadata.hpp"
#include "output_pool.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>

#include <functional>
#include <numeric>

namespace nb = nanobind;
using namespace nb::literals;

using Bytes = nb::ndarray<std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Json = vcr::zarr::Json;
using HostPoints = nb::ndarray<nb::ro, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template<class T> static void sample_u8_points(vcr::Array& a, const HostPoints& points,
                                              const vcr::Index& offset, Bytes out, int order)
{
    vcr::RebasedCoordinates<vcr::PointCoordinates<T>> coords;
    coords.data = static_cast<const T*>(points.data()); coords.ndim = 3;
    std::copy_n(offset.begin(), 3, coords.offset.begin());
    nb::gil_scoped_release release;
    a.sample_u8_float32(coords, points.shape(0), order, out.data());
}

#ifdef VC_READER_CUDA
#include "reader_cuda/bindings.hpp"
#endif
#include "sources/bindings.hpp"

static nb::object py_json(const Json& value)
{
    return nb::module_::import_("json").attr("loads")(value.dump());
}

static Json to_json(const nb::object& value)
{
    return Json::parse(nb::cast<std::string>(nb::module_::import_("json").attr("dumps")(value)));
}

static std::string access_kind(const vcr::zarr::Access& access)
{
    using Kind = vcr::zarr::Access::Kind;
    return access.kind == Kind::S3 ? "s3" : access.kind == Kind::Basic ? "basic" : "anonymous";
}

static vcr::ElementType sample_type(const std::string& name, std::size_t itemsize, bool swap)
{
    const char kind = name.size() == 1 ? name[0] : '?';
    const bool ok = (kind == 'u' || kind == 'i') ? (itemsize == 1 || itemsize == 2 || itemsize == 4 || itemsize == 8)
                    : kind == 'f'             ? (itemsize == 4 || itemsize == 8)
                                              : (kind == 'b' && itemsize == 1);
    if (!ok)
        throw nb::type_error("sample() needs a bool, integer, float32 or float64 array");
    return {kind, itemsize, swap};
}



static std::vector<std::size_t> output_counts(const std::vector<vcr::Index>& shapes,
                                            const std::vector<Bytes>& outs, std::size_t itemsize)
{
    if (shapes.size() != outs.size()) throw nb::value_error("one output is required per region");
    std::vector<std::size_t> counts;
    for (std::size_t j = 0; j < shapes.size(); ++j) {
        std::size_t n = 1;
        for (auto dim : shapes[j]) {
            if (dim < 0 || (dim && n > std::numeric_limits<std::size_t>::max() / dim))
                throw nb::value_error("invalid or overflowing output shape");
            n *= dim;
        }
        if (n > std::numeric_limits<std::size_t>::max() / itemsize || outs[j].size() != n * itemsize)
            throw nb::value_error("an output buffer does not match its region");
        counts.push_back(n);
    }
    return counts;
}

#include "aligned/bindings.hpp"
#include "surface/bindings.hpp"
#include "sheet_normal/bindings.hpp"
#include "geometry/bindings.hpp"
#include "grid/bindings.hpp"

NB_MODULE(_reader, m)
{
    m.doc() = "Zarr reader: local and HTTP chunks, disk cache layers, memory cache.";
    m.def("_local_points", [](HostPoints points, std::optional<vcr::Index> origin) -> nb::object {
        if (points.shape(1) != 3) throw nb::value_error("expected points of shape (N, 3)");
        const auto local = [&](const auto* p) -> nb::object {
            auto owned = std::make_unique<vcr::LocalPoints>();
            {
                nb::gil_scoped_release release;
                *owned = vcr::localPoints(p, points.shape(0), origin);
            }
            if (owned->dispersed)
                return nb::none();
            const auto base = owned->base;
            float* data = owned->local.data();
            nb::capsule holder(owned.release(), [](void* q) noexcept { delete static_cast<vcr::LocalPoints*>(q); });
            return nb::make_tuple(nb::ndarray<nb::numpy, float, nb::ndim<2>>(data, {points.shape(0), std::size_t(3)}, holder),
                                  vcr::Index(base.begin(), base.end()));
        };
        if (points.dtype() == nb::dtype<float>()) return local(static_cast<const float*>(points.data()));
        if (points.dtype() == nb::dtype<double>()) return local(static_cast<const double*>(points.data()));
        throw nb::type_error("expected float32 or float64 points");
    }, "points"_a, "origin"_a.none() = nb::none());
    m.def("_check_crop_shape", [](const vcr::Index& shape) { return vcr::checkedCropShape(shape); }, "shape"_a);
    m.def("_check_coordinate_origin", [](const vcr::Index& origin) { return vcr::checkedCoordinateOrigin(origin); }, "origin"_a);
    m.def("_local_origin", [](nb::ndarray<nb::ro, nb::ndim<1>, nb::c_contig, nb::device::cpu> origin, std::optional<vcr::Index> coordinate_origin) {
        if (origin.size() != 3) throw nb::value_error("origin and axes must be finite with shapes (3,) and (3, 3)");
        const auto local = origin.dtype() == nb::dtype<float>()
            ? vcr::localOrigin(static_cast<const float*>(origin.data()), coordinate_origin)
            : origin.dtype() == nb::dtype<double>() ? vcr::localOrigin(static_cast<const double*>(origin.data()), coordinate_origin)
                                                     : throw nb::type_error("expected a float32 or float64 origin");
        return std::make_pair(local.base, local.origin);
    }, "origin"_a, "coordinate_origin"_a.none() = nb::none());
#ifdef VC_READER_CUDA
    vcr::reader_cuda::bind(m);
#else
    m.def("reader_cuda_available", &vcr::cuda::available);
#endif
    vcr::aligned::bind(m);
    vcr::surface::bind(m);
    vcr::sheet_normal::bind(m);
    vcr::geometry::bind(m);
    vcr::grid::python::bind(m);
    vcr::sources::python::bind(m);
    // Engines inherited by a forked process are deliberately kept alive and never
    // destroyed there (their threads do not exist in the child), so they are
    // still referenced at exit; that is not a leak worth reporting.
    nb::set_leak_warnings(false);

    // C++ exceptions with a Python meaning: offline cache misses and refused requests.
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const vcr::CacheMiss& e) {
            nb::object cls = nb::module_::import_("vc_reader.errors").attr("CacheMissError");
            PyErr_SetString(cls.ptr(), e.what());
        } catch (const vcr::HttpDenied& e) {
            PyErr_SetString(PyExc_PermissionError, e.what());
        } catch (const vcr::zarr::NotFound& e) {
            PyErr_SetString(PyExc_FileNotFoundError, e.what());
        } catch (const std::filesystem::filesystem_error& e) {
            errno = e.code().value();
            PyErr_SetFromErrnoWithFilename(PyExc_OSError, e.path1().c_str());
        }
    });

    m.def(
        "empty",
        [](std::size_t nbytes) {
            // A buffer from the output pool, as a 1-D uint8 numpy array; it goes back
            // to the pool when the last array viewing it is freed.
            const std::size_t n = nbytes ? nbytes : 1;
            void* p = vcr::OutputPool::instance().acquire(n);
            nb::capsule owner(p, [](void* q) noexcept { vcr::OutputPool::instance().release(q); });
            return nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<1>>(p, {nbytes}, owner);
        },
        "nbytes"_a, "An uninitialized uint8 array from the recycled output pool.");
    m.def("set_output_pool_bytes", [](std::size_t n) { vcr::OutputPool::instance().set_max_bytes(n); }, "max_bytes"_a,
          "How many bytes of freed output buffers to keep for reuse (0 disables).");
    m.def("output_pool_stats", [] {
        auto& pool = vcr::OutputPool::instance();
        nb::dict d;
        d["max_bytes"] = pool.max_bytes();
        d["retained_bytes"] = pool.retained();
        d["hits"] = pool.hits();
        d["misses"] = pool.misses();
        return d;
    });

    nb::class_<vcr::MemoryCache>(m, "MemoryCache", "Byte-bounded LRU of decoded chunks, shareable between arrays.")
        .def(nb::init<std::size_t>(), "max_bytes"_a)
        .def_prop_ro("max_bytes", &vcr::MemoryCache::max_bytes)
        .def_prop_ro("nbytes", &vcr::MemoryCache::nbytes)
        .def_prop_ro("hits", &vcr::MemoryCache::hits)
        .def_prop_ro("misses", &vcr::MemoryCache::misses)
        .def("__len__", &vcr::MemoryCache::size)
        .def("clear", &vcr::MemoryCache::clear);

    // Finding remote arrays and their disk caches (vc_reader/zarr.hpp).
    m.def("is_remote", &vcr::zarr::isRemote, "location"_a);
    m.def("http_url", &vcr::zarr::httpUrl, "location"_a);
    m.def("is_public_location", &vcr::zarr::isPublicLocation, "location"_a);
    m.def("load_basic_auth", [](const std::string& path) { return vcr::zarr::loadBasicAuth(path); }, "path"_a);
    m.def("cache_roots", [](const std::vector<std::string>& cache_dirs) {
        std::vector<std::string> out;
        for (const auto& d : vcr::zarr::cacheRoots(cache_dirs))
            out.push_back(d.string());
        return out;
    }, "cache_dirs"_a);
    m.def("decoded_cache_path", [](const std::string& cache_dir, const std::string& url) {
        return vcr::zarr::decodedCachePath(cache_dir, url).string();
    }, "cache_dir"_a, "url"_a);
    m.def("vc3d_volume_id", &vcr::zarr::vc3dVolumeId, "url"_a);
    m.def("vc3d_cache_root", []() -> std::optional<std::string> {
        const auto root = vcr::zarr::vc3dCacheRoot();
        return root ? std::optional<std::string>(root->string()) : std::nullopt;
    });
    m.def("disk_budget", &vcr::zarr::diskBudget, "root"_a, "max_bytes"_a.none(),
          nb::call_guard<nb::gil_scoped_release>());

    using vcr::zarr::Access;
    nb::class_<Access>(m, "Access", "How requests under one root are made. Picklable; keys from the environment are read again in the new process.")
        .def_prop_ro("kind", &access_kind)
        .def_prop_ro("region", [](const Access& a) { return a.auth.region; })
        .def_prop_ro("endpoint", [](const Access& a) { return a.endpoint; })
        .def("url", &Access::url, "location"_a)
        .def("cache_url", &Access::cacheUrl, "location"_a)
        .def("__repr__", [](const Access& a) {
            if (a.kind != Access::Kind::S3)
                return "Access(" + access_kind(a) + ")";
            return std::string("Access(s3, ") + (a.fromEnvironment ? "AWS environment" : "fixed keys") + ", region='" + a.auth.region + "')";
        })
        .def("__getstate__", [](const Access& a) {
            const bool keep = !a.fromEnvironment;
            return nb::make_tuple(access_kind(a), keep ? a.auth.user : "", keep ? a.auth.secret : "", keep ? a.auth.token : "",
                                  a.auth.region, a.auth.kind, a.endpoint, a.fromEnvironment);
        })
        .def("__setstate__", [](Access& a, const std::tuple<std::string, std::string, std::string, std::string, std::string,
                                                            std::string, std::string, bool>& state) {
            const auto& [kind, user, secret, token, region, authKind, endpoint, environment] = state;
            new (&a) Access();
            a.kind = kind == "s3" ? Access::Kind::S3 : kind == "basic" ? Access::Kind::Basic : Access::Kind::Anonymous;
            a.auth = vcr::Auth{authKind, user, secret, token, region};
            a.endpoint = endpoint;
            a.fromEnvironment = environment;
            vcr::zarr::reloadEnvironmentKeys(a);
        });
    m.def("anonymous_access", [] { return Access{}; });
    m.def("resolve_access", [](const std::string& location, const std::string& mode, bool offline) {
        if (mode != "auto" && mode != "aws")
            throw nb::value_error("auth mode must be 'auto' or 'aws'");
        nb::gil_scoped_release release;
        return vcr::zarr::resolveAccess(location, mode == "aws" ? vcr::zarr::AuthMode::Aws : vcr::zarr::AuthMode::Auto, offline);
    }, "location"_a, "mode"_a, "offline"_a = false);
    m.def("s3_access", [](const std::string& location, const std::string& key, const std::string& secret,
                          const std::string& token, const std::string& region, const std::string& endpoint) {
        nb::gil_scoped_release release;
        return vcr::zarr::s3Access(location, vcr::Auth{"s3", key, secret, token, region}, endpoint);
    }, "location"_a, "access_key_id"_a, "secret_access_key"_a, "session_token"_a, "region"_a, "endpoint"_a);
    m.def("basic_access", &vcr::zarr::basicAccess, "user"_a, "password"_a);

    using vcr::zarr::ArrayMeta;
    nb::class_<ArrayMeta>(m, "ArrayMeta", "Zarr array metadata as parsed by vcr::zarr::parseArrayMeta.")
        .def_prop_ro("shape", [](const ArrayMeta& a) { return a.shape; })
        .def_prop_ro("chunks", [](const ArrayMeta& a) { return a.chunks; })
        .def_prop_ro("dtype", [](const ArrayMeta& a) { return a.dtype; })
        .def_prop_ro("fill", [](const ArrayMeta& a) { return nb::bytes(reinterpret_cast<const char*>(a.fill.data()), a.fill.size()); })
        .def_prop_ro("order", [](const ArrayMeta& a) { return a.order; })
        .def_prop_ro("separator", [](const ArrayMeta& a) { return a.separator; })
        .def_prop_ro("compressor", [](const ArrayMeta& a) { return py_json(a.compressor); })
        .def_prop_ro("filters", [](const ArrayMeta& a) { return py_json(a.filters); })
        .def_prop_ro("raw", [](const ArrayMeta& a) { return py_json(a.raw); })
        .def_prop_ro("chunk_nbytes", &ArrayMeta::chunkBytes)
        .def("codec", &ArrayMeta::codec)
        .def("decoded_zarray", [](const ArrayMeta& a) { return py_json(a.decodedZarray()); });
    m.def("parse_array_meta", [](const nb::object& metadata) { return vcr::zarr::parseArrayMeta(to_json(metadata)); }, "metadata"_a);

    using vcr::zarr::CacheLayer;
    nb::class_<CacheLayer>(m, "CacheLayer", "One cache directory holding chunks of one array, in one format.")
        .def_ro("kind", &CacheLayer::kind)
        .def_ro("root", &CacheLayer::root)
        .def_ro("stored", &CacheLayer::stored)
        .def_ro("empty_root", &CacheLayer::emptyRoot)
        .def_ro("allow_lossy", &CacheLayer::allowLossy)
        .def_ro("budget_root", &CacheLayer::budgetRoot)
        .def("__repr__", [](const CacheLayer& l) { return "CacheLayer('" + l.kind + "', '" + l.root + "')"; });

    using vcr::zarr::OpenOptions;
    nb::class_<OpenOptions>(m, "OpenOptions")
        .def("__init__", [](OpenOptions* self, Access access, std::vector<std::string> cache_dirs, bool cache_writes,
                            std::optional<std::int64_t> cache_max_bytes, bool compress_cache, bool allow_lossy_cache,
                            std::shared_ptr<vcr::MemoryCache> memory_cache, std::size_t threads, int connections,
                            std::string read_mode, bool offline) {
            new (self) OpenOptions();
            self->access = std::move(access);
            self->cacheDirs = std::move(cache_dirs);
            self->cacheWrites = cache_writes;
            self->cacheMaxBytes = cache_max_bytes;
            self->compressCache = compress_cache;
            self->allowLossyCache = allow_lossy_cache;
            self->memoryCache = std::move(memory_cache);
            self->threads = threads;
            self->connections = connections;
            self->readMode = std::move(read_mode);
            self->offline = offline;
        }, "access"_a, "cache_dirs"_a, "cache_writes"_a, "cache_max_bytes"_a.none(), "compress_cache"_a, "allow_lossy_cache"_a,
           "memory_cache"_a.none(), "threads"_a, "connections"_a, "read_mode"_a, "offline"_a);

    using vcr::zarr::OpenedArray;
    nb::class_<OpenedArray>(m, "OpenedArray")
        .def_ro("location", &OpenedArray::location)
        .def_ro("meta", &OpenedArray::meta)
        .def_ro("store", &OpenedArray::store)
        .def_ro("level_key", &OpenedArray::levelKey)
        .def_ro("writer", &OpenedArray::writer)
        .def_ro("busy", &OpenedArray::busy)
        .def_prop_ro("array", [](OpenedArray& o) -> vcr::Array& { return *o.array; }, nb::rv_policy::reference_internal)
        .def_prop_ro("layers", [](const OpenedArray& o) {
            auto layers = o.layers;  // a view: the leases stay with the opened array
            for (auto& layer : layers)
                layer.lease.reset();
            return layers;
        })
        .def_prop_ro("placement_scale", [](const OpenedArray& o) { return o.placement.scale; })
        .def("attributes", [](OpenedArray& o, const OpenOptions& options) {
            {
                nb::gil_scoped_release release;
                vcr::zarr::arrayAttributes(o, options);
            }
            return py_json(o.attributes);
        }, "options"_a)
        .def_prop_ro("known_attributes", [](const OpenedArray& o) -> nb::object {
            return o.attributesRead ? py_json(o.attributes) : nb::none();
        })
        .def("close", &OpenedArray::close, nb::call_guard<nb::gil_scoped_release>());
    m.def("open_array", [](const std::string& location, const OpenOptions& options, std::optional<std::string> group,
                           std::optional<std::string> level_key, nb::object metadata, nb::object attributes) {
        vcr::zarr::ArrayHints hints;
        hints.group = std::move(group);
        hints.levelKey = std::move(level_key);
        if (!metadata.is_none())
            hints.metadata = to_json(metadata);
        if (!attributes.is_none())
            hints.attributes = to_json(attributes);
        nb::gil_scoped_release release;
        return vcr::zarr::openArray(location, options, std::move(hints));
    }, "location"_a, "options"_a, "group"_a.none() = nb::none(), "level_key"_a.none() = nb::none(),
       "metadata"_a.none() = nb::none(), "attributes"_a.none() = nb::none());

    using vcr::zarr::Group;
    nb::class_<Group>(m, "Group")
        .def_static("open", [](const std::string& location, const OpenOptions& options) {
            nb::gil_scoped_release release;
            return Group::open(location, options);
        }, "location"_a, "options"_a)
        .def_static("from_state", [](const std::string& state, const OpenOptions& options) {
            return Group::fromState(Json::parse(state), options);
        }, "state"_a, "options"_a)
        .def("state", [](const Group& g, bool inherited) { return g.state(inherited).dump(); }, "inherited"_a = false)
        .def_prop_ro("location", [](const Group& g) { return g.location(); })
        .def_prop_ro("version", &Group::version)
        .def_prop_ro("attributes", [](const Group& g) { return py_json(g.attributes()); })
        .def_prop_ro("multiscales", [](const Group& g) { return py_json(g.multiscales()); })
        .def("levels", &Group::levels, "probe_levels"_a = 0, nb::call_guard<nb::gil_scoped_release>())
        .def("keys", &Group::keys, nb::call_guard<nb::gil_scoped_release>())
        .def("node", [](Group& g, const std::string& key) -> nb::object {
            std::optional<std::pair<Group::NodeKind, Json>> node;
            {
                nb::gil_scoped_release release;
                node = g.node(key);
            }
            if (!node)
                return nb::none();
            return nb::cast(std::string(node->first == Group::NodeKind::Array ? "array" : "group"));
        }, "key"_a, "'array' or 'group' for a child path; None when there is none.")
        .def("array", &Group::array, "key"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("group", &Group::group, "key"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("metadata_json", [](Group& g, bool refresh) {
            std::string text;
            {
                nb::gil_scoped_release release;
                text = vcr::zarr::storeMetadataJson(g.metadata(refresh)).dump();
            }
            return text;
        }, "refresh"_a = false);
    m.def("volume_metadata_json", [](const std::string& location, const OpenOptions& options, bool refresh) {
        nb::gil_scoped_release release;
        return vcr::zarr::storeMetadataJson(vcr::zarr::volumeMetadata(location, options, refresh)).dump();
    }, "location"_a, "options"_a, "refresh"_a = false,
       "What is known about the volume at a location, as JSON (see vcr::zarr::volumeMetadata).");
    m.def("forget_volume_metadata", &vcr::zarr::forgetVolumeMetadata);
    m.def("level_placement_scale", [](const nb::object& group_attributes, const std::string& path) {
        return vcr::zarr::levelPlacement(to_json(group_attributes), path).scale;
    }, "group_attributes"_a, "path"_a);

    nb::class_<vcr::DiskCache>(m, "DiskCache")
        .def(nb::init<const std::string&, std::int64_t>(), "root"_a, "max_bytes"_a,
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_max_bytes", &vcr::DiskCache::set_max_bytes, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("max_bytes", &vcr::DiskCache::max_bytes, nb::call_guard<nb::gil_scoped_release>())
        .def("open_file", [](vcr::DiskCache& cache, const std::string& path, const std::string& location,
                              const vcr::zarr::Access& access) {
            nb::gil_scoped_release release;
            return cache.open_file(path, access.url(location), access.auth);
        }, "path"_a, "location"_a, "access"_a);

    nb::class_<vcr::Array>(m, "Array")
        .def(
            "read_many",
            [](vcr::Array& a, const std::vector<vcr::Index>& starts, const std::vector<vcr::Index>& sizes,
               std::vector<Bytes> outs, std::size_t itemsize) {
                if (starts.size() != sizes.size() || outs.size() != sizes.size())
                    throw nb::value_error("starts, sizes and outs must have the same length");
                std::vector<std::uint8_t*> ptrs(outs.size());
                for (std::size_t i = 0; i < outs.size(); ++i) {
                    const std::int64_t n = std::accumulate(sizes[i].begin(), sizes[i].end(), std::int64_t{1},
                                                           std::multiplies<>());
                    if (static_cast<std::int64_t>(outs[i].size()) != n * static_cast<std::int64_t>(itemsize))
                        throw nb::value_error("an output buffer does not match its region");
                    ptrs[i] = outs[i].data();
                }
                nb::gil_scoped_release release;
                a.read_many(starts, sizes, ptrs);
            },
            "starts"_a, "sizes"_a, "outs"_a, "itemsize"_a,
            "Read regions into flat uint8 views of C-order outputs (the GIL is released).")
        .def(
            "read_many_float",
            [](vcr::Array& a, const std::vector<vcr::Index>& starts, const std::vector<vcr::Index>& sizes,
               std::vector<Bytes> outs, const std::string& kind, std::size_t itemsize, bool swap,
               bool out64) {
                if (starts.size() != sizes.size()) throw nb::value_error("starts and sizes must match");
                if (itemsize != a.itemsize()) throw nb::value_error("source itemsize does not match");
                const auto type = sample_type(kind, itemsize, swap);
                output_counts(sizes, outs, out64 ? 8 : 4);
                std::vector<std::uint8_t*> ptrs;
                for (auto& out : outs) ptrs.push_back(out.data());
                nb::gil_scoped_release release;
                a.read_many_float(starts, sizes, ptrs, type, out64);
            },
            "starts"_a, "sizes"_a, "outs"_a, "kind"_a, "itemsize"_a, "swap"_a, "out64"_a,
            "Read directly into floating outputs.")
        .def(
            "read_points",
            [](vcr::Array& a, nb::ndarray<std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> coords, Bytes out,
               std::size_t itemsize) {
                if (static_cast<std::size_t>(out.size()) != coords.shape(0) * itemsize)
                    throw nb::value_error("the output buffer does not match the points");
                if (coords.shape(1) != a.ndim())
                    throw nb::value_error("points must have one index per array dimension");
                nb::gil_scoped_release release;
                a.read_points(coords.data(), coords.shape(0), out.data());
            },
            "coords"_a, "out"_a, "itemsize"_a,
            "Read the elements at (n, ndim) int64 points into a flat uint8 view of n outputs (the GIL is released).")
        .def(
            "sample_nearest",
            [](vcr::Array& a, nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> coords, Bytes out,
               std::size_t itemsize) {
                if (static_cast<std::size_t>(out.size()) != coords.shape(0) * itemsize)
                    throw nb::value_error("the output buffer does not match the points");
                if (coords.shape(1) != a.ndim())
                    throw nb::value_error("points must have one coordinate per array dimension");
                nb::gil_scoped_release release;
                a.sample_nearest(coords.data(), coords.shape(0), out.data());
            },
            "coords"_a, "out"_a, "itemsize"_a, "Nearest elements at (n, ndim) float64 points (the GIL is released).")
        .def(
            "sample_linear",
            [](vcr::Array& a, nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> coords, Bytes out,
               const std::string& kind_name, std::size_t itemsize, bool swap, bool out64) {
                if (static_cast<std::size_t>(out.size()) != coords.shape(0) * (out64 ? 8 : 4))
                    throw nb::value_error("the output buffer does not match the points");
                if (coords.shape(1) != a.ndim())
                    throw nb::value_error("points must have one coordinate per array dimension");
                const auto type = sample_type(kind_name, itemsize, swap);
                nb::gil_scoped_release release;
                a.sample_linear(coords.data(), coords.shape(0), type, out64, out.data());
            },
            "coords"_a, "out"_a, "kind"_a, "itemsize"_a, "swap"_a, "out64"_a,
            "N-linear interpolation at (n, ndim) float64 points into float32/float64 outputs (the GIL is released).")
        .def(
            "read_affine_many",
            [](vcr::Array& a, const std::vector<vcr::Index>& shapes,
               nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> origins,
               nb::ndarray<double, nb::ndim<3>, nb::c_contig, nb::device::cpu> axes,
               std::vector<Bytes> outs, int order, const std::string& kind, std::size_t itemsize,
               bool swap, bool out64, bool float_output) {
                if (a.ndim() != 3 || origins.shape(1) != 3 || axes.shape(1) != 3 || axes.shape(2) != 3)
                    throw nb::value_error("affine reads require 3-D shapes, origins and axes");
                if (shapes.size() != outs.size() || shapes.size() != origins.shape(0) || shapes.size() != axes.shape(0))
                    throw nb::value_error("shapes, origins, axes and outs must have the same length");
                if (order != 0 && order != 1)
                    throw nb::value_error("order must be 0 (nearest) or 1 (linear)");
                if (itemsize != a.itemsize())
                    throw nb::value_error("itemsize does not match the source array");
                const auto type = order == 1 || float_output ? sample_type(kind, itemsize, swap) : vcr::ElementType{};
                const std::size_t outsize = order == 0 && !float_output ? itemsize : (out64 ? 8 : 4);
                std::vector<vcr::AffineCoordinates> grids(shapes.size());
                std::vector<std::size_t> counts(shapes.size());
                // Validate the whole batch before writing any output.
                for (std::size_t j = 0; j < shapes.size(); ++j) {
                    if (shapes[j].size() != 3)
                        throw nb::value_error("shape must have three non-negative integers");
                    auto& grid = grids[j];
                    std::size_t n = 1;
                    for (std::size_t i = 0; i < 3; ++i) {
                        const auto size = shapes[j][i];
                        if (size < 0 || (size && n > std::numeric_limits<std::size_t>::max() / size))
                            throw nb::value_error("invalid or overflowing affine shape");
                        n *= size;
                        grid.shape[i] = size;
                        grid.origin[i] = origins(j, i);
                        if (!std::isfinite(grid.origin[i]))
                            throw nb::value_error("origin must be finite");
                        for (std::size_t k = 0; k < 3; ++k) {
                            grid.axes[3*i+k] = axes(j, i, k);
                            if (!std::isfinite(grid.axes[3*i+k]))
                                throw nb::value_error("axes must be finite");
                        }
                    }
                    if (n > std::numeric_limits<std::size_t>::max() / outsize || outs[j].size() != n * outsize)
                        throw nb::value_error("an output buffer does not match its affine shape");
                    counts[j] = n;
                }
                nb::gil_scoped_release release;
                for (std::size_t j = 0; j < grids.size(); ++j)
                    a.read_affine(grids[j], counts[j], order, type, out64, outs[j].data(), float_output);
            },
            "shapes"_a, "origins"_a, "axes"_a, "outs"_a, "order"_a, "kind"_a, "itemsize"_a, "swap"_a, "out64"_a,
            "float_output"_a = false,
            "Sample regular affine grids with generated coordinates (the GIL is released).")
        // Float32 CT implementation shared by the public CPU and CUDA fallback.
        .def("_sample_u8_float32", [](vcr::Array& a,
             HostPoints points,
             const vcr::Index& offset, Bytes out, int order) {
            if (points.shape(1) != 3 || offset.size() != 3 || (order != 0 && order != 1))
                throw nb::value_error("expected 3-D points/origin and order 0 or 1");
            if (out.size() != points.shape(0)*(order ? sizeof(float) : 1))
                throw nb::value_error("output size does not match points");
            for (int i=0;i<3;++i) {
                if (offset[i] < 0 || offset[i] > (1LL<<30)) throw nb::value_error("invalid integer origin");
            }
            if (points.dtype() == nb::dtype<float>()) sample_u8_points<float>(a,points,offset,out,order);
            else if (points.dtype() == nb::dtype<double>()) sample_u8_points<double>(a,points,offset,out,order);
            else throw nb::type_error("expected float32 or float64 points");
        })
        .def("_affine_u8_float32", [](vcr::Array& a, const vcr::Index& shape,
             const std::vector<float>& origin, const std::vector<float>& axes,
             const vcr::Index& offset, Bytes out, int order) {
            if (shape.size()!=3 || origin.size()!=3 || axes.size()!=9 || offset.size()!=3 || (order!=0 && order!=1))
                throw nb::value_error("expected 3-D affine geometry and order 0 or 1");
            const auto counts=output_counts({shape},{out},order ? sizeof(float) : 1);
            vcr::FloatAffine grid;
            for (int i=0;i<3;++i) {
                if (!std::isfinite(origin[i]) || offset[i]<0 || offset[i]>(1LL<<30))
                    throw nb::value_error("invalid affine origin");
                grid.shape[i]=shape[i]; grid.origin[i]=origin[i]; grid.offset[i]=offset[i];
            }
            for (int i=0;i<9;++i) {
                if (!std::isfinite(axes[i])) throw nb::value_error("axes must be finite");
                grid.axes[i]=axes[i];
            }
            nb::gil_scoped_release release;
            a.sample_u8_float32(grid,counts[0],order,out.data());
        })
        .def_prop_ro("threads", &vcr::Array::threads)
        .def("flush", &vcr::Array::flush, nb::call_guard<nb::gil_scoped_release>(),
             "Wait for every queued disk-cache write.")
        .def("stats", [](vcr::Array& a) {
            const auto& s = a.stats();
            nb::dict d;
            d["memory_hits"] = s.memory_hits.load();
            d["pending_hits"] = s.pending_hits.load();
            d["pending_writes"] = a.pending_writes();
            d["disk_hits"] = s.disk_hits.load();
            d["remote_fetches"] = s.remote_fetches.load();
            d["remote_missing"] = s.remote_missing.load();
            d["remote_bytes"] = s.remote_bytes.load();
            d["write_errors"] = s.write_errors.load();
            d["last_write_error"] = a.last_write_error();
            return d;
        });
}
