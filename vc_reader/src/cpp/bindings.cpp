// Python bindings for the C++ read path (module vc_reader._reader).
#include "array.hpp"
#include "cuda_available.hpp"
#include "vc_reader/composite.hpp"
#include "vc_reader/computed.hpp"
#include "vc_reader/reader.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_metadata.hpp"
#include "vc_reader/zarr_write.hpp"
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

#include <cstring>
#include <functional>
#include <map>
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
#include "tifxyz/bindings.hpp"

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

#include "sheet_normal/bindings.hpp"
#include "geometry/bindings.hpp"

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
    vcr::sheet_normal::bind(m);
    vcr::geometry::bind(m);
    vcr::sources::python::bind(m);
    vcr::tifxyz::python::bind(m);
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
                          const std::string& token, const std::string& region, const std::string& endpoint, bool offline) {
        nb::gil_scoped_release release;
        return vcr::zarr::s3Access(location, vcr::Auth{"s3", key, secret, token, region}, endpoint, offline);
    }, "location"_a, "access_key_id"_a, "secret_access_key"_a, "session_token"_a, "region"_a, "endpoint"_a, "offline"_a = false);
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
    m.def("composite_reduce", [](nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> values, const std::string& method,
                                 float iso_cutoff, float alpha_min, float alpha_max, float alpha_opacity, float alpha_cutoff,
                                 float extinction, float emission, float ambient, nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> out) {
        namespace composite = vcr::composite;
        static const std::map<std::string, composite::Method> methods{
            {"mean", composite::Method::Mean}, {"max", composite::Method::Max}, {"min", composite::Method::Min},
            {"median", composite::Method::Median}, {"min_abs", composite::Method::MinAbs},
            {"alpha", composite::Method::Alpha}, {"beer_lambert", composite::Method::BeerLambert}};
        const auto found = methods.find(method);
        if (found == methods.end())
            throw nb::value_error(("unknown composite method '" + method + "'").c_str());
        if (out.shape(0) != values.shape(1))
            throw nb::value_error("composite output does not match the values");
        composite::Options options;
        options.method = found->second;
        options.isoCutoff = iso_cutoff;
        options.alphaMin = alpha_min;
        options.alphaMax = alpha_max;
        options.alphaOpacity = alpha_opacity;
        options.alphaCutoff = alpha_cutoff;
        options.extinction = extinction;
        options.emission = emission;
        options.ambient = ambient;
        nb::gil_scoped_release release;
        composite::reduce(values.data(), values.shape(0), values.shape(1), options, out.data());
    }, "values"_a, "method"_a, "iso_cutoff"_a, "alpha_min"_a, "alpha_max"_a, "alpha_opacity"_a, "alpha_cutoff"_a,
       "extinction"_a, "emission"_a, "ambient"_a, "out"_a);
    m.def("open_computed", [](const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& chunks, const std::string& dtype,
                              nb::object fill_value, const std::string& identity, nb::callable compute,
                              std::optional<std::string> cache_dir, std::optional<std::int64_t> cache_max_bytes, bool compress_cache,
                              std::shared_ptr<vcr::MemoryCache> memory_cache, std::size_t threads) {
        vcr::computed::Options options;
        options.shape = shape;
        options.chunks = chunks;
        options.dtype = dtype;
        options.fillValue = to_json(fill_value);
        options.identity = identity;
        if (cache_dir)
            options.cacheDir = *cache_dir;
        options.cacheMaxBytes = cache_max_bytes;
        options.compressCache = compress_cache;
        options.memoryCache = std::move(memory_cache);
        options.threads = threads;
        OpenedArray opened;
        opened.location = "computed:" + identity;
        opened.meta = vcr::zarr::parseArrayMeta({{"zarr_format", 2}, {"shape", shape}, {"chunks", chunks}, {"dtype", dtype},
            {"compressor", nullptr}, {"filters", nullptr}, {"order", "C"}, {"fill_value", options.fillValue}, {"dimension_separator", "/"}});
        const std::size_t bytes = opened.meta.chunkBytes();
        // The callback outlives this call inside the array; the GIL guards its reference count.
        auto held = std::shared_ptr<nb::callable>(new nb::callable(std::move(compute)), [](nb::callable* c) {
            nb::gil_scoped_acquire acquire;
            delete c;
        });
        opened.array = vcr::computed::open(options, [held, bytes](const vcr::Index& chunk, std::uint8_t* out) {
            nb::gil_scoped_acquire acquire;
            nb::tuple index = nb::steal<nb::tuple>(PyTuple_New(Py_ssize_t(chunk.size())));
            for (std::size_t i = 0; i < chunk.size(); ++i)
                PyTuple_SET_ITEM(index.ptr(), Py_ssize_t(i), PyLong_FromLongLong(chunk[i]));
            nb::object result = (*held)(index);
            if (result.is_none())
                return false;
            const auto data = nb::cast<nb::bytes>(result);
            if (data.size() != bytes)
                throw std::invalid_argument("a computed chunk has the wrong size");
            std::memcpy(out, data.c_str(), bytes);
            return true;
        });
        return opened;
    }, "shape"_a, "chunks"_a, "dtype"_a, "fill_value"_a, "identity"_a, "compute"_a, "cache_dir"_a.none(), "cache_max_bytes"_a.none(),
       "compress_cache"_a, "memory_cache"_a.none(), "threads"_a);
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
        .def("placement", [](const Group& g, const std::string& path) {
            const auto placement = g.placement(path);
            return nb::make_tuple(placement.scale, placement.offset);
        }, "path"_a)
        .def("keys", &Group::keys, nb::call_guard<nb::gil_scoped_release>())
        .def("array_keys", &Group::arrayKeys, nb::call_guard<nb::gil_scoped_release>())
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
    m.def("is_array_node", &vcr::zarr::isArrayNode, "location"_a, "options"_a, nb::call_guard<nb::gil_scoped_release>());
    m.def("forget_kept_metadata", &vcr::zarr::forgetKeptMetadata, "location"_a, "options"_a);
    m.def("level_voxel_size_zyx", [](std::optional<double> voxel_size, const nb::object& documents, const std::string& level_key) {
        vcr::zarr::StoreMetadata metadata;
        metadata.voxelSizeUm = voxel_size;
        metadata.documents = to_json(documents);
        return vcr::zarr::levelVoxelSizeZYX(metadata, level_key);
    }, "voxel_size"_a.none(), "documents"_a, "level_key"_a);

    using vcr::zarr::Writer;
    using ConstBytes = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;
    m.def("compress_bytes", [](ConstBytes data, const nb::object& compressor, std::size_t itemsize, const vcr::Index& shape) {
        const auto config = to_json(compressor);
        std::vector<std::uint8_t> encoded;
        {
            nb::gil_scoped_release release;
            encoded = vcr::zarr::compressBytes(config, {static_cast<const std::uint8_t*>(data.data()), data.nbytes()}, itemsize, shape);
        }
        return nb::bytes(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    }, "data"_a, "compressor"_a.none(), "itemsize"_a, "shape"_a);
    m.def("decompress_bytes", [](ConstBytes data, const nb::object& compressor, std::size_t size) {
        const auto config = to_json(compressor);
        auto decoded = nb::steal<nb::bytes>(PyBytes_FromStringAndSize(nullptr, Py_ssize_t(size)));
        if (!decoded.is_valid())
            throw nb::python_error();
        auto* out = reinterpret_cast<std::uint8_t*>(PyBytes_AS_STRING(decoded.ptr()));
        {
            nb::gil_scoped_release release;
            vcr::zarr::decompressBytes(config, {static_cast<const std::uint8_t*>(data.data()), data.nbytes()}, {out, size});
        }
        return decoded;
    }, "data"_a, "compressor"_a.none(), "size"_a);
    nb::class_<Writer>(m, "Writer", "A Zarr array, local or on S3, opened for writing.")
        .def_static("create", [](const std::string& path, const vcr::Index& shape, const vcr::Index& chunks,
                                 const std::string& dtype, const std::string& compressor, int level, const std::string& cname,
                                 int shuffle, bool overwrite, const std::string& separator, const nb::object& fill, int version,
                                 const vcr::Index& shards, bool packbits, const vcr::zarr::Access& access) {
            vcr::zarr::CreateOptions options;
            options.shape = shape;
            options.chunks = chunks;
            options.dtype = dtype;
            options.compressor = compressor;
            options.level = level;
            options.cname = cname;
            options.shuffle = shuffle;
            options.overwrite = overwrite;
            options.separator = separator;
            options.fillValue = to_json(fill);
            options.version = version;
            options.shardShape = shards;
            options.packbits = packbits;
            nb::gil_scoped_release release;
            return Writer::create(path, options, access);
        }, "path"_a, "shape"_a, "chunks"_a, "dtype"_a, "compressor"_a, "level"_a, "cname"_a, "shuffle"_a, "overwrite"_a,
           "separator"_a, "fill_value"_a.none(), "version"_a, "shards"_a, "packbits"_a, "access"_a)
        .def_static("create_like", [](const std::string& path, const ArrayMeta& like, const std::optional<vcr::Index>& shape,
                                      const std::optional<vcr::Index>& chunks, const std::optional<std::string>& dtype,
                                      const nb::object& fill, const std::optional<std::string>& separator, bool overwrite,
                                      const vcr::zarr::Access& access) {
            auto options = vcr::zarr::createOptionsLike(like);
            if (shape) options.shape = *shape;
            if (chunks) options.chunks = *chunks;
            if (dtype) {
                options.dtype = *dtype;
                options.packbits = options.packbits && *dtype == "|b1";
            }
            if (!fill.is_none()) options.fillValue = to_json(fill);
            if (separator) options.separator = *separator;
            options.overwrite = overwrite;
            nb::gil_scoped_release release;
            return Writer::create(path, options, access);
        }, "path"_a, "like"_a, "shape"_a.none(), "chunks"_a.none(), "dtype"_a.none(), "fill_value"_a.none(), "separator"_a.none(),
           "overwrite"_a, "access"_a)
        .def_static("open", [](const std::string& path, const vcr::zarr::Access& access) {
            nb::gil_scoped_release release;
            return Writer::open(path, access);
        }, "path"_a, "access"_a)
        .def_prop_ro("meta", &Writer::meta, nb::rv_policy::reference_internal)
        .def_prop_ro("path", [](const Writer& w) { return w.path().string(); })
        .def("write_region", [](Writer& w, const vcr::Index& start, const vcr::Index& shape, ConstBytes data, bool skip_empty) {
            std::int64_t n = static_cast<std::int64_t>(w.meta().itemsize());
            for (const auto extent : shape) n *= extent;
            if (static_cast<std::int64_t>(data.nbytes()) != n)
                throw nb::value_error("the data does not match its region");
            nb::gil_scoped_release release;
            w.writeRegion(start, shape, static_cast<const std::uint8_t*>(data.data()), skip_empty);
        }, "start"_a, "shape"_a, "data"_a, "skip_empty"_a = false)
        .def("write_chunk", [](Writer& w, const vcr::Index& chunk, ConstBytes data, bool skip_empty) {
            nb::gil_scoped_release release;
            const auto* bytes = static_cast<const std::uint8_t*>(data.data());
            if (skip_empty)
                w.writeChunkSkipEmpty(chunk, bytes, data.nbytes());
            else
                w.writeChunk(chunk, bytes, data.nbytes());
        }, "chunk"_a, "data"_a, "skip_empty"_a = false)
        .def("chunk_exists", &Writer::chunkExists, "chunk"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("stored_chunks", &Writer::storedChunks, nb::call_guard<nb::gil_scoped_release>())
        .def("remove_chunk", &Writer::removeChunk, "chunk"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("set_fill_value", [](Writer& w, const nb::object& value) {
            const auto fill = to_json(value);
            nb::gil_scoped_release release;
            w.setFillValue(fill);
        }, "value"_a.none());
    using vcr::zarr::Store;
    nb::class_<Store>(m, "Store", "The objects below a local directory or an s3:// prefix.")
        .def("__init__", [](Store* store, const std::string& location, const vcr::zarr::Access& access, int connections) {
            new (store) Store(location, access, connections);
        }, "location"_a, "access"_a, "connections"_a = 16)
        .def_prop_ro("location", &Store::location)
        .def_prop_ro("access", [](const Store& s) { return s.access(); })
        .def_prop_ro("remote", &Store::remote)
        .def("read", [](const Store& s, const std::string& name) -> nb::object {
            std::optional<std::string> data;
            {
                nb::gil_scoped_release release;
                data = s.read(name);
            }
            if (!data) return nb::none();
            return nb::bytes(data->data(), data->size());
        }, "name"_a)
        .def("write", [](const Store& s, const std::string& name, const nb::bytes& data) {
            const std::string_view view(data.c_str(), data.size());
            nb::gil_scoped_release release;
            s.write(name, view);
        }, "name"_a, "data"_a)
        .def("write_file", &Store::writeFile, "name"_a, "file"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("read_file", &Store::readFile, "name"_a, "file"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("exists", &Store::exists, "name"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("remove", &Store::remove, "name"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("remove_all", &Store::removeAll, nb::call_guard<nb::gil_scoped_release>())
        .def("objects", [](const Store& s, const std::string& prefix) {
            std::vector<vcr::s3::Object> objects;
            {
                nb::gil_scoped_release release;
                objects = s.list(prefix);
            }
            nb::list out;
            for (const auto& object : objects) {
                nb::dict entry;
                entry["key"] = object.key;
                entry["size"] = object.size;
                entry["etag"] = object.etag;
                entry["last_modified"] = object.lastModified;
                out.append(entry);
            }
            return out;
        }, "prefix"_a = "")
        .def("list", [](const Store& s) {
            std::vector<std::pair<std::string, std::uint64_t>> out;
            {
                nb::gil_scoped_release release;
                for (auto& object : s.list())
                    out.emplace_back(std::move(object.key), object.size);
            }
            return out;
        })
        .def("children", &Store::children, nb::call_guard<nb::gil_scoped_release>())
        .def("is_node", &Store::isNode, nb::call_guard<nb::gil_scoped_release>())
        .def("is_array", &Store::isArray, nb::call_guard<nb::gil_scoped_release>());
    using vcr::zarr::Access;
    m.def("read_attributes", [](const std::string& path, const Access& access) {
        Json attributes;
        {
            nb::gil_scoped_release release;
            attributes = vcr::zarr::readAttributes(path, access);
        }
        return py_json(attributes);
    }, "path"_a, "access"_a);
    m.def("write_attributes", [](const std::string& path, const nb::object& attributes, const Access& access) {
        const auto json = to_json(attributes);
        nb::gil_scoped_release release;
        vcr::zarr::writeAttributes(path, json, access);
    }, "path"_a, "attributes"_a, "access"_a);
    m.def("create_group", [](const std::string& path, const nb::object& attributes, int version, const Access& access) {
        const auto json = attributes.is_none() ? std::nullopt : std::optional<Json>(to_json(attributes));
        nb::gil_scoped_release release;
        vcr::zarr::createGroup(path, json, version, access);
    }, "path"_a, "attributes"_a.none(), "version"_a, "access"_a);
    m.def("remove_node", [](const std::string& path, const Access& access) {
        nb::gil_scoped_release release;
        return vcr::zarr::removeNode(path, access);
    }, "path"_a, "access"_a);
    m.def("node_children", [](const std::string& path, const Access& access) { return vcr::zarr::nodeChildren(path, access); },
          "path"_a, "access"_a, nb::call_guard<nb::gil_scoped_release>());

    const auto pyramidOptions = [](int levels, const std::array<int, 3>& factors, const std::string& reduction, std::size_t threads) {
        vcr::zarr::PyramidOptions options;
        options.levels = levels;
        options.factors = factors;
        options.reduction = vcr::zarr::parseReduction(reduction);
        options.threads = threads;
        return options;
    };
    m.def("create_pyramid_levels", [pyramidOptions](const std::string& root, int levels, const std::array<int, 3>& factors, bool overwrite) {
        const auto options = pyramidOptions(levels, factors, "mean", 0);
        nb::gil_scoped_release release;
        vcr::zarr::createPyramidLevels(root, options, overwrite);
    }, "root"_a, "levels"_a, "factors"_a, "overwrite"_a);
    m.def("write_pyramid_level", [pyramidOptions](const std::string& root, int level, const std::array<int, 3>& factors,
                                                  const std::string& reduction, std::size_t threads,
                                                  const std::optional<vcr::Index>& begin, const std::optional<vcr::Index>& end) {
        const auto options = pyramidOptions(level, factors, reduction, threads);
        std::optional<vcr::zarr::ChunkRange> range;
        if (begin && end)
            range = vcr::zarr::ChunkRange{*begin, *end};
        nb::gil_scoped_release release;
        vcr::zarr::writePyramidLevel(root, level, options, range);
    }, "root"_a, "level"_a, "factors"_a, "reduction"_a, "threads"_a, "begin"_a.none(), "end"_a.none());
    m.def("write_pyramid", [pyramidOptions](const std::string& root, int levels, const std::array<int, 3>& factors,
                                            const std::string& reduction, std::size_t threads, bool overwrite) {
        const auto options = pyramidOptions(levels, factors, reduction, threads);
        nb::gil_scoped_release release;
        vcr::zarr::writePyramid(root, options, overwrite);
    }, "root"_a, "levels"_a, "factors"_a, "reduction"_a, "threads"_a, "overwrite"_a);
    m.def("ome_multiscales", [](const std::string& name, int levels, const std::array<int, 3>& factors, const std::array<double, 3>& scale,
                                const std::string& unit, const std::string& reduction, const std::array<std::string, 3>& axes) {
        vcr::zarr::MultiscalesOptions options;
        options.name = name;
        options.levels = levels;
        options.factors = factors;
        options.scale = scale;
        options.unit = unit;
        options.reduction = vcr::zarr::parseReduction(reduction);
        options.axes = axes;
        return py_json(vcr::zarr::omeMultiscales(options));
    }, "name"_a, "levels"_a, "factors"_a, "scale"_a, "unit"_a, "reduction"_a, "axes"_a);

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
            d["corrupt"] = s.corrupt.load();
            d["last_corrupt_chunk"] = a.last_corrupt_chunk();
            return d;
        });
}
