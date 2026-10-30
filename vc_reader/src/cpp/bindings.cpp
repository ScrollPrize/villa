// Python bindings for the C++ read path (module vc_reader._reader).
#include "array.hpp"
#include "cuda_available.hpp"
#include "vc_reader/reader.hpp"
#include "vc_reader/zarr.hpp"
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

#include <functional>
#include <numeric>

namespace nb = nanobind;
using namespace nb::literals;

using Bytes = nb::ndarray<std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using LayerTuple = std::tuple<std::string, std::string, std::string, std::string, bool>;
// (kind, user or access key id, password or secret key, session token, region); see vcr::Auth.
using AuthTuple = std::tuple<std::string, std::string, std::string, std::string, std::string>;
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

static vcr::Auth to_auth(const AuthTuple& t)
{
    auto& [kind, user, secret, token, region] = t;
    return vcr::Auth{kind, user, secret, token, region};
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
    m.def("normalize_zarr_metadata", &vcr::normalizeZarrMetadata);
    m.doc() = "Zarr reader: local and HTTP chunks, disk cache layers, memory cache.";
    m.def("_point_origin", [](HostPoints points) -> nb::object {
        if (points.shape(1) != 3) throw nb::value_error("expected points of shape (N, 3)");
        const auto origin = [&](const auto* p) -> nb::object {
            std::optional<std::array<std::int64_t, 3>> base;
            {
                nb::gil_scoped_release release;
                base = vcr::pointOrigin(p, points.shape(0));
            }
            return base ? nb::cast(vcr::Index(base->begin(), base->end())) : nb::none();
        };
        if (points.dtype() == nb::dtype<float>()) return origin(static_cast<const float*>(points.data()));
        if (points.dtype() == nb::dtype<double>()) return origin(static_cast<const double*>(points.data()));
        throw nb::type_error("expected float32 or float64 points");
    });
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

    m.def(
        "http_get",
        [](const std::string& url, const AuthTuple& auth, long stall_timeout_s, int retries) -> nb::object {
            std::vector<std::uint8_t> body;
            bool found;
            {
                nb::gil_scoped_release release;
                vcr::HttpOptions options;
                options.connections = 1;
                options.stall_timeout_s = stall_timeout_s;
                options.retries = retries;
                options.auth = to_auth(auth);
                vcr::HttpClient client(options);
                found = client.get(url, body);
            }
            if (!found)
                return nb::none();
            return nb::bytes(reinterpret_cast<const char*>(body.data()), body.size());
        },
        "url"_a, "auth"_a = AuthTuple{}, "stall_timeout_s"_a = 10, "retries"_a = 5,
        "GET a small object (e.g. .zarray); None if it does not exist (404). PermissionError on 401/403.");

    nb::class_<vcr::MemoryCache>(m, "MemoryCache", "Byte-bounded LRU of decoded chunks, shareable between arrays.")
        .def(nb::init<std::size_t>(), "max_bytes"_a)
        .def_prop_ro("max_bytes", &vcr::MemoryCache::max_bytes)
        .def_prop_ro("nbytes", &vcr::MemoryCache::nbytes)
        .def_prop_ro("hits", &vcr::MemoryCache::hits)
        .def_prop_ro("misses", &vcr::MemoryCache::misses)
        .def("__len__", &vcr::MemoryCache::size)
        .def("clear", &vcr::MemoryCache::clear);

    // Disk-cache layers of remote arrays (vc_reader/zarr.hpp); the Python reader's
    // vc_reader.reader.disk_cache is a thin layer over these.
    nb::class_<vcr::zarr::Layers>(m, "CacheLayers")
        .def("__len__", [](const vcr::zarr::Layers& l) { return l.layers.size(); })
        .def_prop_ro("writer", [](const vcr::zarr::Layers& l) { return l.writer; })
        .def_prop_ro("busy", [](const vcr::zarr::Layers& l) { return l.busy; })
        .def("layer", [](const vcr::zarr::Layers& l, std::size_t i) {
            const auto& c = l.layers.at(i);
            return std::make_tuple(c.kind, c.root, c.stored.raw.dump(), c.emptyRoot, c.allowLossy, c.budgetRoot);
        }, "index"_a)
        .def("release", [](vcr::zarr::Layers& l, std::size_t i) { l.layers.at(i).lease.reset(); }, "index"_a);
    m.def("open_layers",
          [](const std::vector<std::string>& cache_dirs, const std::string& url, const std::string& source,
             std::optional<std::string> group_url, std::optional<std::string> level_key, bool writable,
             bool allow_lossy, bool compress, std::optional<std::int64_t> max_bytes) {
              vcr::zarr::LayerOptions options;
              options.groupUrl = std::move(group_url);
              options.levelKey = std::move(level_key);
              options.writable = writable;
              options.allowLossy = allow_lossy;
              options.compress = compress;
              options.maxBytes = max_bytes;
              const auto meta = vcr::zarr::parseArrayMeta(vcr::zarr::Json::parse(source));
              nb::gil_scoped_release release;
              return vcr::zarr::openLayers(cache_dirs, url, meta, options);
          },
          "cache_dirs"_a, "url"_a, "source"_a, "group_url"_a.none(), "level_key"_a.none(), "writable"_a,
          "allow_lossy"_a, "compress"_a, "max_bytes"_a.none());
    m.def("cached_source_zarray",
          [](const std::vector<std::string>& cache_dirs, const std::string& url, std::optional<std::string> group_url,
             std::optional<std::string> level_key) -> std::optional<std::string> {
              const auto found = vcr::zarr::cachedSourceZarray(cache_dirs, url, group_url, level_key);
              return found ? std::optional<std::string>(found->dump()) : std::nullopt;
          },
          "cache_dirs"_a, "url"_a, "group_url"_a.none(), "level_key"_a.none());
    m.def("cached_node_dirs", [](const std::vector<std::string>& cache_dirs, const std::string& url) {
        std::vector<std::string> out;
        for (const auto& d : vcr::zarr::cachedNodeDirs(cache_dirs, url))
            out.push_back(d.string());
        return out;
    }, "cache_dirs"_a, "url"_a);
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
    m.def("store_metadata_json",
          [](const std::string& location, std::function<std::optional<std::string>(const std::string&, const std::string&)> read) {
              const auto m = vcr::zarr::storeMetadata(location, read);
              const auto optional = [](const auto& value) { return value ? vcr::zarr::Json(*value) : vcr::zarr::Json(); };
              return vcr::zarr::Json{{"voxel_size_um", optional(m.voxelSizeUm)},
                                     {"energy_kev", optional(m.energyKeV)},
                                     {"detector_distance_mm", optional(m.detectorDistanceMm)},
                                     {"beamline", m.beamline},
                                     {"scan_name", m.scanName},
                                     {"acquisition_date", m.acquisitionDate},
                                     {"value_window", optional(m.valueWindow)},
                                     {"z_crop", optional(m.zCrop)},
                                     {"documents", m.documents}}.dump();
          },
          "location"_a, "read"_a,
          "What the store's sidecar documents say, as JSON; read(location, name) returns a document's "
          "text or None.");
    m.def("vc3d_cache_root", []() -> std::optional<std::string> {
        const auto root = vcr::zarr::vc3dCacheRoot();
        return root ? std::optional<std::string>(root->string()) : std::nullopt;
    });
    m.def("disk_budget", &vcr::zarr::diskBudget, "root"_a, "max_bytes"_a.none(),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("atomic_write", [](const std::string& path, nb::bytes data) {
        vcr::zarr::atomicWrite(path, std::string(data.c_str(), data.size()));
    }, "path"_a, "data"_a);

    nb::class_<vcr::DiskCache>(m, "DiskCache")
        .def(nb::init<const std::string&, std::int64_t>(), "root"_a, "max_bytes"_a,
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_max_bytes", &vcr::DiskCache::set_max_bytes, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("max_bytes", &vcr::DiskCache::max_bytes, nb::call_guard<nb::gil_scoped_release>())
        .def("open_file", [](vcr::DiskCache& cache, const std::string& path, const std::string& url,
                              const AuthTuple& auth) {
            nb::gil_scoped_release release;
            return cache.open_file(path, url, to_auth(auth));
        }, "path"_a, "url"_a, "auth"_a);

    nb::class_<vcr::Array>(m, "Array")
        .def(
            "__init__",
            [](vcr::Array* self, vcr::Index shape, vcr::Index chunks, std::size_t itemsize, nb::bytes fill,
               std::string separator, std::string identity, std::vector<LayerTuple> layers, int write_layer,
               std::string remote_url, std::string remote_codec, AuthTuple auth, bool offline,
               std::string read_mode, bool hint, std::size_t threads, int connections, long stall_timeout_s, int retries,
               std::shared_ptr<vcr::MemoryCache> cache, std::size_t writer_threads,
               std::size_t max_pending_write_bytes, std::shared_ptr<vcr::DiskCache> disk_cache,
               std::string source_root, std::string source_prefix, vcr::Index shard_chunks,
               bool shard_index_end, bool shard_index_crc) {
                vcr::ArraySpec spec;
                spec.shape = std::move(shape);
                spec.chunks = std::move(chunks);
                spec.itemsize = itemsize;
                spec.fill.assign(static_cast<const std::uint8_t*>(fill.data()),
                                 static_cast<const std::uint8_t*>(fill.data()) + fill.size());
                spec.separator = std::move(separator);
                spec.identity = std::move(identity);
                for (auto& [kind, root, codec, empty_root, allow_lossy] : layers)
                    spec.layers.push_back({kind, root, codec, empty_root, allow_lossy});
                spec.write_layer = write_layer;
                spec.remote_url = std::move(remote_url);
                spec.remote_codec = std::move(remote_codec);
                spec.source_root = std::move(source_root);
                spec.source_prefix = std::move(source_prefix);
                spec.shard_chunks = std::move(shard_chunks);
                spec.shard_index_end = shard_index_end;
                spec.shard_index_crc = shard_index_crc;
                spec.offline = offline;
                spec.http.auth = to_auth(auth);
                spec.read_mode = std::move(read_mode);
                spec.hint = hint;
                spec.threads = threads;
                spec.writer_threads = writer_threads;
                spec.max_pending_write_bytes = max_pending_write_bytes;
                spec.disk_cache = std::move(disk_cache);
                spec.http.connections = connections;
                spec.http.stall_timeout_s = stall_timeout_s;
                spec.http.retries = retries;
                new (self) vcr::Array(std::move(spec), std::move(cache));
            },
            "shape"_a, "chunks"_a, "itemsize"_a, "fill"_a, "separator"_a, "identity"_a, "layers"_a,
            "write_layer"_a = -1, "remote_url"_a = "", "remote_codec"_a = "", "auth"_a = AuthTuple{},
            "offline"_a = false, "read_mode"_a = "slab",
            "hint"_a = false, "threads"_a = 16, "connections"_a = 16, "stall_timeout_s"_a = 10,
            "retries"_a = 5, "cache"_a.none() = nb::none(), "writer_threads"_a = 4,
            "max_pending_write_bytes"_a = std::size_t{1} << 30, "disk_cache"_a.none() = nb::none(),
            "source_root"_a = "", "source_prefix"_a = "", "shard_chunks"_a = vcr::Index{},
            "shard_index_end"_a = true, "shard_index_crc"_a = false)
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
        .def("set_auth", [](vcr::Array& a, const AuthTuple& auth) { a.set_auth(to_auth(auth)); }, "auth"_a,
             "Credentials for later fetches.")
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
