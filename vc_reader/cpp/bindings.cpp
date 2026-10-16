// Python bindings for the native read path (module vc_reader._native).
#include "array.hpp"
#include "output_pool.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <functional>
#include <numeric>

namespace nb = nanobind;
using namespace nb::literals;

using Bytes = nb::ndarray<std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using LayerTuple = std::tuple<std::string, std::string, std::string, std::string, bool>;
// (kind, user or access key id, password or secret key, session token, region); see vcr::Auth.
using AuthTuple = std::tuple<std::string, std::string, std::string, std::string, std::string>;

static vcr::Auth to_auth(const AuthTuple& t)
{
    auto& [kind, user, secret, token, region] = t;
    return vcr::Auth{kind, user, secret, token, region};
}

NB_MODULE(_native, m)
{
    m.doc() = "Native zarr v2 read path: local and HTTP chunks, disk cache layers, memory cache.";
    // Engines inherited by a forked process are deliberately kept alive and never
    // destroyed there (their threads do not exist in the child), so they are
    // still referenced at exit; that is not a leak worth reporting.
    nb::set_leak_warnings(false);

    // Native errors with a Python meaning: offline cache misses and refused requests.
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

    m.def(
        "sigv4_headers",
        [](const std::string& url, const std::string& access_key, const std::string& secret_key,
           const std::string& session_token, const std::string& region, const std::string& amz_date,
           const std::vector<std::pair<std::string, std::string>>& extra) {
            return vcr::sigv4_get_headers(url, {access_key, secret_key, session_token, region}, amz_date, extra);
        },
        "url"_a, "access_key"_a, "secret_key"_a, "session_token"_a, "region"_a, "amz_date"_a,
        "extra"_a = std::vector<std::pair<std::string, std::string>>{},
        "The headers that sign a GET of url for S3 at amz_date (YYYYMMDDTHHMMSSZ); for tests.");

    nb::class_<vcr::MemoryCache>(m, "MemoryCache", "Byte-bounded LRU of decoded chunks, shareable between arrays.")
        .def(nb::init<std::size_t>(), "max_bytes"_a)
        .def_prop_ro("max_bytes", &vcr::MemoryCache::max_bytes)
        .def_prop_ro("nbytes", &vcr::MemoryCache::nbytes)
        .def_prop_ro("hits", &vcr::MemoryCache::hits)
        .def_prop_ro("misses", &vcr::MemoryCache::misses)
        .def("__len__", &vcr::MemoryCache::size)
        .def("clear", &vcr::MemoryCache::clear);

    nb::class_<vcr::DiskCache>(m, "DiskCache")
        .def(nb::init<const std::string&, std::int64_t>(), "root"_a, "max_bytes"_a,
             nb::call_guard<nb::gil_scoped_release>())
        .def("set_max_bytes", &vcr::DiskCache::set_max_bytes, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<vcr::Array>(m, "Array")
        .def(
            "__init__",
            [](vcr::Array* self, vcr::Index shape, vcr::Index chunks, std::size_t itemsize, nb::bytes fill,
               std::string separator, std::string identity, std::vector<LayerTuple> layers, int write_layer,
               std::string remote_url, std::string remote_codec, AuthTuple auth, bool offline,
               std::string read_mode, bool hint, std::size_t threads, int connections, long stall_timeout_s, int retries,
               std::shared_ptr<vcr::MemoryCache> cache, std::size_t writer_threads,
               std::size_t max_pending_write_bytes, std::shared_ptr<vcr::DiskCache> disk_cache) {
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
            "max_pending_write_bytes"_a = std::size_t{1} << 30, "disk_cache"_a.none() = nb::none())
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
                const char kind = kind_name.size() == 1 ? kind_name[0] : '?';
                if (static_cast<std::size_t>(out.size()) != coords.shape(0) * (out64 ? 8 : 4))
                    throw nb::value_error("the output buffer does not match the points");
                if (coords.shape(1) != a.ndim())
                    throw nb::value_error("points must have one coordinate per array dimension");
                const bool ok = (kind == 'u' || kind == 'i') ? (itemsize == 1 || itemsize == 2 || itemsize == 4 || itemsize == 8)
                                : kind == 'f'                 ? (itemsize == 4 || itemsize == 8)
                                                              : (kind == 'b' && itemsize == 1);
                if (!ok)
                    throw nb::type_error("sample() needs a bool, integer, float32 or float64 array");
                nb::gil_scoped_release release;
                a.sample_linear(coords.data(), coords.shape(0), vcr::ElementType{kind, itemsize, swap}, out64,
                                out.data());
            },
            "coords"_a, "out"_a, "kind"_a, "itemsize"_a, "swap"_a, "out64"_a,
            "N-linear interpolation at (n, ndim) float64 points into float32/float64 outputs (the GIL is released).")
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
