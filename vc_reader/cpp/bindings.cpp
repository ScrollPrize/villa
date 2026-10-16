// Python bindings for the native read path (module vc_reader._native).
#include "array.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <functional>
#include <numeric>

namespace nb = nanobind;
using namespace nb::literals;

using Bytes = nb::ndarray<std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using LayerTuple = std::tuple<std::string, std::string, std::string, std::string, bool>;

NB_MODULE(_native, m)
{
    m.doc() = "Native zarr v2 read path: local and HTTP chunks, disk cache layers, memory cache.";

    m.def(
        "http_get",
        [](const std::string& url, long stall_timeout_s, int retries) -> nb::object {
            std::vector<std::uint8_t> body;
            bool found;
            {
                nb::gil_scoped_release release;
                vcr::HttpOptions options;
                options.connections = 1;
                options.stall_timeout_s = stall_timeout_s;
                options.retries = retries;
                vcr::HttpClient client(options);
                found = client.get(url, body);
            }
            if (!found)
                return nb::none();
            return nb::bytes(reinterpret_cast<const char*>(body.data()), body.size());
        },
        "url"_a, "stall_timeout_s"_a = 10, "retries"_a = 5,
        "GET a small object (e.g. .zarray); None if it does not exist (404).");

    nb::class_<vcr::MemoryCache>(m, "MemoryCache", "Byte-bounded LRU of decoded chunks, shareable between arrays.")
        .def(nb::init<std::size_t>(), "max_bytes"_a)
        .def_prop_ro("max_bytes", &vcr::MemoryCache::max_bytes)
        .def_prop_ro("nbytes", &vcr::MemoryCache::nbytes)
        .def_prop_ro("hits", &vcr::MemoryCache::hits)
        .def_prop_ro("misses", &vcr::MemoryCache::misses)
        .def("__len__", &vcr::MemoryCache::size)
        .def("clear", &vcr::MemoryCache::clear);

    nb::class_<vcr::Array>(m, "Array")
        .def(
            "__init__",
            [](vcr::Array* self, vcr::Index shape, vcr::Index chunks, std::size_t itemsize, nb::bytes fill,
               std::string separator, std::string identity, std::vector<LayerTuple> layers, int write_layer,
               std::string remote_url, std::string remote_codec, std::string read_mode, bool hint,
               std::size_t threads, int connections, long stall_timeout_s, int retries,
               std::shared_ptr<vcr::MemoryCache> cache) {
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
                spec.read_mode = std::move(read_mode);
                spec.hint = hint;
                spec.threads = threads;
                spec.http.connections = connections;
                spec.http.stall_timeout_s = stall_timeout_s;
                spec.http.retries = retries;
                new (self) vcr::Array(std::move(spec), std::move(cache));
            },
            "shape"_a, "chunks"_a, "itemsize"_a, "fill"_a, "separator"_a, "identity"_a, "layers"_a,
            "write_layer"_a = -1, "remote_url"_a = "", "remote_codec"_a = "", "read_mode"_a = "slab",
            "hint"_a = false, "threads"_a = 16, "connections"_a = 16, "stall_timeout_s"_a = 10,
            "retries"_a = 5, "cache"_a.none() = nb::none())
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
        .def("stats", [](vcr::Array& a) {
            const auto& s = a.stats();
            nb::dict d;
            d["memory_hits"] = s.memory_hits.load();
            d["disk_hits"] = s.disk_hits.load();
            d["remote_fetches"] = s.remote_fetches.load();
            d["remote_missing"] = s.remote_missing.load();
            d["remote_bytes"] = s.remote_bytes.load();
            d["write_errors"] = s.write_errors.load();
            d["last_write_error"] = a.last_write_error();
            return d;
        });
}
