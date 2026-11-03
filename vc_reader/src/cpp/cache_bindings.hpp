#pragma once
#include <vc_reader/cache.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace vcr::cache::python {
inline void bind(nanobind::module_& m)
{
    namespace nb = nanobind;
    using namespace nb::literals;
    nb::class_<std::atomic<bool>>(m, "CacheCancellation")
        .def(nb::init<bool>(), "cancelled"_a = false)
        .def("cancel", [](std::atomic<bool>& flag) { flag.store(true); })
        .def("reset", [](std::atomic<bool>& flag) { flag.store(false); });
    nb::enum_<PrefillResult::Status>(m, "PrefillStatus")
        .value("Completed", PrefillResult::Status::Completed)
        .value("Cancelled", PrefillResult::Status::Cancelled)
        .value("Failed", PrefillResult::Status::Failed);
    nb::class_<PrefillResult>(m, "PrefillResult")
        .def_ro("status", &PrefillResult::status)
        .def_ro("total_chunks", &PrefillResult::totalChunks)
        .def_ro("resolved_chunks", &PrefillResult::resolvedChunks)
        .def_ro("data_chunks", &PrefillResult::dataChunks)
        .def_ro("empty_chunks", &PrefillResult::emptyChunks)
        .def_ro("error_chunks", &PrefillResult::errorChunks)
        .def_ro("fetched_chunks", &PrefillResult::fetchedChunks)
        .def_ro("message", &PrefillResult::message);
    nb::class_<DownloadProgress>(m, "DownloadProgress")
        .def_ro("total_files", &DownloadProgress::totalFiles)
        .def_ro("completed_files", &DownloadProgress::completedFiles)
        .def_ro("failed_files", &DownloadProgress::failedFiles)
        .def_ro("total_bytes", &DownloadProgress::totalBytes)
        .def_ro("completed_bytes", &DownloadProgress::completedBytes)
        .def_ro("file_name", &DownloadProgress::fileName)
        .def_ro("status", &DownloadProgress::status);
    nb::class_<DownloadResult>(m, "DownloadResult")
        .def_ro("progress", &DownloadResult::progress)
        .def_ro("cancelled", &DownloadResult::cancelled)
        .def_ro("error", &DownloadResult::error);
    m.def("prefill", nb::overload_cast<Array&, const std::atomic<bool>*, const PrefillProgress&>(&prefill),
          "array"_a, "cancel"_a.none() = nullptr, "progress"_a.none() = PrefillProgress{},
          nb::call_guard<nb::gil_scoped_release>());
    m.def("prefill_chunks", nb::overload_cast<Array&, const std::vector<Index>&, const std::atomic<bool>*,
                                              const PrefillProgress&>(&prefill),
          "array"_a, "chunks"_a, "cancel"_a.none() = nullptr, "progress"_a.none() = PrefillProgress{},
          nb::call_guard<nb::gil_scoped_release>(),
          "Prefill only the listed chunk-grid indices (e.g. Array.offline_misses()).");
    m.def("download_prefix", &downloadPrefix, "location"_a, "options"_a,
          "workers"_a = 16, "progress"_a.none() = DownloadCallback{}, "cancel"_a.none() = nullptr,
          nb::call_guard<nb::gil_scoped_release>());
}
} // namespace vcr::cache::python
