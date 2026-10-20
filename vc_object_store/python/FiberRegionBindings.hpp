#pragma once
#include "StorageBindings.hpp"
#include "vc_object_store/objects/FiberRegion.hpp"

namespace vc::python {
struct FiberRegionCancellation { std::atomic_bool cancelled{false}; };
inline void bindFiberRegion(nb::module_& m)
{
    nb::class_<FiberRegionCancellation>(m, "FiberRegionCancellation")
        .def(nb::init<>())
        .def("cancel", [](FiberRegionCancellation& self) { self.cancelled.store(true); });
    nb::class_<vo::FiberSlice>(m, "FiberSlice")
        .def(nb::init<>())
        .def_rw("origin", &vo::FiberSlice::origin)
        .def_rw("u", &vo::FiberSlice::u)
        .def_rw("v", &vo::FiberSlice::v)
        .def_rw("normal", &vo::FiberSlice::normal)
        .def_rw("local_bounds", &vo::FiberSlice::localBounds);
    nb::class_<vo::FiberRegionOptions>(m, "FiberRegionOptions")
        .def(nb::init<>())
        .def_rw("maximum_fibers", &vo::FiberRegionOptions::maximumFibers)
        .def_rw("selected", &vo::FiberRegionOptions::selected)
        .def_rw("family", &vo::FiberRegionOptions::family)
        .def_rw("min_length", &vo::FiberRegionOptions::minLength)
        .def_rw("min_points", &vo::FiberRegionOptions::minPoints)
        .def_rw("output_byte_budget", &vo::FiberRegionOptions::outputByteBudget)
        .def_rw("processing_byte_budget", &vo::FiberRegionOptions::processingByteBudget)
        .def_rw("candidate_work_budget", &vo::FiberRegionOptions::candidateWorkBudget)
        .def_rw("sql_step_budget", &vo::FiberRegionOptions::sqlStepBudget)
        .def_rw("max_unit_bytes", &vo::FiberRegionOptions::maxUnitBytes);
}
inline nb::dict fiberRegionFromPy(PyObjectStore& self, const vo::FiberSlice& slice,
                                 const vo::FiberRegionOptions& options, FiberRegionCancellation* cancel)
{
    // Copy mutable Python-owned options before releasing the GIL.
    const auto view = slice;
    const auto request = options;
    vo::FiberRegionResult result;
    { auto lease = self.db; self.require(); nb::gil_scoped_release release;
      result = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) {
          return db.fiberRegion(txn, view, request, [cancel](const auto&) { return cancel && cancel->cancelled.load(); });
      }); }
    nb::dict output, stats;
    const auto& s = result.stats;
    stats["output_bytes"] = s.outputBytes;
    stats["processing_bytes"] = s.processingBytes;
    stats["payload_bytes"] = s.payloadBytes;
    stats["verified_bytes"] = s.verifiedBytes;
    stats["metadata_bytes"] = s.metadataBytes;
    stats["reconstructed_bytes"] = s.reconstructedBytes;
    stats["cache_bytes"] = s.cacheBytes;
    stats["candidate_work"] = s.candidateWork;
    stats["candidate_visits"] = s.candidateVisits;
    stats["tree_visits"] = s.treeVisits;
    stats["segment_tests"] = s.segmentTests;
    stats["sql_steps"] = s.sqlSteps;
    nb::list runs;
    for (const auto& run : result.runs) {
        nb::dict item;
        item["object_uuid"] = run.objectUuid; item["revision_uuid"] = run.revisionUuid;
        item["array_name"] = run.arrayName; item["family"] = run.family;
        item["first_segment"] = run.firstSegment; item["points"] = run.points;
        runs.append(item);
    }
    output["runs"] = runs; output["stats"] = stats;
    output["stop"] = vo::fiberRegionStopName(result.stop);
    output["complete"] = result.complete();
    output["intersecting_fibers"] = result.intersectingFibers;
    return output;
}
}
