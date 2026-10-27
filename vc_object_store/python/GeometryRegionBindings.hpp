#pragma once
#include "StorageBindings.hpp"
#include "vc_object_store/objects/GeometryRegion.hpp"

namespace vc::python {
struct RegionCancellation { std::atomic_bool cancelled{false}; };
inline void bindGeometryRegion(nb::module_& m)
{
    nb::class_<RegionCancellation>(m, "RegionCancellation")
        .def(nb::init<>())
        .def("cancel", [](RegionCancellation& self) { self.cancelled.store(true); });
    nb::class_<vo::RegionSlice>(m, "RegionSlice")
        .def(nb::init<>())
        .def_rw("origin", &vo::RegionSlice::origin)
        .def_rw("u", &vo::RegionSlice::u)
        .def_rw("v", &vo::RegionSlice::v)
        .def_rw("normal", &vo::RegionSlice::normal)
        .def_rw("local_bounds", &vo::RegionSlice::localBounds);
    nb::class_<vo::RegionOptions>(m, "RegionOptions")
        .def(nb::init<>())
        .def_rw("kind", &vo::RegionOptions::kind)
        .def_rw("uuids", &vo::RegionOptions::uuids)
        .def_rw("maximum_objects", &vo::RegionOptions::maximumObjects)
        .def_rw("selected", &vo::RegionOptions::selected)
        .def_rw("family", &vo::RegionOptions::family)
        .def_rw("min_length", &vo::RegionOptions::minLength)
        .def_rw("min_points", &vo::RegionOptions::minPoints)
        .def_rw("output_byte_budget", &vo::RegionOptions::outputByteBudget)
        .def_rw("processing_byte_budget", &vo::RegionOptions::processingByteBudget)
        .def_rw("candidate_work_budget", &vo::RegionOptions::candidateWorkBudget)
        .def_rw("sql_step_budget", &vo::RegionOptions::sqlStepBudget)
        .def_rw("max_unit_bytes", &vo::RegionOptions::maxUnitBytes)
        .def_rw("coordinates", &vo::RegionOptions::coordinates)
        .def_rw("volume", &vo::RegionOptions::volume)
        .def_rw("target_scale", &vo::RegionOptions::targetScale);
}
inline nb::dict geometryRegionFromPy(PyObjectStore& self, const vo::RegionSlice& slice,
                                     const vo::RegionOptions& options, RegionCancellation* cancel)
{
    // Copy mutable Python-owned options before releasing the GIL.
    const auto view = slice;
    const auto request = options;
    vo::RegionResult result;
    { auto lease = self.db; self.require(); nb::gil_scoped_release release;
      result = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) {
          return db.geometryRegion(txn, view, request, [cancel](const auto&) { return cancel && cancel->cancelled.load(); });
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
    output["stop"] = vo::regionStopName(result.stop);
    output["complete"] = result.complete();
    output["intersecting_objects"] = result.intersectingObjects;
    return output;
}
}
