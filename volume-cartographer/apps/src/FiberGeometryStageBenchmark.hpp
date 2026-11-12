#pragma once
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include <nlohmann/json.hpp>

// Invoked AFTER the production catalogue/transform/geometry loader, before
// constructing any prediction samplers or importing output graph records.
nlohmann::json benchmarkFiberGeometryStage(
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>&,
    vc::fiber_tracer::FiberTraceConstraintConfig,
    const nlohmann::json&);
