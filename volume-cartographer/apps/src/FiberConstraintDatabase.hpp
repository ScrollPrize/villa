#pragma once
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <vector>

nlohmann::json writeFiberConstraintDatabase(
    const std::filesystem::path& output, const std::vector<int64_t>& sourceIds,
    const vc::fiber_tracer::FiberTraceConstraintReport& report,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads,
    const nlohmann::json& metadata);

void verifyFiberConstraintDatabaseMeasurements(const std::filesystem::path& path,
    const vc::fiber_tracer::FiberTraceConstraintReport& report);
