#pragma once
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/core/util/Umbilicus.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <stop_token>
#include <span>
#include <vc_reader/affine.hpp>

struct FiberConstraintDatabaseOptions {
    int workers=32;
    bool packed=true;
    // One inverse volume mapping per source fiber. Empty explicitly declares
    // identity. Core anchors use stored geometry coordinates; measured points
    // remain in base XYZ for the unchanged scientific calculations.
    std::vector<vcr::catalog::Affine> baseToStored;
};

// Called only by the encoding producer. Empty means successful end of stream;
// failures must throw. Blocking sources must honor cancellation on writer failure.
using FiberConstraintBatchSource = std::function<
    std::vector<vc::fiber_tracer::FiberTraceConstraint>(std::stop_token)>;

nlohmann::json streamFiberConstraintDatabase(
    const std::filesystem::path& output, const std::vector<std::string>& sourceIds,
    std::span<const vc::fiber_tracer::FiberTraceConstraintPiece> pieces,
    const FiberConstraintBatchSource& source,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads,
    const nlohmann::json& metadata,
    const vc::fiber_tracer::FiberTraceConstraintProgressCallback& progress = {},
    const FiberConstraintDatabaseOptions& options = {}, std::size_t expectedEvents = 0);

nlohmann::json writeFiberConstraintDatabase(
    const std::filesystem::path& output, const std::vector<std::string>& sourceIds,
    const vc::fiber_tracer::FiberTraceConstraintReport& report,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads,
    const nlohmann::json& metadata,
    const vc::fiber_tracer::FiberTraceConstraintProgressCallback& progress = {},
    const FiberConstraintDatabaseOptions& options = {});

void verifyFiberConstraintDatabaseMeasurements(const std::filesystem::path& path,
    const vc::fiber_tracer::FiberTraceConstraintReport& report);

nlohmann::json orientFiberConstraintDatabase(
    const std::filesystem::path& input, const std::filesystem::path& output,
    const std::vector<std::string>& sourceIds,
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads,
    double sampleSpacing, const nlohmann::json& provenance,
    const vc::core::util::Umbilicus* priorUmbilicus = nullptr,
    bool exportNormalCurves = false, bool selectedNormalCurves = false);
