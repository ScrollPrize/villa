#pragma once
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/core/util/Umbilicus.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <vector>
#include <stop_token>

struct FiberConstraintDatabaseOptions {
    // Rows per prepared INSERT; bounded to keep statement size portable.
    int insertBatchRows=32;
    bool inMemory = false;
    // Zero retains the SQLite default. Otherwise a power of two from512 to65536.
    int pageSize = 0;
};

// Called only by the encoding producer. Empty means successful end of stream;
// failures must throw. Blocking sources must honor cancellation on writer failure.
using FiberConstraintBatchSource = std::function<
    std::vector<vc::fiber_tracer::FiberTraceConstraint>(std::stop_token)>;

nlohmann::json streamFiberConstraintDatabase(
    const std::filesystem::path& output, const std::vector<int64_t>& sourceIds,
    std::span<const vc::fiber_tracer::FiberTraceConstraintPiece> pieces,
    const FiberConstraintBatchSource& source,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads,
    const nlohmann::json& metadata,
    const vc::fiber_tracer::FiberTraceConstraintProgressCallback& progress = {},
    const FiberConstraintDatabaseOptions& options = {}, std::size_t expectedEvents = 0);

nlohmann::json writeFiberConstraintDatabase(
    const std::filesystem::path& output, const std::vector<int64_t>& sourceIds,
    const vc::fiber_tracer::FiberTraceConstraintReport& report,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads,
    const nlohmann::json& metadata,
    const vc::fiber_tracer::FiberTraceConstraintProgressCallback& progress = {},
    const FiberConstraintDatabaseOptions& options = {});

void verifyFiberConstraintDatabaseMeasurements(const std::filesystem::path& path,
    const vc::fiber_tracer::FiberTraceConstraintReport& report);

nlohmann::json orientFiberConstraintDatabase(
    const std::filesystem::path& input, const std::filesystem::path& output,
    const std::vector<int64_t>& sourceIds,
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads,
    double sampleSpacing, const nlohmann::json& provenance,
    const vc::core::util::Umbilicus* priorUmbilicus = nullptr,
    bool exportNormalCurves = false, bool selectedNormalCurves = false);
