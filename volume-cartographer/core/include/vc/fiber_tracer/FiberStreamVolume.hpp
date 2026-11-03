#pragma once
#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include <atomic>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace vc::fiber_tracer {
// All hot-path scheduling, collision/CT decisions, seed admission and output
// assembly are native. Python may supply metadata and poll cancellation only.
nlohmann::json runFiberStreamVolume(const FiberPredictionSource&,
    const vc::lasagna::NormalSampler&,const FiberTraceConfig&,const FiberChunkTraceConfig&,
    const nlohmann::json& options,const std::vector<std::array<int,3>>& keys,
    const std::filesystem::path& output,std::atomic_bool& cancel);
} // namespace vc::fiber_tracer
