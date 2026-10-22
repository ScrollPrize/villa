#pragma once
#include "utils/zarr.hpp"

namespace vc::lasagna {
// Metadata-only derived view. Reads original local Zarr payloads synchronously
// inside the caller's native fetch task; never nests work on that task pool.
std::shared_ptr<utils::Store> openPooledPredictionStore(const std::filesystem::path& path);
struct PredictionPoolingStats { uint64_t computations, hits, readNanoseconds, poolNanoseconds, residentBytes; };
PredictionPoolingStats predictionPoolingStats();
}
