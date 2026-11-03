#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace vc::fiber_tracer {

struct FiberEmbeddingReference {
    double arcBase;
    std::vector<float> vector;
};
using FiberEmbeddingHistory = std::vector<FiberEmbeddingReference>;

// Inputs are unit vectors normalized after interpolation, not at each corner.
inline double fiberEmbeddingCosine(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.empty() || a.size() != b.size())
        throw std::invalid_argument("invalid embedding dimensions");
    double dot = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            throw std::invalid_argument("nonfinite embedding");
        dot += double(a[i]) * b[i];
    }
    return std::clamp(dot, -1.0, 1.0);
}

inline double fiberEmbeddingCost(const std::vector<float>& vector,
    const FiberEmbeddingHistory& history, double arcBase, double window,
    std::string_view falloff = "uniform", bool best = false)
{
    if (!std::isfinite(arcBase) || !std::isfinite(window) || window <= 0)
        throw std::invalid_argument("invalid embedding history window/arc");
    if (falloff != "uniform" && falloff != "linear" && falloff != "gaussian")
        throw std::invalid_argument("unknown embedding history falloff");
    if (best && falloff != "uniform")
        throw std::invalid_argument("best history requires uniform falloff");
    double cost = 0;
    double weightSum = 0;
    double minimum = 2;
    for (const auto& reference : history) {
        if (arcBase - reference.arcBase > window) continue;
        const double relative = (arcBase - reference.arcBase) / window;
        const double weight = falloff == "linear" ? std::max(0.0, 1.0-relative) :
            falloff == "gaussian" ? std::exp(-2.0*relative*relative) : 1.0;
        const double cosine = fiberEmbeddingCosine(vector, reference.vector);
        minimum = std::min(minimum, 1.0 - cosine);
        cost += weight * (1.0 - cosine);
        weightSum += weight;
    }
    return weightSum ? (best ? minimum : cost / weightSum) : 0.0;
}

inline void advanceFiberEmbeddingHistory(FiberEmbeddingHistory& history,
    const std::vector<float>& vector, double arcBase, double spacing, double window, size_t count)
{
    if (!std::isfinite(arcBase) || !std::isfinite(spacing) || spacing <= 0 ||
        !std::isfinite(window) || window < spacing || count == 0 || count > 64 ||
        (!history.empty() && arcBase < history.back().arcBase))
        throw std::invalid_argument("invalid embedding history update");
    if (!vector.empty()) {
        (void)fiberEmbeddingCosine(vector, vector);
        if (!history.empty() && vector.size() != history.back().vector.size())
            throw std::invalid_argument("embedding dimensions changed");
    }
    std::erase_if(history, [&](const auto& r) { return arcBase - r.arcBase > window; });
    if (!vector.empty() && (history.empty() || arcBase - history.back().arcBase >= spacing))
        history.push_back({arcBase, vector});
    if (history.size() > count) history.erase(history.begin(), history.end() - count);
}

} // namespace vc::fiber_tracer
