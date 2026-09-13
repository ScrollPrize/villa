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

// Unit vectors enter here after interpolation, never normalize each corner.
inline double fiberEmbeddingCost(const std::vector<float>& vector,
    const FiberEmbeddingHistory& history, double arcBase, double window,
    std::string_view falloff = "uniform")
{
    if (falloff != "uniform" && falloff != "linear" && falloff != "gaussian")
        throw std::invalid_argument("unknown embedding history falloff");
    double cost = 0;
    double weightSum = 0;
    for (const auto& reference : history) {
        if (arcBase - reference.arcBase > window) continue;
        const double relative = (arcBase - reference.arcBase) / window;
        const double weight = falloff == "linear" ? std::max(0.0, 1.0-relative) :
            falloff == "gaussian" ? std::exp(-2.0*relative*relative) : 1.0;
        if (reference.vector.size() != vector.size()) throw std::invalid_argument("embedding dimensions differ");
        double cosine = 0;
        for (size_t i = 0; i < vector.size(); ++i) cosine += double(vector[i]) * reference.vector[i];
        cost += weight * (1.0 - std::clamp(cosine, -1.0, 1.0));
        weightSum += weight;
    }
    return weightSum ? cost / weightSum : 0.0;
}

inline void advanceFiberEmbeddingHistory(FiberEmbeddingHistory& history,
    const std::vector<float>& vector, double arcBase, double spacing, double window, size_t count)
{
    std::erase_if(history, [&](const auto& r) { return arcBase - r.arcBase > window; });
    if (!vector.empty() && (history.empty() || arcBase - history.back().arcBase >= spacing))
        history.push_back({arcBase, vector});
    if (history.size() > count) history.erase(history.begin(), history.end() - count);
}
} // namespace vc::fiber_tracer
