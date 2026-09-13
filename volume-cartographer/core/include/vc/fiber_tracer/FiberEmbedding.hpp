#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <utility>

namespace vc::fiber_tracer {
struct FiberEmbeddingReference {
    double arcBase;
    std::vector<float> vector;
};
using FiberEmbeddingHistory = std::vector<FiberEmbeddingReference>;

inline double fiberEmbeddingCosine(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.empty() || a.size() != b.size()) throw std::invalid_argument("invalid embedding dimensions");
    double dot = 0;
    for (size_t i = 0; i < a.size(); ++i) dot += double(a[i]) * b[i];
    return std::clamp(dot, -1.0, 1.0);
}

struct FiberEmbeddingMemory {
    std::vector<float> ema, seed;
};

inline void advanceFiberEmbeddingMemory(FiberEmbeddingMemory& state, const std::vector<float>& vector,
    double distance, double halfLife, double presence, bool gated)
{
    if (vector.empty() || !std::isfinite(distance) || distance < 0 ||
        !std::isfinite(halfLife) || halfLife <= 0 || !std::isfinite(presence) ||
        std::any_of(vector.begin(), vector.end(), [](float v) { return !std::isfinite(v); }))
        throw std::invalid_argument("invalid embedding EMA update");
    if (state.ema.empty()) { state.ema = state.seed = vector; return; }
    const double cosine = fiberEmbeddingCosine(vector, state.ema);
    if (gated && (presence < .5 || cosine < .5)) return;
    const double decay = std::exp2(-distance / halfLife);
    double norm2 = 0;
    std::vector<float> mixed(vector.size());
    for (size_t i = 0; i < vector.size(); ++i) {
        mixed[i] = static_cast<float>(decay * state.ema[i] + (1-decay) * vector[i]);
        norm2 += double(mixed[i]) * mixed[i];
    }
    // Exact cancellation retains the previous valid reference, never a zero vector.
    if (norm2 == 0) return;
    for (auto& v : mixed) v = static_cast<float>(v / std::sqrt(norm2));
    state.ema = std::move(mixed);
}

inline double fiberEmbeddingMemoryCost(const std::vector<float>& vector,
    const FiberEmbeddingMemory& state, bool withSeed)
{
    const double cost = 1 - fiberEmbeddingCosine(vector, state.ema);
    return withSeed ? .5 * (cost + 1 - fiberEmbeddingCosine(vector, state.seed)) : cost;
}

// Unit vectors enter here after interpolation, never normalize each corner.
inline double fiberEmbeddingCost(const std::vector<float>& vector,
    const FiberEmbeddingHistory& history, double arcBase, double window,
    std::string_view falloff = "uniform", bool best = false)
{
    if (falloff != "uniform" && falloff != "linear" && falloff != "gaussian")
        throw std::invalid_argument("unknown embedding history falloff");
    if (best && falloff != "uniform") throw std::invalid_argument("best history requires uniform falloff");
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
    std::erase_if(history, [&](const auto& r) { return arcBase - r.arcBase > window; });
    if (!vector.empty() && (history.empty() || arcBase - history.back().arcBase >= spacing))
        history.push_back({arcBase, vector});
    if (history.size() > count) history.erase(history.begin(), history.end() - count);
}
} // namespace vc::fiber_tracer
