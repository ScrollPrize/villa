#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <utility>
#include <limits>

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

// Greedy facility-location medoids: cover observations, not just farthest outliers.
// Stable earliest-index ties; bounded bank makes the O(K*N*N) selection explicit.
inline std::vector<size_t> selectFiberEmbeddingAnchors(const FiberEmbeddingHistory& bank, size_t count)
{
    if (count < 1 || count > 8 || bank.empty() || bank.size() > 64)
        throw std::invalid_argument("invalid embedding anchor bank/count");
    const size_t n = bank.size();
    std::vector<double> distances(n*n), nearest(n, 2.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            distances[i*n+j] = 1 - fiberEmbeddingCosine(bank[i].vector, bank[j].vector);
    std::vector<size_t> selected;
    for (size_t k = 0; k < std::min(count, n); ++k) {
        size_t best = n;
        double bestGain = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::find(selected.begin(), selected.end(), i) != selected.end()) continue;
            double gain = 0;
            for (size_t j = 0; j < n; ++j) gain += std::max(0.0, nearest[j]-distances[i*n+j]);
            if (gain > bestGain) { bestGain = gain; best = i; }
        }
        if (best == n) break;
        selected.push_back(best);
        for (size_t j = 0; j < n; ++j) nearest[j] = std::min(nearest[j], distances[best*n+j]);
    }
    return selected;
}

inline double fiberEmbeddingAnchorCost(const std::vector<float>& vector,
    const FiberEmbeddingHistory& bank, const std::vector<size_t>& anchors)
{
    if (anchors.empty()) throw std::invalid_argument("empty embedding anchors");
    double cost = 2;
    for (size_t i : anchors) cost = std::min(cost, 1-fiberEmbeddingCosine(vector, bank.at(i).vector));
    return cost;
}

inline void advanceFiberEmbeddingAnchors(FiberEmbeddingHistory& bank, std::vector<size_t>& anchors,
    const std::vector<float>& vector, double arc, double spacing, double window, size_t bankSize, size_t count)
{
    const auto size = bank.size();
    const double first = bank.empty() ? -1 : bank.front().arcBase;
    const double last = bank.empty() ? -1 : bank.back().arcBase;
    advanceFiberEmbeddingHistory(bank, vector, arc, spacing, window, bankSize);
    if (bank.size() != size || bank.empty() || bank.front().arcBase != first || bank.back().arcBase != last)
        anchors = selectFiberEmbeddingAnchors(bank, count);
}

struct FiberEmbeddingLru {
    FiberEmbeddingHistory entries;
    std::vector<double> lastUsed;
    double lastAdmission = 0;
};

inline void advanceFiberEmbeddingLru(FiberEmbeddingLru& state, const std::vector<float>& vector,
    double arc, double spacing, size_t capacity, double novelty = 0)
{
    if (capacity < 1 || capacity > 8 || vector.empty() || !std::isfinite(arc) ||
        !std::isfinite(spacing) || spacing <= 0 || !std::isfinite(novelty) || novelty < 0 || novelty > 2)
        throw std::invalid_argument("invalid embedding LRU admission");
    if (!state.entries.empty() && arc-state.lastAdmission < spacing) return;
    state.lastAdmission = arc;
    if (!state.entries.empty()) {
        size_t nearest = 0;
        double closest = -2;
        for (size_t i=0; i<state.entries.size(); ++i) {
            const double cosine = fiberEmbeddingCosine(vector,state.entries[i].vector);
            if (cosine > closest) { closest = cosine; nearest = i; }
        }
        state.lastUsed[nearest] = arc;
        if (state.entries[nearest].vector == vector || 1-closest < novelty) return;
        if (state.entries.size() >= capacity) {
            const size_t oldest = std::min_element(state.lastUsed.begin(),state.lastUsed.end())-state.lastUsed.begin();
            state.entries.erase(state.entries.begin()+oldest);
            state.lastUsed.erase(state.lastUsed.begin()+oldest);
        }
    }
    state.entries.push_back({arc,vector});
    state.lastUsed.push_back(arc);
}

inline double fiberEmbeddingLruCost(const std::vector<float>& vector, const FiberEmbeddingLru& state, bool average)
{
    if (state.entries.empty()) throw std::invalid_argument("empty embedding LRU state");
    // LRU has no age expiry: reuse the shared normalized history aggregation.
    return fiberEmbeddingCost(vector,state.entries,state.lastAdmission,
        std::numeric_limits<double>::infinity(),"uniform",!average);
}
// Streaming spherical clustering: retain all admitted mass, never expire by age.
struct FiberEmbeddingCluster {
    std::vector<double> sum;
    std::vector<float> centroid;
    size_t support = 1;
};
struct FiberEmbeddingAppearance {
    std::vector<FiberEmbeddingCluster> clusters;
    double lastAdmission = 0;
};

inline double fiberEmbeddingSumNorm(const std::vector<double>& sum)
{
    double square = 0;
    for (double v : sum) square += v*v;
    return std::sqrt(square);
}

inline void advanceFiberEmbeddingAppearance(FiberEmbeddingAppearance& state,
    const std::vector<float>& vector, double arc, double spacing, size_t capacity)
{
    if (capacity < 1 || capacity > 8 || vector.empty() || !std::isfinite(arc) ||
        !std::isfinite(spacing) || spacing <= 0 ||
        std::any_of(vector.begin(),vector.end(),[](float v){return !std::isfinite(v);}))
        throw std::invalid_argument("invalid embedding appearance admission");
    if (!state.clusters.empty() && state.clusters.front().sum.size() != vector.size())
        throw std::invalid_argument("inconsistent embedding appearance dimensions");
    if (!state.clusters.empty() && arc-state.lastAdmission < spacing) return;
    state.lastAdmission = arc;
    state.clusters.push_back({std::vector<double>(vector.begin(),vector.end()),vector,1});
    if (state.clusters.size() <= capacity) return;
    size_t a = 0, b = 1;
    double best = std::numeric_limits<double>::infinity();
    std::vector<double> merged(vector.size());
    for (size_t i=0; i<state.clusters.size(); ++i)
        for (size_t j=i+1; j<state.clusters.size(); ++j) {
            for (size_t d=0; d<merged.size(); ++d)
                merged[d] = state.clusters[i].sum[d]+state.clusters[j].sum[d];
            const double increase = fiberEmbeddingSumNorm(state.clusters[i].sum) +
                fiberEmbeddingSumNorm(state.clusters[j].sum)-fiberEmbeddingSumNorm(merged);
            if (increase < best) { best = increase; a = i; b = j; }
        }
    auto& target = state.clusters[a];
    for (size_t d=0; d<target.sum.size(); ++d) target.sum[d] += state.clusters[b].sum[d];
    target.support += state.clusters[b].support;
    const double norm = fiberEmbeddingSumNorm(target.sum);
    // Cancellation retains a valid representative but does not change true sums.
    if (norm > 0)
        for (size_t d=0; d<target.sum.size(); ++d) target.centroid[d] = static_cast<float>(target.sum[d]/norm);
    state.clusters.erase(state.clusters.begin()+b);
}

inline double fiberEmbeddingAppearanceCost(const std::vector<float>& vector,
    const FiberEmbeddingAppearance& state, double temperature = 0)
{
    if (state.clusters.empty()) throw std::invalid_argument("empty embedding appearance");
    if (!std::isfinite(temperature) || temperature < 0)
        throw std::invalid_argument("invalid embedding appearance temperature");
    double cost = 2;
    for (const auto& cluster : state.clusters)
        cost = std::min(cost,1-fiberEmbeddingCosine(vector,cluster.centroid));
    if (temperature > 0) {
        double mass = 0, likelihood = 0;
        for (const auto& cluster : state.clusters) {
            if (!cluster.support) throw std::invalid_argument("zero embedding cluster support");
            mass += cluster.support;
            const double exponent = -(1-fiberEmbeddingCosine(vector,cluster.centroid)-cost)/temperature;
            likelihood += cluster.support * (temperature >= 1 ? std::expm1(exponent) : std::exp(exponent));
        }
        const double logMean = temperature >= 1 ? std::log1p(likelihood/mass) : std::log(likelihood/mass);
        return std::clamp(cost-temperature*logMean,0.0,2.0);
    }
    return cost;
}
} // namespace vc::fiber_tracer
