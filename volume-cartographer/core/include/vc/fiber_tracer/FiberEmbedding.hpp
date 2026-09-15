#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <string>
#include <vector>
#include <utility>
#include <limits>

namespace vc::fiber_tracer {
inline double fiberEmbeddingDistancePenalty(double distance, double threshold, double temperature)
{
    if (temperature == 0) return distance;
    const auto softplus = [](double x) { return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x))); };
    // Anchored at zero: no per-step constant can bias paths of different lengths.
    return temperature * (softplus((distance-threshold)/temperature) - softplus(-threshold/temperature));
}

struct FiberEmbeddingReference {
    double arcBase;
    std::vector<float> vector;
};
using FiberEmbeddingHistory = std::vector<FiberEmbeddingReference>;
inline double fiberEmbeddingCost(const std::vector<float>&, const FiberEmbeddingHistory&, double, double, std::string_view, bool);
inline void advanceFiberEmbeddingHistory(FiberEmbeddingHistory&, const std::vector<float>&, double, double, double, size_t);

inline double fiberEmbeddingCosine(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.empty() || a.size() != b.size()) throw std::invalid_argument("invalid embedding dimensions");
    double dot = 0;
    for (size_t i = 0; i < a.size(); ++i) dot += double(a[i]) * b[i];
    return std::clamp(dot, -1.0, 1.0);
}

struct FiberEmbeddingMemory {
    std::vector<float> ema, seed;
    // Allocated only by opt-in integration experiments.
    std::vector<float> secondary;
    FiberEmbeddingHistory integrationHistory;
    double integrationArc = 0, integrationElapsed = 0;
};

struct FiberEmbeddingIntegrationConfig {
    std::string kind = "ema";
    double spacing = 0;
    double gateBias = 3, gatePresence = 2, gateDistance = 2;
    double capDegrees = 10;
    double longHalfLife = 1024, mixBias = 0, mixSlope = 2;
    double window = 512;
    bool active() const { return kind != "ema" || spacing != 0; }
};

inline void blendFiberEmbeddingVector(std::vector<float>& reference,
    const std::vector<float>& vector, double decay)
{
    double norm2 = 0;
    std::vector<float> mixed(vector.size());
    for (size_t i = 0; i < vector.size(); ++i) {
        mixed[i] = static_cast<float>(decay * reference[i] + (1-decay) * vector[i]);
        norm2 += double(mixed[i]) * mixed[i];
    }
    if (norm2 == 0) return;
    for (auto& v : mixed) v = static_cast<float>(v / std::sqrt(norm2));
    reference = std::move(mixed);
}

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
    // Exact cancellation retains the previous valid reference, never a zero vector.
    blendFiberEmbeddingVector(state.ema, vector, decay);
}

inline double fiberEmbeddingSigmoid(double x)
{
    return x >= 0 ? 1/(1+std::exp(-x)) : std::exp(x)/(1+std::exp(x));
}

inline double integratedFiberEmbeddingCost(const std::vector<float>& vector,
    const FiberEmbeddingMemory& state, double candidateDistance,
    const FiberEmbeddingIntegrationConfig& config)
{
    const double fast = 1-fiberEmbeddingCosine(vector,state.ema);
    if (config.kind == "dual") {
        const double slow = state.secondary.empty() ? fast : 1-fiberEmbeddingCosine(vector,state.secondary);
        const double mix = fiberEmbeddingSigmoid(config.mixBias+config.mixSlope*(slow-fast));
        return mix*fast+(1-mix)*slow;
    }
    if (config.kind == "window") {
        const double arc = state.integrationArc+candidateDistance;
        if (state.integrationHistory.empty()) return arc <= config.window ? 1-fiberEmbeddingCosine(vector,state.seed) : 0;
        return fiberEmbeddingCost(vector,state.integrationHistory,arc,config.window,"uniform",false);
    }
    return fast;
}

inline void advanceIntegratedFiberEmbedding(FiberEmbeddingMemory& state,
    const std::vector<float>& vector, double distance, double halfLife, double presence,
    const FiberEmbeddingIntegrationConfig& config)
{
    if (config.kind == "window" && state.integrationHistory.empty() && state.integrationArc == 0)
        state.integrationHistory.push_back({0,state.seed});
    state.integrationArc += distance;
    state.integrationElapsed += distance;
    // Admission uses accumulated actual distance. No tolerance or step-count
    // approximation; exporter records these same distances for fitting parity.
    if (state.integrationElapsed < config.spacing) return;
    const double elapsed = std::exchange(state.integrationElapsed,0.0);
    const double decay = std::exp2(-elapsed/halfLife);
    if (config.kind == "window") {
        advanceFiberEmbeddingHistory(state.integrationHistory,vector,state.integrationArc,0,config.window,4096);
        return;
    }
    if (config.kind == "dual") {
        if (state.secondary.empty()) state.secondary = state.ema;
        blendFiberEmbeddingVector(state.secondary,vector,std::exp2(-elapsed/config.longHalfLife));
    }
    if (config.kind == "gate") {
        const double disagreement = 1-fiberEmbeddingCosine(vector,state.ema);
        const double gate = fiberEmbeddingSigmoid(config.gateBias+config.gatePresence*(presence-.5)-config.gateDistance*disagreement);
        blendFiberEmbeddingVector(state.ema,vector,1-(1-decay)*gate);
    } else if (config.kind == "angular") {
        auto proposed = state.ema;
        blendFiberEmbeddingVector(proposed,vector,decay);
        const double cosine = fiberEmbeddingCosine(proposed,state.ema);
        // An exactly antipodal update has no unique rotation plane, even if
        // the configured cap exceeds pi. Match the differentiable replay.
        if (cosine <= -1) return;
        const double angle = std::acos(cosine);
        const double limit = config.capDegrees * (3.14159265358979323846/180) * elapsed/8;
        if (angle <= limit) { state.ema = std::move(proposed); return; }
        // Undefined antipodal plane retains the previous reference.
        if (std::sin(angle) <= 1e-8) return;
        const double a = std::sin(angle-limit)/std::sin(angle), b = std::sin(limit)/std::sin(angle);
        std::vector<float> capped(vector.size());
        double norm2 = 0;
        for (size_t i=0;i<capped.size();++i) { capped[i]=a*state.ema[i]+b*proposed[i]; norm2+=double(capped[i])*capped[i]; }
        if (norm2 > 0) { for(auto& x:capped)x=static_cast<float>(x/std::sqrt(norm2)); state.ema=std::move(capped); }
    } else blendFiberEmbeddingVector(state.ema,vector,decay);
}

inline double fiberEmbeddingMemoryCost(const std::vector<float>& vector,
    const FiberEmbeddingMemory& state, bool withSeed)
{
    const double cost = 1 - fiberEmbeddingCosine(vector, state.ema);
    return withSeed ? .5 * (cost + 1 - fiberEmbeddingCosine(vector, state.seed)) : cost;
}

// Retain observation intervals, not release intervals: delayed application must
// perform exactly the same EMA arithmetic on the same ordered observations.
struct FiberEmbeddingPending {
    double arcBase, distance;
    std::vector<float> vector;
};
struct FiberEmbeddingDelay {
    static constexpr size_t capacity = 4096;
    // Empty vector avoids deque's allocation for every beam in unrelated modes.
    std::vector<FiberEmbeddingPending> pending;
};

inline void advanceFiberEmbeddingDelayedMemory(FiberEmbeddingMemory& state,
    FiberEmbeddingDelay& delayState, const std::vector<float>& vector,
    double arcBase, double distance, double delay, double halfLife)
{
    if (!std::isfinite(delay) || delay < 0 || delay > 2048 ||
        !std::isfinite(arcBase) || arcBase < 0 ||
        !std::isfinite(distance) || distance < 0 ||
        !std::isfinite(halfLife) || halfLife <= 0 || vector.empty() ||
        vector.size() != state.ema.size() ||
        std::any_of(vector.begin(), vector.end(), [](float v) { return !std::isfinite(v); }) ||
        (!delayState.pending.empty() && arcBase <= delayState.pending.back().arcBase))
        throw std::invalid_argument("invalid delayed embedding EMA update");
    if (delay == 0) {
        advanceFiberEmbeddingMemory(state, vector, distance, halfLife, 1, false);
        return;
    }
    size_t released = 0;
    while (released < delayState.pending.size() && delayState.pending[released].arcBase <= arcBase-delay) {
        const auto& observation = delayState.pending[released];
        advanceFiberEmbeddingMemory(state, observation.vector, observation.distance, halfLife, 1, false);
        ++released;
    }
    if (released) delayState.pending.erase(delayState.pending.begin(), delayState.pending.begin()+released);
    if (delayState.pending.size() >= FiberEmbeddingDelay::capacity)
        throw std::runtime_error("delayed embedding EMA pending capacity exceeded");
    delayState.pending.push_back({arcBase, distance, vector});
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

inline double fiberEmbeddingSumNorm(const std::vector<double>& sum)
{
    double square = 0;
    for (double v : sum) square += v*v;
    return std::sqrt(square);
}

struct FiberEmbeddingLru {
    FiberEmbeddingHistory entries;
    std::vector<double> lastUsed;
    double lastAdmission = 0;
    std::vector<float> meanDirection;
    double coherence = 0;
};

inline void refreshFiberEmbeddingLruMean(FiberEmbeddingLru& state)
{
    if (state.entries.empty()) throw std::invalid_argument("empty embedding LRU state");
    std::vector<double> sum(state.entries.front().vector.size(),0);
    for (const auto& entry : state.entries) {
        if (entry.vector.size()!=sum.size()) throw std::invalid_argument("inconsistent embedding LRU dimensions");
        for (size_t d=0; d<sum.size(); ++d) sum[d]+=entry.vector[d];
    }
    const double norm=fiberEmbeddingSumNorm(sum);
    if (!std::isfinite(norm)) throw std::invalid_argument("nonfinite embedding LRU mean");
    state.coherence=norm/state.entries.size();
    state.meanDirection.resize(sum.size());
    for (size_t d=0; d<sum.size(); ++d)
        state.meanDirection[d]=norm>0 ? static_cast<float>(sum[d]/norm) : 0;
}

inline void advanceFiberEmbeddingLru(FiberEmbeddingLru& state, const std::vector<float>& vector,
    double arc, double spacing, size_t capacity, double novelty = 0, bool hitOnly = false)
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
        const bool familiar = state.entries[nearest].vector == vector || 1-closest < novelty;
        if (!hitOnly || familiar) state.lastUsed[nearest] = arc;
        if (familiar) return;
        if (state.entries.size() >= capacity) {
            const size_t oldest = std::min_element(state.lastUsed.begin(),state.lastUsed.end())-state.lastUsed.begin();
            state.entries.erase(state.entries.begin()+oldest);
            state.lastUsed.erase(state.lastUsed.begin()+oldest);
        }
    }
    state.entries.push_back({arc,vector});
    state.lastUsed.push_back(arc);
    refreshFiberEmbeddingLruMean(state);
}

inline double fiberEmbeddingLruCost(const std::vector<float>& vector, const FiberEmbeddingLru& state, bool average,
    bool normalizedMean = false)
{
    if (state.entries.empty()) throw std::invalid_argument("empty embedding LRU state");
    if (normalizedMean) {
        if (state.meanDirection.empty()) throw std::invalid_argument("uninitialized embedding LRU mean");
        return state.coherence==0 ? 1 : 1-fiberEmbeddingCosine(vector,state.meanDirection);
    }
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
