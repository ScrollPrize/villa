#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"
#include <nlohmann/json.hpp>
#include <memory>
#include <semaphore>
#include <chrono>
#include <unordered_set>

struct FiberReadinessProfile {
    size_t calls = 0, localHits = 0, bridgeCalls = 0, submittedDependencies = 0, cacheClears = 0;
    double localLookupSeconds = 0, bridgeSeconds = 0, bridgeLockSeconds = 0, readyWaitSeconds = 0;
};

// Owned by ONE live case thread. These are publication proofs, not decoded
// payloads; never share this object across cases or inference sessions.
class FiberCaseReadiness {
public:
    explicit FiberCaseReadiness(size_t capacity = 4096, FiberReadinessProfile* profile = nullptr)
        : capacity_(capacity), profile_(profile) {}

    template <typename Wait, typename CheckActive>
    void await(const std::vector<vc::fiber_tracer::FiberPredictionChunkDependency>& chunks,
               Wait&& wait, CheckActive&& checkActive)
    {
        checkActive(); // atomic fast check, including on all-known calls
        if (profile_) ++profile_->calls;
        const auto start = profile_ ? Clock::now() : Clock::time_point{};
        std::vector<vc::fiber_tracer::FiberPredictionChunkDependency> missing;
        if (capacity_) {
            for (const auto& chunk : chunks)
                if (!ready_.contains(chunk)) missing.push_back(chunk);
        }
        const auto& requested = capacity_ ? missing : chunks;
        if (profile_) profile_->localLookupSeconds += seconds(start);
        if (requested.empty()) {
            if (profile_) ++profile_->localHits;
            checkActive();
            return; // no bridge mutex, message, or CPU-permit operation
        }
        if (profile_) {
            ++profile_->bridgeCalls;
            profile_->submittedDependencies += requested.size();
        }
        const auto waitStart = profile_ ? Clock::now() : Clock::time_point{};
        wait(requested);
        if (profile_) profile_->bridgeSeconds += seconds(waitStart);
        checkActive();
        // Only successful acknowledgements establish proof. Eviction is safe:
        // forgotten chunks are checked again, never inferred from file presence.
        if (capacity_ && requested.size() <= capacity_) {
            if (ready_.size() + requested.size() > capacity_) {
                ready_.clear();
                if (profile_) ++profile_->cacheClears;
            }
            for (const auto& chunk : requested) ready_.insert(chunk);
        }
    }
private:
    using Clock = std::chrono::steady_clock;
    static double seconds(Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }
    size_t capacity_;
    FiberReadinessProfile* profile_;
    struct KeyHash {
        size_t operator()(const vc::fiber_tracer::FiberPredictionChunkDependency& key) const {
            size_t h = std::filesystem::hash_value(key.arrayPath);
            for (uint32_t v : key.indexZYX) h ^= size_t(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    std::unordered_set<vc::fiber_tracer::FiberPredictionChunkDependency, KeyHash> ready_;
};

// Control-only localhost transport. Prediction bytes remain ordinary Zarr.
// Construct before case threads, destroy after joining them.
class FiberLiveBridge {
public:
    explicit FiberLiveBridge(const nlohmann::json& endpoint);
    ~FiberLiveBridge();
    void await(const std::string& source, size_t caseIndex, double remainingBase,
               const std::vector<vc::fiber_tracer::FiberPredictionChunkDependency>& chunks,
               FiberReadinessProfile* profile = nullptr);
    void checkActive() const;
    void cancel(const std::string& reason);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class FiberCaseCpuPermit {
public:
    explicit FiberCaseCpuPermit(std::counting_semaphore<>& pool) : pool_(pool) { pool_.acquire(); }
    ~FiberCaseCpuPermit() { pool_.release(); }
    FiberCaseCpuPermit(const FiberCaseCpuPermit&) = delete;
    // Called only by this case's thread; nested candidate pools are disabled.
    template <typename Wait> void suspend(Wait&& wait) {
        pool_.release();
        try { wait(); }
        catch (...) { pool_.acquire(); throw; }
        pool_.acquire();
    }
private:
    std::counting_semaphore<>& pool_;
};

// Construction and replay share admission, but not dependency-waiting threads.
class FiberConstructionCpuPermit : public FiberCaseCpuPermit {
public:
    explicit FiberConstructionCpuPermit(std::counting_semaphore<>& pool)
        : FiberCaseCpuPermit(pool), previous_(current) { current = this; }
    ~FiberConstructionCpuPermit() { current = previous_; }
    static inline thread_local FiberConstructionCpuPermit* current = nullptr;
private:
    FiberConstructionCpuPermit* previous_;
};
