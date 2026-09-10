#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"
#include <nlohmann/json.hpp>
#include <memory>
#include <semaphore>

// Control-only localhost transport. Prediction bytes remain ordinary Zarr.
// Construct before case threads, destroy after joining them.
class FiberLiveBridge {
public:
    explicit FiberLiveBridge(const nlohmann::json& endpoint);
    ~FiberLiveBridge();
    void await(const std::string& source, size_t caseIndex, double remainingBase,
               const std::vector<vc::fiber_tracer::FiberPredictionChunkDependency>& chunks);
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
