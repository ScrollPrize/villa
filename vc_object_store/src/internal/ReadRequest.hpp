#pragma once
#include "vc_object_store/objects/GeometryRegion.hpp"
#include <stdexcept>

namespace vc::objects::detail {
struct ReadStopped {};
struct ReadRequest {
    const RegionOptions& options;
    RegionResult& result;
    const RegionCancel& cancel;
    void stop(RegionStop reason) { result.stop = reason; throw ReadStopped{}; }
    void poll() {
        if (cancel && cancel(result.stats)) stop(RegionStop::Cancelled);
    }
    void bytes(uint64_t count, uint64_t RegionStats::* counter) {
        poll();
        auto& s = result.stats;
        if (count > options.maxUnitBytes || count > options.processingByteBudget - s.processingBytes)
            stop(RegionStop::ProcessingBudget);
        s.processingBytes += count;
        s.*counter += count;
    }
    void work(uint64_t RegionStats::* counter) {
        poll();
        if (result.stats.candidateWork == options.candidateWorkBudget) stop(RegionStop::CandidateBudget);
        ++result.stats.candidateWork;
        ++(result.stats.*counter);
    }
};
inline thread_local ReadRequest* readRequest = nullptr;
inline void requestPoll() { if (readRequest) readRequest->poll(); }
inline void requestBytes(uint64_t n, uint64_t RegionStats::* counter) { if (readRequest) readRequest->bytes(n, counter); }
inline void requestWork(uint64_t RegionStats::* counter) { if (readRequest) readRequest->work(counter); }
}
