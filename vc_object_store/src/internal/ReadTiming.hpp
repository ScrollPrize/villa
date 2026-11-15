#pragma once
// Wall-clock split of a selected-inputs read, reported on stderr when the
// environment variable VC_OBJECT_STORE_TIMING is set. Measurement only: no
// code path changes with it, and it costs two clock reads per timed step.
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace vc::objects::detail {
struct ReadTiming {
    bool enabled = std::getenv("VC_OBJECT_STORE_TIMING") != nullptr;
    // seconds
    double objects = 0, relationshipResolve = 0, members = 0, scopeResolve = 0, incident = 0, recorded = 0,
           archives = 0, applies = 0, order = 0;
    // counts
    long incidentEdges = 0, recordedCalls = 0, appliesCalls = 0, archiveDecodes = 0;
    void reset() { *this = ReadTiming{}; }
    void report(const char* what, size_t objectCount, size_t relationshipCount) const
    {
        if (!enabled) return;
        std::fprintf(stderr,
            "VC_OBJECT_STORE_TIMING %s: objects=%zu relationships=%zu | objectsAtVersions %.1fs | relationshipsAt resolve %.1fs"
            " | members %.1fs | scope resolve %.1fs | incidentEdges %.1fs (%ld edges seen) | recordedRelationship %.1fs (%ld calls)"
            " | archives %.1fs (%ld decodes) | endpointApplies %.1fs (%ld calls) | order %.1fs\n",
            what, objectCount, relationshipCount, objects, relationshipResolve, members, scopeResolve, incident,
            incidentEdges, recorded, recordedCalls, archives, archiveDecodes, applies, appliesCalls, order);
        std::fflush(stderr);
    }
};
inline ReadTiming& readTiming()
{
    thread_local ReadTiming timing;
    return timing;
}
class ScopedReadTimer {
public:
    explicit ScopedReadTimer(double& accumulator)
        : accumulator_(readTiming().enabled ? &accumulator : nullptr),
          start_(accumulator_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    ~ScopedReadTimer()
    {
        if (accumulator_) *accumulator_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }
    ScopedReadTimer(const ScopedReadTimer&) = delete;
    ScopedReadTimer& operator=(const ScopedReadTimer&) = delete;
private:
    double* accumulator_;
    std::chrono::steady_clock::time_point start_;
};
}
