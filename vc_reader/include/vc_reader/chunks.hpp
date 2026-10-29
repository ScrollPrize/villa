#pragma once
// A process-wide service that holds decoded chunks of many sources in one bounded
// memory budget and fetches missing ones on worker threads, in priority order.
// Interactive viewers probe without blocking, publish the chunks each view needs
// (coarse levels and chunks near the view's focus first) and repaint when ready
// listeners fire; batch code and tools fetch blocking or prefetch.
//
// Sources are Zarr pyramids (one vcr::Array per level; give those arrays no
// MemoryCache, the service is their decoded cache) or keyed loaders of
// application records.

#include "vc_reader/native/array.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vcr::chunks {

using SourceId = std::uint32_t;

struct Key {
    SourceId source = 0;
    std::int32_t level = 0;
    std::int64_t z = 0, y = 0, x = 0;
    friend bool operator==(const Key&, const Key&) = default;
};
struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept;
};

enum class Status {
    Unavailable,  // not in memory and not queued
    Queued,       // queued or currently being fetched
    Fill,         // not stored, or stored entirely as the fill value
    Data,
    Error,
};

struct Chunk {
    Status status = Status::Unavailable;
    ChunkPtr bytes;  // Data: the decoded chunk (or record), shared and immutable
    std::string error;
};

// Loads one record of a keyed source; nullptr when it does not exist. Throws on failure.
using Loader = std::function<ChunkPtr(std::int32_t level, std::int64_t z, std::int64_t y, std::int64_t x)>;

struct SourceOptions {
    // Under memory pressure this source's chunks are evicted before any other
    // source's (bulk background readers that must not evict what a viewer shows).
    bool preferSelfEviction = false;
    // Data chunks that hold only the fill value are kept as Fill (no bytes).
    bool detectFill = true;
    // Keyed sources: whether loading may touch the network (remote workers and admission).
    bool remote = false;
};

// Who asks for a chunk. A view's demand is replaced as the view changes;
// foreground work (e.g. tracing) shares fetch admission fairly with views;
// everything else is background work ordered by priority (higher first).
struct Demand {
    std::uint64_t view = 0;  // 0: not a view
    bool foreground = false;
    int priority = 0;
};

// One chunk a view needs, where it appears in the view, and how much coarser its
// level is than the level the view asked for (coarser fallbacks paint first).
struct ViewSample {
    Key key;
    std::array<float, 2> position{};
    int relativeLevel = 0;
};

struct AdaptiveState {
    std::size_t limit = 0;          // settled remote admission limit
    double bytesPerSecond = 0.0;    // throughput at that limit
};

struct Stats {
    std::size_t decodedBytes = 0, decodedCapacity = 0, resident = 0;
    std::size_t queued = 0, remoteInFlight = 0, localInFlight = 0, remoteAdmission = 0;
    double remoteBytesPerSecond = 0.0;
    std::uint64_t fetched = 0, failed = 0, evicted = 0;
    std::vector<std::size_t> unresolvedByLevel;  // queued or running, for one source
};

class Service {
public:
    struct Options {
        std::size_t decodedBytes = 512ull << 20;
        std::size_t remoteWorkers = 16;     // threads for sources that touch the network
        std::size_t localWorkers = 8;       // threads for local sources, never network-limited
        std::size_t remoteAdmission = 16;   // concurrent remote fetches (the adaptive upper bound)
        bool adaptive = false;              // tune remote admission to measured throughput
        std::optional<AdaptiveState> adaptiveState;  // a previous run's settled admission
    };

    explicit Service(Options options);
    Service() : Service(Options{}) {}
    ~Service();
    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;

    SourceId addSource(std::vector<std::shared_ptr<Array>> levels, SourceOptions options = {});
    SourceId addSource(Loader loader, std::int32_t levels, SourceOptions options = {});
    // Cancels the source's queued work and drops its chunks; running fetches are discarded.
    void removeSource(SourceId source);
    // Drops the source's decoded chunks (e.g. after its data changed on disk).
    void invalidate(SourceId source);

    // The chunk if resident, without I/O, queueing or touching LRU order.
    Chunk probe(const Key& key);
    // The chunk if resident (promoted in LRU order); otherwise queues it for `demand`.
    Chunk request(const Key& key, const Demand& demand = {});
    // Waits for the chunk, fetching it with foreground priority when needed.
    Chunk get(const Key& key, const Demand& demand = {.foreground = true});
    void prefetch(std::span<const Key> keys, const Demand& demand = {}, bool wait = false);

    // Replaces everything `view` asked for at an older version with `samples`.
    // Queued chunks no longer demanded by anyone are cancelled. Stale versions are ignored.
    void replaceViewDemand(std::uint64_t view, std::uint64_t version, std::array<float, 2> focus,
                           std::vector<ViewSample> samples);
    // Withdraws a view's demand (for one source, or every source).
    void clearViewDemand(std::uint64_t view, std::uint64_t version, std::optional<SourceId> source = {});
    // The most recently used view's demand is served first.
    void markViewActive(std::uint64_t view);

    using ListenerId = std::uint64_t;
    // Called on a worker thread after a chunk of `source` resolves (data, fill or error).
    // Listener exceptions are isolated; callbacks must not block waiting for service work.
    ListenerId addReadyListener(SourceId source, std::function<void(const Key&)> listener);
    // Called on a worker thread when a remote fetch starts (true) and ends (false).
    ListenerId addFetchListener(SourceId source, std::function<void(const Key&, bool)> listener);
    void removeListener(ListenerId id);

    // Changes the memory budget and remote admission in place; queued and running work is kept.
    void configure(std::size_t decodedBytes, std::size_t remoteAdmission, bool adaptive);
    AdaptiveState adaptiveState() const;
    Stats stats(std::optional<SourceId> source = {}) const;
    // Waits until no work of the source is queued or running.
    void drain(SourceId source);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

// The process-wide service most callers share, created on first use.
std::shared_ptr<Service> processService();
// Replaces the process-wide service's options before its first use, or
// reconfigures it in place (decodedBytes, remoteAdmission, adaptive) afterwards.
std::shared_ptr<Service> configureProcessService(Service::Options options);

}  // namespace vcr::chunks
