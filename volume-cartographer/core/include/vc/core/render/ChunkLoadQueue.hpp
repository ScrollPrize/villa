#pragma once
#include "vc/core/render/IChunkedArray.hpp"
#include <future>

namespace vc::render {

// Optional batch admission layer above existing cache/read/decode stages.
// Source callbacks must never block for IO or call back into this queue.
class ChunkLoadQueue final {
public:
    struct Query {
        ChunkKey key;
        bool foreground = false;
        int priority = 0;
    };
    struct Source {
        std::uint64_t id = 0;
        // Strong exception guarantee: acquire all reservations or none.
        std::function<void(const std::vector<ChunkKey>&)> acquire;
        std::function<std::vector<ChunkResult>(const std::vector<Query>&)> poll;
        // Must drop every supplied reservation before potentially throwing.
        // Callback failure may terminate the queue, not just this source.
        std::function<void(const std::vector<ChunkKey>&)> release;
        // Subscribe before the first poll; return an unsubscribe callback.
        std::function<std::function<void()>(std::function<void()>)> subscribe;
    };
    struct Stats {
        std::uint64_t requests=0, keys=0, coalesced=0, completed=0, polls=0, wakes=0;
        std::size_t inFlight=0, peakInFlight=0, pending=0, peakPending=0;
        std::size_t replyBytes=0, peakReplyBytes=0;
    };
    explicit ChunkLoadQueue(std::size_t maxInFlight);
    ~ChunkLoadQueue();
    ChunkLoadQueue(const ChunkLoadQueue&)=delete;
    ChunkLoadQueue& operator=(const ChunkLoadQueue&)=delete;
    std::future<std::vector<ChunkResult>> submit(Source source,
        std::vector<ChunkKey> keys, bool foreground, int priority, bool retainResults=true);
    Stats stats() const;
    void cancelSource(std::uint64_t sourceId);
    void stop();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
