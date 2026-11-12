// Process-local policy shared by every live reader of one array identity.
#pragma once
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "platform_io.hpp"
#include "single_flight.hpp"
#include "memory_cache.hpp"
#include "channel_decode.hpp"

namespace vcr {
struct ChunkLoad {
    bool available=true,complete=false;
    ChunkPtr whole;
    std::vector<DecodedChannel> channels;
    std::shared_ptr<MemoryCache> cache;
};
struct ChannelPolicy {
    std::atomic<bool> enabled{false};
    std::mutex mutex;
    std::set<std::int64_t> relevant;
    // Shared by whole/selected readers. Keys include the decoded-cache namespace
    // so different cache quality/encoding views cannot exchange wrong values.
    SingleFlight<std::shared_ptr<const ChunkLoad>> reads;
    static std::shared_ptr<ChannelPolicy> for_array(const std::string& identity)
    {
        struct Registry {
            long pid = platform::process_id();
            std::mutex mutex;
            std::unordered_map<std::string, std::weak_ptr<ChannelPolicy>> policies;
        };
        // A child must not touch locks inherited from vanished parent threads.
        // The inherited registry is deliberately abandoned, just like inherited
        // reader engines in the Python fork handling.
        static std::atomic<Registry*> registry{new Registry};
        auto* current = registry.load(std::memory_order_acquire);
        if (current->pid != platform::process_id()) {
            auto* fresh = new Registry;
            if (registry.compare_exchange_strong(current,fresh,std::memory_order_acq_rel)) current = fresh;
            else delete fresh;
        }
        std::lock_guard lock(current->mutex);
        auto& policies = current->policies;
        if (auto existing = policies[identity].lock()) return existing;
        if (policies.size() > 256)
            for (auto it = policies.begin(); it != policies.end();)
                if (it->second.expired()) it = policies.erase(it); else ++it;
        auto policy = std::make_shared<ChannelPolicy>();
        policies[identity] = policy;
        return policy;
    }
};
} // namespace vcr
