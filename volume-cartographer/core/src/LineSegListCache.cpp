#include "vc/core/util/LineSegListCache.hpp"
#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/LineSegList.hpp"

#include <random>
#include <iostream>
#include <iomanip>

namespace vc::core::util {

LineSegListCache::LineSegListCache(size_t max_size_bytes)
    : max_size_bytes_(max_size_bytes), last_stat_time_(std::chrono::steady_clock::now()) {}

std::shared_ptr<LineSegList> LineSegListCache::get(const CacheKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = lookup_.find(key);
    if (it != lookup_.end()) {
        auto& entry = entries_[it->second];
        if (auto data = entry.data.lock()) {
            entry.generation = ++generation_counter_;
            cache_hits_++;
            check_print_stats();
            return data;
        } else {
            // Entry expired, remove it
            current_size_bytes_ -= entry.size_bytes;
            lookup_.erase(it);
        }
    }
    cache_misses_++;
    check_print_stats();
    return nullptr;
}

void LineSegListCache::put(const CacheKey& key, std::shared_ptr<LineSegList> data) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (lookup_.count(key)) {
        return; // Already in cache
    }

    size_t data_size = data->get_memory_usage();

    if (current_size_bytes_ + data_size > max_size_bytes_) {
        evict();
    }

    size_t index = entries_.size();
    entries_.emplace_back(CacheEntry{key, data, ++generation_counter_, data_size});
    lookup_[key] = index;
    current_size_bytes_ += data_size;
}

void LineSegListCache::evict() {
    if (entries_.empty()) {
        return;
    }

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, entries_.size() - 1);

    size_t evict_idx = 0;
    uint64_t min_generation = std::numeric_limits<uint64_t>::max();

    for (int i = 0; i < 10; ++i) {
        size_t rand_idx = dist(gen);
        if (entries_[rand_idx].generation < min_generation) {
            min_generation = entries_[rand_idx].generation;
            evict_idx = rand_idx;
        }
    }

    auto& entry_to_evict = entries_[evict_idx];
    current_size_bytes_ -= entry_to_evict.size_bytes;
    lookup_.erase(entry_to_evict.key);

    // Replace with the last element to avoid vector shifting
    if (evict_idx != entries_.size() - 1) {
        entries_[evict_idx] = std::move(entries_.back());
        lookup_[entries_[evict_idx].key] = evict_idx;
    }
    entries_.pop_back();
}

void LineSegListCache::check_print_stats() {
    if (generation_counter_ % 1000 == 0) {
        auto now = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::seconds>(now - last_stat_time_);
        if (diff.count() >= 1) {
            uint64_t hits = cache_hits_.load();
            uint64_t misses = cache_misses_.load();
            uint64_t total = hits + misses;
            double hit_rate = (total == 0) ? 0.0 : (static_cast<double>(hits) / total) * 100.0;
            double utilization_gb = static_cast<double>(current_size_bytes_) / (1024 * 1024 * 1024);
            std::cout << "[LineSegList Cache] Hits: " << hits << ", Misses: " << misses << ", Total: " << total
                      << ", Hit Rate: " << std::fixed << std::setprecision(2) << hit_rate << "%"
                      << ", Utilization: " << utilization_gb << " GB" << std::endl;
            last_stat_time_ = now;
        }
    }
}

} // namespace vc::core::util