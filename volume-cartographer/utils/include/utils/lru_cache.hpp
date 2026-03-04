#pragma once
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <functional>
#include <optional>
#include <utility>

namespace utils {

// ---------------------------------------------------------------------------
// SizeOf concept: detect V::byte_size() for dynamic-size values
// ---------------------------------------------------------------------------
template<typename V>
concept HasByteSize = requires(const V& v) {
    { v.byte_size() } -> std::convertible_to<std::size_t>;
};

// ---------------------------------------------------------------------------
// LRUCache -- generation-based, byte-budgeted, thread-safe LRU cache
// ---------------------------------------------------------------------------
template<typename K, typename V,
         typename Hash     = std::hash<K>,
         typename KeyEqual = std::equal_to<K>>
class LRUCache final {
public:
    // -- configuration ------------------------------------------------------
    struct Config {
        std::size_t max_bytes = 1ULL << 30;           // 1 GB default
        double      evict_ratio = 15.0 / 16.0;        // hysteresis target
        bool        promote_on_read = true;            // update generation on get()
        std::function<std::size_t(const V&)> size_fn = nullptr;
    };

    explicit LRUCache(Config config = {})
        : config_{std::move(config)}
        , generation_{0}
        , current_bytes_{0}
        , hits_{0}
        , misses_{0}
        , evictions_{0}
    {
    }

    // -- non-copyable, non-movable (contains mutex) -------------------------
    LRUCache(const LRUCache&)            = delete;
    LRUCache& operator=(const LRUCache&) = delete;
    LRUCache(LRUCache&&)                 = delete;
    LRUCache& operator=(LRUCache&&)      = delete;

    // -- read ---------------------------------------------------------------

    /// Non-blocking read. Returns nullopt on miss.
    /// If promote_on_read is true (default), updates generation on hit.
    [[nodiscard]] std::optional<V> get(const K& key) const
    {
        std::shared_lock lock{mutex_};
        auto it = map_.find(key);
        if (it == map_.end()) {
            misses_.fetch_add(1, std::memory_order_relaxed);
            return std::nullopt;
        }
        if (config_.promote_on_read) {
            it->second.generation.store(generation_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
        }
        hits_.fetch_add(1, std::memory_order_relaxed);
        return it->second.value;
    }

    /// Non-blocking read returning a pointer to the stored value.
    /// Returns nullptr on miss. The pointer is valid while the caller
    /// holds no exclusive lock and the entry is not evicted.
    /// For shared_ptr<T> values, callers should copy the shared_ptr
    /// while the shared lock is held (which this method does internally).
    [[nodiscard]] V get_or(const K& key, V fallback) const
    {
        std::shared_lock lock{mutex_};
        auto it = map_.find(key);
        if (it == map_.end()) {
            misses_.fetch_add(1, std::memory_order_relaxed);
            return fallback;
        }
        if (config_.promote_on_read) {
            it->second.generation.store(generation_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
        }
        hits_.fetch_add(1, std::memory_order_relaxed);
        return it->second.value;
    }

    /// Check existence without promoting.
    [[nodiscard]] bool contains(const K& key) const noexcept
    {
        std::shared_lock lock{mutex_};
        return map_.contains(key);
    }

    // -- write --------------------------------------------------------------

    /// Insert or update an entry. May trigger eviction.
    void put(const K& key, V value)
    {
        put_impl(key, std::move(value), /*pinned=*/false);
    }

    /// Insert or update a pinned entry (never evicted).
    void put_pinned(const K& key, V value)
    {
        put_impl(key, std::move(value), /*pinned=*/true);
    }

    /// Remove a specific key. Returns true if the key existed.
    bool remove(const K& key)
    {
        std::unique_lock lock{mutex_};
        auto it = map_.find(key);
        if (it == map_.end()) {
            return false;
        }
        current_bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
        map_.erase(it);
        return true;
    }

    /// Clear all entries.
    void clear()
    {
        std::unique_lock lock{mutex_};
        map_.clear();
        current_bytes_.store(0, std::memory_order_relaxed);
    }

    // -- stats --------------------------------------------------------------

    [[nodiscard]] std::size_t size() const noexcept
    {
        std::shared_lock lock{mutex_};
        return map_.size();
    }

    [[nodiscard]] std::size_t byte_size() const noexcept
    {
        return current_bytes_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t max_bytes() const noexcept
    {
        return config_.max_bytes;
    }

    [[nodiscard]] std::uint64_t hits() const noexcept
    {
        return hits_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t misses() const noexcept
    {
        return misses_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t evictions() const noexcept
    {
        return evictions_.load(std::memory_order_relaxed);
    }

    // -- batch operations ---------------------------------------------------

    /// Return keys from [begin, end) that are absent from the cache.
    /// Acquires the shared lock once for the entire range.
    template<typename Iter>
    [[nodiscard]] std::vector<K> missing_keys(Iter begin, Iter end) const
    {
        std::vector<K> result;
        std::shared_lock lock{mutex_};
        for (auto it = begin; it != end; ++it) {
            if (!map_.contains(*it)) {
                result.push_back(*it);
            }
        }
        return result;
    }

    /// Iterate over all entries under shared lock.
    /// Func signature: void(const K&, const V&)
    template<typename F>
    void for_each(F&& func) const
    {
        std::shared_lock lock{mutex_};
        for (const auto& [k, entry] : map_) {
            func(k, entry.value);
        }
    }

private:
    // -- internal entry -----------------------------------------------------
    struct Entry {
        V                              value;
        std::size_t                    bytes;
        mutable std::atomic<std::uint64_t> generation;
        bool                           pinned;

        Entry(V v, std::size_t b, std::uint64_t g, bool p)
            : value(std::move(v)), bytes(b), generation(g), pinned(p) {}
        Entry(Entry&& o) noexcept(std::is_nothrow_move_constructible_v<V>)
            : value(std::move(o.value))
            , bytes(o.bytes)
            , generation(o.generation.load(std::memory_order_relaxed))
            , pinned(o.pinned) {}
        Entry& operator=(Entry&& o) noexcept(std::is_nothrow_move_assignable_v<V>) {
            value = std::move(o.value);
            bytes = o.bytes;
            generation.store(o.generation.load(std::memory_order_relaxed), std::memory_order_relaxed);
            pinned = o.pinned;
            return *this;
        }
    };

    // -- size computation ---------------------------------------------------
    [[nodiscard]] std::size_t compute_size(const V& v) const
    {
        if (config_.size_fn) {
            return config_.size_fn(v);
        }
        if constexpr (HasByteSize<V>) {
            return v.byte_size();
        }
        return sizeof(V);
    }

    // -- put implementation -------------------------------------------------
    void put_impl(const K& key, V value, bool pinned)
    {
        const auto val_bytes = compute_size(value);

        {
            std::unique_lock lock{mutex_};

            // If key already exists, remove its old byte contribution.
            if (auto it = map_.find(key); it != map_.end()) {
                current_bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
                it->second.value      = std::move(value);
                it->second.bytes      = val_bytes;
                it->second.generation.store(generation_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
                it->second.pinned     = pinned;
                current_bytes_.fetch_add(val_bytes, std::memory_order_relaxed);
            } else {
                // New entry.
                auto gen = generation_.fetch_add(1, std::memory_order_relaxed);
                map_.emplace(
                    key,
                    Entry{std::move(value), val_bytes, gen, pinned});
                current_bytes_.fetch_add(val_bytes, std::memory_order_relaxed);
            }
        }

        // Evict outside the unique_lock if over budget.
        if (current_bytes_.load(std::memory_order_relaxed) > config_.max_bytes) {
            evict();
        }
    }

    // -- eviction -----------------------------------------------------------
    void evict()
    {
        const auto target = static_cast<std::size_t>(
            static_cast<double>(config_.max_bytes) * config_.evict_ratio);

        // Phase 1 -- collect candidates under shared lock.
        struct Candidate {
            K             key;
            std::size_t   bytes;
            std::uint64_t generation;
        };
        std::vector<Candidate> candidates;

        {
            std::shared_lock lock{mutex_};
            candidates.reserve(map_.size());
            for (const auto& [k, entry] : map_) {
                if (!entry.pinned) {
                    candidates.push_back({k, entry.bytes, entry.generation.load(std::memory_order_relaxed)});
                }
            }
        }

        // Sort oldest generation first.
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.generation < b.generation;
                  });

        // Phase 2 -- evict under unique lock until under target.
        std::unique_lock lock{mutex_};
        for (const auto& cand : candidates) {
            if (current_bytes_.load(std::memory_order_relaxed) <= target) {
                break;
            }
            auto it = map_.find(cand.key);
            if (it == map_.end() || it->second.pinned) {
                continue; // removed or pinned between phases
            }
            // Guard against a put() that refreshed the entry between phases.
            if (it->second.generation.load(std::memory_order_relaxed) != cand.generation) {
                continue;
            }
            current_bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
            map_.erase(it);
            evictions_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // -- data members -------------------------------------------------------
    Config config_;

    using Map = std::unordered_map<K, Entry, Hash, KeyEqual>;
    mutable Map map_;

    mutable std::shared_mutex          mutex_;
    mutable std::atomic<std::uint64_t> generation_;
    mutable std::atomic<std::size_t>   current_bytes_;
    mutable std::atomic<std::uint64_t> hits_;
    mutable std::atomic<std::uint64_t> misses_;
    mutable std::atomic<std::uint64_t> evictions_;
};

} // namespace utils
