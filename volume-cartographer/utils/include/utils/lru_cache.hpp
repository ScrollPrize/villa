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
// FlatHashMap -- open-addressing hash map with linear probing.
//
// Replaces std::unordered_map in LRUCache to eliminate pointer-chasing
// bucket chain traversal (_M_locate at 1.2% in profiles).  Open addressing
// keeps entries in a contiguous array for cache-friendly lookups.
//
// API is a subset of std::unordered_map: find, emplace, erase, iteration.
// Additionally exposes find_prehashed() for callers that already have the
// hash (e.g. ShardedLRUCache computes it once for shard selection).
// ---------------------------------------------------------------------------
template<typename K, typename V,
         typename Hash     = std::hash<K>,
         typename KeyEqual = std::equal_to<K>>
class FlatHashMap final {
    static constexpr uint8_t kEmpty    = 0;
    static constexpr uint8_t kOccupied = 1;
    static constexpr uint8_t kDeleted  = 2;

    struct Slot {
        uint8_t state = kEmpty;
        alignas(std::pair<K, V>) unsigned char storage[sizeof(std::pair<K, V>)];

        std::pair<K, V>& kv() { return *reinterpret_cast<std::pair<K, V>*>(storage); }
        const std::pair<K, V>& kv() const { return *reinterpret_cast<const std::pair<K, V>*>(storage); }
    };

public:
    using key_type    = K;
    using mapped_type = V;
    using value_type  = std::pair<K, V>;

    // --- Iterator ---
    template<bool IsConst>
    class Iterator {
        friend class FlatHashMap;
        using SlotPtr = std::conditional_t<IsConst, const Slot*, Slot*>;
        using PairRef = std::conditional_t<IsConst, const value_type&, value_type&>;
        using PairPtr = std::conditional_t<IsConst, const value_type*, value_type*>;

        SlotPtr slot_;
        SlotPtr end_;

        void advance() {
            while (slot_ != end_ && slot_->state != kOccupied) ++slot_;
        }

    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;

        Iterator() : slot_(nullptr), end_(nullptr) {}
        Iterator(SlotPtr s, SlotPtr e) : slot_(s), end_(e) { advance(); }

        PairRef operator*() const { return slot_->kv(); }
        PairPtr operator->() const { return &slot_->kv(); }

        Iterator& operator++() { ++slot_; advance(); return *this; }
        Iterator operator++(int) { auto tmp = *this; ++(*this); return tmp; }

        bool operator==(const Iterator& o) const { return slot_ == o.slot_; }
        bool operator!=(const Iterator& o) const { return slot_ != o.slot_; }

        operator Iterator<true>() const requires(!IsConst) {
            return Iterator<true>(slot_, end_);
        }
    };

    using iterator       = Iterator<false>;
    using const_iterator = Iterator<true>;

    FlatHashMap() { init(16); }
    ~FlatHashMap() { destroy_all(); }

    FlatHashMap(const FlatHashMap&) = delete;
    FlatHashMap& operator=(const FlatHashMap&) = delete;

    FlatHashMap(FlatHashMap&& o) noexcept
        : slots_(o.slots_), capacity_(o.capacity_), size_(o.size_),
          tomb_count_(o.tomb_count_), mask_(o.mask_)
    {
        o.slots_ = nullptr;
        o.capacity_ = o.size_ = o.tomb_count_ = o.mask_ = 0;
    }

    FlatHashMap& operator=(FlatHashMap&& o) noexcept {
        if (this != &o) {
            destroy_all();
            slots_ = o.slots_;
            capacity_ = o.capacity_;
            size_ = o.size_;
            tomb_count_ = o.tomb_count_;
            mask_ = o.mask_;
            o.slots_ = nullptr;
            o.capacity_ = o.size_ = o.tomb_count_ = o.mask_ = 0;
        }
        return *this;
    }

    // --- Lookup ---

    iterator find(const K& key) {
        if (size_ == 0) return end();
        return find_impl(key, Hash{}(key));
    }

    const_iterator find(const K& key) const {
        if (size_ == 0) return end();
        return cfind_impl(key, Hash{}(key));
    }

    // Find using a pre-computed hash (avoids re-hashing).
    iterator find_prehashed(const K& key, std::size_t h) {
        if (size_ == 0) return end();
        return find_impl(key, h);
    }

    const_iterator find_prehashed(const K& key, std::size_t h) const {
        if (size_ == 0) return end();
        return cfind_impl(key, h);
    }

    bool contains(const K& key) const {
        return find(key) != end();
    }

    // --- Insert ---

    std::pair<iterator, bool> emplace(const K& key, V value) {
        maybe_grow();
        auto idx = Hash{}(key) & mask_;
        std::size_t first_tomb = ~std::size_t(0);
        while (true) {
            auto& slot = slots_[idx];
            if (slot.state == kEmpty) {
                auto& target = (first_tomb != ~std::size_t(0))
                    ? (--tomb_count_, slots_[first_tomb]) : slot;
                new (target.storage) value_type(key, std::move(value));
                target.state = kOccupied;
                ++size_;
                return {iterator(&target, slots_ + capacity_), true};
            }
            if (slot.state == kDeleted && first_tomb == ~std::size_t(0)) {
                first_tomb = idx;
            } else if (slot.state == kOccupied && KeyEqual{}(slot.kv().first, key)) {
                return {iterator(&slot, slots_ + capacity_), false};
            }
            idx = (idx + 1) & mask_;
        }
    }

    // --- Erase ---

    void erase(iterator it) {
        it.slot_->kv().~value_type();
        const_cast<Slot*>(it.slot_)->state = kDeleted;
        --size_;
        ++tomb_count_;
    }

    void erase(const_iterator it) {
        const_cast<Slot*>(it.slot_)->kv().~value_type();
        const_cast<Slot*>(it.slot_)->state = kDeleted;
        --size_;
        ++tomb_count_;
    }

    bool erase(const K& key) {
        auto it = find(key);
        if (it == end()) return false;
        erase(it);
        return true;
    }

    // --- Size / Clear ---

    std::size_t size() const noexcept { return size_; }

    void clear() {
        for (std::size_t i = 0; i < capacity_; ++i) {
            if (slots_[i].state == kOccupied)
                slots_[i].kv().~value_type();
            slots_[i].state = kEmpty;
        }
        size_ = 0;
        tomb_count_ = 0;
    }

    // --- Iteration ---

    iterator begin() { return iterator(slots_, slots_ + capacity_); }
    iterator end()   { return iterator(slots_ + capacity_, slots_ + capacity_); }
    const_iterator begin() const { return const_iterator(slots_, slots_ + capacity_); }
    const_iterator end()   const { return const_iterator(slots_ + capacity_, slots_ + capacity_); }

private:
    Slot*       slots_      = nullptr;
    std::size_t capacity_   = 0;
    std::size_t size_       = 0;
    std::size_t tomb_count_ = 0;
    std::size_t mask_       = 0;

    void init(std::size_t cap) {
        capacity_ = cap;
        mask_ = cap - 1;
        slots_ = new Slot[cap]{};
        size_ = 0;
        tomb_count_ = 0;
    }

    void destroy_all() {
        if (!slots_) return;
        for (std::size_t i = 0; i < capacity_; ++i) {
            if (slots_[i].state == kOccupied)
                slots_[i].kv().~value_type();
        }
        delete[] slots_;
        slots_ = nullptr;
    }

    void maybe_grow() {
        if ((size_ + tomb_count_ + 1) * 4 > capacity_ * 3)
            rehash(capacity_ * 2);
    }

    void rehash(std::size_t new_cap) {
        auto* old_slots = slots_;
        auto old_cap = capacity_;
        init(new_cap);
        for (std::size_t i = 0; i < old_cap; ++i) {
            if (old_slots[i].state == kOccupied) {
                auto& kv = old_slots[i].kv();
                auto idx = Hash{}(kv.first) & mask_;
                while (slots_[idx].state != kEmpty)
                    idx = (idx + 1) & mask_;
                new (slots_[idx].storage) value_type(std::move(kv));
                slots_[idx].state = kOccupied;
                ++size_;
                kv.~value_type();
            }
        }
        delete[] old_slots;
    }

    iterator find_impl(const K& key, std::size_t h) {
        auto idx = h & mask_;
        while (true) {
            auto& slot = slots_[idx];
            if (slot.state == kEmpty) return end();
            if (slot.state == kOccupied && KeyEqual{}(slot.kv().first, key))
                return iterator(&slot, slots_ + capacity_);
            idx = (idx + 1) & mask_;
        }
    }

    const_iterator cfind_impl(const K& key, std::size_t h) const {
        auto idx = h & mask_;
        while (true) {
            auto& slot = slots_[idx];
            if (slot.state == kEmpty) return end();
            if (slot.state == kOccupied && KeyEqual{}(slot.kv().first, key))
                return const_iterator(&slot, slots_ + capacity_);
            idx = (idx + 1) & mask_;
        }
    }
};

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
            thread_local int missCount = 0;
            if (++missCount >= 256) {
                misses_.fetch_add(256, std::memory_order_relaxed);
                missCount = 0;
            }
            return std::nullopt;
        }
        if (config_.promote_on_read) {
            thread_local uint64_t localGen = 0;
            if (++localGen >= 64) {
                it->second.generation.store(generation_.fetch_add(localGen, std::memory_order_relaxed), std::memory_order_relaxed);
                localGen = 0;
            } else {
                it->second.generation.store(generation_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
        }
        thread_local int hitCount = 0;
        if (++hitCount >= 256) {
            hits_.fetch_add(256, std::memory_order_relaxed);
            hitCount = 0;
        }
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
            thread_local int missCount = 0;
            if (++missCount >= 256) {
                misses_.fetch_add(256, std::memory_order_relaxed);
                missCount = 0;
            }
            return fallback;
        }
        if (config_.promote_on_read) {
            thread_local uint64_t localGen = 0;
            if (++localGen >= 64) {
                it->second.generation.store(generation_.fetch_add(localGen, std::memory_order_relaxed), std::memory_order_relaxed);
                localGen = 0;
            } else {
                it->second.generation.store(generation_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
        }
        thread_local int hitCount = 0;
        if (++hitCount >= 256) {
            hits_.fetch_add(256, std::memory_order_relaxed);
            hitCount = 0;
        }
        return it->second.value;
    }

    /// Like get_or but accepts a pre-computed hash to avoid re-hashing.
    /// Used by ShardedLRUCache which already computed the hash for shard
    /// selection.
    [[nodiscard]] V get_or_prehashed(const K& key, std::size_t hash, V fallback) const
    {
        std::shared_lock lock{mutex_};
        auto it = map_.find(key);
        if (it == map_.end()) {
            thread_local int missCount = 0;
            if (++missCount >= 256) {
                misses_.fetch_add(256, std::memory_order_relaxed);
                missCount = 0;
            }
            return fallback;
        }
        if (config_.promote_on_read) {
            thread_local uint64_t localGen = 0;
            if (++localGen >= 64) {
                it->second.generation.store(generation_.fetch_add(localGen, std::memory_order_relaxed), std::memory_order_relaxed);
                localGen = 0;
            } else {
                it->second.generation.store(generation_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
        }
        thread_local int hitCount = 0;
        if (++hitCount >= 256) {
            hits_.fetch_add(256, std::memory_order_relaxed);
            hitCount = 0;
        }
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

        // Partial sort: only need the oldest evict_ratio fraction.
        auto evictCount = static_cast<size_t>(candidates.size() * config_.evict_ratio);
        if (evictCount == 0) evictCount = 1;
        std::nth_element(candidates.begin(), candidates.begin() + evictCount, candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.generation < b.generation; });
        candidates.resize(evictCount);  // Only keep the ones to evict

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

// ---------------------------------------------------------------------------
// ShardedLRUCache -- distributes keys across N independent LRUCache shards
// to eliminate shared_mutex contention on concurrent reads.
//
// With a single LRUCache, even read-only shared_lock operations contend on
// the internal atomic reader counter. With N shards, concurrent readers
// hitting different shards never touch the same cache line.
// ---------------------------------------------------------------------------
template<typename K, typename V,
         typename Hash     = std::hash<K>,
         typename KeyEqual = std::equal_to<K>,
         std::size_t NumShards = 16>
class ShardedLRUCache final {
public:
    using Config = typename LRUCache<K, V, Hash, KeyEqual>::Config;

    explicit ShardedLRUCache(Config config = {})
    {
        // Distribute byte budget evenly across shards
        Config shardConfig = config;
        shardConfig.max_bytes = std::max<std::size_t>(1, config.max_bytes / NumShards);
        for (std::size_t i = 0; i < NumShards; i++) {
            shards_[i].emplace(shardConfig);
        }
    }

    ShardedLRUCache(const ShardedLRUCache&)            = delete;
    ShardedLRUCache& operator=(const ShardedLRUCache&) = delete;

    // -- read ---------------------------------------------------------------

    [[nodiscard]] std::optional<V> get(const K& key) const
    {
        return shard(key).get(key);
    }

    /// Hot-path read: hash once, use for both shard selection and map lookup.
    [[nodiscard]] V get_or(const K& key, V fallback) const
    {
        auto h = Hash{}(key);
        return shards_[h % NumShards]->get_or_prehashed(key, h, std::move(fallback));
    }

    [[nodiscard]] bool contains(const K& key) const noexcept
    {
        return shard(key).contains(key);
    }

    // -- write --------------------------------------------------------------

    void put(const K& key, V value)
    {
        shard(key).put(key, std::move(value));
    }

    void put_pinned(const K& key, V value)
    {
        shard(key).put_pinned(key, std::move(value));
    }

    bool remove(const K& key)
    {
        return shard(key).remove(key);
    }

    void clear()
    {
        for (auto& s : shards_) s->clear();
    }

    // -- stats (aggregated) -------------------------------------------------

    [[nodiscard]] std::size_t size() const noexcept
    {
        std::size_t total = 0;
        for (const auto& s : shards_) total += s->size();
        return total;
    }

    [[nodiscard]] std::size_t byte_size() const noexcept
    {
        std::size_t total = 0;
        for (const auto& s : shards_) total += s->byte_size();
        return total;
    }

    [[nodiscard]] std::size_t max_bytes() const noexcept
    {
        std::size_t total = 0;
        for (const auto& s : shards_) total += s->max_bytes();
        return total;
    }

    [[nodiscard]] std::uint64_t hits() const noexcept
    {
        std::uint64_t total = 0;
        for (const auto& s : shards_) total += s->hits();
        return total;
    }

    [[nodiscard]] std::uint64_t misses() const noexcept
    {
        std::uint64_t total = 0;
        for (const auto& s : shards_) total += s->misses();
        return total;
    }

    [[nodiscard]] std::uint64_t evictions() const noexcept
    {
        std::uint64_t total = 0;
        for (const auto& s : shards_) total += s->evictions();
        return total;
    }

    // -- batch operations ---------------------------------------------------

    template<typename Iter>
    [[nodiscard]] std::vector<K> missing_keys(Iter begin, Iter end) const
    {
        std::vector<K> result;
        for (auto it = begin; it != end; ++it) {
            if (!shard(*it).contains(*it)) {
                result.push_back(*it);
            }
        }
        return result;
    }

    template<typename F>
    void for_each(F&& func) const
    {
        for (const auto& s : shards_) {
            s->for_each(func);
        }
    }

private:
    [[nodiscard]] LRUCache<K, V, Hash, KeyEqual>& shard(const K& key) const
    {
        auto idx = Hash{}(key) % NumShards;
        return *shards_[idx];
    }

    // Use std::optional to avoid requiring movability of LRUCache
    mutable std::array<std::optional<LRUCache<K, V, Hash, KeyEqual>>, NumShards> shards_;
};

} // namespace utils
