#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace vc {

// Byte-budgeted generation-based LRU cache. Thread-safe via shared_mutex.
// V must be movable. If V has a byte_size() method it's used for sizing,
// otherwise sizeof(V) is used.
template<typename K, typename V,
         typename Hash = std::hash<K>, typename Eq = std::equal_to<K>>
class LRUCache {
    struct Entry {
        V value;
        size_t bytes;
        mutable std::atomic<uint64_t> gen;
        bool pinned;

        Entry(V v, size_t b, uint64_t g, bool p)
            : value(std::move(v)), bytes(b), gen(g), pinned(p) {}
        Entry(Entry&& o) noexcept
            : value(std::move(o.value)), bytes(o.bytes),
              gen(o.gen.load(std::memory_order_relaxed)), pinned(o.pinned) {}
        Entry& operator=(Entry&& o) noexcept {
            value = std::move(o.value);
            bytes = o.bytes;
            gen.store(o.gen.load(std::memory_order_relaxed), std::memory_order_relaxed);
            pinned = o.pinned;
            return *this;
        }
    };

    std::unordered_map<K, Entry, Hash, Eq> map_;
    mutable std::shared_mutex mu_;
    mutable std::atomic<uint64_t> gen_{0};
    mutable std::atomic<size_t> bytes_{0};
    mutable std::atomic<uint64_t> hits_{0}, misses_{0}, evictions_{0};
    size_t max_bytes_;
    std::function<size_t(const V&)> size_fn_;

    size_t val_size(const V& v) const {
        if (size_fn_) return size_fn_(v);
        if constexpr (requires { v.byte_size(); })
            return v.byte_size();
        return sizeof(V);
    }

    void evict() {
        auto target = size_t(max_bytes_ * 0.9375);
        struct Cand { K key; size_t bytes; uint64_t gen; };
        std::vector<Cand> cands;
        {
            std::shared_lock lk(mu_);
            cands.reserve(map_.size());
            for (auto& [k, e] : map_)
                if (!e.pinned)
                    cands.push_back({k, e.bytes, e.gen.load(std::memory_order_relaxed)});
        }
        auto n = std::max<size_t>(1, cands.size() / 8);
        std::partial_sort(cands.begin(), cands.begin() + n, cands.end(),
            [](auto& a, auto& b) { return a.gen < b.gen; });
        std::unique_lock lk(mu_);
        for (size_t i = 0; i < n && bytes_.load(std::memory_order_relaxed) > target; ++i) {
            auto it = map_.find(cands[i].key);
            if (it == map_.end() || it->second.pinned) continue;
            if (it->second.gen.load(std::memory_order_relaxed) != cands[i].gen) continue;
            bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
            map_.erase(it);
            evictions_.fetch_add(1, std::memory_order_relaxed);
        }
    }

public:
    explicit LRUCache(size_t max_bytes = 1ULL << 30,
                      std::function<size_t(const V&)> size_fn = nullptr)
        : max_bytes_(max_bytes), size_fn_(std::move(size_fn)) {}

    std::optional<V> get(const K& key) const {
        std::shared_lock lk(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) { misses_.fetch_add(1, std::memory_order_relaxed); return {}; }
        it->second.gen.store(gen_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
        hits_.fetch_add(1, std::memory_order_relaxed);
        return it->second.value;
    }

    bool contains(const K& key) const {
        std::shared_lock lk(mu_);
        return map_.contains(key);
    }

    void put(const K& key, V value, bool pinned = false) {
        auto sz = val_size(value);
        {
            std::unique_lock lk(mu_);
            if (auto it = map_.find(key); it != map_.end()) {
                bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
                it->second = Entry{std::move(value), sz,
                    gen_.fetch_add(1, std::memory_order_relaxed), pinned};
            } else {
                map_.emplace(key, Entry{std::move(value), sz,
                    gen_.fetch_add(1, std::memory_order_relaxed), pinned});
            }
            bytes_.fetch_add(sz, std::memory_order_relaxed);
        }
        if (bytes_.load(std::memory_order_relaxed) > max_bytes_) evict();
    }

    bool remove(const K& key) {
        std::unique_lock lk(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) return false;
        bytes_.fetch_sub(it->second.bytes, std::memory_order_relaxed);
        map_.erase(it);
        return true;
    }

    void clear() { std::unique_lock lk(mu_); map_.clear(); bytes_.store(0); }

    size_t size() const { std::shared_lock lk(mu_); return map_.size(); }
    size_t byte_size() const { return bytes_.load(std::memory_order_relaxed); }
    size_t max_bytes() const { return max_bytes_; }
    uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
    uint64_t misses() const { return misses_.load(std::memory_order_relaxed); }
    uint64_t evictions() const { return evictions_.load(std::memory_order_relaxed); }

    template<typename Iter>
    std::vector<K> missing_keys(Iter begin, Iter end) const {
        std::vector<K> out;
        std::shared_lock lk(mu_);
        for (auto it = begin; it != end; ++it)
            if (!map_.contains(*it)) out.push_back(*it);
        return out;
    }

    template<typename F>
    void for_each(F&& f) const {
        std::shared_lock lk(mu_);
        for (auto& [k, e] : map_) f(k, e.value);
    }
};

} // namespace vc
