// Cost-bounded LRU storage. Callers own synchronization and resource lifetimes.
#pragma once
#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>

namespace vcr {
template<class T, class Key = std::string>
class Lru {
public:
    explicit Lru(std::size_t limit) : limit_(limit) {}
    Lru(const Lru&) = delete;
    Lru& operator=(const Lru&) = delete;

    const T* peek(const Key& key) const
    {
        const auto it = index_.find(key);
        return it == index_.end() ? nullptr : &it->second->value;
    }

    T* get(const Key& key)
    {
        const auto it = index_.find(key);
        if (it == index_.end()) return nullptr;
        order_.splice(order_.begin(), order_, it->second);
        return &it->second->value;
    }
    void put(const Key& key, T value, std::size_t cost)
    {
        if (cost > limit_) return;
        if (const auto it = index_.find(key); it != index_.end()) {
            bytes_ -= it->second->cost;
            order_.erase(it->second);
            index_.erase(it);
        }
        while (!order_.empty() && bytes_ > limit_ - cost) {
            const auto& last = order_.back();
            bytes_ -= last.cost;
            index_.erase(last.key);
            order_.pop_back();
        }
        order_.push_front({key, std::move(value), cost});
        try { index_.emplace(key, order_.begin()); }
        catch (...) { order_.pop_front(); throw; }
        bytes_ += cost;
    }
    void clear() { index_.clear(); order_.clear(); bytes_ = 0; }
    std::size_t bytes() const { return bytes_; }
    std::size_t size() const { return index_.size(); }
    std::size_t limit() const { return limit_; }
private:
    struct Entry { Key key; T value; std::size_t cost; };
    std::size_t limit_, bytes_ = 0;
    std::list<Entry> order_;
    std::unordered_map<Key, typename std::list<Entry>::iterator> index_;
};
} // namespace vcr
