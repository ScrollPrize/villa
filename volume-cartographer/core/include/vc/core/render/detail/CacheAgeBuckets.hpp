#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>

namespace vc::render::detail {

// Caller supplies monotonically increasing generations and synchronization.
// Circular lists use pooled indices, including 33 permanent sentinel nodes.
// Recycling a generation splices its list into overflow without visiting it.
template<class T>
class CacheAgeBuckets {
public:
    using Handle = std::size_t;
    static constexpr Handle none = std::numeric_limits<Handle>::max();
    static constexpr Handle generations = 32;

    explicit CacheAgeBuckets(std::uint64_t now = 0) : current_(now), nodes_(33)
    {
        for (Handle i = 0; i <= generations; ++i)
            nodes_[i].prev = nodes_[i].next = i;
    }

    void advance(std::uint64_t now)
    {
        if (now <= current_)
            return;
        const auto steps = std::min<std::uint64_t>(now - current_, generations);
        for (std::uint64_t i = 1; i <= steps; ++i) {
            const Handle head = (current_ + i) % generations;
            if (nodes_[head].next == head)
                continue;
            const Handle first = nodes_[head].next, last = nodes_[head].prev;
            const Handle tail = nodes_[generations].prev;
            nodes_[tail].next = first;
            nodes_[first].prev = tail;
            nodes_[last].next = generations;
            nodes_[generations].prev = last;
            nodes_[head].next = nodes_[head].prev = head;
            nonempty_ &= ~(std::uint64_t{1} << head);
            nonempty_ |= std::uint64_t{1} << generations;
        }
        current_ = now;
    }

    Handle insert(T value, std::uint64_t generation)
    {
        Handle id;
        if (free_ == none) {
            id = nodes_.size();
            nodes_.emplace_back();
        } else {
            id = free_;
            free_ = nodes_[id].next;
        }
        nodes_[id] = Node{};
        nodes_[id].value = value;
        nodes_[id].latest = std::min(generation, current_);
        resume(id);
        return id;
    }

    void touch(Handle id, std::uint64_t generation)
    {
        nodes_[id].latest = std::min(generation, current_);
    }

    void suspend(Handle id)
    {
        auto& node = nodes_[id];
        if (!node.linked)
            return;
        const Handle head = bucket(node.placed);
        nodes_[node.prev].next = node.next;
        nodes_[node.next].prev = node.prev;
        node.linked = false;
        if (nodes_[head].next == head)
            nonempty_ &= ~(std::uint64_t{1} << head);
    }

    void resume(Handle id)
    {
        auto& node = nodes_[id];
        if (node.linked)
            return;
        const Handle head = bucket(node.latest), tail = nodes_[head].prev;
        node.prev = tail;
        node.next = head;
        node.placed = node.latest;
        node.linked = true;
        nodes_[tail].next = id;
        nodes_[head].prev = id;
        nonempty_ |= std::uint64_t{1} << head;
    }

    void erase(Handle id)
    {
        suspend(id);
        nodes_[id].value = T{};
        nodes_[id].next = free_;
        free_ = id;
    }

    // Returns an unlinked victim. Lazy re-bucketing is bounded by workBudget;
    // none with !empty() means yield the caller's lock and try again.
    Handle takeOldest(std::size_t workBudget = 128)
    {
        while (nonempty_ && workBudget--) {
            Handle head = generations;
            if (!(nonempty_ & (std::uint64_t{1} << generations))) {
                const auto start = (current_ + 1) % generations;
                const auto mask = static_cast<std::uint32_t>(nonempty_);
                head = (std::countr_zero(std::rotr(mask, int(start))) + start) % generations;
            }
            const Handle id = nodes_[head].next;
            suspend(id);
            if (bucket(nodes_[id].latest) != head) {
                resume(id);
                continue;
            }
            return id;
        }
        return none;
    }

    bool empty() const { return nonempty_ == 0; }
    T value(Handle id) const { return nodes_[id].value; }
    std::uint64_t lastAccess(Handle id) const { return nodes_[id].latest; }

private:
    Handle bucket(std::uint64_t generation) const
    {
        return current_ - generation >= generations ? generations : generation % generations;
    }
    struct Node {
        Handle prev = none, next = none;
        std::uint64_t latest = 0, placed = 0;
        T value{};
        bool linked = false;
    };
    std::uint64_t current_ = 0, nonempty_ = 0;
    std::vector<Node> nodes_;
    Handle free_ = none;
};

} // namespace vc::render::detail
