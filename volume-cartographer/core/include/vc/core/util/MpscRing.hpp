#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vc::util {

// Bounded MPSC ring buffer using Vyukov's sequence-based protocol.
// Multiple producers, single consumer. Capacity must be a power of two.
//
// Each slot carries a sequence counter that serializes the "reserved /
// committed / consumed" states across producers and the consumer without
// a mutex. Producers race on `tail_` via CAS; the consumer owns `head_`
// plain.
//
// T must be trivially copyable so push/pop can use value assignment
// without running destructors on half-written slots.
template <typename T, std::size_t Capacity>
class MpscRing {
    static_assert(Capacity > 0 && (Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable");

public:
    MpscRing() noexcept
    {
        for (std::size_t i = 0; i < Capacity; ++i) {
            slots_[i].seq.store(i, std::memory_order_relaxed);
        }
    }

    // Producer side. Returns false if the ring is full.
    bool try_push(const T& v) noexcept
    {
        std::size_t pos = tail_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& s = slots_[pos & (Capacity - 1)];
            std::size_t seq = s.seq.load(std::memory_order_acquire);
            auto diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);
            if (diff == 0) {
                if (tail_.compare_exchange_weak(pos, pos + 1,
                                                std::memory_order_relaxed)) {
                    s.value = v;
                    s.seq.store(pos + 1, std::memory_order_release);
                    return true;
                }
                // CAS failed; pos was reloaded with new tail, retry.
            } else if (diff < 0) {
                return false;
            } else {
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
    }

    // Consumer side (single-threaded). Returns false if the ring is empty.
    bool try_pop(T& out) noexcept
    {
        Slot& s = slots_[head_ & (Capacity - 1)];
        std::size_t seq = s.seq.load(std::memory_order_acquire);
        auto diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(head_ + 1);
        if (diff == 0) {
            out = s.value;
            s.seq.store(head_ + Capacity, std::memory_order_release);
            ++head_;
            return true;
        }
        return false;
    }

    // Approximate size (producer-side view). Not exact under contention;
    // intended for overflow-rate dashboards, not correctness.
    [[nodiscard]] std::size_t approx_size() const noexcept
    {
        return tail_.load(std::memory_order_relaxed) - head_;
    }

    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return Capacity; }

private:
    struct alignas(64) Slot {
        std::atomic<std::size_t> seq{0};
        T value{};
    };

    std::array<Slot, Capacity> slots_{};
    alignas(64) std::atomic<std::size_t> tail_{0};
    alignas(64) std::size_t head_{0};
};

}  // namespace vc::util
