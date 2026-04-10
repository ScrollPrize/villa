#pragma once
#include <array>
#include <cstddef>
#include <functional>
#include <mutex>
#include <shared_mutex>

namespace vc {

template<size_t N = 64, typename Mutex = std::mutex>
class LockPool {
    static_assert((N & (N - 1)) == 0);
    std::array<Mutex, N> mutexes_{};

public:
    template<typename K, typename Hash = std::hash<K>>
    static constexpr size_t index(const K& key) noexcept {
        return Hash{}(key) & (N - 1);
    }

    template<typename K, typename Hash = std::hash<K>>
    std::unique_lock<Mutex> lock(const K& key) {
        return std::unique_lock{mutexes_[index<K, Hash>(key)]};
    }

    template<typename K, typename Hash = std::hash<K>>
    std::unique_lock<Mutex> try_lock(const K& key) {
        return std::unique_lock{mutexes_[index<K, Hash>(key)], std::try_to_lock};
    }

    template<typename K, typename Hash = std::hash<K>>
        requires requires(Mutex& m) { m.lock_shared(); }
    std::shared_lock<Mutex> lock_shared(const K& key) {
        return std::shared_lock{mutexes_[index<K, Hash>(key)]};
    }
};

template<size_t N = 64>
using SharedLockPool = LockPool<N, std::shared_mutex>;

} // namespace vc
