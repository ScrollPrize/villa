#pragma once
#include <array>
#include <mutex>
#include <shared_mutex>
#include <cstddef>
#include <functional>
#include <concepts>
#include <type_traits>
#include <span>
#include <vector>
#include <algorithm>

namespace utils {

// RAII guard holding multiple locks in sorted order to prevent deadlock.
template<typename MutexType>
class MultiLockGuard final {
    std::vector<std::unique_lock<MutexType>> locks_;

public:
    MultiLockGuard() = default;
    explicit MultiLockGuard(std::vector<std::unique_lock<MutexType>> locks) noexcept
        : locks_(std::move(locks)) {}

    MultiLockGuard(const MultiLockGuard&) = delete;
    MultiLockGuard& operator=(const MultiLockGuard&) = delete;
    MultiLockGuard(MultiLockGuard&&) noexcept = default;
    MultiLockGuard& operator=(MultiLockGuard&&) noexcept = default;

    [[nodiscard]] std::size_t count() const noexcept { return locks_.size(); }
};

// Per-key lock pool. Serializes operations on the same key without a global lock.
// N must be a power of two for efficient index masking.
template<std::size_t N = 64, typename MutexType = std::mutex>
class LockPool final {
    static_assert(N > 0 && (N & (N - 1)) == 0, "Pool size must be a power of two");

    std::array<MutexType, N> mutexes_{};

public:
    constexpr LockPool() noexcept = default;

    LockPool(const LockPool&) = delete;
    LockPool& operator=(const LockPool&) = delete;

    // Get the lock index for a key.
    template<typename K, typename Hash = std::hash<K>>
    [[nodiscard]] static constexpr std::size_t index(const K& key) noexcept {
        return Hash{}(key) & (N - 1);
    }

    // Get the mutex for a key (by hash).
    template<typename K, typename Hash = std::hash<K>>
    [[nodiscard]] MutexType& get(const K& key) noexcept {
        return mutexes_[index<K, Hash>(key)];
    }

    // RAII exclusive lock guard for a specific key.
    template<typename K, typename Hash = std::hash<K>>
    [[nodiscard]] std::unique_lock<MutexType> lock(const K& key) {
        return std::unique_lock<MutexType>{get<K, Hash>(key)};
    }

    // Try-lock for a specific key. Check owns_lock() on the returned guard.
    template<typename K, typename Hash = std::hash<K>>
    [[nodiscard]] std::unique_lock<MutexType> try_lock(const K& key) {
        return std::unique_lock<MutexType>{get<K, Hash>(key), std::try_to_lock};
    }

    // Lock multiple keys. Deduplicates and sorts by index to prevent deadlock.
    template<typename K, typename Hash = std::hash<K>>
    [[nodiscard]] MultiLockGuard<MutexType> lock_multiple(std::span<const K> keys) {
        // Collect unique indices.
        std::vector<std::size_t> indices;
        indices.reserve(keys.size());
        for (const auto& k : keys)
            indices.push_back(index<K, Hash>(k));
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

        // Lock in sorted order.
        std::vector<std::unique_lock<MutexType>> locks;
        locks.reserve(indices.size());
        for (auto idx : indices)
            locks.emplace_back(mutexes_[idx]);
        return MultiLockGuard<MutexType>{std::move(locks)};
    }

    // Shared lock (only available when MutexType supports it).
    template<typename K, typename Hash = std::hash<K>>
        requires requires(MutexType& m) { m.lock_shared(); }
    [[nodiscard]] std::shared_lock<MutexType> lock_shared(const K& key) {
        return std::shared_lock<MutexType>{get<K, Hash>(key)};
    }

    // Number of slots.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }
};

// Convenience alias for a shared-mutex lock pool.
template<std::size_t N = 64>
using SharedLockPool = LockPool<N, std::shared_mutex>;

} // namespace utils
