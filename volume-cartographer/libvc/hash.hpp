#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <type_traits>

namespace vc {

inline constexpr uint64_t fnv_basis = 14695981039346656037ULL;
inline constexpr uint64_t fnv_prime = 1099511628211ULL;

constexpr uint64_t fnv1a(const void* data, size_t len) noexcept {
    auto p = static_cast<const unsigned char*>(data);
    auto h = fnv_basis;
    for (size_t i = 0; i < len; ++i) {
        h ^= uint64_t(p[i]);
        h *= fnv_prime;
    }
    return h;
}

constexpr uint64_t fnv1a(std::string_view sv) noexcept {
    return fnv1a(sv.data(), sv.size());
}

template<typename T> requires std::is_trivially_copyable_v<T>
constexpr uint64_t fnv1a(const T& v) noexcept {
    return fnv1a(&v, sizeof(T));
}

constexpr size_t hash_combine(size_t seed, size_t value) noexcept {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

template<typename... Ts>
constexpr size_t hash_combine(const Ts&... values) noexcept {
    size_t seed = 0;
    ((seed = hash_combine(seed, std::hash<Ts>{}(values))), ...);
    return seed;
}

} // namespace vc
