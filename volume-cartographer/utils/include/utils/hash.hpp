#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace utils {

// --- FNV-1a constants (64-bit) ---
inline constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ULL;
inline constexpr std::uint64_t fnv_prime = 1099511628211ULL;

// --- FNV-1a hash from raw bytes ---
[[nodiscard]] constexpr std::uint64_t
fnv1a(const void* data, std::size_t len) noexcept
{
    auto p = static_cast<const unsigned char*>(data);
    auto h = fnv_offset_basis;
    for (std::size_t i = 0; i < len; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= fnv_prime;
    }
    return h;
}

// --- FNV-1a hash from string_view ---
[[nodiscard]] constexpr std::uint64_t
fnv1a(std::string_view sv) noexcept
{
    return fnv1a(sv.data(), sv.size());
}

} // namespace utils
