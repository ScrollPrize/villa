#pragma once
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <functional>
#include <type_traits>
#include <array>
#include <string_view>
#include <tuple>
#include <utility>

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

// --- FNV-1a hash from a trivially-copyable value ---
template <typename T>
    requires std::is_trivially_copyable_v<T>
[[nodiscard]] constexpr std::uint64_t
fnv1a_value(const T& v) noexcept
{
    if consteval {
        // At compile time we cannot reinterpret memory, so use bit_cast to an
        // array of bytes and hash that.
        auto bytes = __builtin_bit_cast(std::array<unsigned char, sizeof(T)>, v);
        auto h = fnv_offset_basis;
        for (auto b : bytes) {
            h ^= static_cast<std::uint64_t>(b);
            h *= fnv_prime;
        }
        return h;
    } else {
        return fnv1a(&v, sizeof(T));
    }
}

// --- boost-style hash_combine ---
[[nodiscard]] constexpr std::size_t
hash_combine(std::size_t seed, std::size_t value) noexcept
{
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

// --- Variadic hash_combine_values ---
template <typename... Ts>
[[nodiscard]] constexpr std::size_t
hash_combine_values(const Ts&... values) noexcept
{
    std::size_t seed = 0;
    ((seed = hash_combine(seed, std::hash<Ts>{}(values))), ...);
    return seed;
}

// --- Generic Hasher struct ---
namespace detail {

template <typename T, typename = void>
struct is_std_hashable : std::false_type {};

template <typename T>
struct is_std_hashable<
    T, std::void_t<decltype(std::hash<T>{}(std::declval<const T&>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_std_hashable_v = is_std_hashable<T>::value;

// Detect tuple-like types with structured data() member (e.g. Vec).
template <typename T, typename = void>
struct is_tuple_sized : std::false_type {};

template <typename T>
struct is_tuple_sized<T, std::void_t<decltype(std::tuple_size<T>::value)>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_tuple_sized_v = is_tuple_sized<T>::value;

} // namespace detail

struct Hasher {
    using is_transparent = void;

    // Hash anything that std::hash supports directly.
    template <typename T>
        requires detail::is_std_hashable_v<T>
    [[nodiscard]] constexpr std::size_t
    operator()(const T& v) const noexcept
    {
        return std::hash<T>{}(v);
    }

    // Hash std::pair.
    template <typename A, typename B>
    [[nodiscard]] constexpr std::size_t
    operator()(const std::pair<A, B>& p) const noexcept
    {
        return hash_combine((*this)(p.first), (*this)(p.second));
    }

    // Hash std::tuple.
    template <typename... Ts>
    [[nodiscard]] constexpr std::size_t
    operator()(const std::tuple<Ts...>& t) const noexcept
    {
        return std::apply(
            [this](const auto&... elems) {
                std::size_t seed = 0;
                ((seed = hash_combine(seed, (*this)(elems))), ...);
                return seed;
            },
            t);
    }

    // Hash std::array.
    template <typename T, std::size_t N>
    [[nodiscard]] constexpr std::size_t
    operator()(const std::array<T, N>& a) const noexcept
    {
        std::size_t seed = 0;
        for (const auto& elem : a) {
            seed = hash_combine(seed, (*this)(elem));
        }
        return seed;
    }

    // Hash any tuple-sized type with a data() member (e.g. Vec<T,N>).
    // This overload is lower priority than the above; it activates for types
    // that are tuple-sized but NOT directly std::hash-able, and that provide
    // a contiguous data() pointer.
    template <typename T>
        requires(detail::is_tuple_sized_v<T> && !detail::is_std_hashable_v<T> &&
                 requires(const T& t) { t.data(); })
    [[nodiscard]] constexpr std::size_t
    operator()(const T& v) const noexcept
    {
        constexpr auto N = std::tuple_size_v<T>;
        std::size_t seed = 0;
        const auto* p = v.data();
        for (std::size_t i = 0; i < N; ++i) {
            seed = hash_combine(seed, (*this)(p[i]));
        }
        return seed;
    }
};

// --- Compile-time string hash literal ---
[[nodiscard]] consteval std::uint64_t
operator""_hash(const char* str, std::size_t len) noexcept
{
    auto h = fnv_offset_basis;
    for (std::size_t i = 0; i < len; ++i) {
        h ^= static_cast<std::uint64_t>(
            static_cast<unsigned char>(str[i]));
        h *= fnv_prime;
    }
    return h;
}

} // namespace utils
