#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Content hashing shared by the Fiber Map's layout and its sheet-normal
// field adapter: two independent FNV-1a lanes over raw bytes (IEEE-754
// doubles hashed by bit pattern, strings length-prefixed, field order fixed).
// The bytes are what the layout's cache keys and result digests have always
// been; nothing here may change them.
namespace vc3d::fiber_map
{

struct ContentDigest {
    uint64_t a = 0;
    uint64_t b = 0;
    bool operator==(const ContentDigest& other) const
    {
        return a == other.a && b == other.b;
    }
    bool operator!=(const ContentDigest& other) const { return !(*this == other); }
};

inline void hashBytes(ContentDigest& digest, const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    constexpr uint64_t kPrimeA = 1099511628211ULL;
    constexpr uint64_t kPrimeB = 0x100000001b3ULL ^ 0x9e3779b97f4a7c15ULL;
    uint64_t a = digest.a;
    uint64_t b = digest.b;
    for (std::size_t i = 0; i < size; ++i) {
        a = (a ^ bytes[i]) * kPrimeA;
        b = (b ^ bytes[i]) * (kPrimeB | 1ULL);
    }
    digest.a = a;
    digest.b = b;
}

inline void hashU64(ContentDigest& digest, uint64_t value)
{
    hashBytes(digest, &value, sizeof(value));
}

inline void hashDouble(ContentDigest& digest, double value)
{
    hashBytes(digest, &value, sizeof(value));
}

inline void hashString(ContentDigest& digest, const std::string& value)
{
    hashU64(digest, value.size());
    hashBytes(digest, value.data(), value.size());
}

inline ContentDigest seededDigest(uint64_t seed)
{
    ContentDigest digest{14695981039346656037ULL, 0xcbf29ce484222325ULL};
    hashU64(digest, seed);
    return digest;
}

} // namespace vc3d::fiber_map
