#pragma once

// Random draws that give the same values on every standard library.
//
// The C++ standard leaves std::uniform_int_distribution's algorithm to each
// standard library: from the same std::mt19937_64 output, libc++ (macOS) draws
// other integers than libstdc++ (Linux), as libstdc++ itself did before GCC 11.
// std::generate_canonical is specified, but P0952R2 changed its specification,
// GCC 16 follows the new one, and libc++ can return 1. A seeded fit then
// samples different points from one build to another. These helpers write
// both draws out with the rules of libstdc++ in GCC 11 to 15, so 64-bit builds
// with those compilers keep the values they have drawn so far and every other
// build draws the same ones.

#include <cmath>
#include <cstdint>
#include <limits>

namespace vc_spiral {

// High and low 64 bits of the 128-bit product a * b.
inline void multiply_wide(uint64_t a, uint64_t b, uint64_t& high, uint64_t& low)
{
    const uint64_t a_low = a & 0xffffffffULL, a_high = a >> 32;
    const uint64_t b_low = b & 0xffffffffULL, b_high = b >> 32;
    const uint64_t low_low = a_low * b_low;
    const uint64_t low_high = a_low * b_high;
    const uint64_t high_low = a_high * b_low;
    const uint64_t middle = (low_low >> 32) + (low_high & 0xffffffffULL)
        + (high_low & 0xffffffffULL);
    low = (middle << 32) | (low_low & 0xffffffffULL);
    high = a_high * b_high + (low_high >> 32) + (high_low >> 32) + (middle >> 32);
}

// Uniform integer in [0, range), range > 0, from a 64-bit engine: Lemire's
// nearly divisionless method, which libstdc++'s uniform_int_distribution uses
// for 64-bit engines (GCC 11 and later), with the same engine calls.
template <typename Engine>
uint64_t uniform_below(Engine& engine, uint64_t range)
{
    static_assert(Engine::min() == 0
                      && Engine::max() == std::numeric_limits<uint64_t>::max(),
                  "uniform_below needs an engine with 64 random bits");
    uint64_t high = 0;
    uint64_t low = 0;
    multiply_wide(engine(), range, high, low);
    if (low < range) {
        const uint64_t threshold = (0 - range) % range;
        while (low < threshold)
            multiply_wide(engine(), range, high, low);
    }
    return high;
}

// Uniform float in [0, 1) from one 64-bit engine call: the value of
// libstdc++'s generate_canonical<float, 24> up to GCC 15 (libc++ gives the
// same, except that it can return 1 at the very top of the range).
template <typename Engine>
float uniform_unit_float(Engine& engine)
{
    static_assert(Engine::min() == 0
                      && Engine::max() == std::numeric_limits<uint64_t>::max(),
                  "uniform_unit_float needs an engine with 64 random bits");
    const float value = static_cast<float>(engine()) * 0x1p-64f;
    return value < 1.0f ? value : std::nextafter(1.0f, 0.0f);
}

} // namespace vc_spiral
