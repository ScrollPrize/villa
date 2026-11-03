// TIFF-style, MSB-first LZW byte streams (imagecodecs_lzw Zarr chunks).
// Codes grow from 9 to 12 bits using TIFF's early-change convention.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace vcr {

inline void decode_lzw(const std::uint8_t* src, std::size_t size,
                       std::uint8_t* dst, std::size_t expected)
{
    std::array<std::uint16_t, 4096> prefix{};
    std::array<std::uint8_t, 4096> suffix{}, stack{};
    std::uint32_t buffer = 0;
    unsigned available = 0, width = 9, next = 258;
    std::size_t input = 0, output = 0;
    int previous = -1;
    std::uint8_t first = 0;
    bool started = false;
    for (;;) {
        while (available < width) {
            if (input == size)
                throw std::runtime_error("lzw chunk is truncated (missing end code)");
            buffer = (buffer << 8) | src[input++];
            available += 8;
        }
        available -= width;
        const unsigned code = (buffer >> available) & ((1u << width) - 1);
        if (!started && code != 256)
            throw std::runtime_error("lzw chunk must start with a clear code");
        started = true;
        if (code == 256) {
            width = 9;
            next = 258;
            previous = -1;
            continue;
        }
        if (code == 257) {
            if (output != expected)
                throw std::runtime_error("lzw chunk decoded size does not match the array's chunk size");
            return;
        }
        unsigned cursor = code;
        std::size_t length = 0;
        if (code == next && previous >= 0) {
            // The encoder just added previous-string + its first byte.
            stack[length++] = first;
            cursor = static_cast<unsigned>(previous);
        } else if (code >= next) {
            throw std::runtime_error("lzw chunk contains an invalid dictionary code");
        }
        while (cursor >= 258) {
            if (cursor >= next || length == stack.size())
                throw std::runtime_error("lzw chunk contains an invalid dictionary chain");
            stack[length++] = suffix[cursor];
            cursor = prefix[cursor];
        }
        if (cursor >= 256 || length == stack.size())
            throw std::runtime_error("lzw chunk contains an invalid literal");
        first = static_cast<std::uint8_t>(cursor);
        stack[length++] = first;
        if (length > expected - output)
            throw std::runtime_error("lzw chunk decodes beyond the array's chunk size");
        for (std::size_t i = length; i > 0; --i)
            dst[output++] = stack[i - 1];
        if (previous >= 0 && next < 4096) {
            prefix[next] = static_cast<std::uint16_t>(previous);
            suffix[next++] = first;
            if (width < 12 && next == (1u << width) - 1)
                ++width;
        }
        previous = static_cast<int>(code);
    }
}

}  // namespace vcr
