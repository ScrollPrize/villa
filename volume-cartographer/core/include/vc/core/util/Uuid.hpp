#pragma once

#include <array>
#include <algorithm>
#include <stdexcept>
#include <cctype>
#include <random>
#include <string>

namespace vc::util {

inline bool isUuid(const std::string& value)
{
    if (value.size() != 36) return false;
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (value[i] != '-') return false;
        } else if (!std::isxdigit(static_cast<unsigned char>(value[i]))) {
            return false;
        }
    }
    return true;
}

inline std::string canonicalUuid(std::string value)
{
    if (!isUuid(value)) throw std::invalid_argument("invalid UUID: " + value);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return value;
}

inline std::string newUuid()
{
    std::random_device random;
    std::array<unsigned char, 16> bytes{};
    for (std::size_t i = 0; i < bytes.size(); i += 4) {
        const auto word = random();
        for (std::size_t j = 0; j < 4; ++j)
            bytes[i + j] = static_cast<unsigned char>(word >> (8 * j));
    }
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0f) | 0x40);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3f) | 0x80);

    constexpr char hex[] = "0123456789abcdef";
    std::string uuid;
    uuid.reserve(36);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) uuid.push_back('-');
        uuid.push_back(hex[bytes[i] >> 4]);
        uuid.push_back(hex[bytes[i] & 0x0f]);
    }
    return uuid;
}

} // namespace vc::util
