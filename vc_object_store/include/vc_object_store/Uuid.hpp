#pragma once

#include <array>
#include <algorithm>
#include <stdexcept>
#include <cctype>
#include <random>
#include <string>
#include <string_view>

namespace vc::objects::util {

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
    // UUIDs contain ASCII hex, not locale-dependent text. Validate and lower
    // in one pass rather than two ctype calls for every character of every key.
    const auto invalid=[&]{throw std::invalid_argument("invalid UUID: " + value);};
    if(value.size()!=36)invalid();
    for(size_t i=0;i<value.size();++i){
        auto& c=value[i];
        if(i==8||i==13||i==18||i==23){if(c!='-')invalid();}
        else if(c>='0'&&c<='9'){}
        else if(c>='a'&&c<='f'){}
        else if(c>='A'&&c<='F')c=char(c+('a'-'A'));
        else invalid();
    }
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

// Name-based (RFC 4122 version 5) UUID in the object store's namespace: the
// same name always yields the same UUID, so independent imports of one
// UUID-less source agree on its identity.
std::string nameUuid(std::string_view name);
// The same in another namespace (a UUID), e.g. one shared with other tools.
std::string nameUuid(const std::string& namespaceUuid, std::string_view name);
std::string nameUuid(const std::array<unsigned char,16>& namespaceUuid, std::string_view name);

} // namespace vc::objects::util
