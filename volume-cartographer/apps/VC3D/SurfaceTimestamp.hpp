#pragma once

#include <cctype>
#include <string>
#include <string_view>

namespace vc3d {

inline std::string surfaceTimestampForDisplay(std::string_view timestamp)
{
    if (timestamp.size() < 14) {
        return std::string(timestamp);
    }
    for (std::size_t i = 0; i < 14; ++i) {
        if (!std::isdigit(static_cast<unsigned char>(timestamp[i]))) {
            return std::string(timestamp);
        }
    }

    std::string display;
    display.reserve(19);
    display.append(timestamp.substr(0, 4));
    display.push_back('-');
    display.append(timestamp.substr(4, 2));
    display.push_back('-');
    display.append(timestamp.substr(6, 2));
    display.push_back(' ');
    display.append(timestamp.substr(8, 2));
    display.push_back(':');
    display.append(timestamp.substr(10, 2));
    display.push_back(':');
    display.append(timestamp.substr(12, 2));
    return display;
}

} // namespace vc3d
