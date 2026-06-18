#pragma once

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

namespace vc3d::line_annotation {

inline std::string normalizedFiberUsername(std::string username)
{
    const auto first = std::find_if_not(username.begin(), username.end(), [](unsigned char ch) {
        return std::isspace(ch);
    });
    const auto last = std::find_if_not(username.rbegin(), username.rend(), [](unsigned char ch) {
        return std::isspace(ch);
    }).base();
    if (first >= last) {
        return "anon";
    }

    std::string normalized(first, last);
    for (char& ch : normalized) {
        const auto uch = static_cast<unsigned char>(ch);
        if (!std::isalnum(uch) && ch != '-' && ch != '_') {
            ch = '_';
        }
    }

    if (normalized.empty() ||
        std::all_of(normalized.begin(), normalized.end(), [](char ch) { return ch == '_'; })) {
        return "anon";
    }
    return normalized;
}

inline std::string fiberFileStem(const std::string& username,
                                 const std::string& startedAt,
                                 uint64_t sequence)
{
    std::ostringstream stem;
    stem.imbue(std::locale::classic());
    stem << normalizedFiberUsername(username) << '_' << startedAt << '_'
         << std::setw(6) << std::setfill('0') << sequence;
    return stem.str();
}

inline std::string fiberFileName(const std::string& username,
                                 const std::string& startedAt,
                                 uint64_t sequence)
{
    return fiberFileStem(username, startedAt, sequence) + ".json";
}

} // namespace vc3d::line_annotation
