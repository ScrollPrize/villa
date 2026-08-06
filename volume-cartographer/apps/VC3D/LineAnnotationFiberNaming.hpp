#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <locale>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

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

struct FiberFileNameIdentity {
    std::string username;
    std::string startedAt;
    uint64_t sequence = 0;
};

// Inverse of fiberFileName for canonical names
// (<username>_<startedAt>_<sequence>.json). Usernames may themselves
// contain underscores, so the stem is parsed from the right: the last
// group must be the numeric sequence and the one before it the
// YYYYMMDDThhmmsszzz timestamp. Returns nullopt for non-canonical names.
inline std::optional<FiberFileNameIdentity> parsedFiberFileNameIdentity(
    const std::string& fileName)
{
    constexpr std::string_view kSuffix = ".json";
    if (fileName.size() <= kSuffix.size() ||
        fileName.compare(fileName.size() - kSuffix.size(), kSuffix.size(),
                         kSuffix) != 0) {
        return std::nullopt;
    }
    const std::string stem = fileName.substr(0, fileName.size() - kSuffix.size());

    const auto sequenceSep = stem.rfind('_');
    if (sequenceSep == std::string::npos || sequenceSep + 1 >= stem.size()) {
        return std::nullopt;
    }
    const std::string sequenceText = stem.substr(sequenceSep + 1);
    if (!std::all_of(sequenceText.begin(), sequenceText.end(), [](unsigned char ch) {
            return std::isdigit(ch);
        })) {
        return std::nullopt;
    }

    const auto startedAtSep = stem.rfind('_', sequenceSep - 1);
    if (startedAtSep == std::string::npos || startedAtSep == 0) {
        return std::nullopt;
    }
    const std::string startedAt =
        stem.substr(startedAtSep + 1, sequenceSep - startedAtSep - 1);
    constexpr size_t kStartedAtLength = 18;  // YYYYMMDDThhmmsszzz
    if (startedAt.size() != kStartedAtLength || startedAt[8] != 'T') {
        return std::nullopt;
    }
    for (size_t index = 0; index < startedAt.size(); ++index) {
        if (index == 8) {
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(startedAt[index]))) {
            return std::nullopt;
        }
    }

    FiberFileNameIdentity identity;
    identity.username = stem.substr(0, startedAtSep);
    if (identity.username != normalizedFiberUsername(identity.username)) {
        return std::nullopt;
    }
    identity.startedAt = startedAt;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed =
        std::strtoull(sequenceText.c_str(), &end, 10);
    if (errno != 0 || end != sequenceText.c_str() + sequenceText.size()) {
        return std::nullopt;
    }
    identity.sequence = parsed;
    return identity;
}

} // namespace vc3d::line_annotation
