// The few types the reader's interfaces share, without the reader itself.
#pragma once

#include "vc_reader/exception_type.hpp"
#include "element_type.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace vcr {

using Index = std::vector<std::int64_t>;

inline std::string chunkKey(const Index& index, const std::string& separator)
{
    std::string key;
    for (std::size_t axis = 0; axis < index.size(); ++axis) {
        if (axis) key += separator;
        key += std::to_string(index[axis]);
    }
    return key;
}

// The separators between a local path's parts: '/', and on Windows a backslash too (what
// std::filesystem::path::string() writes there). In a URL or an S3 key a backslash is part of a name,
// and on other systems a legal file name character. The platform is an argument so any platform can test it.
#ifdef _WIN32
inline constexpr bool onWindows = true;
#else
inline constexpr bool onWindows = false;
#endif
inline const char* locationSeparators(const std::string& location, bool windowsLocalPaths = onWindows)
{
    return windowsLocalPaths && location.find("://") == std::string::npos ? "/\\" : "/";
}

inline bool isSeparator(char c, const char* separators) { return std::string_view(separators).find(c) != std::string_view::npos; }

// Whether `path[0, end)` ends in a drive spec: a letter, at the start or after a separator, then ':'.
// Only a set with a backslash has drives. The separator after a drive spec is the drive root's own.
inline bool endsInDriveSpec(const std::string& path, std::size_t end, const char* separators)
{
    return separators[1] && end >= 2 && path[end - 1] == ':' && std::isalpha(static_cast<unsigned char>(path[end - 2])) &&
           (end == 2 || isSeparator(path[end - 3], separators));
}

// `path` without trailing separators; a drive root keeps its own (`C:` alone is drive C's current directory).
inline std::string trimSeparators(std::string path, const char* separators)
{
    while (path.size() > 1 && isSeparator(path.back(), separators) && !endsInDriveSpec(path, path.size() - 1, separators))
        path.pop_back();
    return path;
}

struct SplitLocation { std::string parent, leaf; };

// `path` (trimmed) cut at its last separator, none if it has none at or after `from`. The separator
// after a drive spec stays with the parent: `C:\a` is `C:\` and `a`.
inline std::optional<SplitLocation> splitLocation(const std::string& path, const char* separators, std::size_t from = 0)
{
    const auto slash = path.find_last_of(separators);
    if (slash == std::string::npos || slash < from)
        return std::nullopt;
    return SplitLocation{path.substr(0, slash + (endsInDriveSpec(path, slash, separators) ? 1 : 0)), path.substr(slash + 1)};
}

// The parent and the number of a path ending in a numbered level (`.../volume.zarr/5`).
inline std::optional<SplitLocation> numberedLevel(const std::string& path, const char* separators)
{
    auto split = splitLocation(trimSeparators(path, separators), separators);
    if (!split || split->leaf.empty() || !std::all_of(split->leaf.begin(), split->leaf.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
        return std::nullopt;
    return split;
}

// How requests authenticate.
// A key below a remote location: slashes between them collapse, and a query (a signed
// URL's token) stays at the end, its bytes untouched. Anything without a scheme is a
// path, where ? and # are plain characters; a local Windows path is trimmed like parentLocation.
inline std::string joinLocation(const std::string& location, const std::string& key, const char* separators = nullptr)
{
    if (!separators)
        separators = locationSeparators(location);
    const auto query = location.find("://") == std::string::npos ? std::string::npos : location.find_first_of("?#");
    auto end = query == std::string::npos ? location.size() : query;
    while (end > 0 && isSeparator(location[end - 1], separators) && !endsInDriveSpec(location, end - 1, separators))
        --end;
    const auto start = std::min(key.find_first_not_of('/'), key.size());
    const bool driveRoot = end > 0 && isSeparator(location[end - 1], separators);
    return location.substr(0, end) + (driveRoot ? "" : "/") + key.substr(start) + (query == std::string::npos ? "" : location.substr(query));
}

// The location above `location`: its last path segment removed, trailing slashes ignored, a
// query or fragment kept as joinLocation keeps it. A location with nothing above it (a bare
// host, a path without a slash) comes back unchanged.
inline std::string parentLocation(const std::string& location, const char* separators = nullptr)
{
    if (!separators)
        separators = locationSeparators(location);
    const bool url = location.find("://") != std::string::npos;
    const auto query = url ? location.find_first_of("?#") : std::string::npos;
    const std::string suffix = query == std::string::npos ? std::string() : location.substr(query);
    const auto path = trimSeparators(location.substr(0, query), separators);
    const auto split = splitLocation(path, separators, url ? path.find("://") + 3 : 0);
    return (split ? split->parent : path) + suffix;
}

struct Auth {
    std::string kind;  // "" (anonymous), "basic" or "s3"
    std::string user;  // basic: user name; s3: access key id
    std::string secret;  // basic: password; s3: secret access key
    std::string token;   // s3: session token (may be empty)
    std::string region;  // s3: the bucket's region
};

// 401/403: the request was not allowed (missing, wrong or expired credentials).
struct VCR_EXCEPTION_TYPE HttpDenied : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A chunk an offline read needs is in none of the disk layers.
struct VCR_EXCEPTION_TYPE CacheMiss : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A read stopped because its cancel flag was set: chunks not started by then were not read.
struct VCR_EXCEPTION_TYPE Cancelled : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace vcr
