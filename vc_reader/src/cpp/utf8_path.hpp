// Locations travel through the reader as UTF-8 std::string: that is what Python,
// JSON and URLs hand it, and what std::filesystem::path gives and takes on Linux,
// macOS and MinGW. Microsoft's standard library instead reads a narrow string given
// to a path, and writes path::string(), in the process's active code page, so on a
// Windows system whose code page is not UTF-8 a name outside ASCII would become
// another file's name (or fail to convert). Go between the two with these, never
// with path's narrow constructor, operator/ on a std::string, or path::string().
#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace vcr {

inline std::filesystem::path utf8Path(std::string_view text)
{
#ifdef _WIN32
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
#else
    return std::filesystem::path(text);
#endif
}

inline std::string utf8String(const std::filesystem::path& path)
{
#ifdef _WIN32
    const auto text = path.u8string();
    return std::string(text.begin(), text.end());
#else
    return path.string();
#endif
}

// With '/' separators on every platform.
inline std::string utf8GenericString(const std::filesystem::path& path)
{
#ifdef _WIN32
    const auto text = path.generic_u8string();
    return std::string(text.begin(), text.end());
#else
    return path.generic_string();
#endif
}

}  // namespace vcr
