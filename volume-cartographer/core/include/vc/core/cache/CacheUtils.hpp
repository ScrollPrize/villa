#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ChunkKey.hpp"

namespace vc::cache {

// Read an entire file into a byte vector using POSIX I/O.
// Tries O_NOATIME first (avoids atime updates), falls back to plain O_RDONLY.
// Returns nullopt on any failure (missing file, read error, etc.).
[[nodiscard]] inline std::optional<std::vector<uint8_t>> readFileToVector(
    const std::filesystem::path& path)
{
    int fd = ::open(path.c_str(), O_RDONLY | O_NOATIME);
    if (fd < 0) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return std::nullopt;
    }

    struct stat sb;
    if (::fstat(fd, &sb) != 0) {
        ::close(fd);
        return std::nullopt;
    }

    auto fileSize = static_cast<size_t>(sb.st_size);
    if (fileSize == 0) {
        ::close(fd);
        return std::nullopt;
    }

    std::vector<uint8_t> buf(fileSize);
    size_t total = 0;
    while (total < fileSize) {
        ssize_t n = ::read(fd, buf.data() + total, fileSize - total);
        if (n <= 0) {
            ::close(fd);
            return std::nullopt;
        }
        total += static_cast<size_t>(n);
    }
    ::close(fd);
    return buf;
}

// Build a chunk filename from a ChunkKey: "<iz><delim><iy><delim><ix>"
// Used by DiskStore and FileSystemChunkSource.
// Uses a stack buffer to avoid heap allocation on the hot path (Issue 51).
[[nodiscard]] inline std::string chunkFilename(
    const ChunkKey& key,
    const std::string& delimiter)
{
    // Max: 3 * 10 digits + 2 * 1 char delimiter + NUL = 33 bytes, 64 is safe
    char buf[64];
    const char* d = delimiter.c_str();
    int n = std::snprintf(buf, sizeof(buf), "%d%s%d%s%d",
                          key.iz, d, key.iy, d, key.ix);
    return {buf, static_cast<std::string::size_type>(n > 0 ? n : 0)};
}

}  // namespace vc::cache
