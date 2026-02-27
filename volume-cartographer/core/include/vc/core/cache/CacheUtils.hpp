#pragma once

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

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
        if (n < 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            return std::nullopt;
        }
        if (n == 0) {  // unexpected EOF
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
[[nodiscard]] inline std::string chunkFilename(
    const ChunkKey& key,
    const std::string& delimiter)
{
    std::string name;
    name.reserve(32);
    name += std::to_string(key.iz);
    name += delimiter;
    name += std::to_string(key.iy);
    name += delimiter;
    name += std::to_string(key.ix);
    return name;
}

}  // namespace vc::cache
