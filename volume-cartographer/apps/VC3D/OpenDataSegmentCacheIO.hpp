#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace vc3d::opendata::detail {

[[nodiscard]] bool isNonEmptyFile(const std::filesystem::path& path);
[[nodiscard]] std::string readTextFile(const std::filesystem::path& path);

void writeBytesAtomic(const std::filesystem::path& path,
                      const std::vector<std::byte>& bytes);
void writeStringAtomic(const std::filesystem::path& path,
                       const std::string& text);

void writeCachedTifxyzBand(const std::string& baseUrl,
                           const std::string& fileName,
                           const std::filesystem::path& target);
[[nodiscard]] bool cacheOptionalFile(const std::string& baseUrl,
                                     const std::string& fileName,
                                     const std::filesystem::path& target);

} // namespace vc3d::opendata::detail
