#pragma once
#include <filesystem>
#include <functional>
#include <ostream>

namespace vcr::zarr::detail {
// Writes beside the destination and publishes only after the complete file closes.
void atomicWriteFile(const std::filesystem::path& path, const std::function<void(std::ostream&)>& write);
}
