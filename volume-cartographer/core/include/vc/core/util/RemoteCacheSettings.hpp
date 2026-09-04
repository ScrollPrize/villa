#pragma once

#include <filesystem>

namespace vc::settings {

inline constexpr auto kRemoteCacheDirectory = "viewer/remote_cache_dir";

[[nodiscard]] std::filesystem::path settingsFilePath();
[[nodiscard]] std::filesystem::path remoteCachePath();

} // namespace vc::settings
