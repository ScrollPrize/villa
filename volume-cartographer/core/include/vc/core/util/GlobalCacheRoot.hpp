#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

namespace vc::core::util {

inline std::filesystem::path vc3dConfigDirectory()
{
    const char* configured = std::getenv("VC3D_CONFIG_DIR");
    if (configured && *configured) return configured;
    const char* home = std::getenv("HOME");
#ifdef _WIN32
    if (!home || !*home) home = std::getenv("USERPROFILE");
#endif
    if (!home || !*home) throw std::runtime_error("cannot resolve VC3D config directory");
    return std::filesystem::path(home) / ".VC3D";
}

// This explicit host override is distinct from historical per-volpkg/user
// suggestions, which still lose to provisioned mount defaults when unset.
// The resolver does not create directories, so discovery remains read-only.
inline std::filesystem::path globalCacheRoot(
    const std::filesystem::path& suggestion = {},
    const std::filesystem::path& configDirectory = vc3dConfigDirectory())
{
    const auto configPath = configDirectory / "cache.json";
    if (std::filesystem::exists(configPath)) {
        std::ifstream input(configPath);
        if (!input) throw std::runtime_error("cannot read " + configPath.string());
        const auto config = nlohmann::json::parse(input);
        if (config.at("version") != 1)
            throw std::invalid_argument("unsupported VC3D cache configuration version");
        const std::filesystem::path root(config.at("root").get<std::string>());
        if (root.empty() || !root.is_absolute())
            throw std::invalid_argument("VC3D global cache root must be absolute");
        return root.lexically_normal();
    }
    for (const auto* mount : {"/volpkgs", "/ephemeral"}) {
        if (std::filesystem::exists(mount)) {
            if (!std::filesystem::is_directory(mount))
                throw std::runtime_error(std::string(mount) + " exists but is not a directory");
            return std::filesystem::path(mount) / "remote_cache";
        }
    }
    if (!suggestion.empty()) return suggestion;
    // Keep the legacy default separate from an optional redirected config dir.
    const char* home = std::getenv("HOME");
#ifdef _WIN32
    if (!home || !*home) home = std::getenv("USERPROFILE");
#endif
    if (!home || !*home) throw std::runtime_error("cannot resolve VC3D default cache root");
    return std::filesystem::path(home) / ".VC3D" / "remote_cache";
}

} // namespace vc::core::util
