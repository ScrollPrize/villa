#include <fstream>
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <system_error>

// Records the volume a grow tool traced in (`target_volume`) and, for a volume
// inside a volpkg's volumes/ directory, the scroll it belongs to
// (`scroll_source`, from the volpkg's config.json name or its folder name).
// Keys the meta already has are left alone.
inline void add_target_context(nlohmann::json& meta, const std::filesystem::path& volume_path)
{
    std::filesystem::path normalized_volume_path = volume_path.lexically_normal();
    if (normalized_volume_path.filename().empty()) {
        normalized_volume_path = normalized_volume_path.parent_path();
    }

    const std::string volume_name = normalized_volume_path.filename().string();
    if (!volume_name.empty() && !meta.contains("target_volume")) {
        meta["target_volume"] = volume_name;
    }

    const std::filesystem::path volumes_dir = normalized_volume_path.parent_path();
    if (volumes_dir.filename() != "volumes") {
        return;
    }

    const std::filesystem::path volpkg_root = volumes_dir.parent_path();
    std::string scroll_name = volpkg_root.filename().string();

    const std::filesystem::path config_path = volpkg_root / "config.json";
    std::error_code ec;
    if (std::filesystem::is_regular_file(config_path, ec)) {
        try {
            auto cfg = nlohmann::json::parse(std::ifstream(config_path));
            if (cfg.contains("name") && cfg["name"].is_string()) {
                scroll_name = cfg["name"].get<std::string>();
            }
        } catch (...) {
            // Keep folder-based fallback if config.json cannot be parsed.
        }
    }

    if (!scroll_name.empty() && !meta.contains("scroll_source")) {
        meta["scroll_source"] = scroll_name;
    }
}
