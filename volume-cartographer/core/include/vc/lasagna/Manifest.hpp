#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace vc::lasagna {

enum class NormalSourceKind {
    None,
    NormalGrid,
    DenseZarr,
};

struct LasagnaDatasetManifest {
    std::filesystem::path manifestPath;
    std::filesystem::path baseDirectory;

    std::optional<std::filesystem::path> volumePath;
    std::optional<std::filesystem::path> normalPath;
    NormalSourceKind normalSourceKind = NormalSourceKind::None;
    std::string normalSourceKey;

    nlohmann::json raw = nlohmann::json::object();

    [[nodiscard]] bool hasNormalSource() const noexcept;

    static LasagnaDatasetManifest parseFile(const std::filesystem::path& manifestPath);
    static LasagnaDatasetManifest parseText(
        std::string_view jsonText,
        const std::filesystem::path& manifestPath = {});
};

[[nodiscard]] std::string toString(NormalSourceKind kind);

} // namespace vc::lasagna
