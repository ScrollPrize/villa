#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vc::lasagna {

enum class NormalSourceKind {
    None,
    DenseZarr,
};

struct LasagnaChannelGroup {
    std::string name;
    std::filesystem::path zarrPath;
    int scaledown = 0;
    std::vector<std::string> channels;

    [[nodiscard]] int scaleFactor() const noexcept;
    [[nodiscard]] bool hasChannel(std::string_view channel) const noexcept;
    [[nodiscard]] std::optional<size_t> channelIndex(std::string_view channel) const noexcept;
};

struct LasagnaDatasetManifest {
    std::filesystem::path manifestPath;
    std::filesystem::path baseDirectory;

    int version = 0;
    double sourceToBase = 1.0;
    std::optional<std::filesystem::path> initShellDir;
    std::vector<LasagnaChannelGroup> groups;

    // Backward-compatible summary for old callers: a Lasagna dataset's
    // normal source is its manifest when nx/ny channels are present.
    std::optional<std::filesystem::path> normalPath;
    NormalSourceKind normalSourceKind = NormalSourceKind::None;
    std::string normalSourceKey;

    nlohmann::json raw = nlohmann::json::object();

    [[nodiscard]] bool hasNormalSource() const noexcept;
    [[nodiscard]] const LasagnaChannelGroup* groupForChannel(std::string_view channel) const noexcept;

    static LasagnaDatasetManifest parseFile(const std::filesystem::path& manifestPath);
    static LasagnaDatasetManifest parseText(
        std::string_view jsonText,
        const std::filesystem::path& manifestPath = {});
};

[[nodiscard]] std::string toString(NormalSourceKind kind);

} // namespace vc::lasagna
