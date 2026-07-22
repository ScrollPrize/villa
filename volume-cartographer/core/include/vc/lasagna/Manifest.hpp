#pragma once

#include <filesystem>
#include <array>
#include <cstddef>
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
    // Original manifest-relative Zarr key. This remains authoritative for
    // remotely-backed datasets; zarrPath is the resolved local path used by
    // ordinary datasets and diagnostics.
    std::string relativeZarrKey;
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
    // Runtime coordinate adapter. A point in the caller's working space is
    // multiplied by this value to obtain the dataset's base-volume L0 point.
    // It is not serialized into the Lasagna manifest.
    double workingToBaseScale = 1.0;
    std::optional<std::array<std::size_t, 3>> baseShapeZYX;
    std::optional<std::filesystem::path> initShellDir;
    std::vector<LasagnaChannelGroup> groups;

    // Set by LasagnaDataset::open when the manifest is accompanied by the
    // VC read-through-cache marker.
    std::string remoteBaseUrl;
    std::filesystem::path remoteCacheRoot;

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
