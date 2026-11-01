#pragma once

#include <vc_reader/native/reader_types.hpp>

#include <filesystem>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
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
    // Authoritative local path or remote locator serialized into VC projects;
    // a remote group's array is opened at this URL.
    std::string sourceLocation;
    // Remote groups: the vc_reader disk cache directory of their chunks and
    // how requests are signed (explicit keys, else AWS discovery when allowed).
    std::filesystem::path remoteCacheRoot;
    vcr::Auth remoteAuth;
    bool discoverAwsCredentials = true;
    int scaledown = 0;
    std::vector<std::string> channels;

    [[nodiscard]] int scaleFactor() const noexcept;
    [[nodiscard]] bool isRemote() const noexcept;
    [[nodiscard]] bool hasChannel(std::string_view channel) const noexcept;
    [[nodiscard]] std::optional<size_t> channelIndex(std::string_view channel) const noexcept;
};

struct LasagnaDatasetManifest {
    std::filesystem::path manifestPath;
    std::filesystem::path baseDirectory;
    std::string manifestLocation;
    bool manifestIsRemote = false;

    int version = 0;
    double sourceToBase = 1.0;
    // Runtime coordinate adapter. A point in the caller's working space is
    // multiplied by this value to obtain the dataset's base-volume L0 point.
    // It is not serialized into the Lasagna manifest.
    double workingToBaseScale = 1.0;
    std::optional<std::array<std::size_t, 3>> baseShapeZYX;
    std::optional<std::filesystem::path> initShellDir;
    std::vector<LasagnaChannelGroup> groups;

    // Where the manifest's relative groups live: the marker's artifact URL (LasagnaDataset::open
    // with the VC read-through-cache marker) or the remote manifest's parent (openLocation), as
    // requests use it, a signed query included. Empty for a plain local manifest.
    std::string remoteSourceBaseLocation;
    std::filesystem::path remoteCacheRoot;
    // A cached public catalog marker sets this false so later group opens do
    // not accidentally inherit invalid credentials from the host machine.
    bool discoverAwsCredentials = true;

    // Backward-compatible summary for old callers: a Lasagna dataset's
    // normal source is its manifest when nx/ny channels are present.
    std::optional<std::filesystem::path> normalPath;
    NormalSourceKind normalSourceKind = NormalSourceKind::None;
    std::string normalSourceKey;

    nlohmann::json raw = nlohmann::json::object();

    [[nodiscard]] bool hasNormalSource() const noexcept;
    [[nodiscard]] const LasagnaChannelGroup* groupForChannel(std::string_view channel) const noexcept;
    [[nodiscard]] std::vector<std::string> fiberPredictionPrefixes() const;

    static LasagnaDatasetManifest parseFile(const std::filesystem::path& manifestPath);
    static LasagnaDatasetManifest parseText(
        std::string_view jsonText,
        const std::filesystem::path& manifestPath = {});
};

// Lasagna arrays may be padded beyond their logical base-coordinate extent by
// at most one storage chunk on each axis.
[[nodiscard]] bool lasagnaChannelShapeCompatible(
    const std::array<std::size_t, 3>& baseShapeZYX,
    double baseSpacing,
    const std::array<std::size_t, 3>& shapeZYX,
    const std::array<std::size_t, 3>& chunksZYX) noexcept;

// Return the power-of-two coordinate factor that maps points expressed in
// ``fromShapeZYX`` into ``toShapeZYX``. Pyramid construction may round an odd
// extent up or down, so both conventions are accepted on every axis. Shapes
// at the same level may differ by one voxel because published metadata uses
// both voxel-count and inclusive maximum-coordinate conventions. The shapes
// must still identify exactly one level in the requested range.
[[nodiscard]] double dyadicCoordinateScaleBetweenShapes(
    const std::array<std::size_t, 3>& fromShapeZYX,
    const std::array<std::size_t, 3>& toShapeZYX,
    int maximumAbsoluteLevel = 30);

[[nodiscard]] std::string toString(NormalSourceKind kind);

} // namespace vc::lasagna
