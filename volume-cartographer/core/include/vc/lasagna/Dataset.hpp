#pragma once

#include "vc/lasagna/Manifest.hpp"

#include <filesystem>

namespace utils { class ZarrArray; }

namespace vc::lasagna {

inline constexpr const char* kLasagnaRemoteMarker = "lasagna-remote.json";

struct LasagnaDatasetOpenOptions {
    double workingToBaseScale = 1.0;
};

class LasagnaDataset {
public:
    explicit LasagnaDataset(LasagnaDatasetManifest manifest);

    static LasagnaDataset open(
        const std::filesystem::path& manifestPath,
        LasagnaDatasetOpenOptions options = {});

    [[nodiscard]] const LasagnaDatasetManifest& manifest() const noexcept;
    [[nodiscard]] bool hasNormalSource() const noexcept;
    [[nodiscard]] const std::filesystem::path& normalSourcePath() const;

private:
    LasagnaDatasetManifest manifest_;
};

// Open a channel group's Zarr through the local filesystem or, for a
// manifest-backed catalog cache, through its persistent read-through store.
[[nodiscard]] utils::ZarrArray openLasagnaChannelArray(
    const LasagnaDatasetManifest& manifest,
    const LasagnaChannelGroup& group,
    std::size_t dtypeSize = 1);

} // namespace vc::lasagna
