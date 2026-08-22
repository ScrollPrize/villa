#pragma once

#include <cmath>
#include <filesystem>
#include <utility>

#include <tiffio.h>

namespace vc3d::surface_rotation
{
constexpr float kMinimumAngleDegrees = 0.01f;

struct TransformPersistenceCompatibility {
    bool hasMultipageMask{false};
    bool maskInspectionFailed{false};
    bool hasDisconnectedComponents{false};

    [[nodiscard]] bool allowed() const
    {
        return !hasMultipageMask && !maskInspectionFailed && !hasDisconnectedComponents;
    }
};

inline void inspectMaskForTransformPersistence(
    const std::filesystem::path& maskPath,
    TransformPersistenceCompatibility& compatibility)
{
    std::error_code ec;
    const bool exists = std::filesystem::exists(maskPath, ec);
    if (ec) {
        compatibility.maskInspectionFailed = true;
        return;
    }
    if (!exists) {
        return;
    }

    TIFF* tiff = TIFFOpen(maskPath.string().c_str(), "r");
    if (!tiff) {
        compatibility.maskInspectionFailed = true;
        return;
    }

    const auto directoryCount = TIFFNumberOfDirectories(tiff);
    TIFFClose(tiff);
    if (directoryCount == 0) {
        compatibility.maskInspectionFailed = true;
    } else if (directoryCount > 1) {
        compatibility.hasMultipageMask = true;
    }
}

[[nodiscard]] inline TransformPersistenceCompatibility transformPersistenceCompatibility(
    const std::filesystem::path& surfacePath,
    bool hasDisconnectedComponents)
{
    TransformPersistenceCompatibility compatibility;
    compatibility.hasDisconnectedComponents = hasDisconnectedComponents;
    if (!surfacePath.empty()) {
        inspectMaskForTransformPersistence(surfacePath / "mask.tif", compatibility);
        inspectMaskForTransformPersistence(surfacePath / "multilayer_mask.tif", compatibility);
    }
    return compatibility;
}

struct Transform {
    float angleDegrees{0.0f};
    bool flipHorizontally{false};

    [[nodiscard]] bool isNoOp() const { return std::abs(angleDegrees) < kMinimumAngleDegrees && !flipHorizontally; }

    template <typename Surface>
    void applyInMemory(Surface& surface) const
    {
        if (isNoOp()) {
            return;
        }

        if (std::abs(angleDegrees) >= kMinimumAngleDegrees) {
            surface.rotate(angleDegrees);
        }
        if (!flipHorizontally) {
            return;
        }

        // QuadSurface::flipV() also updates disk-backed sidecar TIFFs when a
        // path is present. Preview and Apply flip a clone in memory;
        // persistence remains the caller's single saveOverwrite() step.
        auto backingPath = std::move(surface.path);
        surface.path.clear();
        try {
            surface.flipV();
        } catch (...) {
            surface.path = std::move(backingPath);
            throw;
        }
        surface.path = std::move(backingPath);
    }
};
}  // namespace vc3d::surface_rotation
