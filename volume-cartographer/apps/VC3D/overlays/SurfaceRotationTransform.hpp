#pragma once

#include <cmath>
#include <filesystem>
#include <utility>

namespace vc3d::surface_rotation
{
constexpr float kMinimumAngleDegrees = 0.01f;

struct TransformPersistenceCompatibility {
    bool hasMultipageMask{false};
    bool hasDisconnectedComponents{false};

    [[nodiscard]] bool allowed() const { return !hasMultipageMask && !hasDisconnectedComponents; }
};

[[nodiscard]] inline TransformPersistenceCompatibility transformPersistenceCompatibility(
    const std::filesystem::path& surfacePath,
    bool hasDisconnectedComponents)
{
    std::error_code ec;
    const bool hasMultipageMask = !surfacePath.empty() &&
                                  (std::filesystem::exists(surfacePath / "multilayer_mask.tif", ec) || bool(ec));
    return {hasMultipageMask, hasDisconnectedComponents};
}

struct HorizontalFlipSelection {
    bool selected{false};
    bool selectionCleared{false};
};

[[nodiscard]] inline HorizontalFlipSelection reconcileHorizontalFlipSelection(bool selected, const TransformPersistenceCompatibility& compatibility)
{
    const bool allowed = compatibility.allowed();
    return {selected && allowed, selected && !allowed};
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
