#pragma once

#include <cmath>


namespace vc3d::surface_rotation
{
constexpr float kMinimumAngleDegrees = 0.01f;

// Whether a transformed surface can be published: disconnected component
// ranges in its meta cannot yet be transformed safely.
struct TransformPersistenceCompatibility {
    bool hasDisconnectedComponents{false};

    [[nodiscard]] bool allowed() const { return !hasDisconnectedComponents; }
};

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
        if (flipHorizontally) {
            surface.flipV();
        }
    }
};
}  // namespace vc3d::surface_rotation
