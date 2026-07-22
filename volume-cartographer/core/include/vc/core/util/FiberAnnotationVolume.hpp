#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "utils/Json.hpp"
#include "vc/core/util/SparseAnnotationVolume.hpp"

namespace vc {

struct FiberAnnotationInput {
    // Stable caller-owned identity used for deterministic label assignment.
    std::string identity;
    std::filesystem::path sourcePath;
    std::vector<std::array<double, 3>> linePointsXYZ;
    std::vector<std::array<double, 3>> controlPointsXYZ;
    // Missing/empty identities are legacy and are accepted. An explicit value
    // must match the requested target coordinate space.
    std::optional<std::string> coordinateSpace;
};

struct FiberAnnotationLabel {
    std::uint16_t label{};
    std::string identity;
    std::filesystem::path sourcePath;
};

struct FiberAnnotationBatches {
    std::vector<AnnotationPointBatch> batches;
    std::vector<FiberAnnotationLabel> labels;
};

[[nodiscard]] FiberAnnotationBatches makeFiberAnnotationBatches(
    const std::vector<FiberAnnotationInput>& fibers,
    const std::string& targetCoordinateSpace = {},
    double coordinateScale = 1.0);

[[nodiscard]] utils::Json fiberAnnotationAttributes(
    const FiberAnnotationBatches& adapter,
    const std::string& coordinateSpace = {});

} // namespace vc
