#pragma once

#include "vc_reader/chunks.hpp"
#include "vc_reader/zarr.hpp"

#include <opencv2/core.hpp>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

// A growth constraint's encoded XYZ channels and optional confidence field.
// Owns native sources for the lifetime of the Ceres residuals that reference it.
struct DirectionField
{
    DirectionField(std::string direction, const std::filesystem::path& root,
                   int level, float weight = 1.0f,
                   const std::optional<std::filesystem::path>& confidenceRoot = {});
    ~DirectionField();
    DirectionField(DirectionField&&) noexcept;
    DirectionField& operator=(DirectionField&&) noexcept;

    struct Native {
        ~Native();
        std::shared_ptr<vcr::chunks::Service> service;
        // XYZ, then optional confidence. The aliases registered with the service
        // keep both the opened metadata and its array alive.
        std::array<std::shared_ptr<vcr::zarr::OpenedArray>, 4> arrays;
        std::array<vcr::chunks::SourceId, 4> sources{};
    };

    std::string direction;
    float scale;
    float weight;
    std::unique_ptr<Native> native;
};

struct DirectionConstraintSample {
    cv::Vec3f directionZYX;
    float confidence;
};

// Decode the quantized direction and confidence used by both direction losses.
// Locations are XYZ in the tracing volume; output vectors follow the losses' ZYX order.
DirectionConstraintSample sampleDirectionConstraint(const DirectionField& field, cv::Vec3d xyz);
