#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <optional>

#include "vc/core/util/Surface.hpp"
#include "vc/ui/VCCollection.hpp"

namespace z5 {
class Dataset;
}

struct GrowSegRequest {
    std::filesystem::path volume_path;
    cv::Vec3d origin{0.0, 0.0, 0.0};
    nlohmann::json params;
    std::filesystem::path output_dir;
    nlohmann::json meta;
    std::optional<std::filesystem::path> resume_surface_path;
    const VCCollection* corrections{nullptr};
    std::optional<std::filesystem::path> cache_root_override;
    std::shared_ptr<z5::Dataset> dataset;
    std::optional<float> voxel_size_override;
};

struct GrowSegResult {
    std::unique_ptr<QuadSurface> surface;
    double area_cm2{0.0};
    float voxel_size{1.0f};
};

GrowSegResult run_grow_seg_from_seed(const GrowSegRequest& request);
