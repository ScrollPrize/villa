#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <opencv2/core/types.hpp>

namespace vc {

struct CtSheetNormal {
    cv::Vec3d normal{0, 0, 0};
    bool valid = false;
    double gap = 0;
    std::string reason;
};

// Bounded cached adapter over vc_reader's native structure-tensor estimator.
// XYZ queries are scaled into the selected uint8 CT level. Insufficient
// 109-cubed context or an ambiguous tensor returns an invalid sample.
class CtSheetNormalSampler {
public:
    static constexpr const char* method = "ct-anisotropic-1-24-4-grid3-r48-v4";
    CtSheetNormalSampler(const std::filesystem::path& ctVolume, int level = 0,
                         double workingToCtScale = 1.0);
    ~CtSheetNormalSampler();
    CtSheetNormalSampler(const CtSheetNormalSampler&) = delete;
    CtSheetNormalSampler& operator=(const CtSheetNormalSampler&) = delete;

    CtSheetNormal sampleNormal(const cv::Vec3d& workingXYZ,
                               const std::atomic_bool* cancel = nullptr) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Precomputed lasagna normals: paired uint8 nx/ny volumes at the same level,
// decoded as the spiral fitter does: (v - 128) / 127, nz = sqrt(1 - nx^2 - ny^2),
// and nx = ny = 0 marks no data. XYZ queries are scaled into the selected level
// and read at the nearest voxel; outside the volume returns an invalid sample.
class LasagnaSheetNormalSampler {
public:
    static constexpr const char* method = "lasagna-nxny-u8-nearest-v1";
    LasagnaSheetNormalSampler(const std::filesystem::path& normalX,
                              const std::filesystem::path& normalY, int level = 0,
                              double workingToNormalScale = 1.0);
    ~LasagnaSheetNormalSampler();
    LasagnaSheetNormalSampler(const LasagnaSheetNormalSampler&) = delete;
    LasagnaSheetNormalSampler& operator=(const LasagnaSheetNormalSampler&) = delete;

    CtSheetNormal sampleNormal(const cv::Vec3d& workingXYZ,
                               const std::atomic_bool* cancel = nullptr) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace vc
