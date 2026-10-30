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
} // namespace vc
