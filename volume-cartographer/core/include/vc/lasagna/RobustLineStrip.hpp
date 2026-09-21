#pragma once

#include "vc/lasagna/LineModel.hpp"
#include <opencv2/core.hpp>
#include <functional>

namespace vc::lasagna {

inline constexpr int kRobustLineStripVersion = 7;

struct RobustLineStripOptions {
    double alongSpacing = 2.0;
    double crossSpacing = 2.0;
    std::function<void()> checkCancelled;
};

// Follow the sampled normal field from the center of a straight ribbon, repair
// invalid regions, then apply four locally guarded Huber IRLS normal fits.
// Only the center row and the adjacent row's orientation are taken from ribbon.
// Throws instead of returning an invalid or partially generated surface.
cv::Mat_<cv::Vec3f> buildRobustLineStrip(
    const cv::Mat_<cv::Vec3f>& ribbon, const NormalSampler& sampler,
    const RobustLineStripOptions& options = {});

// The same v7 fitter, exposed independently for regression against repaired
// experimental seeds. The center row remains bit-exact after float conversion.
cv::Mat_<cv::Vec3f> fitRobustLineStrip(
    const cv::Mat_<cv::Vec3f>& seed, const NormalSampler& sampler,
    bool preserveRepairedWidth, const RobustLineStripOptions& options = {});

} // namespace vc::lasagna
