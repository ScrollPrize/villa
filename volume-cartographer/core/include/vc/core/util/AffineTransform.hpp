#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include "utils/Json.hpp"

class QuadSurface;

namespace vc::core::util {

cv::Matx44d parseAffineTransformMatrix(const utils::Json& json);
cv::Matx44d loadAffineTransformMatrix(const std::filesystem::path& path);
cv::Matx44d loadAffineTransformMatrixFromString(const std::string& text);

cv::Matx44d composeAffineTransform(const cv::Matx44d& first, const cv::Matx44d& second);
std::optional<cv::Matx44d> tryInvertAffineTransformMatrix(const cv::Matx44d& matrix);
cv::Matx44d invertAffineTransformMatrix(const cv::Matx44d& matrix);

bool applyAffineTransform(const cv::Vec3d& point,
                          const cv::Matx44d& matrix,
                          cv::Vec3d& transformed);
cv::Vec3f applyAffineTransform(const cv::Vec3f& point,
                               const cv::Matx44d& matrix);
cv::Vec3f transformNormal(const cv::Vec3f& normal,
                          const cv::Matx44d& matrix);
std::optional<double> affineUniformScaleFactor(const cv::Matx44d& matrix);

cv::Vec3f applyPreAffineScale(const cv::Vec3f& point, int scale);
void transformSurfacePoints(QuadSurface* surface,
                            int scale,
                            const std::optional<cv::Matx44d>& matrix);
void refreshTransformedSurfaceState(QuadSurface* surface);
std::shared_ptr<QuadSurface> cloneSurfaceForTransform(const std::shared_ptr<QuadSurface>& source);

} // namespace vc::core::util
