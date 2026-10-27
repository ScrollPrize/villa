#include "vc/core/util/Umbilicus.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace vc::core::util {

Umbilicus Umbilicus::FromPoints(std::vector<cv::Vec3f> control_points, const cv::Vec3i& volume_shape)
{
    return Umbilicus(std::move(control_points), volume_shape);
}

const cv::Vec3i& Umbilicus::volume_shape() const noexcept
{
    return volume_shape_;
}

const std::vector<cv::Vec3f>& Umbilicus::centers() const noexcept
{
    return dense_centers_;
}

cv::Vec3f Umbilicus::vector_to_umbilicus(const cv::Vec3f& point) const
{
    if (dense_centers_.empty()) {
        throw std::logic_error("Umbilicus has no interpolated centers");
    }
    const int index = clamp_z_index(point[2]);
    return dense_centers_[index] - point;
}

Umbilicus::Umbilicus(std::vector<cv::Vec3f> control_points, const cv::Vec3i& volume_shape)
    : volume_shape_(volume_shape), control_points_(std::move(control_points))
{
    if (volume_shape_[0] <= 0 || volume_shape_[1] <= 0 || volume_shape_[2] <= 0) {
        throw std::invalid_argument("Volume shape components must be positive");
    }
    if (control_points_.empty()) {
        throw std::invalid_argument("Umbilicus requires at least one control point");
    }

    std::sort(control_points_.begin(), control_points_.end(), [](const auto& a, const auto& b) {
        return a[2] < b[2];
    });

    interpolate_centers();
}

void Umbilicus::interpolate_centers()
{
    dense_centers_.clear();
    dense_centers_.resize(volume_shape_[0]);
    for (int z = 0; z < volume_shape_[0]; ++z) {
        dense_centers_[z] = interpolate_center(static_cast<double>(z));
    }
}

cv::Vec3f Umbilicus::interpolate_center(double z) const
{
    if (control_points_.empty()) {
        throw std::logic_error("Umbilicus interpolation requested without control points");
    }

    if (control_points_.size() == 1) {
        cv::Vec3f result = control_points_.front();
        result[2] = static_cast<float>(z);
        return result;
    }

    const double min_z = control_points_.front()[2];
    const double max_z = control_points_.back()[2];

    if (z <= min_z) {
        cv::Vec3f result = control_points_.front();
        result[2] = static_cast<float>(z);
        return result;
    }
    if (z >= max_z) {
        cv::Vec3f result = control_points_.back();
        result[2] = static_cast<float>(z);
        return result;
    }

    const float target = static_cast<float>(z);
    auto upper = std::lower_bound(control_points_.begin(), control_points_.end(), target,
                                  [](const cv::Vec3f& lhs, float value) { return lhs[2] < value; });
    if (upper == control_points_.begin()) {
        cv::Vec3f result = *upper;
        result[2] = static_cast<float>(z);
        return result;
    }

    if (upper != control_points_.end() && std::abs((*upper)[2] - target) < 1e-5f) {
        cv::Vec3f result = *upper;
        result[2] = static_cast<float>(z);
        return result;
    }

    const auto& right = (upper == control_points_.end()) ? control_points_.back() : *upper;
    const auto& left = *(upper - 1);

    const double z0 = left[2];
    const double z1 = right[2];

    if (std::abs(z1 - z0) < 1e-5) {
        cv::Vec3f result = left;
        result[2] = static_cast<float>(z);
        return result;
    }

    const double t = (z - z0) / (z1 - z0);
    const double x = left[0] + t * (right[0] - left[0]);
    const double y = left[1] + t * (right[1] - left[1]);
    return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
}

int Umbilicus::clamp_z_index(double z) const
{
    if (dense_centers_.empty()) {
        throw std::logic_error("Umbilicus has no interpolated centers");
    }
    int index = static_cast<int>(std::lround(z));
    index = std::clamp(index, 0, static_cast<int>(dense_centers_.size() - 1));
    return index;
}

} // namespace vc::core::util
