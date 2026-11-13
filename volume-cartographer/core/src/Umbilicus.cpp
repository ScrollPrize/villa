#include "vc/core/util/Umbilicus.hpp"
#include "vc_object_store/geometry/FiberAxialAlignment.hpp"

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
    return vc::geometry::umbilicusCenter(control_points_,z);
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
