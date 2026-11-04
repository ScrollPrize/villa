#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>

namespace vc::core::util {

    class Umbilicus {
    public:
        static Umbilicus FromPoints(std::vector<cv::Vec3f> control_points,
                                    const cv::Vec3i& volume_shape);

        const cv::Vec3i& volume_shape() const noexcept;
        const std::vector<cv::Vec3f>& centers() const noexcept;

        cv::Vec3f vector_to_umbilicus(const cv::Vec3f& point) const;

    private:
        Umbilicus(std::vector<cv::Vec3f> control_points,
                  const cv::Vec3i& volume_shape);

        void interpolate_centers();
        cv::Vec3f interpolate_center(double z) const;
        int clamp_z_index(double z) const;

        cv::Vec3i volume_shape_{}; // [Z, Y, X]
        std::vector<cv::Vec3f> control_points_; // sparse centers sorted by z
        std::vector<cv::Vec3f> dense_centers_;   // dense centers, one per z slice
    };

} // namespace vc::core::util
