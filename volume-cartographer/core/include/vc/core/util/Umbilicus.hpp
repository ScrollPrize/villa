#pragma once

#include <filesystem>
#include <istream>
#include <optional>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace vc::core::util {

    class Umbilicus {
    public:
        enum class SeamDirection {
            PositiveX,
            NegativeX,
            PositiveY,
            NegativeY
        };

        static Umbilicus FromFile(const std::filesystem::path& path,
                                  const cv::Vec3i& volume_shape);
        static Umbilicus FromPoints(std::vector<cv::Vec3f> control_points,
                                    const cv::Vec3i& volume_shape);

        const cv::Vec3i& volume_shape() const noexcept;
        const std::vector<cv::Vec3f>& centers() const noexcept;
        const cv::Vec3f& center_at(int z_index) const;

        cv::Vec3f vector_to_umbilicus(const cv::Vec3f& point) const;
        double distance_to_umbilicus(const cv::Vec3f& point) const;

        void set_seam(SeamDirection direction);
        void set_seam_from_point(const cv::Vec3f& point);
        bool has_seam() const noexcept;
        SeamDirection seam_direction() const;
        std::pair<cv::Vec3f, cv::Vec3f> seam_segment(int z_index) const;
        const std::vector<cv::Vec3f>& seam_endpoints() const;

        double theta(const cv::Vec3f& point, int wrap_count = 0) const;

    private:
        Umbilicus(std::vector<cv::Vec3f> control_points,
                  const cv::Vec3i& volume_shape);

        static std::vector<cv::Vec3f> LoadFile(const std::filesystem::path& path);
        static std::vector<cv::Vec3f> LoadTextFile(std::istream& stream);
        static std::vector<cv::Vec3f> LoadJsonFile(const std::filesystem::path& path);

        void interpolate_centers();
        cv::Vec3f interpolate_center(double z) const;
        int clamp_z_index(double z) const;
        void set_seam_direction_xy(const cv::Vec2f& direction,
                                   std::optional<SeamDirection> hint);
        void compute_seam_endpoints();

        cv::Vec3i volume_shape_{}; // [Z, Y, X]
        std::vector<cv::Vec3f> control_points_; // sparse centers sorted by z
        std::vector<cv::Vec3f> dense_centers_;   // dense centers, one per z slice

        std::optional<cv::Vec2f> seam_direction_xy_{}; // normalized XY direction
        std::optional<SeamDirection> seam_direction_hint_{};
        std::vector<cv::Vec3f> seam_endpoints_;  // matches dense_centers_
    };

} // namespace vc::core::util
