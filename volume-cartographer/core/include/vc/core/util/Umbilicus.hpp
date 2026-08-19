#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace vc::core::util {

    // An umbilicus file's contents plus whatever frame metadata it declares.
    // The optionals stay unset for legacy/unstamped files, which carry no
    // statement about the grid their coordinates index.
    struct UmbilicusFileInfo {
        std::vector<cv::Vec3f> controlPoints;
        // metadata.voxelsize_um: voxel size, in µm, of the grid the
        // coordinates index. Authoritative and self-contained: to express the
        // umbilicus in a target grid, multiply the coordinates by
        // voxelsizeUm / targetVoxelsizeUm.
        std::optional<double> voxelsizeUm;
        // metadata.volume: volume store the umbilicus was annotated on.
        // Provenance only.
        std::optional<std::string> volume;
        // metadata.volume_width / volume_height / volume_slices: the x, y and z
        // voxel counts of the grid the coordinates index. Exact-integer
        // provenance: a consumer can check that its own grid has the same
        // dimensions, or derive a rescale precisely from the dimension ratios
        // in cases where the µm sizes are rounded and therefore ambiguous.
        std::optional<int> volumeWidth;
        std::optional<int> volumeHeight;
        std::optional<int> volumeSlices;
    };

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
        // The raw control points of an umbilicus file (same formats as
        // FromFile), for callers that need to reframe them before building.
        static std::vector<cv::Vec3f> LoadControlPoints(
            const std::filesystem::path& path);
        // Same points plus the file's frame metadata. Text/CSV files and json
        // files without a "metadata" object yield unset metadata, as does any
        // individual metadata field that is missing or malformed.
        static UmbilicusFileInfo LoadFileInfo(const std::filesystem::path& path);

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

        static UmbilicusFileInfo LoadFile(const std::filesystem::path& path);
        static std::vector<cv::Vec3f> LoadTextFile(std::istream& stream);
        static UmbilicusFileInfo LoadJsonFile(const std::filesystem::path& path);

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
