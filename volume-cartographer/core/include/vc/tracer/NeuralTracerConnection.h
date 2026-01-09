#pragma once

#include <string>
#include <optional>
#include <array>
#include <opencv2/core.hpp>

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)


class NeuralTracerConnection
{
public:

    struct NextUvs
    {
        std::vector<cv::Vec3f> next_u_xyzs;
        std::vector<cv::Vec3f> next_v_xyzs;
    };

    struct EdtResult
    {
        xt::xarray<float> distance_transform;  // 3D float32 volume (z, y, x)
        cv::Vec3f min_corner_xyz;
        std::array<int, 3> shape;  // (depth, height, width)
        float scale_factor = 1.0f;  // 2^volume_scale for coordinate conversion
        std::array<int, 3> crop_size = {192, 192, 192};  // Model crop size in model-space voxels (Z, Y, X)
    };

    struct BatchEdtResult
    {
        std::vector<EdtResult> results;      // One EdtResult per input sample
        std::vector<int> valid_indices;      // Indices of successful samples
        std::vector<std::string> errors;     // Error messages (empty string = success)
    };

    explicit NeuralTracerConnection(std::string const & socket_path);
    ~NeuralTracerConnection();

    NeuralTracerConnection(NeuralTracerConnection const &) = delete;
    NeuralTracerConnection &operator =(NeuralTracerConnection const &) = delete;

    std::vector<NextUvs> get_next_points(
        std::vector<cv::Vec3f> const &center,
        std::vector<std::optional<cv::Vec3f>> const &prev_u,
        std::vector<std::optional<cv::Vec3f>> const &prev_v,
        std::vector<std::optional<cv::Vec3f>> const &prev_diag
    ) const;

    // Get distance transform from EDT model
    // center_xyz: center point in volume coordinates
    // conditioning_mask_path: path to the conditioning mask file
    EdtResult get_distance_transform(
        const cv::Vec3f& center_xyz,
        const std::string& conditioning_mask_path
    ) const;

    // Get signed distance transform from conditioning points
    // center_xyz: center point in volume coordinates (full-res XYZ)
    // conditioning_points: list of conditioning points (full-res XYZ)
    EdtResult get_sdt_from_points(
        const cv::Vec3f& center_xyz,
        const std::vector<cv::Vec3f>& conditioning_points
    ) const;

    // Batched API: process multiple (center, conditioning_points) pairs in one call
    // center_xyzs: vector of center points in full-res XYZ coordinates
    // conditioning_points_list: vector of conditioning point lists (full-res XYZ)
    BatchEdtResult get_sdt_from_points_batch(
        const std::vector<cv::Vec3f>& center_xyzs,
        const std::vector<std::vector<cv::Vec3f>>& conditioning_points_list
    ) const;

    // Batched API: process multiple (center, mask_path) pairs in one call
    // center_xyzs: vector of center points in full-res XYZ coordinates
    // conditioning_mask_paths: vector of paths to conditioning mask zarr files
    BatchEdtResult get_distance_transform_batch(
        const std::vector<cv::Vec3f>& center_xyzs,
        const std::vector<std::string>& conditioning_mask_paths
    ) const;

private:
    int sock = -1;
};
