#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <spiral_graph/input_graph.hpp>

namespace spiral::layout {

struct LayoutOptions {
    double contact_tolerance = 2.0;
    std::size_t min_inliers = 16;
    double uv_ransac_tolerance = 3.0;
    double max_refit_rms = 2.0;
    std::size_t ransac_hypotheses = 512;
    std::size_t max_raster_samples = 100'000'000;
    std::size_t theta_batch_size = 1'048'576;
    int workers = 0;
};

struct FiberPointLayout {
    float z = 0.0f;
    float y = 0.0f;
    float x = 0.0f;
    double u = 0.0;
    double v = 0.0;
    std::int32_t winding = 0;
    float fractional_winding = 0.0f;
    bool theta_valid = false;
};

struct FiberLayout {
    std::string id;
    char axis = 'H';
    std::size_t logical_track = 0;
    bool reversed = false;
    double arclength = 0.0;
    std::int32_t winding_offset = 0;
    std::vector<FiberPointLayout> points;
};

struct CrossingKnot {
    std::string first_fiber;
    std::size_t first_point = 0;
    std::string second_fiber;
    std::size_t second_point = 0;
    double u_residual = 0.0;
    double v_residual = 0.0;
};

struct LayoutResult {
    std::vector<FiberLayout> fibers;
    std::vector<CrossingKnot> crossings;
    std::vector<std::string> excluded_fibers;
    std::string root_fiber;
    double total_arclength = 0.0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    std::size_t solver_iterations = 0;
    std::size_t theta_covered_points = 0;
    std::size_t theta_uncovered_points = 0;
};

LayoutResult layout_largest_fiber_component(
    const std::filesystem::path& cache_directory,
    const winding::ThetaProvider& checkpoint_theta,
    const LayoutOptions& options = {},
    std::optional<std::pair<float, float>> checkpoint_z_range = std::nullopt);

} // namespace spiral::layout
