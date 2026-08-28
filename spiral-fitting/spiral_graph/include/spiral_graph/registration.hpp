#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace spiral::registration {

struct Correspondence2d {
    double source_u = 0.0;
    double source_v = 0.0;
    double target_u = 0.0;
    double target_v = 0.0;
};

struct Options {
    std::size_t min_inliers = 16;
    double inlier_tolerance = 3.0;
    double max_refit_rms = 2.0;
    std::size_t max_hypotheses = 512;
    bool allow_reflection = true;
};

struct Result {
    bool accepted = false;
    bool reflected = false;
    double r00 = 1.0;
    double r01 = 0.0;
    double r10 = 0.0;
    double r11 = 1.0;
    double translation_u = 0.0;
    double translation_v = 0.0;
    double rms = 0.0;
    std::size_t inliers = 0;
    std::string rejection;
};

Result fit_rigid_2d(
    std::span<const Correspondence2d> correspondences,
    const Options& options = {});

struct Pose2d {
    double r00 = 1.0;
    double r01 = 0.0;
    double r10 = 0.0;
    double r11 = 1.0;
    double translation_u = 0.0;
    double translation_v = 0.0;
};

struct AbsolutePoseConstraint {
    std::size_t patch = 0;
    double local_u = 0.0;
    double local_v = 0.0;
    double target_u = 0.0;
    double target_v = 0.0;
};

struct RelativePoseConstraint {
    std::size_t first_patch = 0;
    double first_u = 0.0;
    double first_v = 0.0;
    std::size_t second_patch = 0;
    double second_u = 0.0;
    double second_v = 0.0;
};

struct PoseGraphResult {
    bool usable = false;
    std::vector<Pose2d> poses;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    std::size_t iterations = 0;
};

PoseGraphResult refine_pose_graph(
    std::span<const Pose2d> initial,
    std::span<const AbsolutePoseConstraint> absolute,
    std::span<const RelativePoseConstraint> relative,
    double huber_transition = 3.0,
    std::size_t max_iterations = 200,
    std::size_t workers = 1);

} // namespace spiral::registration
