#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>

namespace vc_spiral::patch_merger {

struct MergeOptions {
    double tolerance = 2.0;
    double dense_spacing = 1.0;
    int erode_cells = 1;
    double output_step = 20.0;
    double thinning_spacing = 2.0;
    std::size_t max_correspondences = 4096;
    double uv_inlier_tolerance = 3.0;
    std::size_t min_inliers = 32;
    double min_major_spread = 20.0;
    double min_minor_spread = 5.0;
    double max_refit_rms = 2.0;
    double containment_threshold = 0.90;
    double ransac_confidence = 0.999;
    std::size_t ransac_max_hypotheses = 512;
    bool allow_reflection = true;
    int threads = 0;
    // CLI benchmark telemetry; intentionally excluded from serialized merge
    // settings because it cannot affect results.
    bool progress = false;
};

struct MergeReport {
    std::size_t input_count = 0;
    std::size_t output_count = 0;
    std::size_t accepted_pair_count = 0;
    std::size_t rejected_pair_count = 0;
    std::size_t dropped_invalid_count = 0;
    std::size_t dropped_output_count = 0;
    std::size_t contained_patch_count = 0;
    std::size_t output_nms_suppressed_count = 0;
    std::size_t output_nms_accepted_pair_count = 0;
    std::size_t output_nms_rejected_pair_count = 0;
    double total_duration = 0.0;
    std::map<std::string, double> stage_timings;
    std::map<std::string, std::size_t> rejections;
    std::map<std::string, std::size_t> output_nms_rejections;
};

MergeReport merge_patch_directory(
    const std::filesystem::path& patches_dir,
    const std::filesystem::path& output_dir,
    const MergeOptions& options = {});

std::string report_json(const MergeReport& report);

} // namespace vc_spiral::patch_merger
