#include "patch_merger.hpp"

#include <filesystem>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using vc_spiral::patch_merger::MergeOptions;
using vc_spiral::patch_merger::MergeReport;

NB_MODULE(patch_merger, module)
{
    module.doc() = "Native direct-overlap TIFXYZ patch merger.";

    nb::class_<MergeOptions>(module, "MergeOptions")
        .def(nb::init<>())
        .def_rw("tolerance", &MergeOptions::tolerance)
        .def_rw("dense_spacing", &MergeOptions::dense_spacing)
        .def_rw("erode_cells", &MergeOptions::erode_cells)
        .def_rw("output_step", &MergeOptions::output_step)
        .def_rw("thinning_spacing", &MergeOptions::thinning_spacing)
        .def_rw("max_correspondences", &MergeOptions::max_correspondences)
        .def_rw("uv_inlier_tolerance", &MergeOptions::uv_inlier_tolerance)
        .def_rw("min_inliers", &MergeOptions::min_inliers)
        .def_rw("min_major_spread", &MergeOptions::min_major_spread)
        .def_rw("min_minor_spread", &MergeOptions::min_minor_spread)
        .def_rw("max_rms", &MergeOptions::max_refit_rms)
        .def_rw("max_refit_rms", &MergeOptions::max_refit_rms)
        .def_rw("containment_threshold", &MergeOptions::containment_threshold)
        .def_rw("ransac_confidence", &MergeOptions::ransac_confidence)
        .def_rw("ransac_max_hypotheses", &MergeOptions::ransac_max_hypotheses)
        .def_rw("allow_reflection", &MergeOptions::allow_reflection)
        .def_rw("threads", &MergeOptions::threads);

    nb::class_<MergeReport>(module, "MergeReport")
        .def_ro("input_count", &MergeReport::input_count)
        .def_ro("output_count", &MergeReport::output_count)
        .def_ro("accepted_pair_count", &MergeReport::accepted_pair_count)
        .def_ro("rejected_pair_count", &MergeReport::rejected_pair_count)
        .def_ro("dropped_invalid_count", &MergeReport::dropped_invalid_count)
        .def_ro("dropped_output_count", &MergeReport::dropped_output_count)
        .def_ro("contained_patch_count", &MergeReport::contained_patch_count)
        .def_ro("output_nms_suppressed_count", &MergeReport::output_nms_suppressed_count)
        .def_ro("output_nms_accepted_pair_count", &MergeReport::output_nms_accepted_pair_count)
        .def_ro("output_nms_rejected_pair_count", &MergeReport::output_nms_rejected_pair_count)
        .def_ro("total_duration", &MergeReport::total_duration)
        .def_ro("stage_timings", &MergeReport::stage_timings)
        .def_ro("rejections", &MergeReport::rejections)
        .def_ro("output_nms_rejections", &MergeReport::output_nms_rejections);

    module.def(
        "merge_patch_directory",
        [](const std::filesystem::path& patches_dir,
           const std::filesystem::path& output_dir,
           const MergeOptions& options) {
            nb::gil_scoped_release release;
            return vc_spiral::patch_merger::merge_patch_directory(
                patches_dir, output_dir, options);
        },
        nb::arg("patches_dir"), nb::arg("output_dir"),
        nb::arg("options") = MergeOptions{});
}
