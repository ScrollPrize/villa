#pragma once

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace vc3d::inklabels {

// The flat label pair vesuvius.ink_detection trains on (scrollprize.org
// tutorial 5): <prefix>_inklabels[_vN].tif marks ink, and
// <prefix>_supervision_mask[_vN].tif marks where labels count. Both are 8-bit,
// 0 or 255, in the segment's render pixels. create_label_zarrs turns them into
// the .zarr volumes the trainer reads; the prefix must be the segment folder's
// name.
struct TrainingLabelFiles {
    std::filesystem::path inklabels;
    std::filesystem::path supervisionMask;
};

// version 0 writes the unversioned names; N > 0 appends _vN.
TrainingLabelFiles trainingLabelFiles(const std::filesystem::path& directory,
                                      const std::string& prefix,
                                      int version = 0);

// Write labels (vc3d::inklabels::Label values) as the pair. Each file is
// written beside its final name and renamed into place.
void writeTrainingLabels(const TrainingLabelFiles& files, const cv::Mat_<uint8_t>& labels);

struct LoadedTrainingLabels {
    cv::Mat_<uint8_t> labels;
    // Ink pixels outside the supervision mask: the trainer ignores them, so
    // they load as unlabelled.
    std::size_t inkOutsideSupervision{0};
};

// Read the pair back. Any nonzero value counts, as in the trainer. Throws when
// a file is missing or unreadable, or the two images differ in size.
LoadedTrainingLabels readTrainingLabels(const TrainingLabelFiles& files);

} // namespace vc3d::inklabels
