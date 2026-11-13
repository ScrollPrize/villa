#pragma once

#include <QJsonArray>
#include <QString>
#include <opencv2/core/mat.hpp>

#include <optional>
#include <vector>

namespace vc3d::spiral {

// One winding of a flattened Spiral preview: the grid rows and columns
// [begin, end) its vertices occupy.
struct PreviewComponent {
    int rowBegin = 0;
    int rowEnd = 0;
    int columnBegin = 0;
    int columnEnd = 0;
    int winding = 0;
};

// The declared windings of a preview ({row_begin, row_end, column_begin,
// column_end, winding} per entry), each matching `windingIds` at its index;
// nullopt, with the reason in `error`, when malformed.
std::optional<std::vector<PreviewComponent>> parseWindingBounds(
    const QJsonArray& bounds, const QJsonArray& windingIds, QString* error);

// A winding-ID image (any one-channel depth) as int32: every value an integer
// winding, -1 outside every winding. nullopt, with the reason in `error`,
// otherwise.
std::optional<cv::Mat_<int32_t>> windingIdsFromImage(const cv::Mat& image, QString* error);

// Whether the winding-ID map holds exactly the declared windings, each within
// exactly its declared bounds; the reason in `error` when it does not.
bool windingMapMatchesBounds(const cv::Mat_<int32_t>& windingIds,
                             const std::vector<PreviewComponent>& components,
                             QString* error);

}  // namespace vc3d::spiral
