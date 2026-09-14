#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace vc3d::spiral {

// Connectivity is through complete quads sharing an edge, not through a lone
// vertex. Equal-size components choose the first in row-major order.
inline cv::Mat1b largestPatchQuadComponent(const cv::Mat1b& selected)
{
    cv::Mat1b retained(selected.rows, selected.cols, uchar{0});
    if (selected.rows < 2 || selected.cols < 2) return retained;
    cv::Mat1b quads(selected.rows - 1, selected.cols - 1, uchar{0});
    for (int row = 0; row < quads.rows; ++row)
        for (int col = 0; col < quads.cols; ++col)
            quads(row, col) = selected(row, col) && selected(row, col + 1)
                && selected(row + 1, col) && selected(row + 1, col + 1);
    cv::Mat1i labels;
    // SAUF assigns labels in row-major order; the tie rule is reproducible.
    const int count = cv::connectedComponents(quads, labels, 4, CV_32S, cv::CCL_SAUF);
    std::vector<int> areas(static_cast<std::size_t>(count), 0);
    for (int row = 0; row < labels.rows; ++row)
        for (int col = 0; col < labels.cols; ++col)
            ++areas[labels(row, col)];
    int largest = 0;
    for (int label = 1; label < count; ++label)
        if (largest == 0 || areas[label] > areas[largest]) largest = label;
    if (largest == 0) return retained;
    for (int row = 0; row < labels.rows; ++row) {
        for (int col = 0; col < labels.cols; ++col) {
            if (labels(row, col) != largest) continue;
            retained(row, col) = retained(row, col + 1) = 1;
            retained(row + 1, col) = retained(row + 1, col + 1) = 1;
        }
    }
    return retained;
}

} // namespace vc3d::spiral
