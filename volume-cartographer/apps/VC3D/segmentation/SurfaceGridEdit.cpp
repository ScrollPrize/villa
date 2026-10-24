#include "SurfaceGridEdit.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <string>

namespace segmentation
{
namespace
{

// Pads (positive) or crops (negative) `mat` by the given counts.
cv::Mat resizeBorder(const cv::Mat& mat, int top, int bottom, int left, int right, const cv::Scalar& fill)
{
    cv::Mat padded;
    cv::copyMakeBorder(mat, padded,
                       std::max(0, top), std::max(0, bottom),
                       std::max(0, left), std::max(0, right),
                       cv::BORDER_CONSTANT, fill);
    const int cropTop = std::max(0, -top);
    const int cropBottom = std::max(0, -bottom);
    const int cropLeft = std::max(0, -left);
    const int cropRight = std::max(0, -right);
    if (cropTop == 0 && cropBottom == 0 && cropLeft == 0 && cropRight == 0) {
        return padded;
    }
    const cv::Rect keep(cropLeft, cropTop,
                        padded.cols - cropLeft - cropRight,
                        padded.rows - cropTop - cropBottom);
    return padded(keep).clone();
}

// Integer factor between a channel dimension and the grid dimension, or 0.
int channelFactor(int channelDim, int gridDim)
{
    if (gridDim <= 0 || channelDim <= 0 || channelDim % gridDim != 0) {
        return 0;
    }
    return channelDim / gridDim;
}

} // namespace

bool resizeSurfaceGrid(QuadSurface& surface, int top, int bottom, int left, int right)
{
    cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        return false;
    }
    if (top == 0 && bottom == 0 && left == 0 && right == 0) {
        return true;
    }
    const int newRows = points->rows + top + bottom;
    const int newCols = points->cols + left + right;
    if (newRows <= 0 || newCols <= 0) {
        return false;
    }

    const cv::Size oldSize = points->size();
    for (const std::string& name : surface.channelNames()) {
        // Loads the channel if it is still only on disk, so the saved copy is
        // rewritten at the new size instead of staying misaligned.
        cv::Mat channel = surface.channel(name, SURF_CHANNEL_NORESIZE);
        if (channel.empty()) {
            continue;
        }
        const int fy = channelFactor(channel.rows, oldSize.height);
        const int fx = channelFactor(channel.cols, oldSize.width);
        if (fy == 0 || fx == 0) {
            cv::resize(channel, channel, oldSize, 0, 0, cv::INTER_NEAREST);
            surface.setChannel(name, resizeBorder(channel, top, bottom, left, right, cv::Scalar::all(0)));
        } else {
            surface.setChannel(name, resizeBorder(channel, top * fy, bottom * fy, left * fx, right * fx,
                                                  cv::Scalar::all(0)));
        }
    }

    cv::Mat resized = resizeBorder(*points, top, bottom, left, right, cv::Scalar(-1.0, -1.0, -1.0));
    *points = cv::Mat_<cv::Vec3f>(resized);
    surface.invalidateCache();
    return true;
}

bool markMaskCellsValid(QuadSurface& surface, const std::vector<cv::Point>& cells)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty() || cells.empty()) {
        return false;
    }
    const auto names = surface.channelNames();
    if (std::find(names.begin(), names.end(), std::string("mask")) == names.end()) {
        return false;
    }
    cv::Mat mask = surface.channel("mask", SURF_CHANNEL_NORESIZE);
    if (mask.empty()) {
        return false;
    }
    int fy = channelFactor(mask.rows, points->rows);
    int fx = channelFactor(mask.cols, points->cols);
    if (fy == 0 || fx == 0) {
        cv::resize(mask, mask, points->size(), 0, 0, cv::INTER_NEAREST);
        fy = 1;
        fx = 1;
    }
    mask = mask.clone();
    bool changed = false;
    for (const auto& cell : cells) {
        if (cell.x < 0 || cell.y < 0 || cell.x >= points->cols || cell.y >= points->rows) {
            continue;
        }
        cv::Mat block = mask(cv::Rect(cell.x * fx, cell.y * fy, fx, fy));
        block.setTo(cv::Scalar::all(255));
        changed = true;
    }
    if (changed) {
        surface.setChannel("mask", mask);
        surface.invalidateCache();
    }
    return changed;
}

} // namespace segmentation
