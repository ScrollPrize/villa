#include "SurfaceGridEdit.hpp"
#include "SegmentationUndoHistory.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <map>
#include <string>
#include <utility>

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

cv::Mat editableMask(QuadSurface& surface, const cv::Size& size)
{
    auto mask = surface.channel("mask", SURF_CHANNEL_NORESIZE);
    if (!mask.empty() && (!channelFactor(mask.rows, size.height) || !channelFactor(mask.cols, size.width)))
        cv::resize(mask, mask, size, 0, 0, cv::INTER_NEAREST);
    return mask;
}

cv::Mat maskBlock(cv::Mat& mask, const cv::Size& size, int row, int col)
{
    const int fy = mask.rows / size.height;
    const int fx = mask.cols / size.width;
    return mask(cv::Rect(col * fx, row * fy, fx, fy));
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
    std::vector<std::pair<std::string, cv::Mat>> resizedChannels;
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
            // Resampling would lose pixels that cropping on Cancel cannot restore.
            return false;
        }
        resizedChannels.emplace_back(name, resizeBorder(channel, top * fy, bottom * fy, left * fx, right * fx,
                                                         cv::Scalar::all(0)));
    }

    cv::Mat resized = resizeBorder(*points, top, bottom, left, right, cv::Scalar(-1.0, -1.0, -1.0));
    for (const auto& [name, channel] : resizedChannels) {
        surface.setChannel(name, channel);
    }
    const cv::Rect resizedBounds(0, 0, newCols, newRows);
    std::vector<QuadSurface::MaskedVertex> maskedVertices;
    for (const auto& vertex : surface.maskedVertices) {
        const auto cell = vertex.cell + cv::Point(left, top);
        if (resizedBounds.contains(cell)) maskedVertices.push_back({cell, vertex.coords});
    }
    surface.maskedVertices = std::move(maskedVertices);
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
    auto mask = editableMask(surface, points->size()).clone();
    if (mask.empty()) {
        return false;
    }
    bool changed = false;
    for (const auto& cell : cells) {
        if (cell.x < 0 || cell.y < 0 || cell.x >= points->cols || cell.y >= points->rows) {
            continue;
        }
        cv::Mat block = maskBlock(mask, points->size(), cell.y, cell.x);
        block.setTo(cv::Scalar::all(255));
        changed = true;
    }
    if (changed) {
        surface.setChannel("mask", mask);
        surface.invalidateCache();
    }
    return changed;
}

bool activateRepairedVertices(QuadSurface& surface, std::vector<VertexDelta>& deltas)
{
    const auto& points = *std::as_const(surface).rawPointsPtr();
    const auto size = points.size();
    const cv::Rect bounds(cv::Point{}, size);
    auto mask = editableMask(surface, size);
    cv::Mat_<uint8_t> exclusions = surface.channel("growth_exclusions", SURF_CHANNEL_NORESIZE);
    bool maskChanged = false;
    bool exclusionsChanged = false;
    std::map<std::pair<int, int>, VertexDelta*> repairs;
    for (auto& delta : deltas) {
        const cv::Point cell(delta.col, delta.row);
        if (!bounds.contains(cell) || !vc::geometry::validSurfacePoint(points(cell))) continue;
        repairs.emplace(std::pair{delta.row, delta.col}, &delta);
        if (!mask.empty()) {
            const auto previous = maskBlock(mask, size, delta.row, delta.col);
            if (size_t(cv::countNonZero(previous)) < previous.total()) {
                delta.previousMask = previous.clone();
                if (!maskChanged) mask = mask.clone();
                maskBlock(mask, size, delta.row, delta.col).setTo(255);
                maskChanged = true;
            }
        }
        if (exclusions.size() == size && exclusions(cell)) {
            delta.previousExclusion = exclusions(cell);
            if (!exclusionsChanged) exclusions = exclusions.clone();
            exclusions(cell) = 0;
            exclusionsChanged = true;
        }
    }
    const auto erased = std::erase_if(surface.maskedVertices, [&](const auto& vertex) {
        const auto repair = repairs.find({vertex.cell.y, vertex.cell.x});
        if (repair == repairs.end()) return false;
        repair->second->previousHiddenWorld = vertex.coords;
        return true;
    });
    if (maskChanged) surface.setChannel("mask", mask);
    if (exclusionsChanged) surface.setChannel("growth_exclusions", exclusions);
    if (maskChanged || exclusionsChanged || erased) surface.invalidateCache();
    return maskChanged || exclusionsChanged || erased;
}

bool restoreRepairedVertices(QuadSurface& surface, const std::vector<VertexDelta>& deltas)
{
    if (std::none_of(deltas.begin(), deltas.end(), [](const auto& delta) {
        return !delta.previousMask.empty() || delta.previousExclusion || delta.previousHiddenWorld.has_value();
    })) return false;

    const auto size = std::as_const(surface).rawPointsPtr()->size();
    const cv::Rect bounds(cv::Point{}, size);
    auto mask = editableMask(surface, size);
    cv::Mat_<uint8_t> exclusions = surface.channel("growth_exclusions", SURF_CHANNEL_NORESIZE);
    bool maskChanged = false;
    bool exclusionsChanged = false;
    bool hiddenChanged = false;
    for (const auto& delta : deltas) {
        const cv::Point cell(delta.col, delta.row);
        if (!bounds.contains(cell)) continue;
        if (!delta.previousMask.empty() && !mask.empty()) {
            if (!maskChanged) mask = mask.clone();
            delta.previousMask.copyTo(maskBlock(mask, size, delta.row, delta.col));
            maskChanged = true;
        }
        if (delta.previousExclusion && exclusions.size() == size) {
            if (!exclusionsChanged) exclusions = exclusions.clone();
            exclusions(cell) = delta.previousExclusion;
            exclusionsChanged = true;
        }
        if (delta.previousHiddenWorld) {
            surface.maskedVertices.push_back({cell, *delta.previousHiddenWorld});
            hiddenChanged = true;
        }
    }
    if (maskChanged) surface.setChannel("mask", mask);
    if (exclusionsChanged) surface.setChannel("growth_exclusions", exclusions);
    if (maskChanged || exclusionsChanged || hiddenChanged) surface.invalidateCache();
    return maskChanged || exclusionsChanged || hiddenChanged;
}

void applyGrowthChannels(QuadSurface& surface, QuadSurface& result)
{
    const auto generations = result.channel("generations", SURF_CHANNEL_NORESIZE);
    if (!generations.empty()) surface.setChannel("generations", generations);

    auto mask = result.channel("mask", SURF_CHANNEL_NORESIZE);
    if (mask.empty()) mask = surface.validMask();
    surface.setChannel("mask", mask);

    auto exclusions = result.channel("growth_exclusions", SURF_CHANNEL_NORESIZE);
    if (exclusions.empty()) exclusions = cv::Mat_<uint8_t>(surface.rawPointsPtr()->size(), uint8_t{0});
    surface.setChannel("growth_exclusions", exclusions);
    surface.maskedVertices = result.maskedVertices;
}

} // namespace segmentation
