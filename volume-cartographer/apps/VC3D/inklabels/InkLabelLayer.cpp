#include "inklabels/InkLabelLayer.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace vc3d::inklabels {

int fullResolutionExtent(int stored, float scale)
{
    if (!std::isfinite(scale) || scale <= 0.0f) {
        return std::max(1, stored);
    }
    const double extent = static_cast<double>(stored) * (1.0 / static_cast<double>(scale));
    return std::max(1, static_cast<int>(std::lround(extent)));
}

cv::Size canvasSize(int gridRows, int gridCols, const cv::Vec2f& scale)
{
    return {fullResolutionExtent(gridCols, scale[0]), fullResolutionExtent(gridRows, scale[1])};
}

InkLabelLayer::InkLabelLayer(cv::Size size)
{
    reset(cv::Mat_<uint8_t>(size, uint8_t(0)));
}

void InkLabelLayer::reset(const cv::Mat_<uint8_t>& labels)
{
    _labels = labels.clone();
    _tilesX = (_labels.cols + kTile - 1) / kTile;
    _stroke.reset();
    _undo.clear();
    _redo.clear();
    _undoBytes = 0;
    ++_revision;
    _savedRevision = _revision;
}

cv::Rect InkLabelLayer::tileRect(int index) const
{
    const int tx = index % std::max(1, _tilesX);
    const int ty = index / std::max(1, _tilesX);
    return cv::Rect(tx * kTile, ty * kTile, kTile, kTile) & cv::Rect(0, 0, _labels.cols, _labels.rows);
}

void InkLabelLayer::beginStroke(Label value, float radius)
{
    if (_stroke) {
        endStroke();
    }
    Stroke stroke;
    stroke.value = static_cast<uint8_t>(value);
    stroke.radius = std::max(0.5f, radius);
    _stroke = std::move(stroke);
}

void InkLabelLayer::captureTiles(const cv::Rect& rect)
{
    if (rect.empty()) {
        return;
    }
    const int tx0 = rect.x / kTile;
    const int ty0 = rect.y / kTile;
    const int tx1 = (rect.x + rect.width - 1) / kTile;
    const int ty1 = (rect.y + rect.height - 1) / kTile;
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            const int index = ty * _tilesX + tx;
            if (_stroke->before.count(index) == 0) {
                _stroke->before.emplace(index, _labels(tileRect(index)).clone());
            }
        }
    }
}

cv::Rect InkLabelLayer::stampDisc(const cv::Point2f& centre)
{
    const float r = _stroke->radius;
    const int x0 = std::max(0, static_cast<int>(std::floor(centre.x - r)));
    const int y0 = std::max(0, static_cast<int>(std::floor(centre.y - r)));
    const int x1 = std::min(_labels.cols - 1, static_cast<int>(std::ceil(centre.x + r)));
    const int y1 = std::min(_labels.rows - 1, static_cast<int>(std::ceil(centre.y + r)));
    if (x1 < x0 || y1 < y0) {
        return {};
    }
    const cv::Rect bounds(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    captureTiles(bounds);
    const float r2 = r * r;
    const uint8_t value = _stroke->value;
    int minX = x1 + 1, minY = y1 + 1, maxX = -1, maxY = -1;
    for (int y = y0; y <= y1; ++y) {
        const float dy = static_cast<float>(y) + 0.5f - centre.y;
        uint8_t* row = _labels.ptr<uint8_t>(y);
        for (int x = x0; x <= x1; ++x) {
            const float dx = static_cast<float>(x) + 0.5f - centre.x;
            if (dx * dx + dy * dy <= r2 && row[x] != value) {
                row[x] = value;
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
            }
        }
    }
    if (maxX < 0) {
        return {};
    }
    return {minX, minY, maxX - minX + 1, maxY - minY + 1};
}

cv::Rect InkLabelLayer::strokeTo(const cv::Point2f& point)
{
    if (!_stroke || _labels.empty()) {
        return {};
    }
    cv::Rect changed;
    auto stamp = [&](const cv::Point2f& p) {
        const cv::Rect r = stampDisc(p);
        if (!r.empty()) {
            changed = changed.empty() ? r : (changed | r);
        }
    };
    if (!_stroke->last) {
        stamp(point);
    } else {
        // Stamps no farther apart than a quarter radius keep the swept edge smooth.
        const cv::Point2f from = *_stroke->last;
        const cv::Point2f delta = point - from;
        const float length = std::sqrt(delta.dot(delta));
        const float spacing = std::max(0.25f, _stroke->radius * 0.25f);
        const int steps = std::max(1, static_cast<int>(std::ceil(length / spacing)));
        for (int i = 1; i <= steps; ++i) {
            stamp(from + delta * (static_cast<float>(i) / static_cast<float>(steps)));
        }
    }
    _stroke->last = point;
    if (!changed.empty()) {
        ++_revision;
    }
    return changed;
}

bool InkLabelLayer::endStroke()
{
    if (!_stroke) {
        return false;
    }
    Edit edit;
    for (auto& [index, before] : _stroke->before) {
        cv::Mat_<uint8_t> after = _labels(tileRect(index)).clone();
        if (cv::countNonZero(before != after) == 0) {
            continue;
        }
        edit.bytes += before.total() + after.total();
        edit.tiles.push_back({index, std::move(before), std::move(after)});
    }
    _stroke.reset();
    if (edit.tiles.empty()) {
        return false;
    }
    std::sort(edit.tiles.begin(), edit.tiles.end(),
              [](const TileEdit& a, const TileEdit& b) { return a.index < b.index; });
    _undoBytes += edit.bytes;
    _undo.push_back(std::move(edit));
    _redo.clear();
    trimHistory();
    return true;
}

cv::Rect InkLabelLayer::cancelStroke()
{
    if (!_stroke) {
        return {};
    }
    cv::Rect changed;
    for (const auto& [index, before] : _stroke->before) {
        const cv::Rect r = tileRect(index);
        before.copyTo(_labels(r));
        changed = changed.empty() ? r : (changed | r);
    }
    _stroke.reset();
    ++_revision;
    return changed;
}

cv::Rect InkLabelLayer::apply(const Edit& edit, bool forward)
{
    cv::Rect changed;
    for (const auto& tile : edit.tiles) {
        const cv::Rect r = tileRect(tile.index);
        (forward ? tile.after : tile.before).copyTo(_labels(r));
        changed = changed.empty() ? r : (changed | r);
    }
    ++_revision;
    return changed;
}

cv::Rect InkLabelLayer::undo()
{
    if (!canUndo()) {
        return {};
    }
    Edit edit = std::move(_undo.back());
    _undo.pop_back();
    _undoBytes -= edit.bytes;
    const cv::Rect changed = apply(edit, false);
    _redo.push_back(std::move(edit));
    return changed;
}

cv::Rect InkLabelLayer::redo()
{
    if (!canRedo()) {
        return {};
    }
    Edit edit = std::move(_redo.back());
    _redo.pop_back();
    const cv::Rect changed = apply(edit, true);
    _undoBytes += edit.bytes;
    _undo.push_back(std::move(edit));
    trimHistory();
    return changed;
}

void InkLabelLayer::trimHistory()
{
    while (_undo.size() > kMinUndoStrokes && _undoBytes > _undoBudget) {
        _undoBytes -= _undo.front().bytes;
        _undo.pop_front();
    }
}

std::array<std::size_t, 3> InkLabelLayer::counts() const
{
    std::array<std::size_t, 3> result{0, 0, 0};
    if (_labels.empty()) {
        return result;
    }
    result[1] = static_cast<std::size_t>(cv::countNonZero(_labels == 1));
    result[2] = static_cast<std::size_t>(cv::countNonZero(_labels == 2));
    result[0] = _labels.total() - result[1] - result[2];
    return result;
}

} // namespace vc3d::inklabels
