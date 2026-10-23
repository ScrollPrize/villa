#include "ManualAddTool.hpp"

#include "ThinPlateSpline3d.hpp"

#include "vc/core/util/PlaneSurface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_map>

#include <opencv2/core.hpp>

namespace
{
ManualAddTool::GridPolyline makeLine(int row0, int col0, int row1, int col1)
{
    ManualAddTool::GridPolyline line;
    if (row0 == row1) {
        const int lo = std::min(col0, col1);
        const int hi = std::max(col0, col1);
        line.vertices.reserve(static_cast<std::size_t>(hi - lo + 1));
        for (int col = lo; col <= hi; ++col) {
            line.vertices.emplace_back(col, row0);
        }
    } else if (col0 == col1) {
        const int lo = std::min(row0, row1);
        const int hi = std::max(row0, row1);
        line.vertices.reserve(static_cast<std::size_t>(hi - lo + 1));
        for (int row = lo; row <= hi; ++row) {
            line.vertices.emplace_back(col0, row);
        }
    }
    return line;
}

int rowOf(const cv::Point2i& p) { return p.y; }
int colOf(const cv::Point2i& p) { return p.x; }

// Vertices used to estimate a line's direction at its edge.
constexpr int kExtendTangentCells = 4;
// Neighboring lines (each side) averaged into a line's edge direction.
constexpr int kExtendTangentSmoothing = 2;

// How a side maps onto the grid. Lines are rows for Left and Right and
// columns for Up and Down; `step` is +1 or -1 along the line, outward.
struct ExtendAxes
{
    bool linesAreRows{true};
    int step{1};
};

ExtendAxes extendAxes(ManualAddTool::ExtendSide side)
{
    switch (side) {
    case ManualAddTool::ExtendSide::Up:
        return {false, -1};
    case ManualAddTool::ExtendSide::Down:
        return {false, 1};
    case ManualAddTool::ExtendSide::Left:
        return {true, -1};
    case ManualAddTool::ExtendSide::Right:
    default:
        return {true, 1};
    }
}

bool pointIsValid(const cv::Mat_<cv::Vec3f>& points, int row, int col)
{
    return row >= 0 && row < points.rows && col >= 0 && col < points.cols &&
           !ManualAddTool::isInvalidPoint(points(row, col));
}

// (row, col) of position `index` along `line`.
cv::Point2i lineCell(const ExtendAxes& axes, int line, int index)
{
    return axes.linesAreRows ? cv::Point2i(index, line) : cv::Point2i(line, index);
}

// Index of the last valid vertex of `line` going outward, or -1.
int lineEdge(const cv::Mat_<cv::Vec3f>& points, const ExtendAxes& axes, int line)
{
    const int length = axes.linesAreRows ? points.cols : points.rows;
    for (int i = 0; i < length; ++i) {
        const int index = axes.step > 0 ? length - 1 - i : i;
        const cv::Point2i cell = lineCell(axes, line, index);
        if (pointIsValid(points, cell.y, cell.x)) {
            return index;
        }
    }
    return -1;
}

std::pair<int, int> spanLines(const cv::Mat_<cv::Vec3f>& points,
                              const ExtendAxes& axes,
                              std::optional<ManualAddTool::EdgeSpan> span)
{
    const int count = axes.linesAreRows ? points.rows : points.cols;
    if (!span || span->last < span->first) {
        return {0, count - 1};
    }
    return {std::clamp(span->first, 0, count - 1), std::clamp(span->last, 0, count - 1)};
}
}

ManualAddTool::GridPadding ManualAddTool::paddingForExtension(const cv::Mat_<cv::Vec3f>& points,
                                                              ExtendSide side,
                                                              int cells,
                                                              std::optional<EdgeSpan> span)
{
    GridPadding padding;
    if (points.empty() || cells <= 0) {
        return padding;
    }
    const ExtendAxes axes = extendAxes(side);
    const int length = axes.linesAreRows ? points.cols : points.rows;
    const auto [firstLine, lastLine] = spanLines(points, axes, span);
    int needed = 0;
    for (int line = firstLine; line <= lastLine; ++line) {
        const int edge = lineEdge(points, axes, line);
        if (edge < 0) {
            continue;
        }
        const int target = edge + axes.step * cells;
        needed = std::max(needed, axes.step > 0 ? target - (length - 1) : -target);
    }
    switch (side) {
    case ExtendSide::Up:
        padding.top = needed;
        break;
    case ExtendSide::Down:
        padding.bottom = needed;
        break;
    case ExtendSide::Left:
        padding.left = needed;
        break;
    case ExtendSide::Right:
        padding.right = needed;
        break;
    }
    return padding;
}

cv::Mat_<cv::Vec3f> ManualAddTool::padPoints(const cv::Mat_<cv::Vec3f>& points, const GridPadding& padding)
{
    if (points.empty() || !padding.any()) {
        return points.clone();
    }
    cv::Mat_<cv::Vec3f> padded;
    cv::copyMakeBorder(points, padded,
                       std::max(0, padding.top), std::max(0, padding.bottom),
                       std::max(0, padding.left), std::max(0, padding.right),
                       cv::BORDER_CONSTANT, cv::Scalar(-1.0, -1.0, -1.0));
    return padded;
}

std::optional<ManualAddTool::ExtendSide> ManualAddTool::extensionSide() const
{
    if (!_extension) {
        return std::nullopt;
    }
    return _extension->side;
}

ManualAddTool::Config ManualAddTool::sanitize(Config config)
{
    config.maxPreviewSpan = std::clamp(config.maxPreviewSpan, 4, 4096);
    config.boundaryBand = std::clamp(config.boundaryBand, 1, 20);
    config.regularization = std::clamp(config.regularization, 0.0, 1.0);
    config.sampleCap = std::clamp(config.sampleCap, 16, 10000);
    config.previewThrottleMs = std::clamp(config.previewThrottleMs, 0, 500);
    config.tintOpacity = std::clamp(config.tintOpacity, 0.0f, 1.0f);
    config.planeConstraintRadius = std::clamp(config.planeConstraintRadius, 0.5, 100.0);
    config.planeConstraintReplacementRadius = std::clamp(config.planeConstraintReplacementRadius, 0.0, 100.0);
    const int mode = std::clamp(static_cast<int>(config.linePreviewMode),
                                static_cast<int>(LinePreviewMode::VerticalOnly),
                                static_cast<int>(LinePreviewMode::CrossFill));
    config.linePreviewMode = static_cast<LinePreviewMode>(mode);
    const int interpolation = std::clamp(static_cast<int>(config.interpolationMode),
                                         static_cast<int>(InterpolationMode::ThinPlateSpline),
                                         static_cast<int>(InterpolationMode::TracerRestrictedToFill));
    config.interpolationMode = static_cast<InterpolationMode>(interpolation);
    return config;
}

bool ManualAddTool::begin(const cv::Mat_<cv::Vec3f>& points, Config config)
{
    if (points.empty()) {
        return false;
    }
    _config = sanitize(config);
    _entrySnapshotPoints = points.clone();
    _previewPoints = points.clone();
    rebuildGridCache();
    _hoverPolylines.clear();
    _hoverVertex.reset();
    _hoverFillVertices.clear();
    _committedPolylines.clear();
    _fillVertices.clear();
    _borderSampleVertices.clear();
    _changedVertices.clear();
    _userPlaneConstraints.clear();
    _initialFillCommitted = false;
    _extension.reset();
    clearHoverFillCache();
    touchRevision();
    return true;
}

void ManualAddTool::clear()
{
    _entrySnapshotPoints.release();
    _previewPoints.release();
    _validBounds = cv::Rect();
    _haveValidBounds = false;
    _invalidMask.clear();
    _fillSeenMarks.clear();
    _barrierMarks.clear();
    _sideMarks.clear();
    _otherSideMarks.clear();
    _markerEpoch = 0;
    _hoverPolylines.clear();
    _hoverVertex.reset();
    _hoverFillVertices.clear();
    _committedPolylines.clear();
    _fillVertices.clear();
    _borderSampleVertices.clear();
    _changedVertices.clear();
    _userPlaneConstraints.clear();
    _initialFillCommitted = false;
    _extension.reset();
    clearHoverFillCache();
    touchRevision();
}

bool ManualAddTool::clearPending(Config config)
{
    if (_entrySnapshotPoints.empty()) {
        return false;
    }
    _config = sanitize(config);
    _previewPoints = _entrySnapshotPoints.clone();
    rebuildGridCache();
    _hoverPolylines.clear();
    _hoverVertex.reset();
    _hoverFillVertices.clear();
    _committedPolylines.clear();
    _fillVertices.clear();
    _borderSampleVertices.clear();
    _changedVertices.clear();
    _userPlaneConstraints.clear();
    if (_extension) {
        // A cleared extension leaves nothing behind, so another may start.
        _extension.reset();
        _initialFillCommitted = false;
    }
    clearHoverFillCache();
    touchRevision();
    return true;
}

void ManualAddTool::setConfig(Config config)
{
    const LinePreviewMode previousMode = _config.linePreviewMode;
    _config = sanitize(config);
    if (_config.linePreviewMode != previousMode) {
        clearHoverFillCache();
    }
    if (_hoverVertex) {
        updateHover(_hoverVertex->row, _hoverVertex->col);
    }
}

bool ManualAddTool::isInvalidPoint(const cv::Vec3f& value)
{
    return !std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]) ||
           (value[0] == -1.0f && value[1] == -1.0f && value[2] == -1.0f);
}

bool ManualAddTool::inBounds(int row, int col) const
{
    return !_entrySnapshotPoints.empty() &&
           row >= 0 && row < _entrySnapshotPoints.rows &&
           col >= 0 && col < _entrySnapshotPoints.cols;
}

bool ManualAddTool::isInvalid(int row, int col) const
{
    return inBounds(row, col) && isInvalidPoint(_entrySnapshotPoints(row, col));
}

bool ManualAddTool::isValid(int row, int col) const
{
    return inBounds(row, col) && !isInvalidPoint(_entrySnapshotPoints(row, col));
}

int ManualAddTool::flatIndex(int row, int col) const
{
    return row * _entrySnapshotPoints.cols + col;
}

void ManualAddTool::rebuildGridCache()
{
    _validBounds = cv::Rect();
    _haveValidBounds = false;
    _invalidMask.clear();
    _fillSeenMarks.clear();
    _barrierMarks.clear();
    _sideMarks.clear();
    _otherSideMarks.clear();
    _markerEpoch = 0;
    clearHoverFillCache();

    if (_entrySnapshotPoints.empty()) {
        return;
    }

    const int rows = _entrySnapshotPoints.rows;
    const int cols = _entrySnapshotPoints.cols;
    const std::size_t size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    _invalidMask.assign(size, 0);
    _fillSeenMarks.assign(size, 0);
    _barrierMarks.assign(size, 0);
    _sideMarks.assign(size, 0);
    _otherSideMarks.assign(size, 0);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int index = flatIndex(row, col);
            const bool invalid = isInvalidPoint(_entrySnapshotPoints(row, col));
            _invalidMask[static_cast<std::size_t>(index)] = invalid ? 1 : 0;
            if (invalid) {
                continue;
            }
            const cv::Rect cell(col, row, 1, 1);
            _validBounds = _haveValidBounds ? (_validBounds | cell) : cell;
            _haveValidBounds = true;
        }
    }
}

void ManualAddTool::clearHoverFillCache()
{
    _cachedHoverFillLines.clear();
    _cachedHoverFillVertices.clear();
    _cachedHoverFillMode = _config.linePreviewMode;
    _hasCachedHoverFill = false;
}

bool ManualAddTool::hoverFillCacheMatches(const std::vector<GridPolyline>& lines) const
{
    if (!_hasCachedHoverFill || _cachedHoverFillMode != _config.linePreviewMode ||
        _cachedHoverFillLines.size() != lines.size()) {
        return false;
    }
    return std::equal(lines.begin(), lines.end(), _cachedHoverFillLines.begin(), [](const auto& a, const auto& b) {
        return a.vertices == b.vertices &&
               a.committed == b.committed &&
               a.floodFillComponent == b.floodFillComponent;
    });
}

void ManualAddTool::storeHoverFillCache(const std::vector<GridPolyline>& lines, const std::vector<GridKey>& fill)
{
    _cachedHoverFillLines = lines;
    _cachedHoverFillVertices = fill;
    _cachedHoverFillMode = _config.linePreviewMode;
    _hasCachedHoverFill = true;
}

uint32_t ManualAddTool::nextMarkerEpoch() const
{
    if (_markerEpoch >= UINT32_MAX - 4) {
        std::fill(_fillSeenMarks.begin(), _fillSeenMarks.end(), 0);
        std::fill(_barrierMarks.begin(), _barrierMarks.end(), 0);
        std::fill(_sideMarks.begin(), _sideMarks.end(), 0);
        std::fill(_otherSideMarks.begin(), _otherSideMarks.end(), 0);
        _markerEpoch = 0;
    }
    return ++_markerEpoch;
}

std::optional<ManualAddTool::GridPolyline> ManualAddTool::discoverAxisLine(int row, int col, bool horizontal) const
{
    int negRow = row;
    int negCol = col;
    int posRow = row;
    int posCol = col;
    const int dRow = horizontal ? 0 : 1;
    const int dCol = horizontal ? 1 : 0;

    while (inBounds(negRow, negCol) && isInvalid(negRow, negCol)) {
        negRow -= dRow;
        negCol -= dCol;
    }
    while (inBounds(posRow, posCol) && isInvalid(posRow, posCol)) {
        posRow += dRow;
        posCol += dCol;
    }
    if (!isValid(negRow, negCol) || !isValid(posRow, posCol)) {
        return std::nullopt;
    }
    const int span = std::abs(posRow - negRow) + std::abs(posCol - negCol);
    if (span < 2 || span > _config.maxPreviewSpan) {
        return std::nullopt;
    }
    return makeLine(negRow, negCol, posRow, posCol);
}

bool ManualAddTool::updateHover(int row, int col)
{
    if (_initialFillCommitted) {
        if (!_hoverPolylines.empty() || _hoverVertex.has_value()) {
            _hoverPolylines.clear();
            _hoverVertex.reset();
            _hoverFillVertices.clear();
            touchRevision();
            return true;
        }
        return false;
    }

    if (!inBounds(row, col) || !isInvalid(row, col)) {
        if (!_hoverPolylines.empty() || _hoverVertex.has_value()) {
            _hoverPolylines.clear();
            _hoverVertex.reset();
            _hoverFillVertices.clear();
            touchRevision();
            return true;
        }
        return false;
    }

    std::vector<GridPolyline> next;
    if (_config.linePreviewMode == LinePreviewMode::HorizontalOnly ||
        _config.linePreviewMode == LinePreviewMode::Cross ||
        _config.linePreviewMode == LinePreviewMode::CrossFill) {
        if (auto line = discoverAxisLine(row, col, true)) {
            next.push_back(*line);
        }
    }
    if (_config.linePreviewMode == LinePreviewMode::VerticalOnly ||
        _config.linePreviewMode == LinePreviewMode::Cross ||
        _config.linePreviewMode == LinePreviewMode::CrossFill) {
        if (auto line = discoverAxisLine(row, col, false)) {
            next.push_back(*line);
        }
    }
    std::vector<GridKey> nextHoverFill;
    if (_config.linePreviewMode != LinePreviewMode::CrossFill) {
        if (hoverFillCacheMatches(next)) {
            nextHoverFill = _cachedHoverFillVertices;
        } else {
            nextHoverFill = computeFillVerticesForLines(next, false);
            storeHoverFillCache(next, nextHoverFill);
        }
    }

    const bool vertexChanged = !_hoverVertex ||
                               _hoverVertex->row != row ||
                               _hoverVertex->col != col;
    const bool linesChanged = next.size() != _hoverPolylines.size() ||
                              !std::equal(next.begin(), next.end(), _hoverPolylines.begin(), [](const auto& a, const auto& b) {
                             return a.vertices == b.vertices && a.committed == b.committed;
                         });
    const bool hoverFillChanged = nextHoverFill != _hoverFillVertices;
    const bool changed = vertexChanged || linesChanged || hoverFillChanged;
    if (changed) {
        _hoverVertex = GridKey{row, col};
        _hoverPolylines = std::move(next);
        _hoverFillVertices = std::move(nextHoverFill);
        touchRevision();
    }
    return changed;
}

bool ManualAddTool::commitHover(std::string* status)
{
    if (_extension) {
        if (status) {
            *status = "An edge extension is pending. Apply it with Shift+E or clear it first.";
        }
        return false;
    }
    if (_initialFillCommitted) {
        if (status) {
            *status = "Manual Add initial fill has already been used. Press Shift+E to save, then enable Manual Add again.";
        }
        return false;
    }
    if (_hoverPolylines.empty()) {
        if (status) {
            *status = "No Manual Add bridge is available at the cursor.";
        }
        return false;
    }
    const auto previousCommittedPolylines = _committedPolylines;
    const auto previousHoverPolylines = _hoverPolylines;
    const auto previousHoverVertex = _hoverVertex;
    const auto previousPreview = _previewPoints.clone();
    const auto previousFillVertices = _fillVertices;
    const auto previousBorderSampleVertices = _borderSampleVertices;
    const auto previousChangedVertices = _changedVertices;

    for (auto line : _hoverPolylines) {
        line.committed = true;
        line.floodFillComponent = _config.linePreviewMode == LinePreviewMode::CrossFill;
        _committedPolylines.push_back(std::move(line));
    }
    _hoverPolylines.clear();
    _hoverVertex.reset();
    _hoverFillVertices.clear();
    touchRevision();
    if (recompute(status)) {
        _initialFillCommitted = true;
        touchRevision();
        return true;
    }

    _committedPolylines = previousCommittedPolylines;
    _hoverPolylines = previousHoverPolylines;
    _hoverVertex = previousHoverVertex;
    _hoverFillVertices = computeFillVerticesForLines(_hoverPolylines, false);
    _previewPoints = previousPreview;
    _fillVertices = previousFillVertices;
    _borderSampleVertices = previousBorderSampleVertices;
    _changedVertices = previousChangedVertices;
    touchRevision();
    return false;
}

std::vector<ManualAddTool::GridKey> ManualAddTool::computeFillVerticesForLines(
    const std::vector<GridPolyline>& lines,
    bool includeUserConstraints) const
{
    if (_entrySnapshotPoints.empty() || !_haveValidBounds || _invalidMask.empty()) {
        return {};
    }

    const int rows = _entrySnapshotPoints.rows;
    const int cols = _entrySnapshotPoints.cols;
    const std::size_t size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (_fillSeenMarks.size() != size || _barrierMarks.size() != size ||
        _sideMarks.size() != size || _otherSideMarks.size() != size) {
        return {};
    }

    std::vector<GridKey> fillVertices;
    const uint32_t seenEpoch = nextMarkerEpoch();
    const uint32_t barrierEpoch = nextMarkerEpoch();

    auto indexOf = [&](int row, int col) -> int {
        return row * cols + col;
    };
    auto fillAllowed = [&](int row, int col) {
        if (row < _validBounds.y || row >= _validBounds.y + _validBounds.height ||
            col < _validBounds.x || col >= _validBounds.x + _validBounds.width) {
            return false;
        }
        const int index = indexOf(row, col);
        return _invalidMask[static_cast<std::size_t>(index)] != 0;
    };
    auto marked = [&](const std::vector<uint32_t>& marks, int index, uint32_t epoch) {
        return marks[static_cast<std::size_t>(index)] == epoch;
    };

    for (const auto& line : lines) {
        for (const auto& p : line.vertices) {
            const int row = rowOf(p);
            const int col = colOf(p);
            if (row >= 0 && row < rows && col >= 0 && col < cols) {
                _barrierMarks[static_cast<std::size_t>(indexOf(row, col))] = barrierEpoch;
            }
        }
    }

    auto addFill = [&](int row, int col) {
        if (!fillAllowed(row, col)) {
            return;
        }
        const int index = indexOf(row, col);
        if (marked(_fillSeenMarks, index, seenEpoch)) {
            return;
        }
        _fillSeenMarks[static_cast<std::size_t>(index)] = seenEpoch;
        fillVertices.push_back(GridKey{row, col});
    };

    struct SideSearch
    {
        std::vector<GridKey> cells;
        std::queue<GridKey> queue;
        std::vector<uint32_t>* marks{nullptr};
        uint32_t epoch{0};
        bool complete{false};
    };

    auto pushSide = [&](SideSearch& side, int row, int col) {
        if (!fillAllowed(row, col)) {
            return;
        }
        const int index = indexOf(row, col);
        if (!side.marks || marked(_barrierMarks, index, barrierEpoch) || marked(*side.marks, index, side.epoch)) {
            return;
        }
        (*side.marks)[static_cast<std::size_t>(index)] = side.epoch;
        side.queue.push(GridKey{row, col});
    };

    auto collectPreferredSide = [&](const GridPolyline& line, int negRow, int negCol, int posRow, int posCol) {
        SideSearch negative;
        SideSearch positive;
        negative.marks = &_sideMarks;
        positive.marks = &_otherSideMarks;
        negative.epoch = nextMarkerEpoch();
        positive.epoch = nextMarkerEpoch();

        for (const auto& p : line.vertices) {
            pushSide(negative, rowOf(p) + negRow, colOf(p) + negCol);
            pushSide(positive, rowOf(p) + posRow, colOf(p) + posCol);
        }

        constexpr int kDirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        auto step = [&](SideSearch& side) {
            if (side.complete) {
                return;
            }
            if (side.queue.empty()) {
                side.complete = true;
                return;
            }
            const GridKey key = side.queue.front();
            side.queue.pop();
            side.cells.push_back(key);
            for (const auto& dir : kDirs) {
                pushSide(side, key.row + dir[0], key.col + dir[1]);
            }
            if (side.queue.empty()) {
                side.complete = true;
            }
        };

        auto shouldStop = [&]() {
            if (!negative.complete && !positive.complete) {
                return false;
            }
            if (negative.complete && positive.complete) {
                return true;
            }
            const SideSearch& done = negative.complete ? negative : positive;
            const SideSearch& running = negative.complete ? positive : negative;
            if (done.cells.empty()) {
                return false;
            }
            return running.cells.size() > done.cells.size();
        };

        while (!shouldStop()) {
            step(negative);
            step(positive);
        }

        const std::vector<GridKey>* chosen = nullptr;
        if (negative.complete && positive.complete) {
            if (negative.cells.empty()) {
                chosen = &positive.cells;
            } else if (positive.cells.empty()) {
                chosen = &negative.cells;
            } else {
                chosen = (positive.cells.size() <= negative.cells.size()) ? &positive.cells : &negative.cells;
            }
        } else if (negative.complete) {
            chosen = &negative.cells;
        } else if (positive.complete) {
            chosen = &positive.cells;
        }

        if (chosen) {
            for (const auto& key : *chosen) {
                addFill(key.row, key.col);
            }
        }
    };

    auto collectComponentFromLine = [&](const GridPolyline& line) {
        std::queue<GridKey> queue;
        auto push = [&](int row, int col) {
            if (!fillAllowed(row, col)) {
                return;
            }
            const int index = indexOf(row, col);
            if (marked(_fillSeenMarks, index, seenEpoch)) {
                return;
            }
            _fillSeenMarks[static_cast<std::size_t>(index)] = seenEpoch;
            queue.push(GridKey{row, col});
        };

        for (const auto& p : line.vertices) {
            push(rowOf(p), colOf(p));
        }

        constexpr int kDirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        while (!queue.empty()) {
            const GridKey key = queue.front();
            queue.pop();
            fillVertices.push_back(key);
            for (const auto& dir : kDirs) {
                push(key.row + dir[0], key.col + dir[1]);
            }
        }
    };

    for (const auto& line : lines) {
        if (line.floodFillComponent) {
            collectComponentFromLine(line);
            continue;
        }
        for (const auto& p : line.vertices) {
            addFill(rowOf(p), colOf(p));
        }
        if (line.vertices.size() < 2) {
            continue;
        }

        const bool horizontal = rowOf(line.vertices.front()) == rowOf(line.vertices.back());
        const bool vertical = colOf(line.vertices.front()) == colOf(line.vertices.back());
        if (horizontal) {
            collectPreferredSide(line, -1, 0, 1, 0);
        } else if (vertical) {
            collectPreferredSide(line, 0, -1, 0, 1);
        }
    }
    if (includeUserConstraints) {
        for (const auto& constraint : _userPlaneConstraints) {
            addFill(constraint.row, constraint.col);
        }
    }
    return fillVertices;
}

void ManualAddTool::extractFillAndBorder()
{
    _fillVertices.clear();
    _borderSampleVertices.clear();
    if (_entrySnapshotPoints.empty()) {
        return;
    }

    const int rows = _entrySnapshotPoints.rows;
    const int cols = _entrySnapshotPoints.cols;
    _fillVertices = computeFillVerticesForLines(_committedPolylines, true);

    std::set<std::pair<int, int>> border;
    auto addBorder = [&](int row, int col) {
        if (row >= 0 && row < rows && col >= 0 && col < cols && isValid(row, col)) {
            border.insert({row, col});
        }
    };
    for (const auto& key : _fillVertices) {
        for (int dr = -_config.boundaryBand; dr <= _config.boundaryBand; ++dr) {
            for (int dc = -_config.boundaryBand; dc <= _config.boundaryBand; ++dc) {
                addBorder(key.row + dr, key.col + dc);
            }
        }
    }
    if (_config.includeTouchedValidBorder) {
        auto fullyValidQuad = [&](int row, int col) {
            return row >= 0 && row + 1 < rows && col >= 0 && col + 1 < cols &&
                   isValid(row, col) && isValid(row + 1, col) &&
                   isValid(row, col + 1) && isValid(row + 1, col + 1);
        };
        auto quadTouchesVertex = [](int quadRow, int quadCol, const GridKey& key) {
            for (int dr = 0; dr <= 1; ++dr) {
                for (int dc = 0; dc <= 1; ++dc) {
                    if (std::abs((quadRow + dr) - key.row) <= 1 &&
                        std::abs((quadCol + dc) - key.col) <= 1) {
                        return true;
                    }
                }
            }
            return false;
        };
        for (const auto& key : _fillVertices) {
            for (int qr = key.row - 2; qr <= key.row + 1; ++qr) {
                for (int qc = key.col - 2; qc <= key.col + 1; ++qc) {
                    if (!fullyValidQuad(qr, qc) || !quadTouchesVertex(qr, qc, key)) {
                        continue;
                    }
                    addBorder(qr, qc);
                    addBorder(qr + 1, qc);
                    addBorder(qr, qc + 1);
                    addBorder(qr + 1, qc + 1);
                }
            }
        }
    }
    _borderSampleVertices.reserve(border.size());
    for (const auto& [row, col] : border) {
        _borderSampleVertices.push_back(GridKey{row, col});
    }
}

std::vector<ManualAddTool::Constraint3d> ManualAddTool::buildFitSamples() const
{
    std::map<std::pair<int, int>, Constraint3d> samples;
    auto add = [&](Constraint3d sample) {
        const auto key = std::make_pair(sample.row, sample.col);
        auto it = samples.find(key);
        if (it == samples.end() || static_cast<int>(sample.source) > static_cast<int>(it->second.source)) {
            samples[key] = sample;
        }
    };

    for (const auto& key : _borderSampleVertices) {
        add(Constraint3d{key.row, key.col, _entrySnapshotPoints(key.row, key.col), Constraint3d::Source::Boundary});
    }
    for (const auto& line : _committedPolylines) {
        if (line.vertices.empty()) {
            continue;
        }
        const cv::Point2i endpoints[2] = {line.vertices.front(), line.vertices.back()};
        for (const auto& p : endpoints) {
            const int row = rowOf(p);
            const int col = colOf(p);
            if (isValid(row, col)) {
                add(Constraint3d{row, col, _entrySnapshotPoints(row, col), Constraint3d::Source::CommittedLine});
            }
        }
    }
    for (const auto& constraint : _userPlaneConstraints) {
        add(constraint);
    }

    std::vector<Constraint3d> result;
    result.reserve(samples.size());
    for (const auto& entry : samples) {
        result.push_back(entry.second);
    }
    return downsampleSamples(std::move(result));
}

std::vector<ManualAddTool::Constraint3d> ManualAddTool::downsampleSamples(std::vector<Constraint3d> samples) const
{
    if (static_cast<int>(samples.size()) <= _config.sampleCap) {
        return samples;
    }
    std::vector<Constraint3d> required;
    std::vector<Constraint3d> border;
    for (const auto& sample : samples) {
        if (sample.source == Constraint3d::Source::Boundary) {
            border.push_back(sample);
        } else {
            required.push_back(sample);
        }
    }
    std::sort(border.begin(), border.end(), [](const auto& a, const auto& b) {
        return std::tie(a.row, a.col) < std::tie(b.row, b.col);
    });
    const int remaining = std::max(0, _config.sampleCap - static_cast<int>(required.size()));
    std::vector<Constraint3d> result = std::move(required);
    if (remaining <= 0 || border.empty()) {
        result.resize(std::min<int>(result.size(), _config.sampleCap));
        return result;
    }
    for (int i = 0; i < remaining; ++i) {
        const std::size_t index = static_cast<std::size_t>(i) * border.size() / static_cast<std::size_t>(remaining);
        result.push_back(border[std::min(index, border.size() - 1)]);
    }
    return result;
}

bool ManualAddTool::recompute(std::string* status)
{
    if (_entrySnapshotPoints.empty()) {
        return false;
    }
    if (_extension) {
        return recomputeExtension(status);
    }
    extractFillAndBorder();
    if (_fillVertices.empty()) {
        if (status) {
            *status = "Manual Add line does not touch an invalid component.";
        }
        return false;
    }

    if (_config.interpolationMode == InterpolationMode::TracerRestrictedToFill) {
        _previewPoints = _entrySnapshotPoints.clone();
        _changedVertices.clear();
        if (status) {
            *status = "Manual Add tracer fill mask updated.";
        }
        touchRevision();
        return true;
    }

    const auto samples = buildFitSamples();
    std::vector<ThinPlateSpline3d::Sample> tpsSamples;
    tpsSamples.reserve(samples.size());
    for (const auto& sample : samples) {
        tpsSamples.push_back({cv::Point2d(sample.col, sample.row),
                              cv::Vec3d(sample.world[0], sample.world[1], sample.world[2])});
    }

    ThinPlateSpline3d tps;
    if (!tps.fit(tpsSamples, _config.regularization)) {
        if (status) {
            *status = "Manual Add requires at least three non-collinear fit samples.";
        }
        return false;
    }

    _previewPoints = _entrySnapshotPoints.clone();
    _changedVertices.clear();
    _changedVertices.reserve(_fillVertices.size());
    for (const auto& key : _fillVertices) {
        auto predicted = tps.evaluate(cv::Point2d(key.col, key.row));
        if (!predicted) {
            if (status) {
                *status = "Manual Add interpolation produced a non-finite point.";
            }
            return false;
        }
        _previewPoints(key.row, key.col) = *predicted;
        _changedVertices.push_back(key);
    }

    if (_config.allowBoundarySmoothing) {
        for (const auto& key : _borderSampleVertices) {
            auto predicted = tps.evaluate(cv::Point2d(key.col, key.row));
            if (!predicted) {
                return false;
            }
            _previewPoints(key.row, key.col) = *predicted;
            _changedVertices.push_back(key);
        }
    }

    if (status) {
        *status = "Manual Add preview updated.";
    }
    touchRevision();
    return true;
}

std::optional<SegmentationEditManager::GridKey>
ManualAddTool::pickPlaneConstraintVertex(PlaneSurface& plane, const cv::Vec2f& clickPlane, double radius) const
{
    const double radiusSq = radius * radius;
    double bestDistSq = std::numeric_limits<double>::max();
    std::optional<GridKey> best;
    for (const auto& key : _fillVertices) {
        if (key.row < 0 || key.row >= _previewPoints.rows || key.col < 0 || key.col >= _previewPoints.cols) {
            continue;
        }
        const cv::Vec3f& vertex = _previewPoints(key.row, key.col);
        const cv::Vec3f projected = plane.project(vertex, 1.0f, 1.0f);
        const double dx = static_cast<double>(projected[0]) - clickPlane[0];
        const double dy = static_cast<double>(projected[1]) - clickPlane[1];
        const double inPlaneSq = dx * dx + dy * dy;
        if (inPlaneSq > radiusSq) {
            continue;
        }
        const double offPlane = static_cast<double>(plane.scalarp(vertex));
        const double distSq = inPlaneSq + offPlane * offPlane;
        if (distSq < bestDistSq) {
            bestDistSq = distSq;
            best = key;
        }
    }
    return best;
}

bool ManualAddTool::addOrReplacePlaneConstraint(int row, int col, const cv::Vec3f& world, std::string* status)
{
    if (!inBounds(row, col) || std::find_if(_fillVertices.begin(), _fillVertices.end(), [&](const GridKey& key) {
            return key.row == row && key.col == col;
        }) == _fillVertices.end()) {
        if (status) {
            *status = "No extrapolated Manual Add vertex is close enough for a plane constraint.";
        }
        return false;
    }
    const auto previousConstraints = _userPlaneConstraints;
    const auto previousPreview = _previewPoints.clone();
    const auto previousFillVertices = _fillVertices;
    const auto previousBorderSampleVertices = _borderSampleVertices;
    const auto previousChangedVertices = _changedVertices;

    Constraint3d constraint{row, col, world, Constraint3d::Source::PlaneUser};
    const double replacementRadiusSq = _config.planeConstraintReplacementRadius * _config.planeConstraintReplacementRadius;
    _userPlaneConstraints.erase(std::remove_if(_userPlaneConstraints.begin(), _userPlaneConstraints.end(), [&](const Constraint3d& c) {
                                    const cv::Vec3f delta = c.world - world;
                                    return c.row == row && c.col == col || delta.dot(delta) <= replacementRadiusSq;
                                }),
                                _userPlaneConstraints.end());
    _userPlaneConstraints.push_back(constraint);
    if (recompute(status)) {
        return true;
    }

    _userPlaneConstraints = previousConstraints;
    _previewPoints = previousPreview;
    _fillVertices = previousFillVertices;
    _borderSampleVertices = previousBorderSampleVertices;
    _changedVertices = previousChangedVertices;
    touchRevision();
    return false;
}

bool ManualAddTool::removePlaneConstraintNear(const cv::Vec3f& world, double radius, std::string* status)
{
    const auto previousConstraints = _userPlaneConstraints;
    const auto previousPreview = _previewPoints.clone();
    const auto previousFillVertices = _fillVertices;
    const auto previousBorderSampleVertices = _borderSampleVertices;
    const auto previousChangedVertices = _changedVertices;

    const double radiusSq = radius * radius;
    auto best = _userPlaneConstraints.end();
    double bestDistSq = radiusSq;
    for (auto it = _userPlaneConstraints.begin(); it != _userPlaneConstraints.end(); ++it) {
        const cv::Vec3f delta = it->world - world;
        const double distSq = delta.dot(delta);
        if (distSq <= bestDistSq) {
            bestDistSq = distSq;
            best = it;
        }
    }
    if (best == _userPlaneConstraints.end()) {
        if (status) {
            *status = "No Manual Add plane constraint is close enough to remove.";
        }
        return false;
    }
    _userPlaneConstraints.erase(best);
    if (recompute(status)) {
        return true;
    }

    _userPlaneConstraints = previousConstraints;
    _previewPoints = previousPreview;
    _fillVertices = previousFillVertices;
    _borderSampleVertices = previousBorderSampleVertices;
    _changedVertices = previousChangedVertices;
    touchRevision();
    return false;
}

bool ManualAddTool::removeLastPlaneConstraint(std::string* status)
{
    if (_userPlaneConstraints.empty()) {
        if (status) {
            *status = "No Manual Add plane constraints to undo.";
        }
        return false;
    }

    const auto previousConstraints = _userPlaneConstraints;
    const auto previousPreview = _previewPoints.clone();
    const auto previousFillVertices = _fillVertices;
    const auto previousBorderSampleVertices = _borderSampleVertices;
    const auto previousChangedVertices = _changedVertices;

    _userPlaneConstraints.pop_back();
    if (recompute(status)) {
        if (status) {
            *status = "Undid last Manual Add plane constraint.";
        }
        return true;
    }

    _userPlaneConstraints = previousConstraints;
    _previewPoints = previousPreview;
    _fillVertices = previousFillVertices;
    _borderSampleVertices = previousBorderSampleVertices;
    _changedVertices = previousChangedVertices;
    touchRevision();
    return false;
}

bool ManualAddTool::beginExtension(ExtendSide side,
                                   int cells,
                                   std::optional<EdgeSpan> span,
                                   std::string* status)
{
    auto fail = [&](const char* message) {
        if (status) {
            *status = message;
        }
        return false;
    };
    if (_entrySnapshotPoints.empty()) {
        return fail("Manual Add is not active.");
    }
    if (_extension || _initialFillCommitted || !_committedPolylines.empty()) {
        return fail("Apply or clear the pending Manual Add geometry before extending an edge.");
    }
    cells = std::clamp(cells, 1, _config.maxPreviewSpan);

    const ExtendAxes axes = extendAxes(side);
    const int length = axes.linesAreRows ? _entrySnapshotPoints.cols : _entrySnapshotPoints.rows;
    const auto [firstLine, lastLine] = spanLines(_entrySnapshotPoints, axes, span);

    struct EdgeLine
    {
        int line{0};
        int edge{0};
        cv::Vec3f tangent{0.0f, 0.0f, 0.0f};
    };
    std::vector<EdgeLine> lines;
    for (int line = firstLine; line <= lastLine; ++line) {
        const int edge = lineEdge(_entrySnapshotPoints, axes, line);
        if (edge < 0) {
            continue;
        }
        // Direction at the edge from the run of valid vertices behind it.
        int run = 0;
        for (int back = 1; back <= kExtendTangentCells; ++back) {
            const cv::Point2i cell = lineCell(axes, line, edge - axes.step * back);
            if (!isValid(cell.y, cell.x)) {
                break;
            }
            run = back;
        }
        if (run == 0) {
            continue;
        }
        const cv::Point2i edgeCell = lineCell(axes, line, edge);
        const cv::Point2i backCell = lineCell(axes, line, edge - axes.step * run);
        const cv::Vec3f tangent = (_entrySnapshotPoints(edgeCell.y, edgeCell.x) -
                                   _entrySnapshotPoints(backCell.y, backCell.x)) / static_cast<float>(run);
        const int target = edge + axes.step * cells;
        if (target < 0 || target >= length) {
            return fail("The grid has no room for this extension yet.");
        }
        lines.push_back(EdgeLine{line, edge, tangent});
    }
    if (lines.size() < 2) {
        return fail("Fewer than two lines on that side end in two valid vertices; nothing to extend.");
    }

    // Average each line's direction with its neighbors so one noisy edge
    // vertex does not steer its whole line.
    std::vector<cv::Vec3f> smoothed(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        cv::Vec3f sum(0.0f, 0.0f, 0.0f);
        int count = 0;
        for (std::size_t j = 0; j < lines.size(); ++j) {
            if (std::abs(lines[j].line - lines[i].line) <= kExtendTangentSmoothing) {
                sum += lines[j].tangent;
                ++count;
            }
        }
        smoothed[i] = sum / static_cast<float>(std::max(1, count));
    }

    Extension extension;
    extension.side = side;
    extension.cells = cells;
    _fillVertices.clear();
    _changedVertices.clear();
    _borderSampleVertices.clear();
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const auto& line = lines[i];
        const cv::Point2i edgeCell = lineCell(axes, line.line, line.edge);
        const cv::Vec3f origin = _entrySnapshotPoints(edgeCell.y, edgeCell.x);
        for (int k = 1; k <= cells; ++k) {
            const cv::Point2i cell = lineCell(axes, line.line, line.edge + axes.step * k);
            _fillVertices.push_back(GridKey{cell.y, cell.x});
            extension.guess.push_back(origin + smoothed[i] * static_cast<float>(k));
        }
        for (int back = 0; back <= _config.boundaryBand; ++back) {
            const cv::Point2i cell = lineCell(axes, line.line, line.edge - axes.step * back);
            if (!isValid(cell.y, cell.x)) {
                break;
            }
            extension.anchors.push_back(GridKey{cell.y, cell.x});
        }
    }
    _borderSampleVertices = extension.anchors;
    _extension = std::move(extension);
    _initialFillCommitted = true;
    _hoverPolylines.clear();
    _hoverVertex.reset();
    _hoverFillVertices.clear();
    _userPlaneConstraints.clear();
    if (!recomputeExtension(status)) {
        _extension.reset();
        _initialFillCommitted = false;
        _fillVertices.clear();
        _borderSampleVertices.clear();
        _changedVertices.clear();
        _previewPoints = _entrySnapshotPoints.clone();
        touchRevision();
        return false;
    }
    if (status) {
        *status = "Extending " + std::to_string(lines.size()) + " lines by " + std::to_string(cells) +
                  " vertices. Shift+click the sheet in a slice view to pull the new strip onto it.";
    }
    return true;
}

bool ManualAddTool::recomputeExtension(std::string* status)
{
    if (!_extension || _fillVertices.size() != _extension->guess.size()) {
        if (status) {
            *status = "No edge extension is pending.";
        }
        return false;
    }

    // An extension is always previewed and applied by the spline, whatever
    // the fill method: tracer growth would replace the grid under it.
    _previewPoints = _entrySnapshotPoints.clone();
    _changedVertices = _fillVertices;
    std::unordered_map<int, std::size_t> guessIndex;
    guessIndex.reserve(_fillVertices.size());
    for (std::size_t i = 0; i < _fillVertices.size(); ++i) {
        const auto& key = _fillVertices[i];
        _previewPoints(key.row, key.col) = _extension->guess[i];
        guessIndex[flatIndex(key.row, key.col)] = i;
    }

    if (!_userPlaneConstraints.empty()) {
        // Fit how far the clicks move the continuation, holding the edge in
        // place, and apply that displacement to the whole new strip.
        std::vector<ThinPlateSpline3d::Sample> samples;
        const int constraintCount = static_cast<int>(_userPlaneConstraints.size());
        const int anchorBudget = std::max(3, _config.sampleCap - constraintCount);
        const auto& anchors = _extension->anchors;
        const int anchorCount = static_cast<int>(anchors.size());
        const int anchorTake = std::min(anchorCount, anchorBudget);
        samples.reserve(static_cast<std::size_t>(anchorTake + constraintCount));
        for (int i = 0; i < anchorTake; ++i) {
            const std::size_t index = static_cast<std::size_t>(i) * static_cast<std::size_t>(anchorCount) /
                                      static_cast<std::size_t>(anchorTake);
            const auto& key = anchors[std::min(index, anchors.size() - 1)];
            samples.push_back({cv::Point2d(key.col, key.row), cv::Vec3d(0.0, 0.0, 0.0)});
        }
        for (const auto& constraint : _userPlaneConstraints) {
            auto it = guessIndex.find(flatIndex(constraint.row, constraint.col));
            if (it == guessIndex.end()) {
                continue;
            }
            const cv::Vec3f delta = constraint.world - _extension->guess[it->second];
            samples.push_back({cv::Point2d(constraint.col, constraint.row),
                               cv::Vec3d(delta[0], delta[1], delta[2])});
        }

        ThinPlateSpline3d tps;
        if (!tps.fit(samples, _config.regularization)) {
            if (status) {
                *status = "Manual Add could not fit the edge extension to the plane constraints.";
            }
            return false;
        }
        for (std::size_t i = 0; i < _fillVertices.size(); ++i) {
            const auto& key = _fillVertices[i];
            auto delta = tps.evaluate(cv::Point2d(key.col, key.row));
            if (!delta) {
                if (status) {
                    *status = "Manual Add interpolation produced a non-finite point.";
                }
                return false;
            }
            _previewPoints(key.row, key.col) = _extension->guess[i] + *delta;
        }
    }

    if (status) {
        *status = "Manual Add edge extension preview updated.";
    }
    touchRevision();
    return true;
}
