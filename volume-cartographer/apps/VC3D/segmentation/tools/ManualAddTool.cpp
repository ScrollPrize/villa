#include "ManualAddTool.hpp"

#include "ThinPlateSpline3d.hpp"

#include <ceres/ceres.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <tuple>

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

struct DistanceResidual
{
    DistanceResidual(double target, double weight)
        : target(target)
        , weight(weight)
    {
    }

    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const
    {
        const T dx = a[0] - b[0];
        const T dy = a[1] - b[1];
        const T dz = a[2] - b[2];
        residual[0] = T(weight) * (ceres::sqrt(dx * dx + dy * dy + dz * dz + T(1e-12)) - T(target));
        return true;
    }

    double target;
    double weight;
};

struct DistanceToFixedResidual
{
    DistanceToFixedResidual(cv::Vec3d fixed, double target, double weight)
        : fixed(fixed)
        , target(target)
        , weight(weight)
    {
    }

    template <typename T>
    bool operator()(const T* const a, T* residual) const
    {
        const T dx = a[0] - T(fixed[0]);
        const T dy = a[1] - T(fixed[1]);
        const T dz = a[2] - T(fixed[2]);
        residual[0] = T(weight) * (ceres::sqrt(dx * dx + dy * dy + dz * dz + T(1e-12)) - T(target));
        return true;
    }

    cv::Vec3d fixed;
    double target;
    double weight;
};

struct AnchorResidual
{
    AnchorResidual(cv::Vec3d target, double weight)
        : target(target)
        , weight(weight)
    {
    }

    template <typename T>
    bool operator()(const T* const p, T* residual) const
    {
        residual[0] = T(weight) * (p[0] - T(target[0]));
        residual[1] = T(weight) * (p[1] - T(target[1]));
        residual[2] = T(weight) * (p[2] - T(target[2]));
        return true;
    }

    cv::Vec3d target;
    double weight;
};
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
    config.spacingRelaxationHalo = std::clamp(config.spacingRelaxationHalo, 0, 8);
    config.spacingRelaxationMaxVertices = std::clamp(config.spacingRelaxationMaxVertices, 16, 10000);
    config.spacingRelaxationIterations = std::clamp(config.spacingRelaxationIterations, 1, 100);
    config.spacingRelaxationDistanceWeight = std::clamp(config.spacingRelaxationDistanceWeight, 0.0, 10.0);
    config.spacingRelaxationAnchorWeight = std::clamp(config.spacingRelaxationAnchorWeight, 0.0, 10.0);
    config.spacingRelaxationHaloAnchorWeight = std::clamp(config.spacingRelaxationHaloAnchorWeight, 0.0, 10.0);
    const int mode = std::clamp(static_cast<int>(config.linePreviewMode),
                                static_cast<int>(LinePreviewMode::VerticalOnly),
                                static_cast<int>(LinePreviewMode::CrossFill));
    config.linePreviewMode = static_cast<LinePreviewMode>(mode);
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
    _hoverPolylines.clear();
    _hoverVertex.reset();
    _committedPolylines.clear();
    _fillVertices.clear();
    _borderSampleVertices.clear();
    _changedVertices.clear();
    _userPlaneConstraints.clear();
    _initialFillCommitted = false;
    touchRevision();
    return true;
}

void ManualAddTool::clear()
{
    _entrySnapshotPoints.release();
    _previewPoints.release();
    _hoverPolylines.clear();
    _hoverVertex.reset();
    _committedPolylines.clear();
    _fillVertices.clear();
    _borderSampleVertices.clear();
    _changedVertices.clear();
    _userPlaneConstraints.clear();
    _initialFillCommitted = false;
    touchRevision();
}

bool ManualAddTool::clearPending(Config config)
{
    if (_entrySnapshotPoints.empty()) {
        return false;
    }
    _config = sanitize(config);
    _previewPoints = _entrySnapshotPoints.clone();
    _hoverPolylines.clear();
    _hoverVertex.reset();
    _committedPolylines.clear();
    _fillVertices.clear();
    _borderSampleVertices.clear();
    _changedVertices.clear();
    _userPlaneConstraints.clear();
    touchRevision();
    return true;
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
            touchRevision();
            return true;
        }
        return false;
    }

    if (!inBounds(row, col) || !isInvalid(row, col)) {
        if (!_hoverPolylines.empty() || _hoverVertex.has_value()) {
            _hoverPolylines.clear();
            _hoverVertex.reset();
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

    const bool vertexChanged = !_hoverVertex ||
                               _hoverVertex->row != row ||
                               _hoverVertex->col != col;
    const bool linesChanged = next.size() != _hoverPolylines.size() ||
                              !std::equal(next.begin(), next.end(), _hoverPolylines.begin(), [](const auto& a, const auto& b) {
                             return a.vertices == b.vertices && a.committed == b.committed;
                         });
    const bool changed = vertexChanged || linesChanged;
    if (changed) {
        _hoverVertex = GridKey{row, col};
        _hoverPolylines = std::move(next);
        touchRevision();
    }
    return changed;
}

bool ManualAddTool::commitHover(std::string* status)
{
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
    touchRevision();
    if (recompute(status)) {
        _initialFillCommitted = true;
        touchRevision();
        return true;
    }

    _committedPolylines = previousCommittedPolylines;
    _hoverPolylines = previousHoverPolylines;
    _hoverVertex = previousHoverVertex;
    _previewPoints = previousPreview;
    _fillVertices = previousFillVertices;
    _borderSampleVertices = previousBorderSampleVertices;
    _changedVertices = previousChangedVertices;
    touchRevision();
    return false;
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
    std::set<std::pair<int, int>> seen;
    std::set<std::pair<int, int>> barriers;

    for (const auto& line : _committedPolylines) {
        for (const auto& p : line.vertices) {
            barriers.insert({rowOf(p), colOf(p)});
        }
    }

    auto addFill = [&](int row, int col) {
        if (!isInvalid(row, col) || seen.count({row, col}) != 0) {
            return;
        }
        seen.insert({row, col});
        _fillVertices.push_back(GridKey{row, col});
    };

    auto collectSide = [&](const GridPolyline& line, int dRow, int dCol) {
        std::vector<GridKey> side;
        std::queue<GridKey> queue;
        std::set<std::pair<int, int>> sideSeen;

        auto push = [&](int row, int col) {
            const auto key = std::make_pair(row, col);
            if (!isInvalid(row, col) || barriers.count(key) != 0 || sideSeen.count(key) != 0) {
                return;
            }
            sideSeen.insert(key);
            queue.push(GridKey{row, col});
        };

        for (const auto& p : line.vertices) {
            push(rowOf(p) + dRow, colOf(p) + dCol);
        }

        constexpr int kDirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        while (!queue.empty()) {
            const GridKey key = queue.front();
            queue.pop();
            side.push_back(key);
            for (const auto& dir : kDirs) {
                push(key.row + dir[0], key.col + dir[1]);
            }
        }
        return side;
    };

    auto collectComponentFromLine = [&](const GridPolyline& line) {
        std::queue<GridKey> queue;
        auto push = [&](int row, int col) {
            if (!isInvalid(row, col) || seen.count({row, col}) != 0) {
                return;
            }
            seen.insert({row, col});
            queue.push(GridKey{row, col});
        };

        for (const auto& p : line.vertices) {
            push(rowOf(p), colOf(p));
        }

        constexpr int kDirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        while (!queue.empty()) {
            const GridKey key = queue.front();
            queue.pop();
            _fillVertices.push_back(key);
            for (const auto& dir : kDirs) {
                push(key.row + dir[0], key.col + dir[1]);
            }
        }
    };

    for (const auto& line : _committedPolylines) {
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
        if (!horizontal && !vertical) {
            continue;
        }

        const auto negativeSide = horizontal ? collectSide(line, -1, 0) : collectSide(line, 0, -1);
        const auto positiveSide = horizontal ? collectSide(line, 1, 0) : collectSide(line, 0, 1);
        const std::vector<GridKey>* chosen = nullptr;
        if (negativeSide.empty()) {
            chosen = &positiveSide;
        } else if (positiveSide.empty()) {
            chosen = &negativeSide;
        } else {
            chosen = (positiveSide.size() <= negativeSide.size()) ? &positiveSide : &negativeSide;
        }
        if (chosen) {
            for (const auto& key : *chosen) {
                addFill(key.row, key.col);
            }
        }
    }
    for (const auto& constraint : _userPlaneConstraints) {
        addFill(constraint.row, constraint.col);
    }

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

void ManualAddTool::relaxSpacing()
{
    if (!_config.spacingRelaxationEnabled ||
        _config.spacingRelaxationDistanceWeight <= 0.0 ||
        _previewPoints.empty() ||
        _fillVertices.empty()) {
        if (!_fillVertices.empty()) {
            std::cout << "Manual Add spacing relaxation skipped: disabled or no distance weight" << std::endl;
        }
        return;
    }

    const int rows = _previewPoints.rows;
    const int cols = _previewPoints.cols;
    std::set<std::pair<int, int>> fill;
    for (const auto& key : _fillVertices) {
        fill.insert({key.row, key.col});
    }
    std::set<std::pair<int, int>> userConstrained;
    for (const auto& constraint : _userPlaneConstraints) {
        userConstrained.insert({constraint.row, constraint.col});
    }

    std::set<std::pair<int, int>> variableKeys;
    const int halo = _config.spacingRelaxationHalo;
    for (const auto& key : _fillVertices) {
        for (int dr = -halo; dr <= halo; ++dr) {
            for (int dc = -halo; dc <= halo; ++dc) {
                const int row = key.row + dr;
                const int col = key.col + dc;
                if (row < 0 || row >= rows || col < 0 || col >= cols) {
                    continue;
                }
                if (!fill.count({row, col}) && !isValid(row, col)) {
                    continue;
                }
                if (!isInvalidPoint(_previewPoints(row, col))) {
                    variableKeys.insert({row, col});
                }
            }
        }
    }
    if (variableKeys.empty()) {
        std::cout << "Manual Add spacing relaxation skipped: no valid vertices in fill/halo" << std::endl;
        return;
    }
    if (static_cast<int>(variableKeys.size()) > _config.spacingRelaxationMaxVertices) {
        std::cout << "Manual Add spacing relaxation skipped: " << variableKeys.size()
                  << " vertices exceeds cap " << _config.spacingRelaxationMaxVertices << std::endl;
        return;
    }

    constexpr int kEdges[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    double fallbackTarget[4] = {0.0, 0.0, 0.0, 0.0};
    for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex) {
        std::vector<double> lengths;
        const int dr = kEdges[edgeIndex][0];
        const int dc = kEdges[edgeIndex][1];
        for (const auto& [row, col] : variableKeys) {
            const int nr = row + dr;
            const int nc = col + dc;
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols ||
                !isValid(row, col) || !isValid(nr, nc)) {
                continue;
            }
            lengths.push_back(cv::norm(_entrySnapshotPoints(row, col) - _entrySnapshotPoints(nr, nc)));
        }
        if (!lengths.empty()) {
            const auto mid = lengths.begin() + static_cast<std::ptrdiff_t>(lengths.size() / 2);
            std::nth_element(lengths.begin(), mid, lengths.end());
            fallbackTarget[edgeIndex] = std::max(1e-6, *mid);
        }
    }
    if (fallbackTarget[0] <= 0.0 && fallbackTarget[1] <= 0.0) {
        std::cout << "Manual Add spacing relaxation skipped: could not infer target edge length" << std::endl;
        return;
    }
    if (fallbackTarget[0] <= 0.0) {
        fallbackTarget[0] = fallbackTarget[1];
    }
    if (fallbackTarget[1] <= 0.0) {
        fallbackTarget[1] = fallbackTarget[0];
    }
    const double diagonalFallback = std::sqrt(fallbackTarget[0] * fallbackTarget[0] +
                                             fallbackTarget[1] * fallbackTarget[1]);
    if (fallbackTarget[2] <= 0.0) {
        fallbackTarget[2] = diagonalFallback;
    }
    if (fallbackTarget[3] <= 0.0) {
        fallbackTarget[3] = diagonalFallback;
    }

    struct Variable
    {
        int row;
        int col;
        cv::Vec3d initial;
        std::array<double, 3> value;
        bool fill;
        bool userConstrained;
    };

    std::vector<Variable> variables;
    variables.reserve(variableKeys.size());
    std::map<std::pair<int, int>, int> indexByKey;
    for (const auto& [row, col] : variableKeys) {
        const cv::Vec3f p = _previewPoints(row, col);
        const cv::Vec3d initial(p[0], p[1], p[2]);
        indexByKey[{row, col}] = static_cast<int>(variables.size());
        variables.push_back(Variable{row, col,
                                     initial,
                                     {initial[0], initial[1], initial[2]},
                                     fill.count({row, col}) != 0,
                                     userConstrained.count({row, col}) != 0});
    }

    auto targetForEdge = [&](int row, int col, int nr, int nc, int edgeIndex) {
        if (isValid(row, col) && isValid(nr, nc)) {
            return std::max(1e-6, static_cast<double>(cv::norm(_entrySnapshotPoints(row, col) - _entrySnapshotPoints(nr, nc))));
        }
        return fallbackTarget[edgeIndex];
    };

    ceres::Problem problem;
    int distanceResiduals = 0;
    for (auto& variable : variables) {
        const double anchorWeight = variable.fill
                                        ? _config.spacingRelaxationAnchorWeight
                                        : _config.spacingRelaxationHaloAnchorWeight;
        if (anchorWeight > 0.0) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<AnchorResidual, 3, 3>(
                    new AnchorResidual(variable.initial, anchorWeight)),
                nullptr,
                variable.value.data());
        }
        if (variable.userConstrained) {
            problem.SetParameterBlockConstant(variable.value.data());
        }
    }

    for (int i = 0; i < static_cast<int>(variables.size()); ++i) {
        const auto& variable = variables[i];
        for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex) {
            const int nr = variable.row + kEdges[edgeIndex][0];
            const int nc = variable.col + kEdges[edgeIndex][1];
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
                continue;
            }
            const double target = targetForEdge(variable.row, variable.col, nr, nc, edgeIndex);
            if (target <= 0.0) {
                continue;
            }
            const auto otherIt = indexByKey.find({nr, nc});
            if (otherIt != indexByKey.end()) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<DistanceResidual, 1, 3, 3>(
                        new DistanceResidual(target, _config.spacingRelaxationDistanceWeight)),
                    nullptr,
                    variables[i].value.data(),
                    variables[otherIt->second].value.data());
                ++distanceResiduals;
            } else if (isValid(nr, nc)) {
                const cv::Vec3f p = _entrySnapshotPoints(nr, nc);
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<DistanceToFixedResidual, 1, 3>(
                        new DistanceToFixedResidual(cv::Vec3d(p[0], p[1], p[2]), target,
                                                    _config.spacingRelaxationDistanceWeight)),
                    nullptr,
                    variables[i].value.data());
                ++distanceResiduals;
            }
        }
    }
    std::cout << "Manual Add spacing relaxation: solving " << variables.size()
              << " vertices, halo " << halo
              << ", " << distanceResiduals
              << " distance residuals, max " << _config.spacingRelaxationIterations
              << " iterations" << std::endl;

    ceres::Solver::Options options;
    options.max_num_iterations = _config.spacingRelaxationIterations;
    options.linear_solver_type = ceres::DENSE_QR;
    options.logging_type = ceres::SILENT;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        std::cout << "Manual Add spacing relaxation failed: "
                  << summary.BriefReport() << std::endl;
        return;
    }
    std::cout << "Manual Add spacing relaxation finished: "
              << summary.iterations.size() << " iterations, cost "
              << summary.initial_cost << " -> " << summary.final_cost << std::endl;

    std::set<std::pair<int, int>> changed;
    for (const auto& key : _changedVertices) {
        changed.insert({key.row, key.col});
    }
    for (const auto& variable : variables) {
        const cv::Vec3f relaxed(static_cast<float>(variable.value[0]),
                                static_cast<float>(variable.value[1]),
                                static_cast<float>(variable.value[2]));
        if (!std::isfinite(relaxed[0]) || !std::isfinite(relaxed[1]) || !std::isfinite(relaxed[2])) {
            continue;
        }
        _previewPoints(variable.row, variable.col) = relaxed;
        const cv::Vec3f original = _entrySnapshotPoints(variable.row, variable.col);
        if (variable.fill || cv::norm(relaxed - original) > 1e-5f) {
            changed.insert({variable.row, variable.col});
        }
    }
    _changedVertices.clear();
    _changedVertices.reserve(changed.size());
    for (const auto& [row, col] : changed) {
        _changedVertices.push_back(GridKey{row, col});
    }
}

bool ManualAddTool::recompute(std::string* status)
{
    if (_entrySnapshotPoints.empty()) {
        return false;
    }
    extractFillAndBorder();
    if (_fillVertices.empty()) {
        if (status) {
            *status = "Manual Add line does not touch an invalid component.";
        }
        return false;
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

bool ManualAddTool::finalizeSpacingRelaxation(std::string* status)
{
    if (_entrySnapshotPoints.empty()) {
        return false;
    }
    relaxSpacing();
    if (status) {
        *status = "Manual Add finalized.";
    }
    touchRevision();
    return true;
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
