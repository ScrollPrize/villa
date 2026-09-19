#pragma once

#include <QPointF>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

// A small retained spatial index for device-space point hit testing. Overlay
// rebuilds populate it while they already have projected point positions;
// mouse moves then inspect only the cells touched by the hit radius.
class ScreenSpacePointIndex
{
public:
    static constexpr qreal kDefaultCellSize = 16.0;

    struct Candidate {
        QPointF devicePosition;
        std::uint64_t stablePrimary{0};
        std::uint64_t stableSecondary{0};
        std::size_t payloadIndex{0};
    };

    explicit ScreenSpacePointIndex(qreal cellSize = kDefaultCellSize)
        : _cellSize(std::isfinite(cellSize) && cellSize > 0.0
                        ? cellSize : kDefaultCellSize)
    {
    }

    void clear()
    {
        _candidates.clear();
        _cells.clear();
    }

    void reserve(std::size_t count)
    {
        _candidates.reserve(count);
        _cells.reserve(count);
    }

    void insert(Candidate candidate)
    {
        const auto cell = cellFor(candidate.devicePosition);
        if (!cell) return;
        const std::size_t index = _candidates.size();
        _candidates.push_back(std::move(candidate));
        _cells[*cell].push_back(index);
    }

    template <typename Accept>
    std::optional<std::size_t> closest(
        const QPointF& devicePosition, qreal radius, Accept&& accept) const
    {
        if (!finitePoint(devicePosition) || !std::isfinite(radius) || radius < 0.0)
            return std::nullopt;

        const auto minCell = cellFor(devicePosition - QPointF(radius, radius));
        const auto maxCell = cellFor(devicePosition + QPointF(radius, radius));
        if (!minCell || !maxCell) return std::nullopt;

        const qreal radiusSquared = radius * radius;
        qreal bestDistance = radiusSquared;
        std::optional<std::size_t> bestCandidate;
        auto consider = [&](std::size_t candidateIndex) {
            const Candidate& candidate = _candidates[candidateIndex];
            if (!accept(candidate.payloadIndex)) return;
            const QPointF delta = candidate.devicePosition - devicePosition;
            const qreal distance = delta.x() * delta.x() + delta.y() * delta.y();
            if (distance > radiusSquared) return;
            const bool equalDistance = bestCandidate && distance == bestDistance;
            if (!bestCandidate || distance < bestDistance
                || (equalDistance
                    && std::tie(candidate.stablePrimary, candidate.stableSecondary)
                        < std::tie(_candidates[*bestCandidate].stablePrimary,
                                   _candidates[*bestCandidate].stableSecondary))) {
                bestCandidate = candidateIndex;
                bestDistance = distance;
            }
        };

        constexpr long double kMaximumCellsPerAxis = 4096.0L;
        const long double columns = static_cast<long double>(maxCell->x)
            - static_cast<long double>(minCell->x) + 1.0L;
        const long double rows = static_cast<long double>(maxCell->y)
            - static_cast<long double>(minCell->y) + 1.0L;
        if (columns <= 0.0L || rows <= 0.0L) return std::nullopt;
        if (columns > kMaximumCellsPerAxis || rows > kMaximumCellsPerAxis) {
            for (std::size_t index = 0; index < _candidates.size(); ++index)
                consider(index);
        } else {
            for (std::int64_t y = minCell->y;; ++y) {
                for (std::int64_t x = minCell->x;; ++x) {
                    const auto found = _cells.find(Cell{x, y});
                    if (found != _cells.end()) {
                        for (std::size_t index : found->second) consider(index);
                    }
                    if (x == maxCell->x) break;
                }
                if (y == maxCell->y) break;
            }
        }
        return bestCandidate
            ? std::optional<std::size_t>(_candidates[*bestCandidate].payloadIndex)
            : std::nullopt;
    }

    std::optional<std::size_t> closest(
        const QPointF& devicePosition, qreal radius) const
    {
        return closest(devicePosition, radius, [](std::size_t) { return true; });
    }

    [[nodiscard]] std::size_t size() const { return _candidates.size(); }

private:
    struct Cell {
        std::int64_t x{0};
        std::int64_t y{0};

        friend bool operator==(const Cell&, const Cell&) = default;
    };

    struct CellHash {
        std::size_t operator()(const Cell& cell) const
        {
            const std::size_t x = std::hash<std::int64_t>{}(cell.x);
            const std::size_t y = std::hash<std::int64_t>{}(cell.y);
            return x ^ (y + 0x9e3779b9U + (x << 6U) + (x >> 2U));
        }
    };

    static bool finitePoint(const QPointF& point)
    {
        return std::isfinite(point.x()) && std::isfinite(point.y());
    }

    std::optional<Cell> cellFor(const QPointF& point) const
    {
        if (!finitePoint(point)) return std::nullopt;
        const long double x = std::floor(
            static_cast<long double>(point.x()) / _cellSize);
        const long double y = std::floor(
            static_cast<long double>(point.y()) / _cellSize);
        constexpr long double lo = static_cast<long double>(
            std::numeric_limits<std::int64_t>::min());
        constexpr long double hi = static_cast<long double>(
            std::numeric_limits<std::int64_t>::max());
        if (x < lo || x > hi || y < lo || y > hi) return std::nullopt;
        return Cell{static_cast<std::int64_t>(x), static_cast<std::int64_t>(y)};
    }

    qreal _cellSize{kDefaultCellSize};
    std::vector<Candidate> _candidates;
    std::unordered_map<Cell, std::vector<std::size_t>, CellHash> _cells;
};
