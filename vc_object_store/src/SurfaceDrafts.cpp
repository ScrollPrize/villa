#include "internal/SurfaceDrafts.hpp"
#include "internal/ReadRequest.hpp"
#include "internal/VolumeFrames.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::objects::detail {
namespace {
using vc::geometry::SurfacePoint;
using vc::geometry::validSurfacePoint;

void checkVertex(const SurfacePoint& p)
{
    for (const auto v : p)
        if (!std::isfinite(v)) throw std::invalid_argument("surface draft vertices must be finite");
    if (!validSurfacePoint(p) && p != SurfacePoint{-1, -1, -1})
        throw std::invalid_argument("invalid surface vertices must use (-1,-1,-1)");
}

// The tile's validity and bounds from its points and channel. Bounds cover the
// vertices with coordinates, as stored tiles are indexed (coordinateBounds).
void finish(SurfaceDraft::Tile& tile, SurfaceDraft::TileData& data)
{
    auto& t = data.tile;
    t.valid.resize(t.points.size());
    tile.bounds.reset();
    for (size_t i = 0; i < t.points.size(); ++i) {
        const auto& p = t.points[i];
        const bool has = validSurfacePoint(p);
        t.valid[i] = has && data.channel[i];
        if (!has) continue;
        if (!tile.bounds) tile.bounds = Bounds{p[0], p[1], p[2], p[0], p[1], p[2]};
        else for (int axis = 0; axis < 3; ++axis) {
            (*tile.bounds)[axis] = std::min((*tile.bounds)[axis], double(p[axis]));
            (*tile.bounds)[axis + 3] = std::max((*tile.bounds)[axis + 3], double(p[axis]));
        }
    }
}

void reindex(SurfaceDraft& draft)
{
    std::vector<BoundsIndex::Box> boxes;
    draft.bounded.clear();
    for (size_t i = 0; i < draft.tiles.size(); ++i)
        if (draft.tiles[i].bounds) {
            boxes.push_back(*draft.tiles[i].bounds);
            draft.bounded.push_back(i);
        }
    draft.index.build(std::move(boxes));
}
}

std::vector<size_t> SurfaceDraft::tilesIn(const Bounds& box) const
{
    std::vector<size_t> found;
    index.query(box, [&](size_t i) { found.push_back(bounded[i]); });
    std::sort(found.begin(), found.end());
    return found;
}

std::shared_ptr<const SurfaceDraft> makeSurfaceDraft(const vc::geometry::Surface& surface, const CoordinateJson& coordinates,
                                                     const std::optional<AttachmentIdentity>& attachment)
{
    surface.validate();
    auto draft = std::make_shared<SurfaceDraft>();
    draft->rows = surface.rows;
    draft->columns = surface.columns;
    draft->coordinates = geometryCoordinates(coordinates);
    draft->attachment = attachment;
    for (const auto& rect : surfaceTiles(surface.rows, surface.columns)) {
        SurfaceDraft::Tile tile{rect, std::nullopt, nullptr};
        auto data = std::make_shared<SurfaceDraft::TileData>();
        for (int64_t r = 0; r < rect.rows; ++r)
            for (int64_t c = 0; c < rect.columns; ++c) {
                const auto index = size_t((rect.row + r) * surface.columns + rect.column + c);
                data->tile.points.push_back(surface.points[index]);
                data->channel.push_back(surface.validity.empty() || surface.validity[index] ? 1 : 0);
            }
        finish(tile, *data);
        tile.data = std::move(data);
        draft->tiles.push_back(std::move(tile));
    }
    reindex(*draft);
    return draft;
}

std::shared_ptr<const SurfaceDraft> editSurfaceDraft(const SurfaceDraft& draft, int64_t row, int64_t column, int64_t rows,
                                                     int64_t columns, std::span<const SurfacePoint> points,
                                                     std::span<const uint8_t> validity)
{
    if (row < 0 || column < 0 || rows < 0 || columns < 0 || row + rows > draft.rows || column + columns > draft.columns)
        throw std::out_of_range("surface draft edit outside the grid");
    if (points.size() != size_t(rows * columns) || (!validity.empty() && validity.size() != points.size()))
        throw std::invalid_argument("surface draft edit must have one vertex (and validity byte) per grid position");
    for (const auto& p : points) checkVertex(p);
    auto edited = std::make_shared<SurfaceDraft>(draft);
    for (auto& tile : edited->tiles) {
        const auto& rect = tile.rect;
        const int64_t r0 = std::max(row, rect.row), r1 = std::min(row + rows, rect.row + rect.rows);
        const int64_t c0 = std::max(column, rect.column), c1 = std::min(column + columns, rect.column + rect.columns);
        if (r0 >= r1 || c0 >= c1) continue;
        auto data = std::make_shared<SurfaceDraft::TileData>(*tile.data);
        for (int64_t r = r0; r < r1; ++r)
            for (int64_t c = c0; c < c1; ++c) {
                const auto source = size_t((r - row) * columns + c - column);
                const auto target = size_t((r - rect.row) * rect.columns + c - rect.column);
                data->tile.points[target] = points[source];
                if (!validity.empty()) data->channel[target] = validity[source] ? 1 : 0;
            }
        finish(tile, *data);
        tile.data = std::move(data);
    }
    reindex(*edited);
    return edited;
}

std::optional<DraftPlacement> placeDraft(const SurfaceDraft& draft, const std::optional<std::string>& coordinates,
                                         const std::optional<std::string>& volume, double targetScale,
                                         const std::optional<VolumeAffine>& toTarget, const VolumeTransforms* transforms)
{
    if (volume) {
        if (coordinates) throw std::invalid_argument("volume and coordinates filters are exclusive");
        if (!std::isfinite(targetScale) || targetScale <= 0)
            throw std::invalid_argument("target scale must be finite and positive");
        const auto mapping = frameMapping(draft.coordinates, *volume, transforms);
        if (!mapping) return std::nullopt;
        return DraftPlacement{VolumeAffine::scale(targetScale) * *mapping};
    }
    if (coordinates && geometryCoordinates(CoordinateJson::parse(*coordinates)) != draft.coordinates) return std::nullopt;
    return DraftPlacement{toTarget};
}

bool draftRegion(const std::string& uuid, const SurfaceDraft& draft, const DraftPlacement& placement,
                 const RegionSlice& view, const RegionOptions& options, const RegionCancel& cancel, RegionResult& result)
{
    if (options.family || options.minLength > 0)
        throw std::invalid_argument("surface regions have no family or length filter");
    const auto target = slabBounds(view);
    const auto bounds = placement.toTarget ? transformBounds(target, placement.toTarget->inverse()) : target;
    ReadRequest request{options, result, cancel};
    SurfaceCut cut(view, placement.toTarget, draft.columns);
    bool hit = false;
    try {
        for (const auto i : draft.tilesIn(bounds)) {
            const auto& tile = draft.tiles[i];
            request.poll();
            ++result.stats.candidateVisits;
            const auto& rect = tile.rect;
            if (rect.rows < 2 || rect.columns < 2) continue;
            result.stats.segmentTests += cut.add(tile.data->tile, rect.row, rect.column, rect.rows, rect.columns);
        }
        cut.pieces([&](int64_t segment, bool continues, const auto& start, const auto& end) {
            // Each point is XYZ and its grid position.
            const uint64_t added = continues ? 40 : 80;
            if (added > options.outputByteBudget - result.stats.outputBytes) request.stop(RegionStop::OutputBudget);
            if (!continues) result.runs.push_back({uuid, "", "points", "", segment, {start.point}, {start.grid}});
            result.runs.back().points.push_back(end.point);
            result.runs.back().grid.push_back(end.grid);
            result.stats.outputBytes += added;
            if (!hit) { hit = true; ++result.intersectingObjects; }
        });
    } catch (const ReadStopped&) {} // result.stop says why
    return hit;
}

std::optional<SurfaceProximityHit> draftNearest(const std::string& uuid, const SurfaceDraft& draft,
                                                const DraftPlacement& placement, const std::array<double, 3>& point,
                                                double tolerance)
{
    for (double x : point) if (!std::isfinite(x)) throw std::invalid_argument("surface proximity point must be finite");
    if (!std::isfinite(tolerance) || tolerance < 0)
        throw std::invalid_argument("surface proximity tolerance must be finite and nonnegative");
    SurfaceNearest nearest(point, tolerance, placement.toTarget);
    const auto box = placement.toTarget ? transformBounds(nearest.box(), placement.toTarget->inverse()) : nearest.box();
    for (const auto i : draft.tilesIn(box)) {
        const auto& tile = draft.tiles[i];
        const auto& rect = tile.rect;
        if (rect.rows < 2 || rect.columns < 2) continue;
        nearest.add(tile.data->tile, rect.row, rect.column, rect.rows, rect.columns);
    }
    const auto hit = nearest.take();
    if (!hit) return std::nullopt;
    return SurfaceProximityHit{uuid, "", hit->grid, hit->point, hit->distance};
}
}
