#include "ink_labels/InkLabelRaster.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace vc3d::ink_labels {

namespace {

constexpr int kFixedShift = 8;
constexpr double kFixedScale = 256.0;

// OpenCV puts pixel i's center at integer i; label pixel centers sit at i+0.5.
// `origin` is the label pixel drawn at mask (0,0).
cv::Point toFixed(const cv::Point2d& p, cv::Point origin)
{
    return {int(std::lround((p.x - 0.5 - origin.x) * kFixedScale)), int(std::lround((p.y - 0.5 - origin.y) * kFixedScale))};
}

bool finitePoint(const cv::Point2d& p)
{
    return std::isfinite(p.x) && std::isfinite(p.y);
}

// Finite and representable in the fixed-point rasterizer.
bool usablePoint(const cv::Point2d& p)
{
    return finitePoint(p) && std::abs(p.x) <= kMaxShapeCoordinate && std::abs(p.y) <= kMaxShapeCoordinate;
}

// The radius a disc or capsule stamps with.
double stampRadius(double radius)
{
    return std::clamp(std::isfinite(radius) ? radius : kMinRadius, kMinRadius, kMaxRadius);
}

// Bounding box of label pixels a shape may touch, before canvas clipping.
cv::Rect shapeBounds(const std::vector<cv::Point2d>& points, double pad)
{
    double x0 = points.front().x, x1 = x0, y0 = points.front().y, y1 = y0;
    for (const auto& p : points) {
        x0 = std::min(x0, p.x);
        x1 = std::max(x1, p.x);
        y0 = std::min(y0, p.y);
        y1 = std::max(y1, p.y);
    }
    // Clamp before converting so a far-off point cannot overflow int.
    const auto clampCoord = [](double v) { return std::clamp(v, -1.0e8, 1.0e8); };
    const int ix0 = int(std::floor(clampCoord(x0 - pad))) - 1;
    const int iy0 = int(std::floor(clampCoord(y0 - pad))) - 1;
    const int ix1 = int(std::ceil(clampCoord(x1 + pad))) + 1;
    const int iy1 = int(std::ceil(clampCoord(y1 + pad))) + 1;
    return {ix0, iy0, ix1 - ix0, iy1 - iy0};
}

void setPixelIfInside(cv::Mat_<uint8_t>& mask, const cv::Point2d& p, cv::Point origin)
{
    const int x = int(std::floor(p.x)) - origin.x;
    const int y = int(std::floor(p.y)) - origin.y;
    if (x >= 0 && y >= 0 && x < mask.cols && y < mask.rows)
        mask(y, x) = 255;
}

cv::Rect localRect(const cv::Rect& labelRect, const cv::Rect& tileBounds)
{
    const cv::Rect clipped = labelRect & tileBounds;
    return {clipped.x - tileBounds.x, clipped.y - tileBounds.y, clipped.width, clipped.height};
}

}  // namespace

PaintOp paintOp(PaintClass cls, bool erase, bool backgroundKeepsInk)
{
    switch (cls) {
        case PaintClass::Ink:
            return erase ? PaintOp{0, uint8_t(kInk | kSupervision)} : PaintOp{uint8_t(kInk | kSupervision), 0};
        case PaintClass::Background:
            if (erase)
                return PaintOp{0, uint8_t(kInk | kSupervision)};
            return backgroundKeepsInk ? PaintOp{kSupervision, 0} : PaintOp{kSupervision, kInk};
        case PaintClass::Validation:
            return erase ? PaintOp{0, kValidation} : PaintOp{kValidation, 0};
    }
    return {};
}

// ---------------------------------------------------------------------------
// TileRaster
// ---------------------------------------------------------------------------

TileRaster::TileRaster(cv::Size canvas)
    : _size(std::max(0, canvas.width), std::max(0, canvas.height))
    , _grid((_size.width + kTileSize - 1) / kTileSize, (_size.height + kTileSize - 1) / kTileSize)
    , _tiles(std::size_t(_grid.area()))
    , _revisions(std::size_t(_grid.area()), 0)
{
}

cv::Rect TileRaster::tileRect(int index) const
{
    const int tx = index % _grid.width;
    const int ty = index / _grid.width;
    return cv::Rect(tx * kTileSize, ty * kTileSize, kTileSize, kTileSize) & cv::Rect(0, 0, _size.width, _size.height);
}

uint8_t TileRaster::at(int x, int y) const
{
    if (x < 0 || y < 0 || x >= _size.width || y >= _size.height)
        return 0;
    const TileState& t = _tiles[std::size_t(tileIndex(x / kTileSize, y / kTileSize))];
    if (!t.dense)
        return t.uniform;
    return t.dense->px[std::size_t((y % kTileSize) * kTileSize + (x % kTileSize))];
}

void TileRaster::readRect(const cv::Rect& rect, cv::Mat_<uint8_t>& out) const
{
    out.create(std::max(0, rect.height), std::max(0, rect.width));
    out.setTo(0);
    const cv::Rect clip = rect & cv::Rect(0, 0, _size.width, _size.height);
    if (clip.empty())
        return;
    const int tx0 = clip.x / kTileSize, tx1 = (clip.br().x - 1) / kTileSize;
    const int ty0 = clip.y / kTileSize, ty1 = (clip.br().y - 1) / kTileSize;
    for (int ty = ty0; ty <= ty1; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx) {
            const int index = tileIndex(tx, ty);
            const cv::Rect part = clip & tileRect(index);
            const TileState& t = _tiles[std::size_t(index)];
            cv::Mat_<uint8_t> dst = out(cv::Rect(part.x - rect.x, part.y - rect.y, part.width, part.height));
            if (!t.dense) {
                dst.setTo(t.uniform);
                continue;
            }
            for (int y = 0; y < part.height; ++y) {
                const uint8_t* src = t.dense->px.data() + std::size_t((part.y - ty * kTileSize + y) * kTileSize)
                                     + std::size_t(part.x - tx * kTileSize);
                std::copy(src, src + part.width, dst.ptr(y));
            }
        }
}

std::array<uint64_t, 8> TileRaster::counts() const
{
    std::array<uint64_t, 8> total{};
    for (int i = 0; i < tileCount(); ++i) {
        const TileState& t = _tiles[std::size_t(i)];
        if (t.dense) {
            for (int v = 0; v < 8; ++v)
                total[std::size_t(v)] += t.dense->counts[std::size_t(v)];
        } else {
            total[std::size_t(t.uniform & 7)] += uint64_t(tileRect(i).area());
        }
    }
    return total;
}

std::size_t TileRaster::denseTileCount() const
{
    return std::size_t(std::count_if(_tiles.begin(), _tiles.end(), [](const TileState& t) { return t.dense != nullptr; }));
}

void TileRaster::setTile(int index, TileState state)
{
    _tiles[std::size_t(index)] = std::move(state);
    ++_revisions[std::size_t(index)];
}

// ---------------------------------------------------------------------------
// Tile finalization
// ---------------------------------------------------------------------------

namespace {

uint8_t cellCode(const Tile& tile, int cx, int cy, const cv::Rect& inCanvas)
{
    const cv::Rect cell = cv::Rect(cx * kSummaryCell, cy * kSummaryCell, kSummaryCell, kSummaryCell) & inCanvas;
    if (cell.empty())
        return kNeutralCode;
    uint8_t anyBits = 0, allBits = 0xFF;
    for (int y = cell.y; y < cell.br().y; ++y) {
        const uint8_t* row = tile.px.data() + std::size_t(y * kTileSize);
        for (int x = cell.x; x < cell.br().x; ++x) {
            const uint8_t code = summaryCode(row[x]);
            anyBits |= code;
            allBits &= code;
        }
    }
    return uint8_t((anyBits & 0b1101) | (allBits & 0b0010));
}

}  // namespace

void finalizeTile(Tile& tile, const cv::Rect& inCanvas)
{
    tile.counts.fill(0);
    for (int y = inCanvas.y; y < inCanvas.br().y; ++y) {
        const uint8_t* row = tile.px.data() + std::size_t(y * kTileSize);
        for (int x = inCanvas.x; x < inCanvas.br().x; ++x)
            ++tile.counts[std::size_t(row[x] & 7)];
    }
    for (int cy = 0; cy < kSummarySize; ++cy)
        for (int cx = 0; cx < kSummarySize; ++cx)
            tile.summary[std::size_t(cy * kSummarySize + cx)] = cellCode(tile, cx, cy, inCanvas);
}

TileState publishTile(std::shared_ptr<Tile> tile, const cv::Rect& inCanvas)
{
    if (inCanvas.empty())
        return {};
    const uint8_t first = tile->px[std::size_t(inCanvas.y * kTileSize + inCanvas.x)];
    for (int y = inCanvas.y; y < inCanvas.br().y; ++y) {
        const uint8_t* row = tile->px.data() + std::size_t(y * kTileSize);
        for (int x = inCanvas.x; x < inCanvas.br().x; ++x)
            if (row[x] != first)
                return {std::shared_ptr<const Tile>(std::move(tile)), 0};
    }
    return {nullptr, first};
}

// ---------------------------------------------------------------------------
// EditTxn
// ---------------------------------------------------------------------------

EditTxn::EditTxn(TileRaster& raster, std::string label, Actor actor)
    : _raster(raster)
    , _label(std::move(label))
    , _actor(actor)
{
}

EditTxn::~EditTxn()
{
    if (!_finished)
        rollback();
}

void EditTxn::rollback()
{
    for (const int index : _order)
        _raster.setTile(index, _working[index].before);
    _working.clear();
    _order.clear();
    _finished = true;
}

Tile& EditTxn::workingTile(int index)
{
    if (auto it = _working.find(index); it != _working.end())
        return *it->second.tile;
    const TileState before = _raster._tiles[std::size_t(index)];
    auto tile = std::make_shared<Tile>();
    const cv::Rect bounds = _raster.tileRect(index);
    const cv::Rect inCanvas(bounds.x % kTileSize, bounds.y % kTileSize, bounds.width, bounds.height);
    if (before.dense) {
        *tile = *before.dense;
    } else {
        // Out-of-canvas pixels of edge tiles stay 0.
        for (int y = inCanvas.y; y < inCanvas.br().y; ++y)
            std::fill_n(tile->px.data() + std::size_t(y * kTileSize + inCanvas.x), inCanvas.width, before.uniform);
        finalizeTile(*tile, inCanvas);
    }
    _raster._tiles[std::size_t(index)] = TileState{tile, 0};
    _working.emplace(index, Working{before, tile});
    _order.push_back(index);
    return *tile;
}

void EditTxn::refreshSummary(Tile& tile, const cv::Rect& local, const cv::Rect& inCanvas)
{
    if (local.empty())
        return;
    const int cx0 = local.x / kSummaryCell, cx1 = (local.br().x - 1) / kSummaryCell;
    const int cy0 = local.y / kSummaryCell, cy1 = (local.br().y - 1) / kSummaryCell;
    for (int cy = cy0; cy <= cy1; ++cy)
        for (int cx = cx0; cx <= cx1; ++cx)
            tile.summary[std::size_t(cy * kSummarySize + cx)] = cellCode(tile, cx, cy, inCanvas);
}

uint64_t EditTxn::applyMaskedTile(int index, const MaskView& mask, const cv::Rect& local, PaintOp op)
{
    const TileState& current = _raster._tiles[std::size_t(index)];
    const auto maskAt = [&mask](int lx, int ly) {
        return mask.data[std::size_t(ly + mask.offset.y) * mask.step + std::size_t(lx + mask.offset.x)] != 0;
    };

    // First pass without cloning: does any masked pixel change?
    bool anyChange = false;
    if (!current.dense) {
        if (op.apply(current.uniform) == current.uniform)
            return 0;
        for (int y = local.y; y < local.br().y && !anyChange; ++y)
            for (int x = local.x; x < local.br().x; ++x)
                if (maskAt(x, y)) {
                    anyChange = true;
                    break;
                }
    } else {
        for (int y = local.y; y < local.br().y && !anyChange; ++y) {
            const uint8_t* row = current.dense->px.data() + std::size_t(y * kTileSize);
            for (int x = local.x; x < local.br().x; ++x)
                if (maskAt(x, y) && op.apply(row[x]) != row[x]) {
                    anyChange = true;
                    break;
                }
        }
    }
    if (!anyChange)
        return 0;

    Tile& tile = workingTile(index);
    uint64_t changed = 0;
    int minX = kTileSize, minY = kTileSize, maxX = -1, maxY = -1;
    for (int y = local.y; y < local.br().y; ++y) {
        uint8_t* row = tile.px.data() + std::size_t(y * kTileSize);
        for (int x = local.x; x < local.br().x; ++x) {
            if (!maskAt(x, y))
                continue;
            const uint8_t v = row[x];
            const uint8_t nv = op.apply(v);
            if (nv == v)
                continue;
            row[x] = nv;
            --tile.counts[std::size_t(v & 7)];
            ++tile.counts[std::size_t(nv & 7)];
            ++changed;
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
    }
    if (changed == 0)
        return 0;
    const cv::Rect bounds = _raster.tileRect(index);
    const cv::Rect inCanvas(bounds.x % kTileSize, bounds.y % kTileSize, bounds.width, bounds.height);
    const cv::Rect changedLocal(minX, minY, maxX - minX + 1, maxY - minY + 1);
    refreshSummary(tile, changedLocal, inCanvas);
    const cv::Point tileOrigin(bounds.x - inCanvas.x, bounds.y - inCanvas.y);
    const cv::Rect changedLabel = changedLocal + tileOrigin;
    _dirty = _dirty.empty() ? changedLabel : (_dirty | changedLabel);
    ++_raster._revisions[std::size_t(index)];
    return changed;
}

uint64_t EditTxn::applyToTiles(const cv::Rect& boundsIn,
                               PaintOp op,
                               const std::function<void(cv::Mat_<uint8_t>&, cv::Point)>& draw)
{
    const cv::Rect bounds = boundsIn & cv::Rect(0, 0, _raster._size.width, _raster._size.height);
    if (bounds.empty())
        return 0;
    cv::Mat_<uint8_t> scratch(kTileSize, kTileSize);
    uint64_t changed = 0;
    const int tx0 = bounds.x / kTileSize, tx1 = (bounds.br().x - 1) / kTileSize;
    const int ty0 = bounds.y / kTileSize, ty1 = (bounds.br().y - 1) / kTileSize;
    for (int ty = ty0; ty <= ty1; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx) {
            const int index = _raster.tileIndex(tx, ty);
            const TileState& current = _raster._tiles[std::size_t(index)];
            if (!current.dense && op.apply(current.uniform) == current.uniform)
                continue;
            const cv::Rect tileBounds = _raster.tileRect(index);
            const cv::Point origin(tx * kTileSize, ty * kTileSize);
            const cv::Rect local = localRect(bounds, cv::Rect(origin, cv::Size(kTileSize, kTileSize)) & tileBounds);
            if (local.empty())
                continue;
            scratch.setTo(0);
            draw(scratch, origin);
            const MaskView view{scratch.ptr(0), std::size_t(scratch.step[0]), {0, 0}};
            changed += applyMaskedTile(index, view, local, op);
        }
    _changedPixels += changed;
    return changed;
}

uint64_t EditTxn::apply(const Shape& shape, PaintOp op)
{
    if (_finished || (op.set == 0 && op.clear == 0) || shape.points.empty())
        return 0;
    if (!std::all_of(shape.points.begin(), shape.points.end(), usablePoint))
        return 0;

    if (shape.kind == Shape::Kind::Polygon) {
        if (shape.points.size() < 3)
            return 0;
        return applyToTiles(shapeBounds(shape.points, 0.0), op, [&shape](cv::Mat_<uint8_t>& mask, cv::Point origin) {
            std::vector<cv::Point> pts;
            pts.reserve(shape.points.size());
            for (const auto& p : shape.points)
                pts.push_back(toFixed(p, origin));
            const cv::Point* data = pts.data();
            const int count = int(pts.size());
            cv::fillPoly(mask, &data, &count, 1, cv::Scalar(255), cv::LINE_8, kFixedShift);
        });
    }

    const double radius = stampRadius(shape.radius);
    if (shape.kind == Shape::Kind::Disc) {
        const cv::Point2d c = shape.points.front();
        return applyToTiles(shapeBounds({c}, radius), op, [c, radius](cv::Mat_<uint8_t>& mask, cv::Point origin) {
            cv::circle(mask, toFixed(c, origin), int(std::lround(radius * kFixedScale)), cv::Scalar(255), cv::FILLED, cv::LINE_8,
                       kFixedShift);
            // Every stamp paints at least the pixel under its center.
            setPixelIfInside(mask, c, origin);
        });
    }

    // Capsule: a thick line with round caps, so a fast drag leaves no gaps.
    const cv::Point2d a = shape.points.front();
    const cv::Point2d b = shape.points.size() > 1 ? shape.points[1] : a;
    const int thickness = std::max(1, int(std::lround(2.0 * radius)));
    return applyToTiles(shapeBounds({a, b}, radius), op, [a, b, radius, thickness](cv::Mat_<uint8_t>& mask, cv::Point origin) {
        cv::line(mask, toFixed(a, origin), toFixed(b, origin), cv::Scalar(255), thickness, cv::LINE_8, kFixedShift);
        // Thin capsules: OpenCV's 1 px lines need the end discs to match the
        // stamps they connect.
        const int fixedRadius = int(std::lround(radius * kFixedScale));
        cv::circle(mask, toFixed(a, origin), fixedRadius, cv::Scalar(255), cv::FILLED, cv::LINE_8, kFixedShift);
        cv::circle(mask, toFixed(b, origin), fixedRadius, cv::Scalar(255), cv::FILLED, cv::LINE_8, kFixedShift);
        setPixelIfInside(mask, a, origin);
        setPixelIfInside(mask, b, origin);
    });
}

CanvasOverlap canvasOverlap(const std::vector<Shape>& shapes, cv::Size size)
{
    double x0 = std::numeric_limits<double>::infinity(), y0 = x0;
    double x1 = -x0, y1 = -x0;
    for (const Shape& shape : shapes) {
        const double r = shape.kind == Shape::Kind::Polygon ? 0.0 : stampRadius(shape.radius);
        for (const auto& p : shape.points) {
            if (!finitePoint(p))
                continue;
            x0 = std::min(x0, p.x - r);
            x1 = std::max(x1, p.x + r);
            y0 = std::min(y0, p.y - r);
            y1 = std::max(y1, p.y + r);
        }
    }
    if (!(x0 <= x1) || x1 <= 0.0 || y1 <= 0.0 || x0 >= size.width || y0 >= size.height)
        return CanvasOverlap::Outside;
    if (x0 < 0.0 || y0 < 0.0 || x1 > size.width || y1 > size.height)
        return CanvasOverlap::Clipped;
    return CanvasOverlap::Inside;
}

uint64_t EditTxn::applyMask(cv::Point origin, const cv::Mat_<uint8_t>& mask, PaintOp op)
{
    if (_finished || mask.empty() || (op.set == 0 && op.clear == 0))
        return 0;
    const cv::Rect bounds = cv::Rect(origin, mask.size()) & cv::Rect(0, 0, _raster._size.width, _raster._size.height);
    if (bounds.empty())
        return 0;
    uint64_t changed = 0;
    const int tx0 = bounds.x / kTileSize, tx1 = (bounds.br().x - 1) / kTileSize;
    const int ty0 = bounds.y / kTileSize, ty1 = (bounds.br().y - 1) / kTileSize;
    for (int ty = ty0; ty <= ty1; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx) {
            const int index = _raster.tileIndex(tx, ty);
            const cv::Point tileOrigin(tx * kTileSize, ty * kTileSize);
            const cv::Rect local = localRect(bounds, cv::Rect(tileOrigin, cv::Size(kTileSize, kTileSize)));
            if (local.empty())
                continue;
            const MaskView view{mask.ptr(0), std::size_t(mask.step[0]), tileOrigin - origin};
            changed += applyMaskedTile(index, view, local, op);
        }
    _changedPixels += changed;
    return changed;
}

cv::Rect EditTxn::takeDirtyRect()
{
    const cv::Rect r = _dirty;
    _dirty = {};
    return r;
}

HistoryEntry EditTxn::commit(uint64_t serial)
{
    HistoryEntry entry;
    entry.serial = serial;
    entry.label = _label;
    entry.actor = _actor;
    if (_finished)
        return entry;
    for (const int index : _order) {
        Working& w = _working[index];
        const cv::Rect bounds = _raster.tileRect(index);
        const cv::Rect inCanvas(bounds.x % kTileSize, bounds.y % kTileSize, bounds.width, bounds.height);
        finalizeTile(*w.tile, inCanvas);
        TileState after = publishTile(w.tile, inCanvas);
        if (!after.dense && !w.before.dense && after.uniform == w.before.uniform) {
            _raster.setTile(index, w.before);
            continue;
        }
        entry.bytes += std::size_t(kTilePixels) * ((w.before.dense ? 1 : 0) + (after.dense ? 1 : 0)) + 64;
        entry.changes.push_back({index, w.before, after});
        _raster.setTile(index, std::move(after));
    }
    entry.changedPixels = _changedPixels;
    _working.clear();
    _order.clear();
    _finished = true;
    return entry;
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

void History::push(HistoryEntry entry)
{
    for (const auto& e : _redo)
        _bytes -= e.bytes;
    _redo.clear();
    _bytes += entry.bytes;
    _undo.push_back(std::move(entry));
    while (_bytes > _budget && _undo.size() > 1) {
        _baseSerial = _undo.front().serial;
        _bytes -= _undo.front().bytes;
        _undo.pop_front();
    }
}

bool History::undo(TileRaster& raster)
{
    if (_undo.empty())
        return false;
    HistoryEntry entry = std::move(_undo.back());
    _undo.pop_back();
    for (auto it = entry.changes.rbegin(); it != entry.changes.rend(); ++it)
        raster.setTile(it->index, it->before);
    _redo.push_back(std::move(entry));
    return true;
}

bool History::redo(TileRaster& raster)
{
    if (_redo.empty())
        return false;
    HistoryEntry entry = std::move(_redo.back());
    _redo.pop_back();
    for (const auto& change : entry.changes)
        raster.setTile(change.index, change.after);
    _undo.push_back(std::move(entry));
    return true;
}

void History::reset(uint64_t serial)
{
    _undo.clear();
    _redo.clear();
    _bytes = 0;
    _baseSerial = serial;
    _cleanSerial = serial;
    _nextSerial = serial + 1;
}

}  // namespace vc3d::ink_labels
