#include "overlays/InkLabelOverlayController.hpp"

#include "ink_labels/InkLabelAssist.hpp"
#include "ink_labels/InkLabelController.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include <QGraphicsView>

#include <algorithm>
#include <cmath>

using namespace vc3d::ink_labels;

namespace {

constexpr qreal kPreviewZ = 20.0;
constexpr qreal kSuggestionZ = 19.5;
// Diagonal stripes this many label pixels wide.
constexpr int kHatchWidth = 3;

int ceilDiv(int a, int b)
{
    return (a + b - 1) / b;
}

}  // namespace

InkLabelOverlayController::InkLabelOverlayController(const InkLabelController& labels, QObject* parent)
    : ViewerOverlayControllerBase("ink_labels", parent)
    , _labels(labels)
    , _colorTable(colorTableFor(true, true, true, 0.6))
{
}

int InkLabelOverlayController::lodFor(double devicePxPerLabelPx)
{
    if (!std::isfinite(devicePxPerLabelPx) || devicePxPerLabelPx <= 0.0)
        return 0;
    return std::clamp(int(std::floor(std::log2(1.0 / devicePxPerLabelPx))), 0, kLevels - 1);
}

QList<QRgb> InkLabelOverlayController::colorTableFor(bool ink, bool supervision, bool validation, double opacity)
{
    const auto alpha = [opacity](int a) { return std::clamp(int(std::lround(a * opacity)), 0, 255); };
    QList<QRgb> table(16, qRgba(0, 0, 0, 0));
    for (int code = 0; code < 16; ++code) {
        if ((code & 1) && ink)
            table[code] = qRgba(255, 64, 64, alpha(255));
        else if ((code & 4) && validation)
            table[code] = qRgba(255, 200, 0, alpha(160));
        else if ((code & 2) && supervision)
            table[code] = qRgba(64, 128, 255, alpha(120));
        else if ((code & 8) && supervision)
            table[code] = qRgba(64, 128, 255, alpha(50));
    }
    return table;
}

quint64 InkLabelOverlayController::keyFor(int level, int i, int j)
{
    return (quint64(level) << 56) | (quint64(uint32_t(i)) << 28) | quint64(uint32_t(j));
}

uint32_t InkLabelOverlayController::revision(int level, int i, int j) const
{
    const cv::Size grid = _levelGrid[std::size_t(level)];
    if (i < 0 || j < 0 || i >= grid.width || j >= grid.height)
        return 0;
    return _revisions[std::size_t(level)][std::size_t(j * grid.width + i)];
}

int InkLabelOverlayController::cacheCapacityFor(int visibleTiles)
{
    return std::max(kMinCachedTiles, 2 * std::max(0, visibleTiles));
}

int InkLabelOverlayController::cacheCapacity() const
{
    int visible = 0;
    for (const int n : _visibleTiles)
        visible += n;
    return cacheCapacityFor(visible);
}

void InkLabelOverlayController::resetForSession()
{
    _cache.clear();
    _lru.clear();
    _visibleTiles.clear();
    const TileRaster* raster = _labels.raster();
    const cv::Size size = raster ? raster->size() : cv::Size();
    for (int level = 0; level < kLevels; ++level) {
        const int span = kTileSize << level;
        const cv::Size grid(ceilDiv(size.width, span), ceilDiv(size.height, span));
        _levelGrid[std::size_t(level)] = grid;
        _revisions[std::size_t(level)].assign(std::size_t(grid.area()), 0);
    }
    _polygon.clear();
}

void InkLabelOverlayController::invalidate(const QRect& labelRect)
{
    if (labelRect.isEmpty())
        return;
    for (int level = 0; level < kLevels; ++level) {
        const cv::Size grid = _levelGrid[std::size_t(level)];
        if (grid.area() == 0)
            continue;
        const int span = kTileSize << level;
        const int i0 = std::clamp(labelRect.left() / span, 0, grid.width - 1);
        const int i1 = std::clamp(labelRect.right() / span, 0, grid.width - 1);
        const int j0 = std::clamp(labelRect.top() / span, 0, grid.height - 1);
        const int j1 = std::clamp(labelRect.bottom() / span, 0, grid.height - 1);
        auto& revs = _revisions[std::size_t(level)];
        for (int j = j0; j <= j1; ++j)
            for (int i = i0; i <= i1; ++i)
                ++revs[std::size_t(j * grid.width + i)];
    }
}

void InkLabelOverlayController::setStyle(bool ink, bool supervision, bool validation, double opacity)
{
    const QList<QRgb> table = colorTableFor(ink, supervision, validation, opacity);
    if (table == _colorTable)
        return;
    _colorTable = table;
    // Codes are unchanged: swap the palette and convert pixmaps again lazily.
    for (auto& entry : _cache) {
        entry.image.setColorTable(_colorTable);
        entry.pixmap = QPixmap();
    }
}

void InkLabelOverlayController::setPolygonPreview(std::vector<cv::Point2d> labelPx, QColor color)
{
    if (labelPx.empty() && _polygon.empty())
        return;
    _polygon = std::move(labelPx);
    _polygonColor = color;
    refreshAll();
}

QList<QRgb> InkLabelOverlayController::suggestionColorTable()
{
    QList<QRgb> table(5, qRgba(0, 0, 0, 0));
    table[1] = qRgba(0, 220, 255, 170);
    table[2] = qRgba(0, 220, 255, 45);
    table[3] = qRgba(255, 64, 64, 220);
    table[4] = qRgba(255, 64, 64, 110);
    return table;
}

void InkLabelOverlayController::setSuggestionPreview(const SuggestionSet* suggestions, const std::set<int>& selected)
{
    if (!suggestions || suggestions->labels.empty()) {
        if (_suggestionImage.isNull())
            return;
        _suggestionImage = QImage();
        _suggestionPixmap = QPixmap();
        refreshAll();
        return;
    }
    const cv::Mat_<int>& labels = suggestions->labels;
    std::vector<uint8_t> picked(suggestions->items.size() + 1, 0);
    for (const int id : selected)
        if (id > 0 && std::size_t(id) < picked.size())
            picked[std::size_t(id)] = 1;
    QImage image(labels.cols, labels.rows, QImage::Format_Indexed8);
    image.setColorTable(suggestionColorTable());
    for (int y = 0; y < labels.rows; ++y) {
        const int* src = labels[y];
        uchar* dst = image.scanLine(y);
        const int gy = suggestions->origin.y + y;
        for (int x = 0; x < labels.cols; ++x) {
            const int id = src[x];
            if (id <= 0 || std::size_t(id) >= picked.size()) {
                dst[x] = 0;
                continue;
            }
            const bool stripe = (((suggestions->origin.x + x + gy) / kHatchWidth) & 1) == 0;
            dst[x] = uchar((picked[std::size_t(id)] ? 3 : 1) + (stripe ? 0 : 1));
        }
    }
    _suggestionImage = std::move(image);
    _suggestionPixmap = QPixmap();
    _suggestionOrigin = suggestions->origin;
    refreshAll();
}

QImage InkLabelOverlayController::buildTile(int level, int i, int j) const
{
    const TileRaster* raster = _labels.raster();
    if (!raster)
        return {};
    const cv::Size size = raster->size();
    const int span = kTileSize << level;
    const int step = 1 << level;
    const int x0 = i * span, y0 = j * span;
    const int coverW = std::min(span, size.width - x0);
    const int coverH = std::min(span, size.height - y0);
    if (coverW <= 0 || coverH <= 0)
        return {};
    QImage image(ceilDiv(coverW, step), ceilDiv(coverH, step), QImage::Format_Indexed8);
    image.setColorTable(_colorTable);
    image.fill(kNeutralCode);

    const int tx0 = x0 / kTileSize, tx1 = (x0 + coverW - 1) / kTileSize;
    const int ty0 = y0 / kTileSize, ty1 = (y0 + coverH - 1) / kTileSize;
    for (int ty = ty0; ty <= ty1; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx) {
            const int index = raster->tileIndex(tx, ty);
            const cv::Rect rect = raster->tileRect(index);
            // Image pixels never straddle tiles: 2^level divides 256.
            const int ix0 = (rect.x - x0) / step, ix1 = ceilDiv(rect.br().x - x0, step);
            const int iy0 = (rect.y - y0) / step, iy1 = ceilDiv(rect.br().y - y0, step);
            const TileState& state = raster->tile(index);
            if (!state.dense) {
                const uint8_t code = summaryCode(state.uniform);
                for (int y = iy0; y < iy1; ++y)
                    std::fill_n(image.scanLine(y) + ix0, ix1 - ix0, code);
                continue;
            }
            const Tile& tile = *state.dense;
            const int localX = rect.x - tx * kTileSize, localY = rect.y - ty * kTileSize;
            if (level == 0) {
                for (int y = 0; y < rect.height; ++y) {
                    const uint8_t* src = tile.px.data() + std::size_t((localY + y) * kTileSize + localX);
                    uchar* dst = image.scanLine(iy0 + y) + ix0;
                    for (int x = 0; x < rect.width; ++x)
                        dst[x] = summaryCode(src[x]);
                }
            } else if (level < 3) {
                for (int iy = iy0; iy < iy1; ++iy) {
                    uchar* dst = image.scanLine(iy);
                    const int py0 = y0 + iy * step - ty * kTileSize;
                    const int py1 = std::min(py0 + step, localY + rect.height);
                    for (int ix = ix0; ix < ix1; ++ix) {
                        const int px0 = x0 + ix * step - tx * kTileSize;
                        const int px1 = std::min(px0 + step, localX + rect.width);
                        uint8_t any = 0, all = 0xFF;
                        for (int py = py0; py < py1; ++py) {
                            const uint8_t* row = tile.px.data() + std::size_t(py * kTileSize);
                            for (int px = px0; px < px1; ++px) {
                                const uint8_t c = summaryCode(row[px]);
                                any |= c;
                                all &= c;
                            }
                        }
                        dst[ix] = uint8_t((any & 0b1101) | (all & 0b0010));
                    }
                }
            } else {
                const int cellsPerPixel = step / kSummaryCell;
                for (int iy = iy0; iy < iy1; ++iy) {
                    uchar* dst = image.scanLine(iy);
                    const int cy0 = (y0 + iy * step - ty * kTileSize) / kSummaryCell;
                    for (int ix = ix0; ix < ix1; ++ix) {
                        const int cx0 = (x0 + ix * step - tx * kTileSize) / kSummaryCell;
                        uint8_t code = kNeutralCode;
                        for (int cy = cy0; cy < std::min(cy0 + cellsPerPixel, kSummarySize); ++cy)
                            for (int cx = cx0; cx < std::min(cx0 + cellsPerPixel, kSummarySize); ++cx)
                                code = poolCodes(code, tile.summary[std::size_t(cy * kSummarySize + cx)]);
                        dst[ix] = code;
                    }
                }
            }
        }
    return image;
}

InkLabelOverlayController::CacheEntry& InkLabelOverlayController::ensureTile(int level, int i, int j)
{
    const quint64 key = keyFor(level, i, j);
    const uint32_t rev = revision(level, i, j);
    auto it = _cache.find(key);
    if (it != _cache.end()) {
        _lru.erase(it->lru);
        _lru.push_front(key);
        it->lru = _lru.begin();
        if (it->revision != rev || it->image.isNull()) {
            it->image = buildTile(level, i, j);
            it->pixmap = QPixmap();
            it->revision = rev;
            ++_builds;
        }
        return *it;
    }
    const int capacity = cacheCapacity();
    while (_cache.size() >= capacity && !_lru.empty()) {
        _cache.remove(_lru.back());
        _lru.pop_back();
    }
    _lru.push_front(key);
    CacheEntry entry;
    entry.revision = rev;
    entry.image = buildTile(level, i, j);
    entry.lru = _lru.begin();
    ++_builds;
    return *_cache.insert(key, std::move(entry));
}

bool InkLabelOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    if (!viewer || !_labels.viewerShowsSession(viewer))
        return false;
    return (_labels.showLabels() && _labels.raster()) || !_polygon.empty() || !_suggestionImage.isNull();
}

QPointF InkLabelOverlayController::snapToDevicePixel(const QPointF& scenePos, const QTransform& toDevice)
{
    bool invertible = false;
    const QTransform toScene = toDevice.inverted(&invertible);
    if (!invertible)
        return scenePos;
    const QPointF device = toDevice.map(scenePos);
    return toScene.map(QPointF(std::round(device.x()), std::round(device.y())));
}

void InkLabelOverlayController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    if (!isOverlayEnabledFor(viewer))
        return;
    const TileRaster* raster = _labels.raster();
    auto* view = viewer->graphicsView();
    if (!raster || !view)
        return;
    const cv::Size size = raster->size();
    const QTransform toDevice = view->viewportTransform();

    if (_labels.showLabels() && size.area() > 0) {
        // Visible label rectangle from the four viewport corners.
        const QRectF sceneRect = visibleSceneRect(viewer);
        double minX = 0, minY = 0, maxX = size.width, maxY = size.height;
        bool first = true;
        bool mapped = true;
        for (const QPointF& corner : {sceneRect.topLeft(), sceneRect.topRight(), sceneRect.bottomLeft(), sceneRect.bottomRight()}) {
            const auto label = _labels.sceneToLabel(viewer, corner);
            if (!label) {
                mapped = false;
                break;
            }
            minX = first ? label->x : std::min(minX, label->x);
            maxX = first ? label->x : std::max(maxX, label->x);
            minY = first ? label->y : std::min(minY, label->y);
            maxY = first ? label->y : std::max(maxY, label->y);
            first = false;
        }
        if (!mapped) {
            minX = minY = 0;
            maxX = size.width;
            maxY = size.height;
        }

        // Level of detail from the zoom at the view center.
        const cv::Point2d center((minX + maxX) * 0.5, (minY + maxY) * 0.5);
        double d = 1.0;
        const auto a = _labels.labelToScene(viewer, center);
        const auto b = _labels.labelToScene(viewer, center + cv::Point2d(1.0, 0.0));
        if (a && b) {
            const QPointF delta = toDevice.map(*b) - toDevice.map(*a);
            d = std::hypot(delta.x(), delta.y());
        }
        const int level = lodFor(d);
        const int span = kTileSize << level;
        const int step = 1 << level;
        const cv::Size grid = _levelGrid[std::size_t(level)];
        const auto clampIndex = [](double v, int count) { return std::clamp(int(std::floor(v)), 0, std::max(0, count - 1)); };
        const int i0 = clampIndex(minX / span - 1, grid.width), i1 = clampIndex(maxX / span + 1, grid.width);
        const int j0 = clampIndex(minY / span - 1, grid.height), j1 = clampIndex(maxY / span + 1, grid.height);
        _visibleTiles[viewer] = grid.area() > 0 ? (i1 - i0 + 1) * (j1 - j0 + 1) : 0;
        for (int j = j0; j <= j1 && grid.area() > 0; ++j)
            for (int i = i0; i <= i1; ++i) {
                CacheEntry& entry = ensureTile(level, i, j);
                if (entry.image.isNull())
                    continue;
                const cv::Point2d corner(double(i * span), double(j * span));
                const auto origin = _labels.labelToScene(viewer, corner);
                const auto alongX = _labels.labelToScene(viewer, corner + cv::Point2d(span, 0));
                const auto alongY = _labels.labelToScene(viewer, corner + cv::Point2d(0, span));
                if (!origin || !alongX || !alongY)
                    continue;
                const double pixelsPerSpan = double(span) / step;
                QPointF o = snapToDevicePixel(*origin, toDevice);
                QPointF ax = snapToDevicePixel(*alongX, toDevice);
                QPointF ay = snapToDevicePixel(*alongY, toDevice);
                if (ax.x() == o.x() || ay.y() == o.y()) {
                    // A tile under one device pixel: keep the exact mapping.
                    o = *origin;
                    ax = *alongX;
                    ay = *alongY;
                }
                const qreal scaleX = (ax.x() - o.x()) / pixelsPerSpan;
                const qreal scaleY = (ay.y() - o.y()) / pixelsPerSpan;
                if (entry.pixmap.isNull())
                    entry.pixmap = QPixmap::fromImage(entry.image);
                builder.addImage(entry.image, entry.pixmap, o, scaleX, scaleY, 1.0, kZ);
            }
    }

    if (!_suggestionImage.isNull()) {
        const cv::Point2d corner(_suggestionOrigin.x, _suggestionOrigin.y);
        const auto origin = _labels.labelToScene(viewer, corner);
        const auto alongX = _labels.labelToScene(viewer, corner + cv::Point2d(_suggestionImage.width(), 0));
        const auto alongY = _labels.labelToScene(viewer, corner + cv::Point2d(0, _suggestionImage.height()));
        if (origin && alongX && alongY) {
            if (_suggestionPixmap.isNull())
                _suggestionPixmap = QPixmap::fromImage(_suggestionImage);
            builder.addImage(_suggestionImage, _suggestionPixmap, *origin, (alongX->x() - origin->x()) / _suggestionImage.width(),
                             (alongY->y() - origin->y()) / _suggestionImage.height(), 1.0, kSuggestionZ);
        }
    }

    if (_polygon.size() >= 1) {
        std::vector<QPointF> points;
        points.reserve(_polygon.size());
        for (const auto& p : _polygon)
            if (const auto scene = _labels.labelToScene(viewer, p))
                points.push_back(*scene);
        OverlayStyle line;
        line.penColor = _polygonColor;
        line.penWidth = 2.0;
        line.z = kPreviewZ;
        if (points.size() >= 2)
            builder.addLineStrip(points, false, line);
        OverlayStyle dot = line;
        dot.brushColor = _polygonColor;
        for (const auto& p : points)
            builder.addCircle(p, 3.0, true, dot);
    }
}

#ifdef VC_TESTING
QImage InkLabelOverlayController::displayTileForTesting(int level, int i, int j)
{
    return ensureTile(level, i, j).image;
}
#endif
