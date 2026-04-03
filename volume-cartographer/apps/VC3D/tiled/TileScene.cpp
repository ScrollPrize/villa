#include "TileScene.hpp"

#include <algorithm>

TileScene::TileScene(QGraphicsScene* scene)
    : _scene(scene)
{
}

void TileScene::rebuildGrid(const ContentBounds& bounds, int viewportW, int viewportH)
{
    // Save old bounds before grid rebuild overwrites them
    ContentBounds oldBounds = _grid.bounds();

    bool gridRebuilt = _grid.rebuildGrid(bounds, viewportW, viewportH);

    const auto& b = _grid.bounds();
    float padX = _grid.padX();
    float padY = _grid.padY();

    if (!gridRebuilt && !_items.empty()) {
        // Structure didn't change -- just reposition for updated padding
        float contentPxW = static_cast<float>(b.totalCols) * TILE_PX;
        float contentPxH = static_cast<float>(b.totalRows) * TILE_PX;
        float sceneW = std::max(contentPxW, static_cast<float>(viewportW));
        float sceneH = std::max(contentPxH, static_cast<float>(viewportH));
        _scene->setSceneRect(0, 0, static_cast<qreal>(sceneW), static_cast<qreal>(sceneH));
        for (int r = 0; r < b.totalRows; ++r) {
            for (int c = 0; c < b.totalCols; ++c) {
                if (auto* item = itemAt(c, r)) {
                    item->setPos(static_cast<qreal>(padX + static_cast<float>(c) * TILE_PX),
                                 static_cast<qreal>(padY + static_cast<float>(r) * TILE_PX));
                }
            }
        }
        return;
    }

    // Save old items + their world keys for pixmap carry-over
    auto oldItems = std::move(_items);
    _items.clear();
    clearRetained();

    if (b.totalCols <= 0 || b.totalRows <= 0) {
        for (auto* item : oldItems) {
            if (item && item->scene()) item->scene()->removeItem(item);
            delete item;
        }
        _scene->setSceneRect(0, 0, viewportW, viewportH);
        return;
    }

    const float contentPxW = static_cast<float>(b.totalCols) * TILE_PX;
    const float contentPxH = static_cast<float>(b.totalRows) * TILE_PX;
    const float sceneW = std::max(contentPxW, static_cast<float>(viewportW));
    const float sceneH = std::max(contentPxH, static_cast<float>(viewportH));
    _scene->setSceneRect(0, 0, static_cast<qreal>(sceneW), static_cast<qreal>(sceneH));

    // Build new grid, carrying over pixmaps from old items at matching world positions.
    // Never show empty content — reuse stale content until fresh renders arrive.
    const size_t count = static_cast<size_t>(b.totalRows) * static_cast<size_t>(b.totalCols);
    _items.resize(count, nullptr);

    for (int r = 0; r < b.totalRows; ++r) {
        for (int c = 0; c < b.totalCols; ++c) {
            WorldTileKey wk = b.worldKeyAt(c, r);

            // Check if this world position existed in the old grid
            int oldCol, oldRow;
            QPixmap carried;
            if (oldBounds.gridPosition(wk, oldCol, oldRow) &&
                !oldItems.empty()) {
                size_t oldIdx = static_cast<size_t>(oldRow) * static_cast<size_t>(oldBounds.totalCols) + static_cast<size_t>(oldCol);
                if (oldIdx < oldItems.size() && oldItems[oldIdx]) {
                    carried = oldItems[oldIdx]->pixmap();
                }
            }

            auto* item = _scene->addPixmap(carried.isNull() ? QPixmap() : carried);
            item->setPos(static_cast<qreal>(padX + static_cast<float>(c) * TILE_PX),
                         static_cast<qreal>(padY + static_cast<float>(r) * TILE_PX));
            item->setZValue(0);
            _items[static_cast<size_t>(r) * static_cast<size_t>(b.totalCols) + static_cast<size_t>(c)] = item;
        }
    }

    // Delete old items
    for (auto* item : oldItems) {
        if (item && item->scene()) item->scene()->removeItem(item);
        delete item;
    }
}

bool TileScene::setTile(const TileKey& key, const QPixmap& pixmap,
                         uint64_t epoch, int8_t level)
{
    const auto& b = _grid.bounds();
    if (key.col < 0 || key.col >= b.totalCols ||
        key.row < 0 || key.row >= b.totalRows) {
        return false;
    }

    // Convert QPixmap to pixel vector for TileGrid
    QImage img = pixmap.toImage().convertToFormat(QImage::Format_ARGB32);
    std::vector<uint32_t> pixels(static_cast<size_t>(img.width()) * static_cast<size_t>(img.height()));
    for (int y = 0; y < img.height(); ++y) {
        memcpy(pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(img.width()),
               img.scanLine(y),
               static_cast<size_t>(img.width()) * sizeof(uint32_t));
    }

    bool accepted = _grid.setTile(key, std::move(pixels), img.width(), img.height(), epoch, level);
    if (accepted) {
        const size_t idx = static_cast<size_t>(key.row) * static_cast<size_t>(b.totalCols) + static_cast<size_t>(key.col);
        _items[idx]->setPixmap(pixmap);
    }
    return accepted;
}

bool TileScene::setTileWorld(const WorldTileKey& wk, const QPixmap& pixmap,
                              uint64_t epoch, int8_t level)
{
    int col, row;
    if (!_grid.bounds().gridPosition(wk, col, row)) {
        return false;
    }
    return setTile(TileKey{col, row}, pixmap, epoch, level);
}

bool TileScene::setTilePixmapOnly(const WorldTileKey& wk, const QPixmap& pixmap)
{
    int col, row;
    if (!_grid.bounds().gridPosition(wk, col, row)) {
        return false;
    }
    const auto& b = _grid.bounds();
    if (col < 0 || col >= b.totalCols || row < 0 || row >= b.totalRows)
        return false;
    const size_t idx = static_cast<size_t>(row) * static_cast<size_t>(b.totalCols) + static_cast<size_t>(col);
    _items[idx]->setPixmap(pixmap);
    return true;
}

bool TileScene::tileNeedsContent(const WorldTileKey& wk) const
{
    return _grid.tileNeedsContent(wk);
}

void TileScene::resetMetadata()
{
    _grid.resetMetadata();
}

void TileScene::clearAll()
{
    clearRetained();

    QPixmap placeholder(TILE_PX, TILE_PX);
    placeholder.fill(QColor(64, 64, 64));

    for (auto* item : _items) {
        if (item) {
            item->setPixmap(placeholder);
        }
    }
    resetMetadata();
}

void TileScene::clearRetained()
{
    for (auto* item : _retainedItems) {
        _scene->removeItem(item);
        delete item;
    }
    _retainedItems.clear();
}

void TileScene::sceneCleared()
{
    // The scene already deleted all items -- just forget the pointers.
    _items.clear();
    _retainedItems.clear();
    _grid.clear();
}

QPointF TileScene::surfaceToScene(float surfX, float surfY) const
{
    auto v = _grid.surfaceToGrid(surfX, surfY);
    return {static_cast<qreal>(v[0]), static_cast<qreal>(v[1])};
}

cv::Vec2f TileScene::sceneToSurface(const QPointF& scenePos) const
{
    return _grid.gridToSurface(static_cast<float>(scenePos.x()),
                               static_cast<float>(scenePos.y()));
}

std::vector<WorldTileKey> TileScene::visibleTiles(const QRectF& viewportSceneRect,
                                                    int buffer) const
{
    return _grid.visibleTiles(
        static_cast<float>(viewportSceneRect.left()),
        static_cast<float>(viewportSceneRect.top()),
        static_cast<float>(viewportSceneRect.right()),
        static_cast<float>(viewportSceneRect.bottom()),
        buffer);
}

int TileScene::worstVisibleLevel(const QRectF& viewportSceneRect) const
{
    return _grid.worstVisibleLevel(
        static_cast<float>(viewportSceneRect.left()),
        static_cast<float>(viewportSceneRect.top()),
        static_cast<float>(viewportSceneRect.right()),
        static_cast<float>(viewportSceneRect.bottom()));
}

std::vector<WorldTileKey> TileScene::staleTilesInRect(int desiredLevel, uint64_t epoch,
                                                        const QRectF& viewportSceneRect,
                                                        int buffer) const
{
    return _grid.staleTilesInRect(
        desiredLevel, epoch,
        static_cast<float>(viewportSceneRect.left()),
        static_cast<float>(viewportSceneRect.top()),
        static_cast<float>(viewportSceneRect.right()),
        static_cast<float>(viewportSceneRect.bottom()),
        buffer);
}

QGraphicsPixmapItem* TileScene::itemAt(int col, int row) const
{
    const auto& b = _grid.bounds();
    if (col < 0 || col >= b.totalCols || row < 0 || row >= b.totalRows) {
        return nullptr;
    }
    return _items[static_cast<size_t>(row) * static_cast<size_t>(b.totalCols) + static_cast<size_t>(col)];
}
