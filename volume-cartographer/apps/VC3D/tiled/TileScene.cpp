#include "TileScene.hpp"

#include <algorithm>
#include <cmath>

TileScene::TileScene(QGraphicsScene* scene)
    : _scene(scene)
{
}

void TileScene::rebuildGrid(const ContentBounds& bounds, int viewportW, int viewportH)
{
    // Skip rebuild if bounds haven't changed and padding is compatible
    if (bounds == _bounds && !_items.empty()) {
        // Just update scene rect padding if viewport changed
        float contentPxW = static_cast<float>(bounds.totalCols) * TILE_PX;
        float contentPxH = static_cast<float>(bounds.totalRows) * TILE_PX;
        float sceneW = std::max(contentPxW, static_cast<float>(viewportW));
        float sceneH = std::max(contentPxH, static_cast<float>(viewportH));
        float newPadX = (sceneW - contentPxW) * 0.5f;
        float newPadY = (sceneH - contentPxH) * 0.5f;
        if (std::abs(newPadX - _padX) < 1.0f && std::abs(newPadY - _padY) < 1.0f) {
            return;  // Nothing changed
        }
        // Padding changed (viewport resized) — reposition items without recreating
        _padX = newPadX;
        _padY = newPadY;
        _scene->setSceneRect(0, 0, sceneW, sceneH);
        for (int r = 0; r < _bounds.totalRows; ++r) {
            for (int c = 0; c < _bounds.totalCols; ++c) {
                if (auto* item = itemAt(c, r)) {
                    item->setPos(_padX + c * TILE_PX, _padY + r * TILE_PX);
                }
            }
        }
        return;
    }

    // Remove old items from scene
    for (auto* item : _items) {
        _scene->removeItem(item);
        delete item;
    }
    _items.clear();
    _meta.clear();

    _bounds = bounds;

    if (_bounds.totalCols <= 0 || _bounds.totalRows <= 0) {
        _padX = 0;
        _padY = 0;
        _scene->setSceneRect(0, 0, viewportW, viewportH);
        return;
    }

    const int contentPxW = _bounds.totalCols * TILE_PX;
    const int contentPxH = _bounds.totalRows * TILE_PX;
    const int sceneW = std::max(contentPxW, viewportW);
    const int sceneH = std::max(contentPxH, viewportH);

    _padX = static_cast<float>(sceneW - contentPxW) / 2.0f;
    _padY = static_cast<float>(sceneH - contentPxH) / 2.0f;

    _scene->setSceneRect(0, 0, sceneW, sceneH);

    // Create placeholder pixmap
    QPixmap placeholder(TILE_PX, TILE_PX);
    placeholder.fill(QColor(64, 64, 64));

    const int count = _bounds.totalRows * _bounds.totalCols;
    _items.resize(count, nullptr);
    _meta.resize(count);

    for (int r = 0; r < _bounds.totalRows; ++r) {
        for (int c = 0; c < _bounds.totalCols; ++c) {
            auto* item = _scene->addPixmap(placeholder);
            item->setPos(_padX + c * TILE_PX, _padY + r * TILE_PX);
            item->setZValue(0);
            _items[r * _bounds.totalCols + c] = item;
        }
    }
}

bool TileScene::setTile(const TileKey& key, const QPixmap& pixmap,
                         uint64_t epoch, int8_t level)
{
    if (key.col < 0 || key.col >= _bounds.totalCols ||
        key.row < 0 || key.row >= _bounds.totalRows) {
        return false;
    }

    const int idx = key.row * _bounds.totalCols + key.col;
    auto& m = _meta[idx];

    // Reject stale renders (from an older epoch)
    if (epoch < m.epoch) return false;

    // Same epoch: reject if we already have equal or better (finer) data.
    // Lower level number = finer resolution = better.
    if (epoch == m.epoch && m.level >= 0 && level >= m.level) return false;

    m.epoch = epoch;
    m.level = level;
    _items[idx]->setPixmap(pixmap);
    return true;
}

bool TileScene::setTileWorld(const WorldTileKey& wk, const QPixmap& pixmap,
                              uint64_t epoch, int8_t level)
{
    int col, row;
    if (!_bounds.gridPosition(wk, col, row)) {
        return false;
    }
    return setTile(TileKey{col, row}, pixmap, epoch, level);
}

void TileScene::resetMetadata()
{
    for (auto& m : _meta) {
        m.epoch = 0;
        m.level = -1;
    }
}

void TileScene::clearAll()
{
    QPixmap placeholder(TILE_PX, TILE_PX);
    placeholder.fill(QColor(64, 64, 64));

    for (auto* item : _items) {
        if (item) {
            item->setPixmap(placeholder);
        }
    }
    resetMetadata();
}

void TileScene::sceneCleared()
{
    // The scene already deleted all items — just forget the pointers.
    _items.clear();
    _meta.clear();
    _bounds = ContentBounds{};
    _padX = 0;
    _padY = 0;
}

QPointF TileScene::surfaceToScene(float surfX, float surfY) const
{
    if (_bounds.worldTileSize <= 0) {
        return {_padX, _padY};
    }
    const float gridColFrac = surfX / _bounds.worldTileSize - _bounds.firstWorldCol;
    const float gridRowFrac = surfY / _bounds.worldTileSize - _bounds.firstWorldRow;
    return {gridColFrac * TILE_PX + _padX, gridRowFrac * TILE_PX + _padY};
}

cv::Vec2f TileScene::sceneToSurface(const QPointF& scenePos) const
{
    if (_bounds.worldTileSize <= 0) {
        return {0, 0};
    }
    const float gridColFrac = static_cast<float>(scenePos.x() - _padX) / TILE_PX;
    const float gridRowFrac = static_cast<float>(scenePos.y() - _padY) / TILE_PX;
    const float surfX = (gridColFrac + _bounds.firstWorldCol) * _bounds.worldTileSize;
    const float surfY = (gridRowFrac + _bounds.firstWorldRow) * _bounds.worldTileSize;
    return {surfX, surfY};
}

void TileScene::visibleGridRange(const QRectF& viewportSceneRect, int buffer,
                                  int& firstCol, int& firstRow,
                                  int& lastCol, int& lastRow) const
{
    if (_bounds.totalCols <= 0 || _bounds.totalRows <= 0) {
        firstCol = 0;
        firstRow = 0;
        lastCol = 0;
        lastRow = 0;
        return;
    }

    firstCol = static_cast<int>(std::floor((viewportSceneRect.left() - _padX) / TILE_PX)) - buffer;
    firstRow = static_cast<int>(std::floor((viewportSceneRect.top() - _padY) / TILE_PX)) - buffer;
    lastCol = static_cast<int>(std::floor((viewportSceneRect.right() - _padX) / TILE_PX)) + buffer;
    lastRow = static_cast<int>(std::floor((viewportSceneRect.bottom() - _padY) / TILE_PX)) + buffer;

    firstCol = std::max(0, firstCol);
    firstRow = std::max(0, firstRow);
    lastCol = std::min(_bounds.totalCols - 1, lastCol);
    lastRow = std::min(_bounds.totalRows - 1, lastRow);
}

std::vector<WorldTileKey> TileScene::visibleTiles(const QRectF& viewportSceneRect,
                                                    int buffer) const
{
    int firstCol, firstRow, lastCol, lastRow;
    visibleGridRange(viewportSceneRect, buffer, firstCol, firstRow, lastCol, lastRow);

    std::vector<WorldTileKey> result;
    if (firstCol > lastCol || firstRow > lastRow) return result;
    result.reserve(static_cast<size_t>(lastCol - firstCol + 1) * (lastRow - firstRow + 1));
    for (int r = firstRow; r <= lastRow; ++r) {
        for (int c = firstCol; c <= lastCol; ++c) {
            result.push_back(_bounds.worldKeyAt(c, r));
        }
    }
    return result;
}

std::vector<WorldTileKey> TileScene::staleTilesInRect(int desiredLevel, uint64_t epoch,
                                                        const QRectF& viewportSceneRect,
                                                        int buffer) const
{
    int firstCol, firstRow, lastCol, lastRow;
    visibleGridRange(viewportSceneRect, buffer, firstCol, firstRow, lastCol, lastRow);

    std::vector<WorldTileKey> result;
    if (firstCol > lastCol || firstRow > lastRow) return result;
    result.reserve(static_cast<size_t>(lastCol - firstCol + 1) * (lastRow - firstRow + 1));
    for (int r = firstRow; r <= lastRow; ++r) {
        for (int c = firstCol; c <= lastCol; ++c) {
            size_t idx = static_cast<size_t>(r) * _bounds.totalCols + c;
            if (idx >= _meta.size()) continue;
            const auto& m = _meta[idx];
            // A tile is stale if it has a worse-than-desired level (regardless
            // of epoch), OR if it was rendered at the correct level but with
            // an older epoch (meaning partial data that a new chunk may improve).
            if (m.level < 0 || m.level > desiredLevel || m.epoch < epoch) {
                result.push_back(_bounds.worldKeyAt(c, r));
            }
        }
    }
    return result;
}

QGraphicsPixmapItem* TileScene::itemAt(int col, int row) const
{
    if (col < 0 || col >= _bounds.totalCols || row < 0 || row >= _bounds.totalRows) {
        return nullptr;
    }
    return _items[row * _bounds.totalCols + col];
}
