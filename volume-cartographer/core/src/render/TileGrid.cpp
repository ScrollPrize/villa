#include "vc/core/render/TileGrid.hpp"

#include <algorithm>
#include <cmath>

namespace vc::render {

bool TileGrid::rebuildGrid(const ContentBounds& bounds, int viewportW, int viewportH)
{
    // Only rebuild when grid STRUCTURE changes (tile count / world position).
    // Scale-only changes just update bounds and re-render tiles in place —
    // the old content stays visible until new renders replace it.
    bool structureSame = (bounds.firstWorldCol == _bounds.firstWorldCol &&
                          bounds.firstWorldRow == _bounds.firstWorldRow &&
                          bounds.totalCols == _bounds.totalCols &&
                          bounds.totalRows == _bounds.totalRows);

    if (structureSame && !_meta.empty()) {
        // Update scale/worldTileSize without destroying the grid
        _bounds = bounds;
        float contentPxW = static_cast<float>(bounds.totalCols) * TILE_PX;
        float contentPxH = static_cast<float>(bounds.totalRows) * TILE_PX;
        float sceneW = std::max(contentPxW, static_cast<float>(viewportW));
        float sceneH = std::max(contentPxH, static_cast<float>(viewportH));
        _padX = (sceneW - contentPxW) * 0.5f;
        _padY = (sceneH - contentPxH) * 0.5f;
        return false;
    }

    _meta.clear();
    _tiles.clear();
    _bounds = bounds;

    if (_bounds.totalCols <= 0 || _bounds.totalRows <= 0) {
        _padX = 0;
        _padY = 0;
        _unfilledCount = 0;
        return true;
    }

    const int contentPxW = _bounds.totalCols * TILE_PX;
    const int contentPxH = _bounds.totalRows * TILE_PX;
    const int sceneW = std::max(contentPxW, viewportW);
    const int sceneH = std::max(contentPxH, viewportH);

    _padX = static_cast<float>(sceneW - contentPxW) / 2.0f;
    _padY = static_cast<float>(sceneH - contentPxH) / 2.0f;

    const int count = _bounds.totalRows * _bounds.totalCols;
    _meta.resize(count);
    _tiles.resize(count);
    _unfilledCount = count;

    return true;
}

bool TileGrid::setTile(const TileKey& key, std::vector<uint32_t>&& pixels,
                        int width, int height, uint64_t epoch, int8_t level,
                        std::chrono::steady_clock::time_point submitTime,
                        std::chrono::steady_clock::time_point renderDone)
{
    if (key.col < 0 || key.col >= _bounds.totalCols ||
        key.row < 0 || key.row >= _bounds.totalRows) {
        return false;
    }

    const int idx = key.row * _bounds.totalCols + key.col;
    auto& m = _meta[idx];

    // Accept if newer epoch (any level -- new camera state wins).
    // Accept if same epoch and finer level (progressive refinement).
    // Also accept if slightly older epoch but MUCH finer level.
    if (epoch < m.epoch) {
        constexpr uint64_t kEpochGrace = 3;
        bool finerFromRecentEpoch = (m.epoch - epoch <= kEpochGrace) &&
                                     m.level >= 0 && level >= 0 && level < m.level;
        if (!finerFromRecentEpoch) return false;
    }
    if (epoch == m.epoch && m.level >= 0 && level >= m.level) return false;

    bool wasFilling = (m.level < 0);
    m.epoch = epoch;
    m.level = level;

    auto& td = _tiles[idx];
    td.pixels = std::move(pixels);
    td.width = width;
    td.height = height;
    td.version = _nextVersion++;
    td.submitTime = submitTime;
    td.renderDone = renderDone;

    if (wasFilling && level >= 0) {
        --_unfilledCount;
    }
    return true;
}

bool TileGrid::setTileWorld(const WorldTileKey& wk, std::vector<uint32_t>&& pixels,
                             int width, int height, uint64_t epoch, int8_t level,
                             std::chrono::steady_clock::time_point submitTime,
                             std::chrono::steady_clock::time_point renderDone)
{
    int col, row;
    if (!_bounds.gridPosition(wk, col, row)) {
        return false;
    }
    return setTile(TileKey{col, row}, std::move(pixels), width, height, epoch, level,
                   submitTime, renderDone);
}

bool TileGrid::setTileMeta(const WorldTileKey& wk, uint64_t epoch, int8_t level)
{
    int col, row;
    if (!_bounds.gridPosition(wk, col, row)) return false;
    const int idx = row * _bounds.totalCols + col;
    if (static_cast<size_t>(idx) >= _meta.size()) return false;

    auto& m = _meta[idx];

    // Same staleness check as setTile
    if (epoch < m.epoch) {
        constexpr uint64_t kEpochGrace = 3;
        bool finerFromRecentEpoch = (m.epoch - epoch <= kEpochGrace) &&
                                     m.level >= 0 && level >= 0 && level < m.level;
        if (!finerFromRecentEpoch) return false;
    }
    if (epoch == m.epoch && m.level >= 0 && level >= m.level) return false;

    bool wasFilling = (m.level < 0);
    m.epoch = epoch;
    m.level = level;

    if (wasFilling && level >= 0)
        --_unfilledCount;

    return true;
}

std::chrono::steady_clock::time_point TileGrid::tileSubmitTime(const WorldTileKey& wk) const
{
    int col, row;
    if (!_bounds.gridPosition(wk, col, row)) return {};
    return _tiles[row * _bounds.totalCols + col].submitTime;
}

std::chrono::steady_clock::time_point TileGrid::tileRenderDone(const WorldTileKey& wk) const
{
    int col, row;
    if (!_bounds.gridPosition(wk, col, row)) return {};
    return _tiles[row * _bounds.totalCols + col].renderDone;
}

bool TileGrid::tileNeedsContent(const WorldTileKey& wk) const
{
    int col, row;
    if (!_bounds.gridPosition(wk, col, row)) return false;
    const int idx = row * _bounds.totalCols + col;
    if (static_cast<size_t>(idx) >= _meta.size()) return false;
    return _meta[idx].level < 0;
}

void TileGrid::resetMetadata()
{
    for (auto& m : _meta) {
        m.epoch = 0;
        m.level = -1;
    }
    _unfilledCount = static_cast<int>(_meta.size());
}

void TileGrid::clear()
{
    _meta.clear();
    _tiles.clear();
    _unfilledCount = 0;
    _bounds = ContentBounds{};
    _padX = 0;
    _padY = 0;
}

cv::Vec2f TileGrid::surfaceToGrid(float surfX, float surfY) const
{
    if (_bounds.worldTileSize <= 0) {
        return {_padX, _padY};
    }
    const float gridColFrac = surfX / _bounds.worldTileSize - _bounds.firstWorldCol;
    const float gridRowFrac = surfY / _bounds.worldTileSize - _bounds.firstWorldRow;
    return {gridColFrac * TILE_PX + _padX, gridRowFrac * TILE_PX + _padY};
}

cv::Vec2f TileGrid::gridToSurface(float gridX, float gridY) const
{
    if (_bounds.worldTileSize <= 0) {
        return {0, 0};
    }
    const float gridColFrac = (gridX - _padX) / TILE_PX;
    const float gridRowFrac = (gridY - _padY) / TILE_PX;
    const float surfX = (gridColFrac + _bounds.firstWorldCol) * _bounds.worldTileSize;
    const float surfY = (gridRowFrac + _bounds.firstWorldRow) * _bounds.worldTileSize;
    return {surfX, surfY};
}

void TileGrid::visibleGridRange(float vpL, float vpT, float vpR, float vpB, int buffer,
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

    firstCol = static_cast<int>(std::floor((vpL - _padX) / TILE_PX)) - buffer;
    firstRow = static_cast<int>(std::floor((vpT - _padY) / TILE_PX)) - buffer;
    lastCol  = static_cast<int>(std::floor((vpR - _padX) / TILE_PX)) + buffer;
    lastRow  = static_cast<int>(std::floor((vpB - _padY) / TILE_PX)) + buffer;

    firstCol = std::max(0, firstCol);
    firstRow = std::max(0, firstRow);
    lastCol  = std::min(_bounds.totalCols - 1, lastCol);
    lastRow  = std::min(_bounds.totalRows - 1, lastRow);
}

std::vector<WorldTileKey> TileGrid::visibleTiles(float vpL, float vpT, float vpR, float vpB,
                                                  int buffer) const
{
    int firstCol, firstRow, lastCol, lastRow;
    visibleGridRange(vpL, vpT, vpR, vpB, buffer, firstCol, firstRow, lastCol, lastRow);

    std::vector<WorldTileKey> result;
    if (firstCol > lastCol || firstRow > lastRow) return result;
    result.reserve(static_cast<size_t>(lastCol - firstCol + 1) * (lastRow - firstRow + 1));
    for (int r = firstRow; r <= lastRow; ++r)
        for (int c = firstCol; c <= lastCol; ++c)
            result.push_back(_bounds.worldKeyAt(c, r));
    return result;
}

int TileGrid::worstVisibleLevel(float vpL, float vpT, float vpR, float vpB) const
{
    int firstCol, firstRow, lastCol, lastRow;
    visibleGridRange(vpL, vpT, vpR, vpB, 0, firstCol, firstRow, lastCol, lastRow);
    int worst = -1;
    for (int r = firstRow; r <= lastRow; ++r) {
        for (int c = firstCol; c <= lastCol; ++c) {
            size_t idx = static_cast<size_t>(r) * _bounds.totalCols + c;
            if (idx >= _meta.size()) continue;
            int8_t lvl = _meta[idx].level;
            if (lvl >= 0 && lvl > worst) worst = lvl;
        }
    }
    return worst;
}

std::vector<WorldTileKey> TileGrid::staleTilesInRect(int desiredLevel, uint64_t epoch,
                                                      float vpL, float vpT, float vpR, float vpB,
                                                      int buffer) const
{
    int firstCol, firstRow, lastCol, lastRow;
    visibleGridRange(vpL, vpT, vpR, vpB, buffer, firstCol, firstRow, lastCol, lastRow);

    std::vector<WorldTileKey> result;
    if (firstCol > lastCol || firstRow > lastRow) return result;
    result.reserve(static_cast<size_t>(lastCol - firstCol + 1) * (lastRow - firstRow + 1));
    for (int r = firstRow; r <= lastRow; ++r) {
        for (int c = firstCol; c <= lastCol; ++c) {
            size_t idx = static_cast<size_t>(r) * _bounds.totalCols + c;
            if (idx >= _meta.size()) continue;
            const auto& m = _meta[idx];
            if (m.level < 0 || m.level > desiredLevel || m.epoch < epoch) {
                result.push_back(_bounds.worldKeyAt(c, r));
            }
        }
    }
    return result;
}

const uint32_t* TileGrid::tilePixels(const WorldTileKey& wk) const
{
    int col, row;
    if (!_bounds.gridPosition(wk, col, row)) return nullptr;
    const int idx = row * _bounds.totalCols + col;
    if (static_cast<size_t>(idx) >= _tiles.size()) return nullptr;
    const auto& td = _tiles[idx];
    return td.pixels.empty() ? nullptr : td.pixels.data();
}

std::pair<int,int> TileGrid::tileSize(const WorldTileKey& wk) const
{
    int col, row;
    if (!_bounds.gridPosition(wk, col, row)) return {0, 0};
    const int idx = row * _bounds.totalCols + col;
    if (static_cast<size_t>(idx) >= _tiles.size()) return {0, 0};
    return {_tiles[idx].width, _tiles[idx].height};
}

uint64_t TileGrid::tileVersion(const WorldTileKey& wk) const
{
    int col, row;
    if (!_bounds.gridPosition(wk, col, row)) return 0;
    const int idx = row * _bounds.totalCols + col;
    if (static_cast<size_t>(idx) >= _tiles.size()) return 0;
    return _tiles[idx].version;
}

} // namespace vc::render
