#include "SliceCache.hpp"
#include <utils/hash.hpp>
#include <cmath>

// ============================================================================
// SliceCacheKey
// ============================================================================

SliceCacheKey SliceCacheKey::make(const WorldTileKey& worldTile, const TiledViewerCamera& camera,
                                  uint64_t paramsHash)
{
    SliceCacheKey k;
    k.worldTile = worldTile;

    // Quantize scale to 1/32 log2 stops
    k.scaleQ = static_cast<int16_t>(std::round(std::log2(camera.scale) * 32.0f));

    // Quantize zOff to 0.25 units
    k.zOffQ = static_cast<int16_t>(std::round(camera.zOff * 4.0f));

    k.dsScaleIdx = static_cast<int8_t>(camera.dsScaleIdx);
    k.paramsHash = paramsHash;

    return k;
}

size_t SliceCacheKeyHash::operator()(const SliceCacheKey& k) const
{
    return utils::hash_combine_values(
        k.worldTile.worldCol, k.worldTile.worldRow,
        k.scaleQ, k.zOffQ, k.dsScaleIdx, k.paramsHash);
}

// ============================================================================
// SliceCache
// ============================================================================

SliceCache::SliceCache(size_t maxBytes)
    : cache_({
        .max_bytes = maxBytes,
        .evict_ratio = 15.0 / 16.0,
        .promote_on_read = true,
        .size_fn = [](const QPixmap& pm) -> std::size_t {
            // Estimate actual pixmap memory: width * height * bytes-per-pixel.
            // Null pixmaps get a minimum weight of 1 to avoid zero-cost entries.
            if (pm.isNull())
                return 1;
            int bpp = pm.depth() / 8;  // depth() returns bits per pixel
            if (bpp <= 0) bpp = 4;     // fallback to 32-bit ARGB
            return static_cast<std::size_t>(pm.width()) * pm.height() * bpp;
        },
    })
{
}

SliceCacheLookup SliceCache::getBest(const SliceCacheKey& key, int maxCoarserLevels)
{
    // Try exact match first
    auto result = cache_.get(key);
    if (result) {
        return {std::move(*result), key.dsScaleIdx};
    }

    // Try coarser levels
    for (int delta = 1; delta <= maxCoarserLevels; delta++) {
        SliceCacheKey coarser = key;
        coarser.dsScaleIdx = key.dsScaleIdx + delta;

        auto cresult = cache_.get(coarser);
        if (cresult) {
            return {std::move(*cresult), coarser.dsScaleIdx};
        }
    }

    return {QPixmap(), -1};
}

void SliceCache::put(const SliceCacheKey& key, const QPixmap& pixmap)
{
    cache_.put(key, pixmap);
}

void SliceCache::clear()
{
    cache_.clear();
}

size_t SliceCache::size() const
{
    return cache_.size();
}
