#pragma once
#include "CVolumeViewer.hpp"
#include <omp.h>

#include <list>
#include <mutex>
#include <optional>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <cstddef>

constexpr size_t kAxisAlignedSliceCacheCapacity = 180;


struct AxisAlignedSliceCacheKey
{
    uint8_t planeId = 0;
    uint16_t rotationKey = 0;
    int originX = 0;
    int originY = 0;
    int originZ = 0;
    int roiX = 0;
    int roiY = 0;
    int roiWidth = 0;
    int roiHeight = 0;
    int scaleMilli = 0;
    int dsScaleMilli = 0;
    int zOffsetMilli = 0;
    int dsIndex = 0;
    uintptr_t datasetPtr = 0;
    uint8_t fastInterpolation = 0;
    uint8_t baseWindowLow = 0;
    uint8_t baseWindowHigh = 0;

    bool operator==(const AxisAlignedSliceCacheKey& other) const noexcept
    {
        return planeId == other.planeId && rotationKey == other.rotationKey &&
               originX == other.originX && originY == other.originY && originZ == other.originZ &&
               roiX == other.roiX && roiY == other.roiY &&
               roiWidth == other.roiWidth && roiHeight == other.roiHeight &&
               scaleMilli == other.scaleMilli && dsScaleMilli == other.dsScaleMilli &&
               zOffsetMilli == other.zOffsetMilli && dsIndex == other.dsIndex &&
               datasetPtr == other.datasetPtr && fastInterpolation == other.fastInterpolation &&
               baseWindowLow == other.baseWindowLow && baseWindowHigh == other.baseWindowHigh;
    }
};

struct AxisAlignedSliceCacheKeyHasher
{
    std::size_t operator()(const AxisAlignedSliceCacheKey& key) const noexcept
    {
        std::size_t seed = 0;
        hashCombine(seed, key.planeId);
        hashCombine(seed, key.rotationKey);
        hashCombine(seed, key.originX);
        hashCombine(seed, key.originY);
        hashCombine(seed, key.originZ);
        hashCombine(seed, key.roiX);
        hashCombine(seed, key.roiY);
        hashCombine(seed, key.roiWidth);
        hashCombine(seed, key.roiHeight);
        hashCombine(seed, key.scaleMilli);
        hashCombine(seed, key.dsScaleMilli);
        hashCombine(seed, key.zOffsetMilli);
        hashCombine(seed, key.dsIndex);
        hashCombine(seed, key.datasetPtr);
        hashCombine(seed, key.fastInterpolation);
        hashCombine(seed, key.baseWindowLow);
        hashCombine(seed, key.baseWindowHigh);
        return seed;
    }

private:
    static void hashCombine(std::size_t& seed, std::size_t value) noexcept
    {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }
};

class AxisAlignedSliceCache
{
public:
    explicit AxisAlignedSliceCache(size_t capacity)
        : _capacity(capacity)
    {
    }

    std::optional<cv::Mat> get(const AxisAlignedSliceCacheKey& key)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _entries.find(key);
        if (it == _entries.end()) {
            return std::nullopt;
        }
        _lru.splice(_lru.begin(), _lru, it->second.orderIt);
        return it->second.image.clone();
    }

    void put(const AxisAlignedSliceCacheKey& key, const cv::Mat& image)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _entries.find(key);
        if (it != _entries.end()) {
            it->second.image = image.clone();
            _lru.splice(_lru.begin(), _lru, it->second.orderIt);
            return;
        }

        if (_entries.size() >= _capacity && !_lru.empty()) {
            const AxisAlignedSliceCacheKey& evictKey = _lru.back();
            _entries.erase(evictKey);
            _lru.pop_back();
        }

        _lru.push_front(key);
        Entry entry;
        entry.image = image.clone();
        entry.orderIt = _lru.begin();
        _entries.emplace(_lru.front(), std::move(entry));
    }

private:
    struct Entry {
        cv::Mat image;
        std::list<AxisAlignedSliceCacheKey>::iterator orderIt;
    };

    size_t _capacity;
    std::list<AxisAlignedSliceCacheKey> _lru;
    std::unordered_map<AxisAlignedSliceCacheKey, Entry, AxisAlignedSliceCacheKeyHasher> _entries;
    std::mutex _mutex;
};

static inline AxisAlignedSliceCache& axisAlignedSliceCache()
{
    static AxisAlignedSliceCache cache(kAxisAlignedSliceCacheCapacity);
    return cache;
}