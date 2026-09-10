#pragma once

#include "vc/core/render/ChunkCache.hpp"

#include <QString>

namespace vc3d {

QString formatCacheStorageStats(
    const vc::render::ChunkCache::Stats& stats);
QString formatDownloadQueueStats(
    const vc::render::ChunkCache::Stats& stats);
QString formatNetworkDownloadStats(
    const vc::render::ChunkCache::Stats& stats,
    bool remoteVolume);

} // namespace vc3d
