#pragma once

#include "vc/core/render/ChunkedPlaneSampler.hpp"

#include <QColor>
#include <QImage>

#include <unordered_set>
#include <vector>

namespace vc3d {

QColor downloadQueueDebugLevelColor(int level);

// Alpha-blend active chunks over an existing clean/display framebuffer.
// The caller resets the image from its clean render before applying one or
// more source lookups.
void applyDownloadQueueDebugOverlay(
    QImage& framebuffer,
    const std::vector<vc::render::ChunkedPlaneSampler::ChunkPixelLookupLevel>& lookup,
    const std::unordered_set<vc::render::ChunkKey,
                             vc::render::ChunkKeyHash>& activeChunks,
    cv::Mat_<uint8_t>* paintedPixels = nullptr);

} // namespace vc3d
