#include "elements/DownloadQueueDebugOverlay.hpp"

#include <algorithm>
#include <array>

namespace vc3d {

QColor downloadQueueDebugLevelColor(int level)
{
    static constexpr std::array<std::array<int, 3>, 8> palette{{
        {{255, 64, 64}},
        {{64, 220, 96}},
        {{64, 144, 255}},
        {{255, 208, 64}},
        {{220, 80, 255}},
        {{64, 224, 224}},
        {{255, 136, 48}},
        {{176, 176, 255}},
    }};
    const auto& rgb = palette[static_cast<std::size_t>(
        std::max(0, level) % static_cast<int>(palette.size()))];
    return QColor(rgb[0], rgb[1], rgb[2]);
}

void applyDownloadQueueDebugOverlay(
    QImage& framebuffer,
    const std::vector<vc::render::ChunkedPlaneSampler::ChunkPixelLookupLevel>& lookup,
    const std::unordered_set<vc::render::ChunkKey,
                             vc::render::ChunkKeyHash>& activeChunks,
    cv::Mat_<uint8_t>* paintedPixels)
{
    if (framebuffer.isNull() || activeChunks.empty())
        return;

    cv::Mat_<uint8_t> localPainted;
    if (paintedPixels == nullptr) {
        localPainted = cv::Mat_<uint8_t>::zeros(
            framebuffer.height(), framebuffer.width());
        paintedPixels = &localPainted;
    } else if (paintedPixels->rows != framebuffer.height() ||
               paintedPixels->cols != framebuffer.width()) {
        *paintedPixels = cv::Mat_<uint8_t>::zeros(
            framebuffer.height(), framebuffer.width());
    }

    for (const auto& level : lookup) {
        if (level.pixelIds.empty() || level.chunks.empty() ||
            level.pixelIds.rows != framebuffer.height() ||
            level.pixelIds.cols != framebuffer.width()) {
            continue;
        }

        std::vector<uint8_t> activeIds(level.chunks.size() + 1, uint8_t{0});
        bool anyActive = false;
        for (std::size_t i = 0; i < level.chunks.size(); ++i) {
            if (activeChunks.contains(level.chunks[i])) {
                activeIds[i + 1] = 1;
                anyActive = true;
            }
        }
        if (!anyActive)
            continue;

        const QColor color = downloadQueueDebugLevelColor(level.level);
        for (int y = 0; y < framebuffer.height(); ++y) {
            auto* pixels = reinterpret_cast<QRgb*>(framebuffer.scanLine(y));
            const auto* ids = level.pixelIds.ptr<uint16_t>(y);
            auto* painted = paintedPixels->ptr<uint8_t>(y);
            for (int x = 0; x < framebuffer.width(); ++x) {
                const uint16_t id = ids[x];
                if (painted[x] || id == 0 || id >= activeIds.size() ||
                    !activeIds[id]) {
                    continue;
                }
                const QRgb source = pixels[x];
                pixels[x] = qRgb(
                    (qRed(source) + color.red() + 1) / 2,
                    (qGreen(source) + color.green() + 1) / 2,
                    (qBlue(source) + color.blue() + 1) / 2);
                painted[x] = 1;
            }
        }
    }
}

} // namespace vc3d
