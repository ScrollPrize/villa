#include "RenderPool.hpp"

#include <QImage>

#include "vc/core/util/Surface.hpp"
#include "vc/core/types/Volume.hpp"

RenderPool::RenderPool(int numThreads, QObject* parent)
    : QObject(parent)
    , corePool_(std::make_unique<vc::render::CoreRenderPool>(numThreads))
{
    corePool_->setReadyCallback([this]() { emit tileReady(); });
}

RenderPool::~RenderPool()
{
    cancelAll();
}

void RenderPool::submit(const TileRenderParams& params,
                        const std::shared_ptr<Surface>& surface,
                        const std::shared_ptr<Volume>& volume,
                        int controllerId)
{
    corePool_->submit(params, surface, volume, controllerId);
}

std::vector<QtTileRenderResult> RenderPool::takeResults(int controllerId)
{
    auto coreResults = corePool_->takeResults(controllerId);

    std::vector<QtTileRenderResult> results;
    results.reserve(coreResults.size());

    for (auto& cr : coreResults) {
        QtTileRenderResult qr;
        static_cast<TileRenderResult&>(qr) = std::move(cr);

        if (!qr.pixels.empty()) {
            QImage img(qr.width, qr.height, QImage::Format_RGB32);
            const int srcStride = qr.width * 4;
            const int dstStride = img.bytesPerLine();
            if (srcStride == dstStride) {
                std::memcpy(img.bits(), qr.pixels.data(),
                            static_cast<size_t>(srcStride) * qr.height);
            } else {
                for (int y = 0; y < qr.height; y++) {
                    std::memcpy(img.scanLine(y),
                                reinterpret_cast<const uchar*>(qr.pixels.data()) + y * srcStride,
                                srcStride);
                }
            }
            qr.pixmap = QPixmap::fromImage(std::move(img), Qt::NoFormatConversion);
            qr.pixels.clear();
            qr.pixels.shrink_to_fit();
        }

        results.push_back(std::move(qr));
    }

    return results;
}

void RenderPool::cancelAll()
{
    corePool_->clearAll();
}
