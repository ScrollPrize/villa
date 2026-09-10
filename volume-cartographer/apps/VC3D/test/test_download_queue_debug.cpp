#include "elements/DownloadQueueDebugOverlay.hpp"

#include <QTest>

class DownloadQueueDebugTest : public QObject {
    Q_OBJECT

private slots:
    void colorsAreStableByLevel();
    void activeChunkIsBlendedAtHalfOpacity();
    void overlappingDownloadsAreBlendedOnlyOnce();
};

void DownloadQueueDebugTest::colorsAreStableByLevel()
{
    QCOMPARE(vc3d::downloadQueueDebugLevelColor(0), QColor(255, 64, 64));
    QCOMPARE(vc3d::downloadQueueDebugLevelColor(1), QColor(64, 220, 96));
    QCOMPARE(vc3d::downloadQueueDebugLevelColor(8), QColor(255, 64, 64));
}

void DownloadQueueDebugTest::activeChunkIsBlendedAtHalfOpacity()
{
    const vc::render::VolumeSourceId source{17};
    vc::render::ChunkedPlaneSampler::ChunkPixelLookupLevel level;
    level.level = 1;
    level.chunks = {
        {1, 0, 0, 0, source},
        {1, 0, 0, 1, source},
    };
    level.pixelIds = cv::Mat_<uint16_t>(1, 3);
    level.pixelIds(0, 0) = 1;
    level.pixelIds(0, 1) = 2;
    level.pixelIds(0, 2) = 0;

    QImage framebuffer(3, 1, QImage::Format_RGB32);
    framebuffer.fill(QColor(20, 40, 60));
    std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash> active{
        level.chunks[1],
    };
    vc3d::applyDownloadQueueDebugOverlay(framebuffer, {level}, active);

    QCOMPARE(framebuffer.pixelColor(0, 0), QColor(20, 40, 60));
    QCOMPARE(framebuffer.pixelColor(1, 0), QColor(42, 130, 78));
    QCOMPARE(framebuffer.pixelColor(2, 0), QColor(20, 40, 60));
}

void DownloadQueueDebugTest::overlappingDownloadsAreBlendedOnlyOnce()
{
    const vc::render::VolumeSourceId source{23};
    vc::render::ChunkedPlaneSampler::ChunkPixelLookupLevel fine;
    fine.level = 1;
    fine.chunks = {{1, 0, 0, 0, source}};
    fine.pixelIds = cv::Mat_<uint16_t>(1, 1, uint16_t{1});

    vc::render::ChunkedPlaneSampler::ChunkPixelLookupLevel coarse;
    coarse.level = 2;
    coarse.chunks = {{2, 0, 0, 0, source}};
    coarse.pixelIds = cv::Mat_<uint16_t>(1, 1, uint16_t{1});

    QImage framebuffer(1, 1, QImage::Format_RGB32);
    framebuffer.fill(QColor(20, 40, 60));
    std::unordered_set<vc::render::ChunkKey, vc::render::ChunkKeyHash> active{
        fine.chunks[0],
        coarse.chunks[0],
    };
    cv::Mat_<uint8_t> paintedPixels;
    vc3d::applyDownloadQueueDebugOverlay(
        framebuffer, {fine, coarse}, active, &paintedPixels);

    QCOMPARE(framebuffer.pixelColor(0, 0), QColor(42, 130, 78));
    QCOMPARE(paintedPixels(0, 0), uint8_t{1});

    vc3d::applyDownloadQueueDebugOverlay(
        framebuffer, {coarse}, active, &paintedPixels);
    QCOMPARE(framebuffer.pixelColor(0, 0), QColor(42, 130, 78));
}

QTEST_MAIN(DownloadQueueDebugTest)
#include "test_download_queue_debug.moc"
