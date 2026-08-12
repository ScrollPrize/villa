#include "elements/DownloadQueueStats.hpp"

#include <QtTest/QtTest>

class DownloadQueueStatsTest final : public QObject
{
    Q_OBJECT

private slots:
    void idleIsEmpty()
    {
        vc::render::ChunkCache::Stats stats;
        stats.unresolvedFetchesByLevel = {0, 0, 0};
        QCOMPARE(vc3d::formatDownloadQueueStats(stats), QString{});
    }

    void trimsOuterZerosAndKeepsInteriorZeros()
    {
        vc::render::ChunkCache::Stats stats;
        stats.unresolvedFetchesByLevel = {0, 3, 0, 5, 0};
        QCOMPARE(vc3d::formatDownloadQueueStats(stats),
                 QStringLiteral("q1 3/0/5"));
    }

    void networkStatusHonorsVolumeAndActivity()
    {
        vc::render::ChunkCache::Stats stats;
        stats.unresolvedFetchesByLevel = {0, 3, 0, 5};
        stats.remoteDownloadBytesPerSecond = 12.34 * 1024.0 * 1024.0;

        QCOMPARE(vc3d::formatNetworkDownloadStats(stats, false), QString{});
        QCOMPARE(vc3d::formatNetworkDownloadStats(stats, true),
                 QStringLiteral("net idle"));

        stats.remoteFetchesInFlight = 7;
        QCOMPARE(vc3d::formatNetworkDownloadStats(stats, true),
                 QStringLiteral("net 7@12.3MiB/s q1 3/0/5"));
    }
};

QTEST_APPLESS_MAIN(DownloadQueueStatsTest)
#include "test_download_queue_stats.moc"
