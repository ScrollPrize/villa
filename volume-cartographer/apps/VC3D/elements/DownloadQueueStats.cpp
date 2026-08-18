#include "elements/DownloadQueueStats.hpp"

#include <algorithm>
#include <iterator>
#include <QStringList>

namespace vc3d {

namespace {

QString formatGigabytes(std::size_t bytes)
{
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    return QString::number(static_cast<double>(bytes) / kGiB, 'f', 1);
}

} // namespace

QString formatCacheStorageStats(const vc::render::ChunkCache::Stats& stats)
{
    QString result = QStringLiteral("RAM %1/%2")
        .arg(formatGigabytes(stats.decodedBytes))
        .arg(formatGigabytes(stats.decodedByteCapacity));
    if (!stats.persistentCacheEnabled)
        return result + QStringLiteral(" GiB disk off");

    result += QStringLiteral(" disk %1/")
        .arg(formatGigabytes(stats.persistentCacheBytes));
    result += stats.persistentCacheMaximumBytes
        ? formatGigabytes(*stats.persistentCacheMaximumBytes)
        : QStringLiteral("unlimited");
    result += QStringLiteral(" GiB");
    if (stats.persistentCacheScanInFlight)
        result += QStringLiteral(" (scanning)");
    else if (stats.persistentCacheTrimInFlight)
        result += QStringLiteral(" (trimming)");
    return result;
}

QString formatDownloadQueueStats(const vc::render::ChunkCache::Stats& stats)
{
    const auto first = std::find_if(
        stats.unresolvedFetchesByLevel.begin(),
        stats.unresolvedFetchesByLevel.end(),
        [](std::size_t count) { return count != 0; });
    if (first == stats.unresolvedFetchesByLevel.end())
        return {};

    const auto last = std::find_if(
        stats.unresolvedFetchesByLevel.rbegin(),
        stats.unresolvedFetchesByLevel.rend(),
        [](std::size_t count) { return count != 0; }).base();

    QStringList counts;
    for (auto it = first; it != last; ++it)
        counts << QString::number(*it);

    const auto firstLevel = static_cast<qlonglong>(
        std::distance(stats.unresolvedFetchesByLevel.begin(), first));
    return QStringLiteral("q%1 %2")
        .arg(firstLevel)
        .arg(counts.join('/'));
}

QString formatNetworkDownloadStats(const vc::render::ChunkCache::Stats& stats,
                                   bool remoteVolume)
{
    if (!remoteVolume)
        return {};
    if (stats.remoteFetchesInFlight == 0)
        return QStringLiteral("net idle");

    constexpr double kMiB = 1024.0 * 1024.0;
    QString result = QStringLiteral("net %1x %2MiB/s")
        .arg(stats.remoteFetchesInFlight)
        .arg(std::max(0.0, stats.remoteDownloadBytesPerSecond) / kMiB,
             0, 'f', 1);
    const QString queue = formatDownloadQueueStats(stats);
    if (!queue.isEmpty())
        result += QStringLiteral(" ") + queue;
    return result;
}

} // namespace vc3d
