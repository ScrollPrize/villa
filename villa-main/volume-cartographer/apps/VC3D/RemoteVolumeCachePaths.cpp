#include "RemoteVolumeCachePaths.hpp"

#include "CState.hpp"
#include "VCSettings.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <QSettings>
#include <QString>

#include <cstdint>
#include <sstream>
#include <system_error>

namespace vc3d {

std::string stableHexHash(const std::string& value)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : value) {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

std::string normalizedVolumeCacheIdentity(const std::shared_ptr<Volume>& volume)
{
    if (!volume)
        return {};
    if (volume->isRemote()) {
        return "remote|" + volume->remoteUrl() +
               "|base=" + std::to_string(volume->baseScaleLevel()) +
               "|id=" + volume->id() +
               "|cache_schema=remote_sharded_ranges_v1";
    }

    std::error_code ec;
    auto path = std::filesystem::weakly_canonical(volume->path(), ec);
    if (ec)
        path = std::filesystem::absolute(volume->path(), ec);
    if (ec)
        path = volume->path();
    return "local|" + path.string() + "|id=" + volume->id();
}

std::filesystem::path remoteCacheRootForState(const CState* state)
{
    // Suggestion order: per-volpkg setting first (so projects with an
    // explicit cache stay co-located when no host mount is present), then
    // the user's persisted setting. remoteCachePath() ignores both when
    // /volpkgs or /ephemeral is mounted.
    QString suggestion;
    if (state && state->vpkg()) {
        suggestion = QString::fromStdString(state->vpkg()->remoteCacheRootOrEmpty()).trimmed();
    }
    if (suggestion.isEmpty()) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        suggestion =
            settings.value(vc3d::settings::viewer::REMOTE_CACHE_DIR).toString();
    }
    return vc3d::remoteCachePath(suggestion).toStdString();
}

std::filesystem::path persistentCacheDirForVolume(
    const std::shared_ptr<Volume>& volume,
    const CState* state)
{
    if (!volume || !volume->isRemote())
        return {};
    if (!volume->remoteCacheRoot().empty())
        return volume->remotePersistentCachePath();
    return remoteCacheRootForState(state) /
           stableHexHash(normalizedVolumeCacheIdentity(volume));
}

} // namespace vc3d
