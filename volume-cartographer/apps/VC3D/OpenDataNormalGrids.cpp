#include "OpenDataNormalGrids.hpp"

#include "OpenDataManifest.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/NormalGridVolume.hpp"

#include <nlohmann/json.hpp>

#include <QDir>
#include <QStandardPaths>
#include <QString>
#include <QUrl>
#include <QXmlStreamReader>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <mutex>
#include <system_error>
#include <thread>

namespace vc3d::opendata {
namespace {

std::string trimTrailingSlashes(std::string value)
{
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

std::string artifactUrl(const OpenDataArtifact& artifact)
{
    return trimTrailingSlashes(artifact.resolvedUrl.empty()
                                  ? artifact.sourcePath
                                  : artifact.resolvedUrl);
}

std::string safePathComponent(std::string value)
{
    for (char& c : value) {
        const auto uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '-' && c != '_' && c != '.') {
            c = '_';
        }
    }
    while (!value.empty() && (value.front() == '.' || value.front() == '_')) {
        value.erase(value.begin());
    }
    return value.empty() ? std::string("unnamed") : value;
}

std::string tagValue(const std::vector<std::string>& tags, std::string_view prefix)
{
    for (const auto& tag : tags) {
        if (tag.rfind(prefix, 0) == 0) {
            return tag.substr(prefix.size());
        }
    }
    return {};
}

// Split "https://host/some/key/prefix" into origin ("https://host") and the
// key prefix ("some/key/prefix/").
bool splitRemotePrefix(const std::string& url, std::string& origin, std::string& prefix)
{
    const auto schemePos = url.find("://");
    if (schemePos == std::string::npos) {
        return false;
    }
    const auto hostEnd = url.find('/', schemePos + 3);
    if (hostEnd == std::string::npos) {
        return false;
    }
    origin = url.substr(0, hostEnd);
    prefix = trimTrailingSlashes(url.substr(hostEnd + 1)) + "/";
    return prefix.size() > 1;
}

struct RemoteObject {
    std::string relativePath;  // key with the listing prefix stripped
    std::uint64_t size = 0;
};

// Anonymous S3 ListObjectsV2 over the public bucket. Returns every object
// under `prefix`, paginated 1000 keys at a time.
std::vector<RemoteObject> listRemotePrefix(const std::string& origin,
                                           const std::string& prefix,
                                           const std::atomic<bool>* cancelRequested,
                                           std::string* errorOut)
{
    std::vector<RemoteObject> objects;
    QString continuationToken;

    while (true) {
        if (cancelRequested && cancelRequested->load(std::memory_order_acquire)) {
            if (errorOut) *errorOut = "cancelled";
            return {};
        }

        QString listUrl = QString::fromStdString(origin) +
                          QStringLiteral("/?list-type=2&max-keys=1000&prefix=") +
                          QString::fromUtf8(QUrl::toPercentEncoding(QString::fromStdString(prefix)));
        if (!continuationToken.isEmpty()) {
            listUrl += QStringLiteral("&continuation-token=") +
                       QString::fromUtf8(QUrl::toPercentEncoding(continuationToken));
        }

        std::string body;
        try {
            body = vc::httpGetString(listUrl.toStdString());
        } catch (const std::exception& e) {
            if (errorOut) *errorOut = std::string("listing failed: ") + e.what();
            return {};
        }
        if (body.empty()) {
            if (errorOut) *errorOut = "listing returned no data for " + prefix;
            return {};
        }

        QXmlStreamReader xml(QString::fromStdString(body));
        QString nextToken;
        bool truncated = false;
        while (!xml.atEnd()) {
            xml.readNext();
            if (!xml.isStartElement()) {
                continue;
            }
            if (xml.name() == QStringLiteral("Contents")) {
                RemoteObject object;
                while (!(xml.isEndElement() && xml.name() == QStringLiteral("Contents")) &&
                       !xml.atEnd()) {
                    xml.readNext();
                    if (!xml.isStartElement()) {
                        continue;
                    }
                    if (xml.name() == QStringLiteral("Key")) {
                        const std::string key = xml.readElementText().toStdString();
                        if (key.rfind(prefix, 0) == 0) {
                            object.relativePath = key.substr(prefix.size());
                        }
                    } else if (xml.name() == QStringLiteral("Size")) {
                        object.size = xml.readElementText().toULongLong();
                    }
                }
                if (!object.relativePath.empty() && object.relativePath.back() != '/') {
                    objects.push_back(std::move(object));
                }
            } else if (xml.name() == QStringLiteral("NextContinuationToken")) {
                nextToken = xml.readElementText();
            } else if (xml.name() == QStringLiteral("IsTruncated")) {
                truncated = xml.readElementText() == QStringLiteral("true");
            }
        }
        if (xml.hasError()) {
            if (errorOut) {
                *errorOut = "failed to parse listing: " + xml.errorString().toStdString();
            }
            return {};
        }

        if (!truncated || nextToken.isEmpty()) {
            break;
        }
        continuationToken = nextToken;
    }

    if (objects.empty() && errorOut) {
        *errorOut = "no objects found under " + prefix;
    }
    return objects;
}

std::string encodedObjectUrl(const std::string& origin, const std::string& key)
{
    const QByteArray encoded =
        QUrl::toPercentEncoding(QString::fromStdString(key), QByteArrayLiteral("/"));
    return origin + "/" + std::string(encoded.constData(), encoded.size());
}

void writeFile(const std::filesystem::path& target, const std::vector<std::byte>& bytes)
{
    std::error_code ec;
    std::filesystem::create_directories(target.parent_path(), ec);
    if (ec) {
        throw std::runtime_error("failed to create directory " +
                                 target.parent_path().string() + ": " + ec.message());
    }
    std::ofstream out(target, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open " + target.string() + " for writing");
    }
    if (!bytes.empty()) {
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    out.close();
    if (!out) {
        throw std::runtime_error("failed to write " + target.string());
    }
}

std::filesystem::path makeTempDir(const std::filesystem::path& finalDir)
{
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    return finalDir.parent_path() /
           (finalDir.filename().string() + ".tmp-" + std::to_string(tick));
}

void publishDirectory(const std::filesystem::path& tempDir,
                      const std::filesystem::path& finalDir)
{
    std::error_code ec;
    std::filesystem::create_directories(finalDir.parent_path(), ec);
    if (ec) {
        throw std::runtime_error("failed to create cache parent: " + ec.message());
    }

    const auto backupDir = finalDir.parent_path() /
        (finalDir.filename().string() + ".previous");
    std::filesystem::remove_all(backupDir, ec);
    ec.clear();
    if (std::filesystem::exists(finalDir, ec)) {
        std::filesystem::rename(finalDir, backupDir, ec);
        if (ec) {
            throw std::runtime_error("failed to move existing cache aside: " + ec.message());
        }
    }

    std::filesystem::rename(tempDir, finalDir, ec);
    if (ec) {
        std::error_code restoreEc;
        if (std::filesystem::exists(backupDir, restoreEc) &&
            !std::filesystem::exists(finalDir, restoreEc)) {
            std::filesystem::rename(backupDir, finalDir, restoreEc);
        }
        throw std::runtime_error("failed to publish normal grids cache: " + ec.message());
    }

    std::filesystem::remove_all(backupDir, ec);
}

std::filesystem::path cachedCatalogManifestPath()
{
    QString base = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
    if (base.isEmpty()) {
        base = QDir::home().filePath(QStringLiteral(".VC3D"));
    }
    return std::filesystem::path(base.toStdString()) / "open-data-catalog" / "metadata.json";
}

std::vector<std::string> normalGridsEntryTags(const OpenDataNormalGridsInfo& info)
{
    std::vector<std::string> tags{"open-data"};
    if (!info.sampleId.empty()) {
        tags.push_back(std::string(kOpenDataSampleIdTagPrefix) + info.sampleId);
    }
    if (!info.volumeId.empty()) {
        tags.push_back(std::string(kOpenDataVolumeIdTagPrefix) + info.volumeId);
    }
    return tags;
}

constexpr const char* kCompleteMarker = "normal-grids-complete.json";

// Write the marker NormalGridVolume streams from. Idempotent.
bool writeRemoteMarker(const std::filesystem::path& dir, const std::string& url)
{
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        return false;
    }
    const auto markerPath = dir / vc::core::util::kNormalGridsRemoteMarker;
    if (std::filesystem::is_regular_file(markerPath, ec)) {
        try {
            const auto existing = nlohmann::json::parse(std::ifstream(markerPath));
            if (existing.value("url", std::string()) == url) {
                return true;
            }
        } catch (...) {
        }
    }
    nlohmann::json marker;
    marker["url"] = url;
    std::ofstream out(markerPath, std::ios::trunc);
    out << marker.dump(2);
    out.close();
    return static_cast<bool>(out);
}

void writeCompleteMarker(const std::filesystem::path& dir,
                         int files,
                         std::uint64_t bytes)
{
    nlohmann::json marker;
    marker["files"] = files;
    marker["bytes"] = bytes;
    std::ofstream out(dir / kCompleteMarker, std::ios::trunc);
    out << marker.dump(2);
}

} // namespace

const OpenDataArtifact* normalGridsArtifact(const OpenDataVolume& volume)
{
    return findArtifact(volume.artifacts, kNormalGridsArtifactType);
}

std::string normalGridsArtifactUrl(const OpenDataVolume& volume)
{
    const auto* artifact = normalGridsArtifact(volume);
    return artifact ? artifactUrl(*artifact) : std::string{};
}

std::optional<OpenDataNormalGridsInfo> normalGridsInfoFromTags(
    const std::vector<std::string>& tags)
{
    OpenDataNormalGridsInfo info;
    info.sampleId = tagValue(tags, kOpenDataSampleIdTagPrefix);
    info.volumeId = tagValue(tags, kOpenDataVolumeIdTagPrefix);
    info.url = tagValue(tags, kOpenDataNormalGridsTagPrefix);
    if (info.volumeId.empty()) {
        return std::nullopt;
    }
    return info;
}

std::filesystem::path normalGridsCacheDir(const std::filesystem::path& remoteCacheRoot,
                                          const std::string& sampleId,
                                          const std::string& volumeId)
{
    return remoteCacheRoot / "normal_grids" /
           safePathComponent(sampleId.empty() ? "sample" : sampleId) /
           safePathComponent(volumeId.empty() ? "volume" : volumeId);
}

bool isCachedNormalGridsDir(const std::filesystem::path& dir)
{
    std::error_code ec;
    return std::filesystem::is_directory(dir / "xy", ec) &&
           std::filesystem::is_directory(dir / "xz", ec) &&
           std::filesystem::is_directory(dir / "yz", ec) &&
           std::filesystem::is_regular_file(dir / "metadata.json", ec);
}

bool attachStreamingNormalGridsEntry(VolumePkg& pkg,
                                     const OpenDataNormalGridsInfo& info,
                                     const std::filesystem::path& remoteCacheRoot)
{
    if (remoteCacheRoot.empty() || info.url.empty()) {
        return false;
    }
    const auto dir = normalGridsCacheDir(remoteCacheRoot, info.sampleId, info.volumeId);
    if (!writeRemoteMarker(dir, info.url)) {
        Logger()->warn("Failed to create streaming normal-grid cache at {}", dir.string());
        return false;
    }
    // addNormalGridEntry dedups by location and returns false for duplicates;
    // either way the entry ends up attached.
    pkg.addNormalGridEntry(dir.string(), normalGridsEntryTags(info));
    return true;
}

int attachOpenDataNormalGrids(VolumePkg& pkg,
                              const OpenDataSample& sample,
                              const std::filesystem::path& remoteCacheRoot)
{
    int attached = 0;
    for (const auto& volume : sample.volumes) {
        OpenDataNormalGridsInfo info;
        info.sampleId = sample.id;
        info.volumeId = volume.id;
        info.url = normalGridsArtifactUrl(volume);
        if (info.url.empty()) {
            continue;
        }
        if (attachStreamingNormalGridsEntry(pkg, info, remoteCacheRoot)) {
            ++attached;
        }
    }
    return attached;
}

NormalGridsCacheState normalGridsCacheState(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId,
    const OpenDataVolume& volume)
{
    NormalGridsCacheState state;
    state.hasArtifact = normalGridsArtifact(volume) != nullptr;
    if (!state.hasArtifact || remoteCacheRoot.empty()) {
        return state;
    }

    const auto dir = normalGridsCacheDir(remoteCacheRoot, sampleId, volume.id);
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        return state;
    }

    const auto completeMarker = dir / kCompleteMarker;
    if (std::filesystem::is_regular_file(completeMarker, ec)) {
        try {
            const auto marker = nlohmann::json::parse(std::ifstream(completeMarker));
            state.complete = true;
            state.totalBytes = marker.value("bytes", std::uint64_t{0});
            state.cachedBytes = state.totalBytes;
            state.cachedFiles = marker.value("files", 0);
            return state;
        } catch (...) {
        }
    }

    for (auto it = std::filesystem::recursive_directory_iterator(
             dir, std::filesystem::directory_options::skip_permission_denied, ec);
         it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) {
            break;
        }
        if (!it->is_regular_file(ec)) {
            continue;
        }
        const auto name = it->path().filename().string();
        if (name == vc::core::util::kNormalGridsRemoteMarker ||
            name.ends_with(".missing") ||
            name.find(".tmp-") != std::string::npos) {
            continue;
        }
        state.cachedBytes += it->file_size(ec);
        ++state.cachedFiles;
    }
    return state;
}

std::filesystem::path downloadOpenDataNormalGrids(
    const OpenDataNormalGridsInfo& info,
    const std::filesystem::path& remoteCacheRoot,
    int workers,
    const NormalGridsProgressCallback& progress,
    const std::atomic<bool>* cancelRequested,
    std::string* errorOut)
{
    if (info.url.empty() || remoteCacheRoot.empty()) {
        if (errorOut) *errorOut = "no normal grids URL or cache root available";
        return {};
    }

    const auto finalDir = normalGridsCacheDir(remoteCacheRoot, info.sampleId, info.volumeId);
    std::error_code completeEc;
    if (std::filesystem::is_regular_file(finalDir / kCompleteMarker, completeEc)) {
        return finalDir;
    }

    std::string origin;
    std::string prefix;
    if (!splitRemotePrefix(info.url, origin, prefix)) {
        if (errorOut) *errorOut = "unsupported normal grids URL: " + info.url;
        return {};
    }

    NormalGridsDownloadProgress state;
    state.status = "listing";
    if (progress) {
        progress(state);
    }

    const auto objects = listRemotePrefix(origin, prefix, cancelRequested, errorOut);
    if (objects.empty()) {
        return {};
    }

    state.status = "downloading";
    state.totalFiles = static_cast<int>(objects.size());
    for (const auto& object : objects) {
        state.totalBytes += object.size;
    }
    if (progress) {
        progress(state);
    }

    const auto tempDir = makeTempDir(finalDir);
    std::error_code ec;
    std::filesystem::create_directories(tempDir, ec);
    if (ec) {
        if (errorOut) *errorOut = "failed to create " + tempDir.string() + ": " + ec.message();
        return {};
    }

    const int workerCount = std::clamp(workers, 1, kNormalGridsDownloadWorkers);
    std::atomic<std::size_t> next{0};
    std::atomic<int> completed{0};
    std::atomic<int> failed{0};
    std::atomic<std::uint64_t> completedBytes{0};
    std::mutex progressMutex;
    std::string firstError;

    auto worker = [&]() {
        while (true) {
            if (cancelRequested && cancelRequested->load(std::memory_order_acquire)) {
                return;
            }
            const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= objects.size()) {
                return;
            }
            const auto& object = objects[index];
            try {
                const auto target = tempDir / object.relativePath;
                // Reuse files a streaming session already cached: hardlink from
                // the live dir when the size matches the listing.
                const auto streamed = finalDir / object.relativePath;
                std::error_code linkEc;
                if (std::filesystem::is_regular_file(streamed, linkEc) &&
                    std::filesystem::file_size(streamed, linkEc) == object.size && !linkEc) {
                    std::filesystem::create_directories(target.parent_path(), linkEc);
                    std::filesystem::create_hard_link(streamed, target, linkEc);
                }
                if (linkEc || !std::filesystem::is_regular_file(target, linkEc)) {
                    const auto url = encodedObjectUrl(origin, prefix + object.relativePath);
                    const auto bytes = vc::httpGetBytes(url);
                    // httpGetBytes returns an empty body for 4xx misses; only
                    // accept it when the listing says the object really is empty.
                    if (bytes.empty() && object.size > 0) {
                        throw std::runtime_error("empty response for " + url);
                    }
                    writeFile(target, bytes);
                }
                completed.fetch_add(1, std::memory_order_relaxed);
                completedBytes.fetch_add(object.size, std::memory_order_relaxed);
            } catch (const std::exception& e) {
                failed.fetch_add(1, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(progressMutex);
                if (firstError.empty()) {
                    firstError = e.what();
                }
            }

            if (progress) {
                NormalGridsDownloadProgress update = state;
                update.completedFiles = completed.load(std::memory_order_relaxed);
                update.failedFiles = failed.load(std::memory_order_relaxed);
                update.completedBytes = completedBytes.load(std::memory_order_relaxed);
                update.fileName = object.relativePath;
                progress(update);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workerCount));
    for (int i = 0; i < workerCount; ++i) {
        threads.emplace_back(worker);
    }
    for (auto& thread : threads) {
        thread.join();
    }

    const bool cancelled =
        cancelRequested && cancelRequested->load(std::memory_order_acquire);
    if (cancelled || failed.load() > 0) {
        std::filesystem::remove_all(tempDir, ec);
        if (errorOut) {
            *errorOut = cancelled
                ? "cancelled"
                : ("failed to download " + std::to_string(failed.load()) +
                   " file(s): " + firstError);
        }
        return {};
    }

    try {
        publishDirectory(tempDir, finalDir);
    } catch (const std::exception& e) {
        std::filesystem::remove_all(tempDir, ec);
        if (errorOut) *errorOut = e.what();
        return {};
    }

    if (!isCachedNormalGridsDir(finalDir)) {
        if (errorOut) {
            *errorOut = "downloaded data at " + finalDir.string() +
                        " is not a normal grid store (missing xy/xz/yz or metadata.json)";
        }
        return {};
    }

    // Keep the dir streamable (harmless once complete) and record completion
    // so the catalog can report "all downloaded" without a remote listing.
    writeRemoteMarker(finalDir, info.url);
    writeCompleteMarker(finalDir, static_cast<int>(objects.size()), state.totalBytes);

    Logger()->info("Downloaded normal grids for {}/{} to {}",
                   info.sampleId, info.volumeId, finalDir.string());
    return finalDir;
}

std::vector<OpenDataNormalGridsInfo> normalGridsAvailabilityFromCachedManifest()
{
    std::vector<OpenDataNormalGridsInfo> result;
    const auto manifestPath = cachedCatalogManifestPath();
    std::error_code ec;
    if (!std::filesystem::is_regular_file(manifestPath, ec)) {
        return result;
    }
    OpenDataManifest manifest;
    try {
        manifest = loadOpenDataManifestFile(manifestPath);
    } catch (...) {
        return result;
    }
    for (const auto& sample : manifest.samples) {
        for (const auto& volume : sample.volumes) {
            const auto url = normalGridsArtifactUrl(volume);
            if (!url.empty()) {
                result.push_back({sample.id, volume.id, url});
            }
        }
    }
    return result;
}

} // namespace vc3d::opendata
