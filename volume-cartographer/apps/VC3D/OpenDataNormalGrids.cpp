#include "OpenDataNormalGrids.hpp"

#include "OpenDataManifest.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/NormalGridVolume.hpp"

#include <nlohmann/json.hpp>

#include <vc_reader/s3_listing.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <mutex>
#include <sstream>
#include <system_error>
#include <thread>

namespace vc3d::opendata {
namespace {

struct RemoteObject {
    std::string relativePath;  // key with the listing prefix stripped
    std::uint64_t size = 0;
};

// Every object under `prefix` (anonymous ListObjectsV2), 1000 keys a page.
std::vector<RemoteObject> listRemotePrefix(const vcr::s3::Prefix& prefix,
                                           const std::atomic<bool>* cancelRequested,
                                           std::string* errorOut)
{
    std::vector<RemoteObject> objects;
    std::string continuationToken;
    do {
        if (cancelRequested && cancelRequested->load(std::memory_order_acquire)) {
            if (errorOut) *errorOut = "cancelled";
            return {};
        }
        vcr::s3::ListPage page;
        try {
            page = vcr::s3::parseListPage(vc::httpGetString(vcr::s3::listUrl(prefix, {}, continuationToken)));
        } catch (const std::exception& e) {
            if (errorOut) *errorOut = std::string("listing failed: ") + e.what();
            return {};
        }
        for (const auto& object : page.objects) {
            if (object.key.rfind(prefix.key, 0) != 0 || object.key.size() == prefix.key.size() ||
                object.key.back() == '/') {
                continue;
            }
            objects.push_back({object.key.substr(prefix.key.size()), object.size});
        }
        continuationToken = std::move(page.nextContinuationToken);
    } while (!continuationToken.empty());

    if (objects.empty() && errorOut) {
        *errorOut = "no objects found under " + prefix.key;
    }
    return objects;
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

// <remoteCacheRoot>/normal_grids/<sampleId>/<volumeId>/L<level>-<url hash>
std::filesystem::path normalGridsCacheDir(const std::filesystem::path& remoteCacheRoot,
                                          const std::string& sampleId,
                                          const std::string& volumeId,
                                          int sourceCoordinateLevel,
                                          std::string_view url)
{
    return remoteCacheRoot / "normal_grids" /
           vcr::catalog::safePathComponent(sampleId.empty() ? "sample" : sampleId) /
           vcr::catalog::safePathComponent(volumeId.empty() ? "volume" : volumeId) /
           ("L" + std::to_string(sourceCoordinateLevel) + "-" + cacheKey(url));
}

constexpr const char* kCompleteMarker = "normal-grids-complete.json";

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

std::vector<OpenDataNormalGridsInfo> normalGridsArtifacts(
    const std::string& sampleId,
    const vcr::catalog::OpenDataVolume& volume)
{
    std::vector<OpenDataNormalGridsInfo> result;
    for (std::size_t artifactIndex = 0; artifactIndex < volume.artifacts.size(); ++artifactIndex) {
        const auto& artifact = volume.artifacts[artifactIndex];
        if (vcr::catalog::classifyDerivedRepresentation(artifact) !=
            vcr::catalog::OpenDataRepresentationKind::NormalGrids) {
            continue;
        }
        // Without parameters.level the grids cannot be paired with any volume.
        if (!artifact.sourceCoordinateLevel) {
            Logger()->warn("Skipping normal-grids artifact with missing or invalid parameters.level for {}/{}",
                           sampleId, volume.id);
            continue;
        }
        const auto url = artifact.url();
        if (url.empty()) {
            continue;
        }
        OpenDataNormalGridsInfo info;
        info.sampleId = sampleId;
        info.volumeId = volume.id;
        info.url = url;
        info.sourceCoordinateLevel = *artifact.sourceCoordinateLevel;
        info.artifactIndex = artifactIndex;
        const auto duplicate = std::find_if(result.begin(), result.end(), [&](const auto& item) {
            return item.url == info.url &&
                   item.sourceCoordinateLevel == info.sourceCoordinateLevel;
        });
        if (duplicate == result.end()) {
            result.push_back(std::move(info));
        }
    }
    return result;
}

std::optional<vcr::catalog::OpenDataVolumeView> normalGridsCatalogView(const VolumePkg& pkg,
                                                                       const vc::project::Entry& entry)
{
    if (!pkg.catalog()) {
        return std::nullopt;
    }
    const auto url = vc::core::util::normalGridsRemoteUrl(
        vc::project::resolveLocalPath(entry.location, pkg.path().parent_path()));
    return url ? pkg.catalog()->volumeAt(*url, 0) : std::nullopt;
}

namespace {

// Create the streaming cache dir for this pairing (writing the remote-marker
// file NormalGridVolume streams from) and attach it as a normal-grid project
// entry. Returns true when the entry is attached (or already present).
bool attachStreamingNormalGridsEntry(VolumePkg& pkg,
                                     const OpenDataNormalGridsInfo& info,
                                     const std::filesystem::path& remoteCacheRoot)
{
    if (remoteCacheRoot.empty() || info.url.empty()) {
        return false;
    }
    const auto dir = normalGridsCacheDir(remoteCacheRoot, info.sampleId, info.volumeId,
                                         info.sourceCoordinateLevel, info.url);
    if (!vc::core::util::writeNormalGridsRemoteMarker(dir, info.url)) {
        Logger()->warn("Failed to create streaming normal-grid cache at {}", dir.string());
        return false;
    }
    // addNormalGridEntry dedups by location; either way the entry ends up
    // attached. The marker's URL names its catalog dataset (normalGridsCatalogView).
    pkg.addNormalGridEntry(dir.string());
    return true;
}

} // namespace

int attachOpenDataNormalGrids(VolumePkg& pkg,
                              const vcr::catalog::OpenDataSample& sample,
                              const std::filesystem::path& remoteCacheRoot,
                              const OpenDataResourceSelection* selection)
{
    int attached = 0;
    for (std::size_t volumeIndex = 0; volumeIndex < sample.volumes.size();
         ++volumeIndex) {
        const auto& volume = sample.volumes[volumeIndex];
        if (selection && !selection->allowsVolume(volume.id)) {
            continue;
        }
        for (const auto& info : normalGridsArtifacts(sample.id, volume)) {
            if (selection &&
                !selection->allowsRepresentation(
                    volumeIndex, info.artifactIndex,
                    vcr::catalog::OpenDataRepresentationKind::NormalGrids, volume.id)) {
                continue;
            }
            if (attachStreamingNormalGridsEntry(pkg, info, remoteCacheRoot)) {
                ++attached;
            }
        }
    }
    return attached;
}

NormalGridsCacheState normalGridsCacheState(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId,
    const vcr::catalog::OpenDataVolume& volume)
{
    NormalGridsCacheState state;
    const auto artifacts = normalGridsArtifacts(sampleId, volume);
    state.hasArtifact = !artifacts.empty();
    if (!state.hasArtifact || remoteCacheRoot.empty()) {
        return state;
    }
    state.complete = true;
    for (const auto& artifact : artifacts) {
        const auto dir = normalGridsCacheDir(remoteCacheRoot, sampleId, volume.id,
                                             artifact.sourceCoordinateLevel,
                                             artifact.url);
        std::error_code ec;
        bool artifactComplete = false;
        const auto completeMarker = dir / kCompleteMarker;
        if (std::filesystem::is_regular_file(completeMarker, ec)) {
            try {
                const auto marker = nlohmann::json::parse(std::ifstream(completeMarker));
                const auto bytes = marker.value("bytes", std::uint64_t{0});
                state.totalBytes += bytes;
                state.cachedBytes += bytes;
                state.cachedFiles += marker.value("files", 0);
                artifactComplete = true;
            } catch (...) {
            }
        }
        state.complete = state.complete && artifactComplete;
        if (artifactComplete || !std::filesystem::is_directory(dir, ec))
            continue;
        for (auto it = std::filesystem::recursive_directory_iterator(
                 dir, std::filesystem::directory_options::skip_permission_denied, ec);
             it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
            if (ec)
                break;
            if (!it->is_regular_file(ec))
                continue;
            const auto name = it->path().filename().string();
            if (name == vc::core::util::kNormalGridsRemoteMarker ||
                name.ends_with(".missing") ||
                name.find(".tmp-") != std::string::npos)
                continue;
            state.cachedBytes += it->file_size(ec);
            ++state.cachedFiles;
        }
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

    const auto finalDir = normalGridsCacheDir(remoteCacheRoot, info.sampleId, info.volumeId,
                                              info.sourceCoordinateLevel, info.url);
    std::error_code completeEc;
    if (std::filesystem::is_regular_file(finalDir / kCompleteMarker, completeEc)) {
        return finalDir;
    }

    const auto remote = vcr::s3::splitPrefixUrl(info.url);
    if (!remote || remote->key.empty()) {
        if (errorOut) *errorOut = "unsupported normal grids URL: " + info.url;
        return {};
    }

    NormalGridsDownloadProgress state;
    state.status = "listing";
    if (progress) {
        progress(state);
    }

    const auto objects = listRemotePrefix(*remote, cancelRequested, errorOut);
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
                    const auto url = vcr::s3::objectUrl(remote->origin, remote->key + object.relativePath);
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

    if (!vc::core::util::hasNormalGridFiles(finalDir)) {
        if (errorOut) {
            *errorOut = "downloaded data at " + finalDir.string() +
                        " is not a normal grid store (missing xy/xz/yz or metadata.json)";
        }
        return {};
    }

    // Keep the dir streamable (harmless once complete) and record completion
    // so the catalog can report "all downloaded" without a remote listing.
    vc::core::util::writeNormalGridsRemoteMarker(finalDir, info.url);
    writeCompleteMarker(finalDir, static_cast<int>(objects.size()), state.totalBytes);

    Logger()->info("Downloaded normal grids for {}/{} to {}",
                   info.sampleId, info.volumeId, finalDir.string());
    return finalDir;
}

} // namespace vc3d::opendata
