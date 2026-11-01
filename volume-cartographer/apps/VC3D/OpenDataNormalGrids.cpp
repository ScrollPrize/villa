#include "OpenDataNormalGrids.hpp"

#include "OpenDataManifest.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/RemoteCacheSettings.hpp"

#include <nlohmann/json.hpp>

#include <vc_reader/s3_listing.hpp>
#include <vc_reader/source_reader.hpp>
#include <vc_reader/zarr.hpp>

#include <algorithm>
#include <fstream>
#include <mutex>
#include <thread>

namespace vc3d::opendata {
namespace {

// Where the store's URL caches under the root, as every vc_reader cache does.
std::filesystem::path normalGridsCacheDir(const std::filesystem::path& remoteCacheRoot, const std::string& url)
{
    return vcr::zarr::decodedCachePath(remoteCacheRoot, url);
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
    const auto dir = normalGridsCacheDir(remoteCacheRoot, info.url);
    if (!vc::core::util::writeNormalGridsRemoteMarker(dir, info.url)) {
        Logger()->warn("Failed to create streaming normal-grid cache at {}", dir.string());
        return false;
    }
    // addNormalGridEntry dedups by location; either way the entry ends up
    // attached. The marker's URL names its catalog dataset (normalGridsCatalogView).
    pkg.addNormalGridEntry(dir.string());
    // The same store attached from another cache directory goes.
    std::vector<std::string> stale;
    for (const auto& entry : pkg.normalGridEntries()) {
        if (entry.location != dir.string() && !vcr::zarr::isRemote(entry.location) &&
            vc::core::util::normalGridsRemoteUrl(
                vc::project::resolveLocalPath(entry.location, pkg.path().parent_path())) == info.url)
            stale.push_back(entry.location);
    }
    for (const auto& location : stale) pkg.removeEntry(location);
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
        const auto dir = normalGridsCacheDir(remoteCacheRoot, artifact.url);
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
            // The store's own files: grids and metadata. Not the markers, nor whatever
            // the file cache has in flight.
            const auto name = it->path().filename().string();
            if (!name.ends_with(".grid") && !(name.starts_with("metadata") && name.ends_with(".json")))
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

    const auto finalDir = normalGridsCacheDir(remoteCacheRoot, info.url);
    std::error_code completeEc;
    if (std::filesystem::is_regular_file(finalDir / kCompleteMarker, completeEc)) {
        return finalDir;
    }
    const auto cancelled = [&] {
        return cancelRequested && cancelRequested->load(std::memory_order_acquire);
    };

    NormalGridsDownloadProgress state;
    state.status = "listing";
    if (progress) {
        progress(state);
    }

    const auto access = vcr::zarr::resolveAccess(info.url);
    std::vector<vcr::s3::Object> objects;
    try {
        objects = vcr::zarr::listObjects(access, info.url);
    } catch (const std::exception& e) {
        if (errorOut) *errorOut = std::string("listing failed: ") + e.what();
        return {};
    }
    std::erase_if(objects, [](const vcr::s3::Object& object) {
        return object.key.empty() || object.key.back() == '/';
    });
    if (objects.empty()) {
        if (errorOut) *errorOut = "no objects found under " + info.url;
        return {};
    }
    if (cancelled()) {
        if (errorOut) *errorOut = "cancelled";
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

    // Each object is fetched by vc_reader's file cache to its place under the root, which
    // is where streaming puts it too, under the root's byte limit and free-space floor.
    // Files already there are kept, so a cancelled or failed download resumes next time,
    // and one download serves every worker or tracer asking for the same file.
    vcr::sources::EncodedFiles files([root = remoteCacheRoot, access](const std::string&) {
        auto options = vc::settings::remoteOpenOptions(root);
        options.access = access;
        return options;
    });

    const int workerCount = std::clamp(workers, 1, kNormalGridsDownloadWorkers);
    std::atomic<std::size_t> next{0};
    std::atomic<int> completed{0};
    std::atomic<int> failed{0};
    std::atomic<std::uint64_t> completedBytes{0};
    std::mutex progressMutex;
    std::string firstError;

    auto worker = [&]() {
        while (!cancelled()) {
            const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= objects.size()) {
                return;
            }
            const auto& object = objects[index];
            try {
                (void)files.open(info.url + "/" + object.key);
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
                update.fileName = object.key;
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

    if (cancelled() || failed.load() > 0) {
        if (errorOut) {
            *errorOut = cancelled()
                ? "cancelled"
                : ("failed to download " + std::to_string(failed.load()) +
                   " file(s): " + firstError);
        }
        return {};
    }

    if (!vc::core::util::hasNormalGridFiles(finalDir)) {
        if (errorOut) {
            *errorOut = "downloaded data at " + finalDir.string() +
                        " is not a normal grid store (missing xy/xz/yz or metadata.json)";
        }
        return {};
    }

    // Markers an older VC3D's streaming left for absent files: nothing reads them now.
    std::error_code ec;
    for (auto it = std::filesystem::recursive_directory_iterator(
             finalDir, std::filesystem::directory_options::skip_permission_denied, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_regular_file(ec) && it->path().filename().string().ends_with(".missing")) {
            std::filesystem::remove(it->path(), ec);
        }
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
