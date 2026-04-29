#include "vc/core/util/RemoteScroll.hpp"

#include <chrono>
#include <fstream>

#include "utils/Json.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/Logging.hpp"

namespace vc {

RemoteScrollInfo discoverRemoteScroll(const std::string& httpsUrl, const cache::HttpAuth& auth)
{
    // Normalize: ensure trailing slash
    std::string baseUrl = httpsUrl;
    while (!baseUrl.empty() && baseUrl.back() == '/') baseUrl.pop_back();

    RemoteScrollInfo info;
    info.baseUrl = baseUrl;
    info.auth = auth;

    // Probe volumes/
    Logger()->info("[RemoteScroll] Probing {}/volumes/", baseUrl);
    auto volList = cache::s3ListObjects(baseUrl + "/volumes/", auth);

    // If the very first request has an auth error, bail out early
    if (volList.authError) {
        info.authError = true;
        info.authErrorMessage = volList.errorMessage;
        Logger()->error("[RemoteScroll] Auth error: {}", volList.errorMessage);
        return info;
    }

    for (const auto& name : volList.prefixes) {
        Logger()->info("[RemoteScroll]   volume: {}", name);
        info.volumeNames.push_back(name);
    }

    // Probe paths/ first (full volpkg format)
    Logger()->info("[RemoteScroll] Probing {}/paths/", baseUrl);
    auto pathsList = cache::s3ListObjects(baseUrl + "/paths/", auth);
    if (!pathsList.prefixes.empty()) {
        info.segmentSource = RemoteSegmentSource::Paths;
        for (const auto& name : pathsList.prefixes) {
            Logger()->info("[RemoteScroll]   path segment: {}", name);
            info.segmentIds.push_back(name);
        }
    } else {
        // Fall back to segments/ (lite format)
        Logger()->info("[RemoteScroll] Probing {}/segments/", baseUrl);
        auto segList = cache::s3ListObjects(baseUrl + "/segments/", auth);
        info.segmentSource = RemoteSegmentSource::Segments;
        for (const auto& name : segList.prefixes) {
            Logger()->info("[RemoteScroll]   segment: {}", name);
            info.segmentIds.push_back(name);
        }
    }

    Logger()->info("[RemoteScroll] Found {} volumes, {} segments (source: {})",
                 info.volumeNames.size(), info.segmentIds.size(),
                 info.segmentSource == RemoteSegmentSource::Paths ? "paths" : "segments");

    return info;
}

namespace {

std::filesystem::path remoteSegmentLocalDir(
    const std::filesystem::path& cacheDir,
    const std::string& segmentId,
    RemoteSegmentSource source)
{
    const char* subdir = (source == RemoteSegmentSource::Segments) ? "segments" : "paths";
    return cacheDir / subdir / segmentId;
}

std::string remoteSegmentBaseUrl(
    const std::string& baseUrl,
    const std::string& segmentId,
    RemoteSegmentSource source)
{
    if (source == RemoteSegmentSource::Direct) return baseUrl + "/" + segmentId + "/";
    if (source == RemoteSegmentSource::Paths)  return baseUrl + "/paths/" + segmentId + "/";
    return baseUrl + "/segments/" + segmentId + "/mesh/tifxyz/";
}

void patchSegmentMeta(const std::filesystem::path& metaPath, const std::string& segmentId)
{
    if (!std::filesystem::exists(metaPath)) return;
    try {
        auto meta = utils::Json::parse_file(metaPath);
        bool patched = false;
        if (!meta.contains("type"))   { meta["type"]   = "seg";     patched = true; }
        if (!meta.contains("uuid"))   { meta["uuid"]   = segmentId; patched = true; }
        if (!meta.contains("format")) { meta["format"] = "tifxyz";  patched = true; }
        if (patched) {
            std::ofstream ofs(metaPath);
            ofs << meta.dump(2);
        }
    } catch (const std::exception& e) {
        Logger()->warn("[RemoteScroll] Failed to patch meta.json: {}", e.what());
    }
}

std::filesystem::path downloadRemoteSegmentFiles(
    const std::string& baseUrl,
    const std::string& segmentId,
    const std::filesystem::path& cacheDir,
    const cache::HttpAuth& auth,
    RemoteSegmentSource source,
    const std::vector<std::string>& files)
{
    namespace fs = std::filesystem;
    auto localDir = remoteSegmentLocalDir(cacheDir, segmentId, source);
    fs::create_directories(localDir);

    bool allExist = true;
    for (const auto& f : files) {
        if (!fs::exists(localDir / f)) { allExist = false; break; }
    }
    if (allExist) return localDir;

    auto failMarker = localDir / ".download_failed";
    if (fs::exists(failMarker)) return localDir;

    const std::string remoteBase = remoteSegmentBaseUrl(baseUrl, segmentId, source);
    for (const auto& f : files) {
        auto localPath = localDir / f;
        if (fs::exists(localPath)) continue;
        const std::string url = remoteBase + f;
        Logger()->info("[RemoteScroll]   Downloading {} -> {}", url, localPath.string());
        if (!cache::httpDownloadFile(url, localPath, auth)) {
            Logger()->error("[RemoteScroll]   FAILED to download {}", f);
        }
    }

    auto metaPath = localDir / "meta.json";
    if (!fs::exists(metaPath)) {
        Logger()->warn("[RemoteScroll] Segment {} download failed, marking to skip", segmentId);
        std::ofstream marker(failMarker);
        marker << "Download failed at "
               << std::chrono::system_clock::now().time_since_epoch().count() << "\n";
        return localDir;
    }
    patchSegmentMeta(metaPath, segmentId);
    return localDir;
}

} // namespace

std::filesystem::path downloadRemoteSegment(
    const std::string& baseUrl,
    const std::string& segmentId,
    const std::filesystem::path& cacheDir,
    const cache::HttpAuth& auth,
    RemoteSegmentSource source)
{
    return downloadRemoteSegmentFiles(
        baseUrl, segmentId, cacheDir, auth, source,
        {"meta.json", "x.tif", "y.tif", "z.tif"});
}

std::filesystem::path downloadRemoteSegmentMetadataOnly(
    const std::string& baseUrl,
    const std::string& segmentId,
    const std::filesystem::path& cacheDir,
    const cache::HttpAuth& auth,
    RemoteSegmentSource source)
{
    return downloadRemoteSegmentFiles(
        baseUrl, segmentId, cacheDir, auth, source, {"meta.json"});
}

bool isRemoteSegmentFullyCached(
    const std::filesystem::path& cacheDir,
    const std::string& segmentId,
    RemoteSegmentSource source)
{
    namespace fs = std::filesystem;
    const char* subdir = (source == RemoteSegmentSource::Segments) ? "segments" : "paths";
    auto localDir = cacheDir / subdir / segmentId;

    for (const auto& f : {"meta.json", "x.tif", "y.tif", "z.tif"}) {
        if (!fs::exists(localDir / f)) {
            return false;
        }
    }
    return true;
}

}  // namespace vc
