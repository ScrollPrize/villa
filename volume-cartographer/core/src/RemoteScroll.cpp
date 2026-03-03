#include "vc/core/util/RemoteScroll.hpp"

#include <fstream>

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

std::filesystem::path downloadRemoteSegment(
    const std::string& baseUrl,
    const std::string& segmentId,
    const std::filesystem::path& cacheDir,
    const cache::HttpAuth& auth,
    RemoteSegmentSource source)
{
    namespace fs = std::filesystem;

    // Local destination: cacheDir/{paths|segments}/<segId>/
    const char* subdir = (source == RemoteSegmentSource::Segments) ? "segments" : "paths";
    auto localDir = cacheDir / subdir / segmentId;
    fs::create_directories(localDir);

    // Files to download from the remote tifxyz directory
    const std::vector<std::string> files = {"meta.json", "x.tif", "y.tif", "z.tif"};

    // Check if all files already exist
    bool allExist = true;
    for (const auto& f : files) {
        if (!fs::exists(localDir / f)) {
            allExist = false;
            break;
        }
    }

    if (allExist) {
        Logger()->info("[RemoteScroll] Segment {} already cached at {}",
                     segmentId, localDir.string());
        return localDir;
    }

    // Build remote base URL depending on source format
    std::string remoteBase;
    if (source == RemoteSegmentSource::Direct) {
        // External URL: baseUrl is already the parent of segment dirs
        remoteBase = baseUrl + "/" + segmentId + "/";
    } else if (source == RemoteSegmentSource::Paths) {
        // Full volpkg: paths/<segId>/<file>
        remoteBase = baseUrl + "/paths/" + segmentId + "/";
    } else {
        // Lite format: segments/<segId>/mesh/tifxyz/<file>
        remoteBase = baseUrl + "/segments/" + segmentId + "/mesh/tifxyz/";
    }

    for (const auto& f : files) {
        auto localPath = localDir / f;
        if (fs::exists(localPath)) {
            Logger()->debug("[RemoteScroll]   {} already exists, skipping", f);
            continue;
        }

        std::string url = remoteBase + f;
        Logger()->info("[RemoteScroll]   Downloading {} -> {}",
                     url, localPath.string());

        if (!cache::httpDownloadFile(url, localPath, auth)) {
            Logger()->error("[RemoteScroll]   FAILED to download {}", f);
            // Continue with other files — partial downloads are handled by the
            // caller checking if meta.json exists
        }
    }

    // Patch meta.json if required fields are missing (safety net for lite format)
    auto metaPath = localDir / "meta.json";
    if (fs::exists(metaPath)) {
        try {
            std::ifstream ifs(metaPath);
            auto meta = nlohmann::json::parse(ifs);
            ifs.close();

            bool patched = false;
            if (!meta.contains("type")) {
                meta["type"] = "seg";
                patched = true;
            }
            if (!meta.contains("uuid")) {
                meta["uuid"] = segmentId;
                patched = true;
            }
            if (!meta.contains("format")) {
                meta["format"] = "tifxyz";
                patched = true;
            }

            if (patched) {
                Logger()->info("[RemoteScroll]   Patched meta.json for segment {}", segmentId);
                std::ofstream ofs(metaPath);
                ofs << meta.dump(2);
            }
        } catch (const std::exception& e) {
            Logger()->warn("[RemoteScroll]   Failed to patch meta.json: {}", e.what());
        }
    }

    return localDir;
}

}  // namespace vc
