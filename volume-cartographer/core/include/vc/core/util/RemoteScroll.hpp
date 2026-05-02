#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "vc/core/util/RemoteAuth.hpp"

namespace vc {

// How segment files are laid out relative to their base URL.
//   Paths:    baseUrl/paths/<segId>/{meta.json, x.tif, ...}
//   Segments: baseUrl/segments/<segId>/mesh/tifxyz/{...}
//   Direct:   segmentsBaseUrl/<segId>/{meta.json, x.tif, ...}
enum class RemoteSegmentSource { Paths, Segments, Direct };

struct RemoteScrollInfo {
    std::string baseUrl;                    // HTTPS base URL of the scroll root
    std::vector<std::string> volumeNames;   // e.g. ["20230205180739.zarr"]
    std::vector<std::string> segmentIds;    // e.g. ["20230205180739"]
    RemoteSegmentSource segmentSource = RemoteSegmentSource::Segments;
    std::string segmentsBaseUrl;            // For Direct source: URL containing segment dirs
    vc::HttpAuth auth;
    bool authError = false;                 // true if discovery failed due to auth
    std::string authErrorMessage;           // e.g. "The provided token has expired."
};

// Probe a remote scroll root URL for volumes/ and segments/ subdirectories.
// Returns discovered volume names and segment IDs.
// If the URL doesn't look like a scroll (no volumes/ or segments/), returns
// empty lists.
RemoteScrollInfo discoverRemoteScroll(const std::string& httpsUrl, const vc::HttpAuth& auth);

// Download a single remote segment's tifxyz files to a local cache directory.
// For Paths source: downloads from paths/<segId>/{meta.json, x.tif, y.tif, z.tif}
// For Segments source: downloads from segments/<segId>/mesh/tifxyz/{...}
// Stores to: cacheDir/{paths|segments}/<segId>/
// Skips download if all 4 files already exist locally.
// Returns the local directory containing the downloaded segment files.
std::filesystem::path downloadRemoteSegment(
    const std::string& baseUrl,
    const std::string& segmentId,
    const std::filesystem::path& cacheDir,
    const vc::HttpAuth& auth,
    RemoteSegmentSource source = RemoteSegmentSource::Segments);

// Download only meta.json for a single remote segment (fast, tiny file).
// Used for lazy loading: populate the segment list quickly without downloading
// GBs of TIFF data.  Returns the local directory.  If meta.json already exists
// locally, the download is skipped.
std::filesystem::path downloadRemoteSegmentMetadataOnly(
    const std::string& baseUrl,
    const std::string& segmentId,
    const std::filesystem::path& cacheDir,
    const vc::HttpAuth& auth,
    RemoteSegmentSource source = RemoteSegmentSource::Segments);

// Check whether a segment's TIFF data (x.tif, y.tif, z.tif) is already cached.
bool isRemoteSegmentFullyCached(
    const std::filesystem::path& cacheDir,
    const std::string& segmentId,
    RemoteSegmentSource source);

}  // namespace vc
