#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <array>

#include "vc/core/util/RemoteAuth.hpp"

namespace vc::cache {

// Shard configuration for zarr v3 sharded storage.
struct ShardConfig {
    bool enabled = false;
    std::array<int, 3> shardShape = {0, 0, 0};
};

struct RemoteZarrInfo {
    std::string url;
    std::filesystem::path stagingDir;
    std::string delimiter = ".";
    int numLevels = 0;
    ShardConfig shardConfig;
};

// Result of S3 ListObjectsV2 with delimiter.
struct S3ListResult {
    std::vector<std::string> prefixes;
    std::vector<std::string> objects;
    bool authError = false;
    std::string errorMessage;
};

// Normalize a remote zarr URL for cache keying and source matching.
std::string normalizeRemoteUrl(const std::string& url);

// Derive a stable cache ID for a remote zarr URL.
std::string deriveRemoteVolumeId(const std::string& url);

// Fetch zarr metadata from a remote URL, write to local staging dir.
RemoteZarrInfo fetchRemoteZarrMetadata(
    const std::string& url,
    const std::filesystem::path& stagingRoot,
    const HttpAuth& auth = {});

// Fetch URL body as string. Empty on failure. Throws on auth errors.
std::string httpGetString(const std::string& url, const HttpAuth& auth = {});

// List objects under an S3 prefix using ListObjectsV2.
S3ListResult s3ListObjects(const std::string& httpsBaseUrl, const HttpAuth& auth = {});

// Download a URL to a local file. Returns true on success.
bool httpDownloadFile(const std::string& url, const std::filesystem::path& dest, const HttpAuth& auth = {});

}  // namespace vc::cache
