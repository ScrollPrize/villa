#pragma once

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkFetch.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/util/RemoteAuth.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace utils {
class Store;
class ZarrArray;
}

namespace vc::render {

struct OpenedChunkedZarr {
    std::vector<int> levelNumbers;
    std::vector<IChunkedArray::LevelTransform> transforms;
    std::vector<std::array<int, 3>> shapes;
    std::vector<std::array<int, 3>> chunkShapes;
    std::vector<std::array<int, 3>> storageChunkShapes;
    std::vector<std::shared_ptr<IChunkFetcher>> fetchers;
    std::vector<double> fillValues;
    double fillValue = 0.0;
    ChunkDtype dtype = ChunkDtype::UInt8;
    // True when the physical /0 OME coordinate transform is absent or an
    // identity scale with zero translation. This survives logical rebasing so
    // catalog prediction/source preflight can enforce prediction identity.
    bool physicalLevelZeroTransformIsIdentity = true;
    std::vector<PersistentCacheMetadataObject> zarrMirrorMetadata;
};

struct RemoteZarrOpenOptions {
    vc::HttpAuth auth;
    // Match Volume::NewFromUrl by discovering AWS credentials for S3 when no
    // explicit credentials were supplied. Disable for forced anonymous reads.
    bool discoverAwsCredentials = true;
};

struct OpenedRemoteChunkedZarr {
    OpenedChunkedZarr opened;
    vc::HttpAuth auth;
    vc::RemoteVolumeSpec spec;
};

OpenedChunkedZarr openLocalZarrPyramid(const std::filesystem::path& root);
OpenedChunkedZarr openHttpZarrPyramid(const std::string& url);
OpenedChunkedZarr openHttpZarrPyramid(
    const std::string& url,
    const vc::HttpAuth& auth,
    std::optional<int> baseScaleLevel = std::nullopt);

// Production remote-volume open policy shared by VC3D and standalone tools:
// resolve the locator, try recognized S3 sources anonymously, then retry with
// discovered credentials only when anonymous access is denied.
OpenedRemoteChunkedZarr openRemoteZarrPyramid(
    const std::string& url,
    RemoteZarrOpenOptions options = {});

// S3 deliberately returns AccessDenied rather than NotFound for a missing key
// when the caller lacks ListBucket. Optional discovery probes must therefore
// accept a generic 403 while preserving explicit credential failures. The
// opener remembers suppressed denials and retries authenticated if no required
// array metadata can ultimately be found.
bool isOptionalRemoteMetadataMiss(
    long status,
    std::string_view key,
    std::string_view responseBody = {});

// Enforce the supported contiguous dyadic VC pyramid contract and make
// physical level baseScaleLevel logical level zero. Exposed for deterministic
// synthetic tests; remote nonzero opens call the same implementation.
OpenedChunkedZarr validateAndRebaseVcPyramid(
    OpenedChunkedZarr opened,
    int baseScaleLevel);

// Map multiscales[0].datasets of the store's .zattrs to (physicalLevel, key)
// pairs. All-numeric dataset paths bind by their value — exporters may publish
// only levels >= some scaledown, and positional binding would register the
// coarse array as full resolution. Exposed for deterministic synthetic tests;
// the remote open uses the same implementation.
std::vector<std::pair<int, std::string>> remoteLevelKeysFromZattrs(
    const std::shared_ptr<utils::Store>& store,
    int firstLevel);

std::unique_ptr<ChunkCache> createChunkCache(
    OpenedChunkedZarr opened,
    std::size_t decodedByteCapacity,
    std::size_t maxConcurrentReads = 16);

// Wrap an already-open scalar 3D Zarr array in the same decoded cache and
// process-wide threaded reader used by VC3D volume rendering.
std::unique_ptr<ChunkCache> createChunkCache(
    std::shared_ptr<utils::ZarrArray> array,
    ChunkCache::Options options = {});
std::unique_ptr<ChunkCache> createChunkCache(
    std::shared_ptr<utils::ZarrArray> array,
    ChunkCache::Options options,
    ChunkCacheService::Options serviceOptions);

// Acquire an already-open scalar array from the process-wide cache service.
// Equal source identities share decoded chunks, scheduling, and persistence.
std::shared_ptr<ChunkCache> acquireProcessChunkCache(
    std::string sourceIdentity,
    std::shared_ptr<utils::ZarrArray> array,
    ChunkCache::Options options = {});

// Packed CZYX arrays share one physical decoded chunk; scalar views only add
// an offset into that payload. Legacy 3D arrays keep the existing fast path.
std::shared_ptr<IChunkedArray> acquireProcessZarrChannel(
    std::string sourceIdentity, std::shared_ptr<utils::ZarrArray> array,
    std::size_t channel, ChunkCache::Options options = {});

// Rebind native scalar/packed channel views to another decoded-cache partition,
// preserving normal cache reads and each view's offset/shape/dtype.
std::shared_ptr<IChunkedArray> rebindChunkCache(
    const std::shared_ptr<IChunkedArray>& channel,
    const std::shared_ptr<ChunkCacheService>& partition);
std::shared_ptr<ChunkCacheService> threadLocalDecodedPartition(
    const std::shared_ptr<ChunkCacheService>& parent,std::size_t bytes);

} // namespace vc::render
