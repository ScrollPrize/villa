#include "vc/core/render/ChunkFetch.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

constexpr std::size_t MAXIMUM_WORKERS = 8;

struct Bounds {
    int zMin = 0;
    int zMax = 0;
    int yMin = 0;
    int yMax = 0;
    int xMin = 0;
    int xMax = 0;
};

void validateBounds(const Bounds& bounds)
{
    if (bounds.zMin < 0 || bounds.yMin < 0 || bounds.xMin < 0)
        throw std::invalid_argument("minimum coordinates must be non-negative");
    if (bounds.zMax < bounds.zMin || bounds.yMax < bounds.yMin || bounds.xMax < bounds.xMin) {
        throw std::invalid_argument(
            "each maximum coordinate must be greater than or equal to its minimum");
    }
}

std::vector<vc::render::ChunkKey> chunkKeysForBounds(
    int level,
    const Bounds& requested,
    const std::array<int, 3>& shape,
    const std::array<int, 3>& chunkShape,
    Bounds& clamped)
{
    if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0)
        throw std::runtime_error("requested level has an empty shape");
    if (chunkShape[0] <= 0 || chunkShape[1] <= 0 || chunkShape[2] <= 0)
        throw std::runtime_error("requested level has an invalid chunk shape");
    if (requested.zMin >= shape[0] ||
        requested.yMin >= shape[1] ||
        requested.xMin >= shape[2]) {
        throw std::out_of_range("requested region does not intersect the volume");
    }

    clamped = requested;
    clamped.zMax = std::min(clamped.zMax, shape[0] - 1);
    clamped.yMax = std::min(clamped.yMax, shape[1] - 1);
    clamped.xMax = std::min(clamped.xMax, shape[2] - 1);

    const int chunkZMin = clamped.zMin / chunkShape[0];
    const int chunkZMax = clamped.zMax / chunkShape[0];
    const int chunkYMin = clamped.yMin / chunkShape[1];
    const int chunkYMax = clamped.yMax / chunkShape[1];
    const int chunkXMin = clamped.xMin / chunkShape[2];
    const int chunkXMax = clamped.xMax / chunkShape[2];

    const auto zCount = static_cast<std::size_t>(chunkZMax) - static_cast<std::size_t>(chunkZMin) + 1;
    const auto yCount = static_cast<std::size_t>(chunkYMax) - static_cast<std::size_t>(chunkYMin) + 1;
    const auto xCount = static_cast<std::size_t>(chunkXMax) - static_cast<std::size_t>(chunkXMin) + 1;
    if (zCount > std::numeric_limits<std::size_t>::max() / yCount ||
        zCount * yCount > std::numeric_limits<std::size_t>::max() / xCount) {
        throw std::overflow_error("requested region contains too many chunks");
    }

    std::vector<vc::render::ChunkKey> keys;
    keys.reserve(zCount * yCount * xCount);
    for (int chunkZ = chunkZMin; chunkZ <= chunkZMax; ++chunkZ) {
        for (int chunkY = chunkYMin; chunkY <= chunkYMax; ++chunkY) {
            for (int chunkX = chunkXMin; chunkX <= chunkXMax; ++chunkX)
                keys.push_back({level, chunkZ, chunkY, chunkX});
        }
    }
    return keys;
}

std::string formatBytes(std::uint64_t bytes)
{
    constexpr std::array<const char*, 5> UNITS{"B", "KiB", "MiB", "GiB", "TiB"};
    auto value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < UNITS.size()) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 2)
        << value << ' ' << UNITS[unit];
    return out.str();
}

std::uint64_t voxelCount(const Bounds& bounds)
{
    const auto z = static_cast<std::uint64_t>(bounds.zMax - bounds.zMin) + 1;
    const auto y = static_cast<std::uint64_t>(bounds.yMax - bounds.yMin) + 1;
    const auto x = static_cast<std::uint64_t>(bounds.xMax - bounds.xMin) + 1;
    if (z > std::numeric_limits<std::uint64_t>::max() / y ||
        z * y > std::numeric_limits<std::uint64_t>::max() / x) {
        throw std::overflow_error("requested region contains too many voxels");
    }
    return z * y * x;
}

} // namespace

int main(int argc, char** argv)
{
    fs::path projectPath;
    std::string url;
    int level = 0;
    Bounds requested;
    bool dryRun = false;

    po::options_description options("vc_zarr_download_region options");
    options.add_options()
        ("help,h", "Show help")
        ("project", po::value<fs::path>(&projectPath)->required(), "VC3D project file (*.volpkg.json); supplies remote_cache_root")
        ("url", po::value<std::string>(&url)->required(), "HTTP/S3 OME-Zarr root or concrete array URL")
        ("dry-run,n", po::bool_switch(&dryRun), "Print the cache destination and uncompressed region size without downloading")
        ("level,l", po::value<int>(&level)->default_value(0), "Logical pyramid level; coordinates are measured in this level")
        ("zmin", po::value<int>(&requested.zMin)->required(), "Inclusive minimum Z voxel")
        ("zmax", po::value<int>(&requested.zMax)->required(), "Inclusive maximum Z voxel")
        ("ymin", po::value<int>(&requested.yMin)->required(), "Inclusive minimum Y voxel")
        ("ymax", po::value<int>(&requested.yMax)->required(), "Inclusive maximum Y voxel")
        ("xmin", po::value<int>(&requested.xMin)->required(), "Inclusive minimum X voxel")
        ("xmax", po::value<int>(&requested.xMax)->required(), "Inclusive maximum X voxel");

    try {
        po::variables_map parsed;
        po::store(po::parse_command_line(argc, argv, options), parsed);
        if (parsed.contains("help")) {
            std::cout
                << "Usage: vc_zarr_download_region --project PROJECT --url URL "
                   "--level LEVEL --zmin Z --zmax Z --ymin Y --ymax Y "
                   "--xmin X --xmax X [--dry-run]\n\n"
                << "Bounds are inclusive voxel coordinates at the requested level.\n\n"
                << options << '\n';
            return 0;
        }
        po::notify(parsed);

        if (level < 0)
            throw std::invalid_argument("--level must be non-negative");
        validateBounds(requested);
        if (!fs::is_regular_file(projectPath))
            throw std::invalid_argument(
                "--project is not a file: " + projectPath.string());

        vc::project::LoadOptions loadOptions;
        loadOptions.deferResolution = true;
        const auto project = VolumePkg::load(projectPath, loadOptions);
        if (!project)
            throw std::runtime_error("failed to load project: " + projectPath.string());

        const fs::path configuredCacheRoot = project->remoteCacheRootOrEmpty();
        if (configuredCacheRoot.empty()) {
            throw std::runtime_error(
                "the project has no remote_cache_root; configure its remote cache "
                "directory in VC3D first");
        }

        const auto spec = vc::parseRemoteVolumeSpec(url);
        const auto projectEntry = project->matchingVolumeEntry(url);
        const fs::path volumeCacheRoot = projectEntry
            ? vc::project::remoteVolumeCacheRootForEntry(configuredCacheRoot, *projectEntry)
            : configuredCacheRoot;
        const bool anonymous = projectEntry && vc::project::usesAnonymousRemoteAuth(*projectEntry);
        const auto metadata = projectEntry
            ? vc::project::volumeMetadataFromEntryTags(projectEntry->tags)
            : utils::Json{};
        std::cout << "Opening "
                  << vc::core::util::redactedRemoteLocation(spec.portableLocator)
                  << '\n';
        auto volume = Volume::NewFromUrl(url, volumeCacheRoot, {}, metadata, !anonymous);
        if (!volume->hasScaleLevel(level)) {
            throw std::out_of_range(
                "requested --level is not present in the Zarr pyramid");
        }

        const auto shape = volume->shape(level);
        const auto chunkShape = volume->chunkShape(level);
        Bounds clamped;
        auto keys = chunkKeysForBounds(
            level, requested, shape, chunkShape, clamped);

        std::cout << "Cache: " << volume->remotePersistentCachePath() << '\n'
                  << "Level " << level
                  << " shape [z,y,x]=[" << shape[0] << ',' << shape[1] << ','
                  << shape[2] << "] chunk=[" << chunkShape[0] << ','
                  << chunkShape[1] << ',' << chunkShape[2] << "]\n"
                  << "Region inclusive [z,y,x]=[" << clamped.zMin << ':'
                  << clamped.zMax << ',' << clamped.yMin << ':' << clamped.yMax
                  << ',' << clamped.xMin << ':' << clamped.xMax << "] covers "
                  << keys.size() << " chunks\n";

        if (dryRun) {
            const auto voxels = voxelCount(clamped);
            const auto bytesPerVoxel = volume->dtypeSize();
            if (bytesPerVoxel != 0 &&
                voxels > std::numeric_limits<std::uint64_t>::max() / bytesPerVoxel) {
                throw std::overflow_error("uncompressed region size exceeds uint64 range");
            }
            const auto bytes = voxels * bytesPerVoxel;
            std::cout << "Region size: " << formatBytes(bytes) << ")\n";
            return 0;
        }

        auto cache = volume->sharedChunkCache();
        std::atomic<std::size_t> nextKey{0};
        std::atomic<std::size_t> persisted{0};
        std::atomic<std::size_t> missing{0};
        std::atomic<std::size_t> errors{0};
        std::mutex errorMutex;
        std::string firstError;
        auto recordError = [&](const std::string& message) {
            errors.fetch_add(1, std::memory_order_relaxed);
            if (!message.empty()) {
                std::lock_guard lock(errorMutex);
                if (firstError.empty())
                    firstError = message;
            }
        };

        const auto fetchConcurrency =
            vc::render::processChunkCacheService()->fetchConcurrency();
        const std::size_t workerCount = std::min<std::size_t>(
            keys.size(),
            std::max<std::size_t>(
                1, std::min(MAXIMUM_WORKERS,
                            fetchConcurrency.maxConcurrentReads)));
        auto worker = [&] {
            while (true) {
                const auto index = nextKey.fetch_add(1, std::memory_order_relaxed);
                if (index >= keys.size())
                    return;
                const auto& key = keys[index];
                try {
                    const auto result = cache->persistChunkBlocking(
                        key.level, key.iz, key.iy, key.ix);
                    switch (result.status) {
                    case vc::render::ChunkCache::PersistentRequestStatus::Data:
                        persisted.fetch_add(1, std::memory_order_relaxed);
                        break;
                    case vc::render::ChunkCache::PersistentRequestStatus::Missing:
                        missing.fetch_add(1, std::memory_order_relaxed);
                        break;
                    case vc::render::ChunkCache::PersistentRequestStatus::Error:
                        recordError(result.error);
                        break;
                    }
                } catch (const std::exception& error) {
                    recordError(error.what());
                } catch (...) {
                    recordError("unknown persistent cache error");
                }
            }
        };

        std::vector<std::jthread> workers;
        workers.reserve(workerCount);
        for (std::size_t i = 0; i < workerCount; ++i)
            workers.emplace_back(worker);
        for (auto& thread : workers)
            thread.join();
        cache->waitForPersistentWrites();

        const auto stats = cache->stats();
        if (!stats.persistentCacheWarning.empty()) {
            throw std::runtime_error(
                "persistent cache warning: " + stats.persistentCacheWarning);
        }

        std::cout << "Done. requested=" << keys.size()
                  << " cached=" << persisted.load()
                  << " missing=" << missing.load()
                  << " errors=" << errors.load() << '\n';
        if (errors.load() != 0) {
            if (!firstError.empty())
                std::cerr << "First cache error: " << firstError << '\n';
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_zarr_download_region error: " << error.what()
                  << "\n\n" << options << '\n';
        return 1;
    }
}
