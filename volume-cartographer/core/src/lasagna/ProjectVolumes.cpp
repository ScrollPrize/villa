#include "vc/lasagna/ProjectVolumes.hpp"

#include "vc/core/render/ChunkFetch.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "utils/zarr.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace vc::lasagna
{
namespace
{

class LasagnaChannelChunkFetcher final : public vc::render::IChunkFetcher
{
public:
    LasagnaChannelChunkFetcher(int level, std::shared_ptr<utils::ZarrArray> array)
        : level_(level),
          array_(std::move(array))
    {
    }

    vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey& key) override
    {
        vc::render::ChunkFetchResult result;
        if (key.level != level_ || key.iz < 0 || key.iy < 0 || key.ix < 0) {
            result.status = vc::render::ChunkFetchStatus::Missing;
            return result;
        }
        try {
            const std::array<std::size_t, 3> sourceKey{
                static_cast<std::size_t>(key.iz),
                static_cast<std::size_t>(key.iy),
                static_cast<std::size_t>(key.ix),
            };
            auto source = array_->read_chunk(sourceKey);
            if (!source) {
                result.status = vc::render::ChunkFetchStatus::Missing;
                return result;
            }
            result.status = vc::render::ChunkFetchStatus::Found;
            result.bytes = std::move(*source);
            return result;
        } catch (const std::exception& error) {
            result.status = vc::render::ChunkFetchStatus::DecodeError;
            result.message = error.what();
            return result;
        }
    }

private:
    int level_ = 0;
    std::shared_ptr<utils::ZarrArray> array_;
};

vc::render::ChunkDtype chunkDtype(utils::ZarrDtype dtype)
{
    if (dtype == utils::ZarrDtype::uint8)
        return vc::render::ChunkDtype::UInt8;
    if (dtype == utils::ZarrDtype::uint16)
        return vc::render::ChunkDtype::UInt16;
    throw std::runtime_error("Lasagna project volumes support only uint8 and uint16 channel data");
}

struct ChannelDescriptor {
    std::array<int, 3> shapeZYX{};
    std::array<int, 3> chunksZYX{};
    vc::render::ChunkDtype dtype = vc::render::ChunkDtype::UInt8;
    double fillValue = 0.0;
};

ChannelDescriptor describeChannel(const utils::ZarrArray& array, const LasagnaChannelGroup& group)
{
    const auto& meta = array.metadata();
    ChannelDescriptor descriptor;
    descriptor.dtype = chunkDtype(meta.dtype);
    descriptor.fillValue = meta.fill_value.value_or(0.0);
    if (meta.shape.size() != 3 || meta.chunks.size() != 3)
        throw std::runtime_error("Lasagna project group '" + group.name + "' must reference a 3D (Z,Y,X) array");
    descriptor.shapeZYX = {static_cast<int>(meta.shape[0]), static_cast<int>(meta.shape[1]), static_cast<int>(meta.shape[2])};
    descriptor.chunksZYX = {static_cast<int>(meta.chunks[0]), static_cast<int>(meta.chunks[1]), static_cast<int>(meta.chunks[2])};
    return descriptor;
}

vc::render::IChunkedArray::LevelTransform dyadicTransform(int level)
{
    vc::render::IChunkedArray::LevelTransform transform;
    const double invScale = 1.0 / static_cast<double>(std::uint64_t{1} << level);
    transform.scaleFromLevel0 = {invScale, invScale, invScale};
    return transform;
}

}  // namespace

std::vector<PreparedLasagnaProjectVolume> prepareLasagnaProjectVolumes(
    const LasagnaDataset& dataset,
    std::string manifestLocation)
{
    const auto& manifest = dataset.manifest();
    if (manifestLocation.empty()) {
        manifestLocation = manifest.manifestLocation.empty()
            ? manifest.manifestPath.string()
            : manifest.manifestLocation;
    }
    const std::string provenanceTag =
        std::string(kLasagnaDerivedVolumeTagPrefix) + manifestLocation;
    std::vector<PreparedLasagnaProjectVolume> prepared;

    for (const auto& group : manifest.groups) {
        if (group.channels.size() != 1)
            throw std::runtime_error("Lasagna project group '" + group.name + "' must describe exactly one 3D channel");
        const auto& channel = group.channels.front();
        const std::string location = lasagnaGroupSourceLocation(group);
        const double spacing = static_cast<double>(group.scaleFactor()) * manifest.sourceToBase / manifest.workingToBaseScale;
        if (auto existing = std::find_if(
                prepared.begin(), prepared.end(), [&](const auto& candidate) {
                    return candidate.location == location;
                }); existing != prepared.end()) {
            if (existing->volume->voxelSize() != spacing) {
                throw std::runtime_error(
                    "Lasagna groups sharing source '" + location +
                    "' must use one spacing");
            }
            continue;
        }
        auto initialArray = std::make_shared<utils::ZarrArray>(openLasagnaChannelArray(manifest, group, 1));
        auto descriptor = describeChannel(*initialArray, group);
        if (descriptor.dtype == vc::render::ChunkDtype::UInt16) {
            initialArray = std::make_shared<utils::ZarrArray>(openLasagnaChannelArray(manifest, group, 2));
            descriptor = describeChannel(*initialArray, group);
        }

        const auto manifestCopy = manifest;
        const auto groupCopy = group;
        const auto sourceFactory = [manifestCopy, groupCopy, descriptor]() mutable {
            const auto bytes = descriptor.dtype == vc::render::ChunkDtype::UInt16 ? std::size_t{2} : std::size_t{1};
            auto array = std::make_shared<utils::ZarrArray>(openLasagnaChannelArray(manifestCopy, groupCopy, bytes));
            vc::render::OpenedChunkedZarr opened;
            const int level = groupCopy.scaledown;
            const std::size_t count = static_cast<std::size_t>(level) + 1;
            opened.levelNumbers.resize(count);
            opened.shapes.resize(count, {0, 0, 0});
            opened.chunkShapes.resize(count, {1, 1, 1});
            opened.storageChunkShapes.resize(count, {1, 1, 1});
            opened.transforms.resize(count);
            opened.fetchers.resize(count);
            opened.fillValues.resize(count, descriptor.fillValue);
            for (int i = 0; i <= level; ++i) {
                opened.levelNumbers[static_cast<std::size_t>(i)] = i;
                opened.transforms[static_cast<std::size_t>(i)] = dyadicTransform(i);
            }
            const auto index = static_cast<std::size_t>(level);
            opened.shapes[index] = descriptor.shapeZYX;
            opened.chunkShapes[index] = descriptor.chunksZYX;
            opened.storageChunkShapes[index] = descriptor.chunksZYX;
            opened.fetchers[index] =
                std::make_shared<LasagnaChannelChunkFetcher>(level, std::move(array));
            opened.fillValue = descriptor.fillValue;
            opened.dtype = descriptor.dtype;
            return opened;
        };

        utils::Json metadata = utils::Json::object();
        metadata["uuid"] = "lasagna:" + location;
        metadata["name"] = group.name + ":" + channel;
        metadata["voxelsize"] = spacing;
        if (manifest.baseShapeZYX) {
            metadata["slices"] = static_cast<int>((*manifest.baseShapeZYX)[0]);
            metadata["height"] = static_cast<int>((*manifest.baseShapeZYX)[1]);
            metadata["width"] = static_cast<int>((*manifest.baseShapeZYX)[2]);
        }
        prepared.push_back({
            location,
            {provenanceTag},
            Volume::NewFromPreparedChunkedSource(sourceFactory, metadata),
        });
    }
    return prepared;
}

std::vector<std::string> reconcileLasagnaProjectVolumes(VolumePkg& package)
{
    std::vector<std::string> diagnostics;
    const auto entries = package.allLasagnaDatasetEntries();
    for (const auto& entry : entries) {
        try {
            LasagnaDatasetOpenOptions options;
            options.remoteCacheRoot = package.remoteCacheRootOrEmpty();
            const std::string resolved = vc::project::isLocationRemote(entry.location)
                                             ? entry.location
                                             : vc::project::resolveLocalPath(entry.location, package.path().parent_path()).string();
            const auto dataset = LasagnaDataset::openLocation(resolved, options);
            std::vector<VolumePkg::PreparedVolumeAttachment> volumes;
            for (auto& prepared : prepareLasagnaProjectVolumes(dataset, entry.location)) {
                volumes.push_back({std::move(prepared.location), std::move(prepared.tags), std::move(prepared.volume)});
            }
            const auto result =
                package.attachPreparedLasagnaDataset(entry.location, entry.tags, vc::project::isFiberLasagnaEntry(entry), volumes, options.remoteCacheRoot, false, false);
            if (result == VolumePkg::AttachLasagnaResult::VolumeIdConflict) {
                diagnostics.push_back(entry.location + ": derived volume id conflict");
            }
        } catch (const std::exception& error) {
            diagnostics.push_back(entry.location + ": " + error.what());
        }
    }
    return diagnostics;
}

}  // namespace vc::lasagna
