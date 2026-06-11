#include "vc/lasagna/MaxflowGraph.hpp"

#include "utils/zarr.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace vc::lasagna {
namespace {

[[nodiscard]] size_t voxelOffset(
    size_t z,
    size_t y,
    size_t x,
    const std::array<size_t, 3>& shapeZYX)
{
    return (z * shapeZYX[1] + y) * shapeZYX[2] + x;
}

[[nodiscard]] uint64_t checkedVoxelCount(const std::array<size_t, 3>& shapeZYX)
{
    const uint64_t z = static_cast<uint64_t>(shapeZYX[0]);
    const uint64_t y = static_cast<uint64_t>(shapeZYX[1]);
    const uint64_t x = static_cast<uint64_t>(shapeZYX[2]);
    if (z != 0 && y > std::numeric_limits<uint64_t>::max() / z) {
        throw std::runtime_error("Maxflow volume shape overflows uint64");
    }
    const uint64_t zy = z * y;
    if (zy != 0 && x > std::numeric_limits<uint64_t>::max() / zy) {
        throw std::runtime_error("Maxflow volume shape overflows uint64");
    }
    return zy * x;
}

[[nodiscard]] int32_t checkedNodeId(uint64_t value)
{
    if (value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Maxflow graph exceeds int32 node id range required by ECL-MaxFlow");
    }
    return static_cast<int32_t>(value);
}

void checkDirectedEdges(uint64_t value)
{
    if (value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Maxflow graph exceeds int32 edge count required by ECL-MaxFlow");
    }
}

[[nodiscard]] MaxflowBox3 voxelBoxXYZ(size_t z, size_t y, size_t x)
{
    return {
        {static_cast<int64_t>(x), static_cast<int64_t>(y), static_cast<int64_t>(z)},
        {static_cast<int64_t>(x + 1), static_cast<int64_t>(y + 1), static_cast<int64_t>(z + 1)},
    };
}

[[nodiscard]] int64_t clampInt64(int64_t value, int64_t lo, int64_t hi)
{
    return std::min(std::max(value, lo), hi);
}

[[nodiscard]] std::string pointString(const MaxflowDouble3& point)
{
    std::ostringstream out;
    out << '(' << point.x << ", " << point.y << ", " << point.z << ')';
    return out.str();
}

[[nodiscard]] std::string intPointString(const MaxflowInt3& point)
{
    std::ostringstream out;
    out << '(' << point.x << ", " << point.y << ", " << point.z << ')';
    return out.str();
}

[[nodiscard]] std::string boxString(const MaxflowBox3& box)
{
    std::ostringstream out;
    out << '[' << intPointString(box.begin) << " -> " << intPointString(box.end) << ')';
    return out.str();
}

[[nodiscard]] std::string shapeString(const std::array<size_t, 3>& shapeZYX)
{
    std::ostringstream out;
    out << "(z=" << shapeZYX[0]
        << ", y=" << shapeZYX[1]
        << ", x=" << shapeZYX[2]
        << ')';
    return out.str();
}

[[nodiscard]] MaxflowBox3 baseCropForShape(
    const MaxflowManifestBuildOptions& options,
    const std::array<size_t, 3>& shapeZYX,
    double spacingBase)
{
    std::vector<MaxflowDouble3> points;
    points.reserve(std::max<size_t>(2, options.sourcesBase.size() + options.sinksBase.size()));
    if (options.sourcesBase.empty() && options.sinksBase.empty()) {
        points.push_back(options.sourceBase);
        points.push_back(options.sinkBase);
    } else {
        points.insert(points.end(), options.sourcesBase.begin(), options.sourcesBase.end());
        points.insert(points.end(), options.sinksBase.begin(), options.sinksBase.end());
    }
    if (points.empty()) {
        throw std::runtime_error("at least one source and one sink are required");
    }

    MaxflowDouble3 lo = points.front();
    MaxflowDouble3 hi = points.front();
    for (const auto& point : points) {
        lo.x = std::min(lo.x, point.x);
        lo.y = std::min(lo.y, point.y);
        lo.z = std::min(lo.z, point.z);
        hi.x = std::max(hi.x, point.x);
        hi.y = std::max(hi.y, point.y);
        hi.z = std::max(hi.z, point.z);
    }

    const double margin = static_cast<double>(options.marginBaseVoxels);
    MaxflowBox3 crop{
        {
            static_cast<int64_t>(std::floor(lo.x - margin)),
            static_cast<int64_t>(std::floor(lo.y - margin)),
            static_cast<int64_t>(std::floor(lo.z - margin)),
        },
        {
            static_cast<int64_t>(std::ceil(hi.x + margin + 1.0)),
            static_cast<int64_t>(std::ceil(hi.y + margin + 1.0)),
            static_cast<int64_t>(std::ceil(hi.z + margin + 1.0)),
        },
    };
    const MaxflowInt3 maxBase{
        static_cast<int64_t>(std::ceil(static_cast<double>(shapeZYX[2]) * spacingBase)),
        static_cast<int64_t>(std::ceil(static_cast<double>(shapeZYX[1]) * spacingBase)),
        static_cast<int64_t>(std::ceil(static_cast<double>(shapeZYX[0]) * spacingBase)),
    };
    crop.begin.x = clampInt64(crop.begin.x, 0, maxBase.x);
    crop.begin.y = clampInt64(crop.begin.y, 0, maxBase.y);
    crop.begin.z = clampInt64(crop.begin.z, 0, maxBase.z);
    crop.end.x = clampInt64(crop.end.x, crop.begin.x, maxBase.x);
    crop.end.y = clampInt64(crop.end.y, crop.begin.y, maxBase.y);
    crop.end.z = clampInt64(crop.end.z, crop.begin.z, maxBase.z);
    return crop;
}

[[nodiscard]] MaxflowBox3 predCropForBaseCrop(
    const MaxflowBox3& baseCrop,
    const std::array<size_t, 3>& shapeZYX,
    double spacingBase)
{
    MaxflowBox3 pred{
        {
            static_cast<int64_t>(std::floor(static_cast<double>(baseCrop.begin.x) / spacingBase)),
            static_cast<int64_t>(std::floor(static_cast<double>(baseCrop.begin.y) / spacingBase)),
            static_cast<int64_t>(std::floor(static_cast<double>(baseCrop.begin.z) / spacingBase)),
        },
        {
            static_cast<int64_t>(std::ceil(static_cast<double>(baseCrop.end.x) / spacingBase)),
            static_cast<int64_t>(std::ceil(static_cast<double>(baseCrop.end.y) / spacingBase)),
            static_cast<int64_t>(std::ceil(static_cast<double>(baseCrop.end.z) / spacingBase)),
        },
    };
    pred.begin.x = clampInt64(pred.begin.x, 0, static_cast<int64_t>(shapeZYX[2]));
    pred.begin.y = clampInt64(pred.begin.y, 0, static_cast<int64_t>(shapeZYX[1]));
    pred.begin.z = clampInt64(pred.begin.z, 0, static_cast<int64_t>(shapeZYX[0]));
    pred.end.x = clampInt64(pred.end.x, pred.begin.x, static_cast<int64_t>(shapeZYX[2]));
    pred.end.y = clampInt64(pred.end.y, pred.begin.y, static_cast<int64_t>(shapeZYX[1]));
    pred.end.z = clampInt64(pred.end.z, pred.begin.z, static_cast<int64_t>(shapeZYX[0]));
    return pred;
}

struct PredDtBinding {
    const LasagnaChannelGroup* group = nullptr;
    size_t channelIndex = 0;
    utils::ZarrArray array;
    bool hasChannelDimension = false;
    std::array<size_t, 3> shapeZYX{0, 0, 0};
    std::array<size_t, 3> chunksZYX{0, 0, 0};
    double spacingBase = 1.0;
};

[[nodiscard]] PredDtBinding bindPredDt(const LasagnaDatasetManifest& manifest)
{
    const LasagnaChannelGroup* group = manifest.groupForChannel("pred_dt");
    if (group == nullptr) {
        throw std::runtime_error("Lasagna dataset missing required channel 'pred_dt'");
    }
    const auto channelIndex = group->channelIndex("pred_dt");
    if (!channelIndex.has_value()) {
        throw std::runtime_error("Internal Lasagna pred_dt channel lookup failure");
    }

    PredDtBinding binding{
        group,
        *channelIndex,
        utils::ZarrArray::open(group->zarrPath, vc::buildZarrCodecRegistry(1)),
        false,
        {0, 0, 0},
        {0, 0, 0},
        static_cast<double>(group->scaleFactor()) * manifest.sourceToBase,
    };
    const auto& meta = binding.array.metadata();
    if (meta.dtype != utils::ZarrDtype::uint8) {
        throw std::runtime_error("Lasagna channel 'pred_dt' must be uint8");
    }
    if (meta.shape.size() == 3) {
        if (meta.chunks.size() != 3) {
            throw std::runtime_error("Lasagna pred_dt zarr has invalid chunks");
        }
        binding.shapeZYX = {meta.shape[0], meta.shape[1], meta.shape[2]};
        binding.chunksZYX = {meta.chunks[0], meta.chunks[1], meta.chunks[2]};
    } else if (meta.shape.size() == 4) {
        if (meta.chunks.size() != 4 || *channelIndex >= meta.shape[0]) {
            throw std::runtime_error("Lasagna pred_dt zarr has invalid channel dimension");
        }
        binding.hasChannelDimension = true;
        binding.shapeZYX = {meta.shape[1], meta.shape[2], meta.shape[3]};
        binding.chunksZYX = {meta.chunks[1], meta.chunks[2], meta.chunks[3]};
    } else {
        throw std::runtime_error("Lasagna pred_dt zarr must be 3D or channel-first 4D");
    }
    if (binding.chunksZYX[0] == 0 || binding.chunksZYX[1] == 0 || binding.chunksZYX[2] == 0) {
        throw std::runtime_error("Lasagna pred_dt zarr has zero-sized chunks");
    }
    return binding;
}

[[nodiscard]] size_t chunkOffset(
    const PredDtBinding& binding,
    size_t localZ,
    size_t localY,
    size_t localX)
{
    const auto& chunks = binding.array.metadata().chunks;
    if (binding.hasChannelDimension) {
        return (((binding.channelIndex % chunks[0]) * chunks[1] + localZ) * chunks[2] + localY) *
                   chunks[3] +
               localX;
    }
    return (localZ * binding.chunksZYX[1] + localY) * binding.chunksZYX[2] + localX;
}

[[nodiscard]] std::vector<size_t> chunkIndices(
    const PredDtBinding& binding,
    size_t cz,
    size_t cy,
    size_t cx)
{
    if (binding.hasChannelDimension) {
        return {binding.channelIndex / binding.array.metadata().chunks[0], cz, cy, cx};
    }
    return {cz, cy, cx};
}

} // namespace

MaxflowGraph buildMaxflowGraphFromPassability(
    std::span<const uint8_t> passableZYX,
    std::array<size_t, 3> shapeZYX,
    const MaxflowGraphBuildOptions& options)
{
    const auto start = std::chrono::steady_clock::now();
    const uint64_t voxelCount = checkedVoxelCount(shapeZYX);
    if (passableZYX.size() != voxelCount) {
        throw std::runtime_error("Maxflow passability volume size does not match shape");
    }
    if (voxelCount > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("Maxflow volume is too large for this platform");
    }

    MaxflowGraph graph;
    std::vector<int32_t> nodeForVoxel(static_cast<size_t>(voxelCount), -1);

    for (size_t z = 0; z < shapeZYX[0]; ++z) {
        for (size_t y = 0; y < shapeZYX[1]; ++y) {
            for (size_t x = 0; x < shapeZYX[2]; ++x) {
                const size_t off = voxelOffset(z, y, x, shapeZYX);
                if (passableZYX[off] == 0) {
                    continue;
                }
                const int32_t node = checkedNodeId(graph.stats.graphNodes);
                nodeForVoxel[off] = node;
                ++graph.stats.graphNodes;
                ++graph.stats.passableVoxels;
                ++graph.stats.leafVoxels;
                if (options.storeNodeMetadata) {
                    graph.nodeMetadata.push_back({MaxflowNodeKind::Voxel, voxelBoxXYZ(z, y, x), -1});
                }
            }
        }
    }

    graph.stats.uncontractedPassableVoxelGraphNodes = graph.stats.passableVoxels;
    graph.nindex.assign(static_cast<size_t>(graph.stats.graphNodes) + 1, 0);

    uint64_t directedEdges = 0;
    uint64_t undirectedEdges = 0;
    const auto countEdge = [&](size_t off, size_t neighbor) {
        if (nodeForVoxel[neighbor] >= 0) {
            ++graph.nindex[static_cast<size_t>(nodeForVoxel[off]) + 1];
            ++directedEdges;
        }
    };
    for (size_t z = 0; z < shapeZYX[0]; ++z) {
        for (size_t y = 0; y < shapeZYX[1]; ++y) {
            for (size_t x = 0; x < shapeZYX[2]; ++x) {
                const size_t off = voxelOffset(z, y, x, shapeZYX);
                if (nodeForVoxel[off] < 0) {
                    continue;
                }
                if (x > 0) countEdge(off, voxelOffset(z, y, x - 1, shapeZYX));
                if (x + 1 < shapeZYX[2]) countEdge(off, voxelOffset(z, y, x + 1, shapeZYX));
                if (y > 0) countEdge(off, voxelOffset(z, y - 1, x, shapeZYX));
                if (y + 1 < shapeZYX[1]) countEdge(off, voxelOffset(z, y + 1, x, shapeZYX));
                if (z > 0) countEdge(off, voxelOffset(z - 1, y, x, shapeZYX));
                if (z + 1 < shapeZYX[0]) countEdge(off, voxelOffset(z + 1, y, x, shapeZYX));

                if (x + 1 < shapeZYX[2] && nodeForVoxel[voxelOffset(z, y, x + 1, shapeZYX)] >= 0) ++undirectedEdges;
                if (y + 1 < shapeZYX[1] && nodeForVoxel[voxelOffset(z, y + 1, x, shapeZYX)] >= 0) ++undirectedEdges;
                if (z + 1 < shapeZYX[0] && nodeForVoxel[voxelOffset(z + 1, y, x, shapeZYX)] >= 0) ++undirectedEdges;
            }
        }
    }

    checkDirectedEdges(directedEdges);
    for (size_t i = 1; i < graph.nindex.size(); ++i) {
        graph.nindex[i] += graph.nindex[i - 1];
    }

    graph.nlist.assign(static_cast<size_t>(directedEdges), 0);
    graph.capacity.assign(static_cast<size_t>(directedEdges), 1);
    std::vector<uint64_t> cursor = graph.nindex;
    const auto addEdge = [&](size_t off, size_t neighbor) {
        const int32_t dst = nodeForVoxel[neighbor];
        if (dst < 0) {
            return;
        }
        const int32_t src = nodeForVoxel[off];
        graph.nlist[static_cast<size_t>(cursor[static_cast<size_t>(src)]++)] = dst;
    };

    for (size_t z = 0; z < shapeZYX[0]; ++z) {
        for (size_t y = 0; y < shapeZYX[1]; ++y) {
            for (size_t x = 0; x < shapeZYX[2]; ++x) {
                const size_t off = voxelOffset(z, y, x, shapeZYX);
                if (nodeForVoxel[off] < 0) {
                    continue;
                }
                if (x > 0) addEdge(off, voxelOffset(z, y, x - 1, shapeZYX));
                if (x + 1 < shapeZYX[2]) addEdge(off, voxelOffset(z, y, x + 1, shapeZYX));
                if (y > 0) addEdge(off, voxelOffset(z, y - 1, x, shapeZYX));
                if (y + 1 < shapeZYX[1]) addEdge(off, voxelOffset(z, y + 1, x, shapeZYX));
                if (z > 0) addEdge(off, voxelOffset(z - 1, y, x, shapeZYX));
                if (z + 1 < shapeZYX[0]) addEdge(off, voxelOffset(z + 1, y, x, shapeZYX));
            }
        }
    }

    graph.stats.directedEdges = directedEdges;
    graph.stats.undirectedEdges = undirectedEdges;
    graph.stats.totalExternalCapacity = undirectedEdges;
    graph.stats.uncontractedPassableVoxelGraphUndirectedEdges = undirectedEdges;
    graph.stats.averageEdgesPerNode =
        graph.stats.graphNodes == 0 ? 0.0
                                    : static_cast<double>(directedEdges) /
                                          static_cast<double>(graph.stats.graphNodes);
    graph.stats.contractionRatio = graph.stats.graphNodes == 0 ? 0.0 : 1.0;
    graph.stats.buildTime = std::chrono::steady_clock::now() - start;
    return graph;
}

MaxflowPredDtVolume loadPredDtPassability(
    const LasagnaDatasetManifest& manifest,
    const MaxflowManifestBuildOptions& options)
{
    if (options.marginBaseVoxels < 0) {
        throw std::runtime_error("margin-base-voxels must be non-negative");
    }
    const PredDtBinding binding = bindPredDt(manifest);
    MaxflowPredDtVolume volume;
    volume.zarrPath = binding.group->zarrPath;
    volume.shapeZYX = binding.shapeZYX;
    volume.chunksZYX = binding.chunksZYX;
    volume.spacingBase = binding.spacingBase;
    volume.cropBaseXYZ = baseCropForShape(options, binding.shapeZYX, binding.spacingBase);
    volume.cropPredXYZ = predCropForBaseCrop(volume.cropBaseXYZ, binding.shapeZYX, binding.spacingBase);

    const size_t cropX = static_cast<size_t>(volume.cropPredXYZ.end.x - volume.cropPredXYZ.begin.x);
    const size_t cropY = static_cast<size_t>(volume.cropPredXYZ.end.y - volume.cropPredXYZ.begin.y);
    const size_t cropZ = static_cast<size_t>(volume.cropPredXYZ.end.z - volume.cropPredXYZ.begin.z);
    volume.passable.assign(cropZ * cropY * cropX, 0);
    if (cropX == 0 || cropY == 0 || cropZ == 0) {
        return volume;
    }

    const size_t z0 = static_cast<size_t>(volume.cropPredXYZ.begin.z);
    const size_t y0 = static_cast<size_t>(volume.cropPredXYZ.begin.y);
    const size_t x0 = static_cast<size_t>(volume.cropPredXYZ.begin.x);
    const size_t z1 = static_cast<size_t>(volume.cropPredXYZ.end.z);
    const size_t y1 = static_cast<size_t>(volume.cropPredXYZ.end.y);
    const size_t x1 = static_cast<size_t>(volume.cropPredXYZ.end.x);

    const size_t cz0 = z0 / binding.chunksZYX[0];
    const size_t cy0 = y0 / binding.chunksZYX[1];
    const size_t cx0 = x0 / binding.chunksZYX[2];
    const size_t cz1 = (z1 - 1) / binding.chunksZYX[0];
    const size_t cy1 = (y1 - 1) / binding.chunksZYX[1];
    const size_t cx1 = (x1 - 1) / binding.chunksZYX[2];

    for (size_t cz = cz0; cz <= cz1; ++cz) {
        for (size_t cy = cy0; cy <= cy1; ++cy) {
            for (size_t cx = cx0; cx <= cx1; ++cx) {
                const auto indices = chunkIndices(binding, cz, cy, cx);
                const auto chunk = binding.array.read_chunk(indices);
                if (!chunk.has_value()) {
                    continue;
                }
                const size_t chunkBaseZ = cz * binding.chunksZYX[0];
                const size_t chunkBaseY = cy * binding.chunksZYX[1];
                const size_t chunkBaseX = cx * binding.chunksZYX[2];
                const size_t oz0 = std::max(z0, chunkBaseZ);
                const size_t oy0 = std::max(y0, chunkBaseY);
                const size_t ox0 = std::max(x0, chunkBaseX);
                const size_t oz1 = std::min(z1, chunkBaseZ + binding.chunksZYX[0]);
                const size_t oy1 = std::min(y1, chunkBaseY + binding.chunksZYX[1]);
                const size_t ox1 = std::min(x1, chunkBaseX + binding.chunksZYX[2]);
                for (size_t z = oz0; z < oz1; ++z) {
                    for (size_t y = oy0; y < oy1; ++y) {
                        for (size_t x = ox0; x < ox1; ++x) {
                            const size_t src = chunkOffset(
                                binding,
                                z - chunkBaseZ,
                                y - chunkBaseY,
                                x - chunkBaseX);
                            if (src >= chunk->size()) {
                                throw std::runtime_error("Lasagna pred_dt chunk is smaller than expected");
                            }
                            const size_t dst = ((z - z0) * cropY + (y - y0)) * cropX + (x - x0);
                            const bool passable = static_cast<uint8_t>((*chunk)[src]) >= options.threshold;
                            volume.passable[dst] = passable ? 1 : 0;
                            volume.passableVoxels += passable ? 1U : 0U;
                        }
                    }
                }
            }
        }
    }

    return volume;
}

MaxflowTerminal findNearestPassableNode(
    std::span<const uint8_t> passableZYX,
    std::array<size_t, 3> shapeZYX,
    const MaxflowBox3& cropPredXYZ,
    const MaxflowDouble3& pointBase,
    double predDtSpacingBase)
{
    const uint64_t voxelCount = checkedVoxelCount(shapeZYX);
    if (passableZYX.size() != voxelCount) {
        throw std::runtime_error("Maxflow passability volume size does not match shape");
    }
    if (!(predDtSpacingBase > 0.0)) {
        throw std::runtime_error("pred_dt spacing must be positive");
    }

    const MaxflowInt3 pred{
        clampInt64(static_cast<int64_t>(std::floor(pointBase.x / predDtSpacingBase)),
                   cropPredXYZ.begin.x,
                   cropPredXYZ.end.x - 1),
        clampInt64(static_cast<int64_t>(std::floor(pointBase.y / predDtSpacingBase)),
                   cropPredXYZ.begin.y,
                   cropPredXYZ.end.y - 1),
        clampInt64(static_cast<int64_t>(std::floor(pointBase.z / predDtSpacingBase)),
                   cropPredXYZ.begin.z,
                   cropPredXYZ.end.z - 1),
    };
    const int64_t localX = pred.x - cropPredXYZ.begin.x;
    const int64_t localY = pred.y - cropPredXYZ.begin.y;
    const int64_t localZ = pred.z - cropPredXYZ.begin.z;
    if (localX < 0 || localY < 0 || localZ < 0 ||
        localX >= static_cast<int64_t>(shapeZYX[2]) ||
        localY >= static_cast<int64_t>(shapeZYX[1]) ||
        localZ >= static_cast<int64_t>(shapeZYX[0])) {
        throw std::runtime_error("source/sink point is outside the pred_dt crop");
    }

    MaxflowTerminal best;
    best.predVoxelXYZ = pred;
    uint64_t rank = 0;
    uint64_t bestDistance = std::numeric_limits<uint64_t>::max();
    for (size_t z = 0; z < shapeZYX[0]; ++z) {
        for (size_t y = 0; y < shapeZYX[1]; ++y) {
            for (size_t x = 0; x < shapeZYX[2]; ++x) {
                const size_t off = voxelOffset(z, y, x, shapeZYX);
                if (passableZYX[off] == 0) {
                    continue;
                }
                const int64_t dx = static_cast<int64_t>(x) - localX;
                const int64_t dy = static_cast<int64_t>(y) - localY;
                const int64_t dz = static_cast<int64_t>(z) - localZ;
                const uint64_t d2 = static_cast<uint64_t>(dx * dx + dy * dy + dz * dz);
                if (d2 < bestDistance) {
                    bestDistance = d2;
                    best.node = checkedNodeId(rank);
                    best.predVoxelXYZ = {
                        cropPredXYZ.begin.x + static_cast<int64_t>(x),
                        cropPredXYZ.begin.y + static_cast<int64_t>(y),
                        cropPredXYZ.begin.z + static_cast<int64_t>(z),
                    };
                    best.exact = d2 == 0;
                    if (best.exact) {
                        return best;
                    }
                }
                ++rank;
            }
        }
    }
    if (best.node < 0) {
        std::ostringstream message;
        message << "pred_dt crop contains no passable voxels"
                << "; point_base_xyz=" << pointString(pointBase)
                << "; pred_dt_spacing_base=" << predDtSpacingBase
                << "; initial_pred_voxel_xyz=" << intPointString(pred)
                << "; local_pred_voxel_xyz=" << intPointString({localX, localY, localZ})
                << "; crop_pred_xyz=" << boxString(cropPredXYZ)
                << "; crop_shape_zyx=" << shapeString(shapeZYX)
                << "; passable_buffer_voxels=" << passableZYX.size();
        throw std::runtime_error(message.str());
    }
    return best;
}

MaxflowGraphBuildReport buildMaxflowGraphFromManifest(
    const std::filesystem::path& manifestPath,
    const MaxflowManifestBuildOptions& options)
{
    MaxflowGraphBuildReport report;
    report.manifest = LasagnaDatasetManifest::parseFile(manifestPath);
    report.predDt = loadPredDtPassability(report.manifest, options);
    const std::array<size_t, 3> cropShapeZYX{
        static_cast<size_t>(report.predDt.cropPredXYZ.end.z - report.predDt.cropPredXYZ.begin.z),
        static_cast<size_t>(report.predDt.cropPredXYZ.end.y - report.predDt.cropPredXYZ.begin.y),
        static_cast<size_t>(report.predDt.cropPredXYZ.end.x - report.predDt.cropPredXYZ.begin.x),
    };
    report.graph = buildMaxflowGraphFromPassability(report.predDt.passable, cropShapeZYX);

    const auto& sourcePoints = options.sourcesBase.empty()
        ? std::vector<MaxflowDouble3>{options.sourceBase}
        : options.sourcesBase;
    const auto& sinkPoints = options.sinksBase.empty()
        ? std::vector<MaxflowDouble3>{options.sinkBase}
        : options.sinksBase;

    report.sources.reserve(sourcePoints.size());
    for (const auto& source : sourcePoints) {
        report.sources.push_back(findNearestPassableNode(
            report.predDt.passable,
            cropShapeZYX,
            report.predDt.cropPredXYZ,
            source,
            report.predDt.spacingBase));
    }
    report.sinks.reserve(sinkPoints.size());
    for (const auto& sink : sinkPoints) {
        report.sinks.push_back(findNearestPassableNode(
            report.predDt.passable,
            cropShapeZYX,
            report.predDt.cropPredXYZ,
            sink,
            report.predDt.spacingBase));
    }
    if (!report.sources.empty()) {
        report.source = report.sources.front();
    }
    if (!report.sinks.empty()) {
        report.sink = report.sinks.front();
    }
    return report;
}

const char* toString(MaxflowNodeKind kind) noexcept
{
    switch (kind) {
    case MaxflowNodeKind::Voxel:
        return "voxel";
    case MaxflowNodeKind::ChunkCenter:
        return "chunk_center";
    case MaxflowNodeKind::ChunkPortal:
        return "chunk_portal";
    }
    return "unknown";
}

} // namespace vc::lasagna
