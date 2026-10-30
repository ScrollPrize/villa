// Writing multiscale pyramids: coarser levels as block means of the level below.
#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_write.hpp"
#include "vc_reader/array.hpp"
#include "vc_reader/native/rounding.hpp"
#include "vc_reader/native/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;

namespace vcr::zarr {

namespace {

void checkFactors(const PyramidOptions& options)
{
    if (options.levels < 0 ||
        std::any_of(options.factors.begin(), options.factors.end(), [](int f) { return f < 1; }))
        throw std::invalid_argument("pyramid levels must be nonnegative and factors positive");
}

std::int64_t ceilDiv(std::int64_t value, std::int64_t divisor) { return (value + divisor - 1) / divisor; }

// Each voxel of `dst` (dstShape, in C order) is the mean of the `factors` block of
// `src` under it, clipped to srcShape; voxels with no block keep their value.
template <class T>
void reduceBlocks(const T* src, const Index& srcShape, T* dst, const Index& dstShape, const std::array<int, 3>& f)
{
    const std::int64_t rowStride = srcShape[2], planeStride = srcShape[1] * srcShape[2];
    for (std::int64_t z = 0; z < dstShape[0]; ++z) {
        const std::int64_t z0 = z * f[0], z1 = std::min(z0 + f[0], srcShape[0]);
        for (std::int64_t y = 0; y < dstShape[1]; ++y) {
            const std::int64_t y0 = y * f[1], y1 = std::min(y0 + f[1], srcShape[1]);
            T* out = dst + (z * dstShape[1] + y) * dstShape[2];
            for (std::int64_t x = 0; x < dstShape[2]; ++x) {
                const std::int64_t x0 = x * f[2], x1 = std::min(x0 + f[2], srcShape[2]);
                if (z0 >= z1 || y0 >= y1 || x0 >= x1)
                    continue;
                if constexpr (std::is_floating_point_v<T>) {
                    float sum = 0.0f;
                    for (std::int64_t k = z0; k < z1; ++k)
                        for (std::int64_t j = y0; j < y1; ++j)
                            for (std::int64_t i = x0; i < x1; ++i)
                                sum += src[k * planeStride + j * rowStride + i];
                    out[x] = sum / float((z1 - z0) * (y1 - y0) * (x1 - x0));
                } else {
                    std::uint64_t sum = 0;
                    for (std::int64_t k = z0; k < z1; ++k)
                        for (std::int64_t j = y0; j < y1; ++j)
                            for (std::int64_t i = x0; i < x1; ++i)
                                sum += src[k * planeStride + j * rowStride + i];
                    out[x] = T(divide_round_half_even(sum, std::uint64_t((z1 - z0) * (y1 - y0) * (x1 - x0))));
                }
            }
        }
    }
}

}  // namespace

void createPyramidLevels(const fs::path& root, const CreateOptions& level0, const PyramidOptions& options)
{
    checkFactors(options);
    if (level0.shape.size() != 3 || level0.chunks.size() != 3)
        throw std::invalid_argument("pyramid levels must be 3-D");
    auto level = level0;
    for (int l = 1; l <= options.levels; ++l) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            level.shape[axis] = ceilDiv(level.shape[axis], options.factors[axis]);
            level.chunks[axis] = std::min(level0.chunks[axis], level.shape[axis]);
        }
        Writer::create(root / std::to_string(l), level);
    }
}

void writePyramidLevel(const fs::path& root, int level, const PyramidOptions& options, const std::optional<ChunkRange>& range,
                       const std::function<void(std::size_t, std::size_t)>& progress)
{
    checkFactors(options);
    if (level < 1)
        throw std::invalid_argument("pyramid level to write must be at least 1");
    OpenOptions readOptions;
    readOptions.threads = 1;  // parallel across output chunks below
    auto src = openArray((root / std::to_string(level - 1)).string(), readOptions);
    auto dst = Writer::open(root / std::to_string(level));
    const auto& srcMeta = src.meta;
    const auto& meta = dst.meta();
    if (srcMeta.shape.size() != 3 || meta.shape.size() != 3 || srcMeta.dtype != meta.dtype)
        throw std::invalid_argument("pyramid levels must be 3-D arrays of one dtype");
    if (meta.dtype != "|u1" && meta.dtype != "<u2" && meta.dtype != "<f4")
        throw std::invalid_argument("pyramids are written for uint8, uint16 and float32: " + meta.dtype);

    Index begin{0, 0, 0}, end(3);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto grid = ceilDiv(meta.shape[axis], meta.chunks[axis]);
        end[axis] = range ? std::min(range->end[axis], grid) : grid;
        begin[axis] = range ? std::min(range->begin[axis], end[axis]) : 0;
    }
    const auto count = std::size_t((end[0] - begin[0]) * (end[1] - begin[1]) * (end[2] - begin[2]));
    const auto itemsize = meta.itemsize();
    std::atomic<std::size_t> done{0};

    ThreadPool pool(options.threads ? options.threads : std::max(1u, std::thread::hardware_concurrency()));
    pool.parallel_for(count, [&](std::size_t index) {
        const auto rows = end[1] - begin[1], columns = end[2] - begin[2];
        const Index chunk{begin[0] + std::int64_t(index) / (rows * columns),
                          begin[1] + std::int64_t(index) / columns % rows,
                          begin[2] + std::int64_t(index) % columns};
        // The source voxels under this output chunk.
        Index start(3), srcSize(3), dstSize(3);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            dstSize[axis] = std::min(meta.chunks[axis], meta.shape[axis] - chunk[axis] * meta.chunks[axis]);
            start[axis] = chunk[axis] * meta.chunks[axis] * options.factors[axis];
            srcSize[axis] = std::min(dstSize[axis] * options.factors[axis], srcMeta.shape[axis] - start[axis]);
        }
        std::vector<std::uint8_t> input(std::size_t(srcSize[0] * srcSize[1] * srcSize[2]) * itemsize);
        src.array->read_many({start}, {srcSize}, {input.data()});
        // The whole chunk: fill past the array's end, the block means inside it.
        std::vector<std::uint8_t> output(meta.chunkBytes());
        for (std::size_t i = 0; i < output.size(); i += itemsize)
            std::memcpy(output.data() + i, meta.fill.data(), itemsize);
        const Index chunkShape(meta.chunks.begin(), meta.chunks.end());
        const auto reduce = [&]<class T>(T) {
            // Compute into a dstSize block, then place its rows in the chunk.
            std::vector<T> block(std::size_t(dstSize[0] * dstSize[1] * dstSize[2]));
            reduceBlocks(reinterpret_cast<const T*>(input.data()), srcSize, block.data(), dstSize, options.factors);
            for (std::int64_t z = 0; z < dstSize[0]; ++z)
                for (std::int64_t y = 0; y < dstSize[1]; ++y)
                    std::memcpy(output.data() + std::size_t((z * chunkShape[1] + y) * chunkShape[2]) * sizeof(T),
                                block.data() + (z * dstSize[1] + y) * dstSize[2], std::size_t(dstSize[2]) * sizeof(T));
        };
        if (meta.dtype == "|u1")
            reduce(std::uint8_t{});
        else if (meta.dtype == "<u2")
            reduce(std::uint16_t{});
        else
            reduce(float{});
        dst.writeChunkSkipEmpty(chunk, output.data(), output.size());
        if (progress)
            progress(++done, count);
    });
}

Json omeMultiscales(const MultiscalesOptions& options)
{
    if (options.levels < 0)
        throw std::invalid_argument("multiscales need a nonnegative level count");
    Json axes = Json::array();
    for (const auto* name : {"z", "y", "x"}) {
        Json axis{{"name", name}, {"type", "space"}};
        if (!options.unit.empty())
            axis["unit"] = options.unit;
        axes.push_back(std::move(axis));
    }
    Json datasets = Json::array();
    std::array<double, 3> factor{1, 1, 1};
    for (int level = 0; level <= options.levels; ++level) {
        Json scale = Json::array(), translation = Json::array();
        for (std::size_t axis = 0; axis < 3; ++axis) {
            scale.push_back(options.scale[axis] * factor[axis]);
            // A block mean's voxel is centred on the block it reduces.
            translation.push_back(options.scale[axis] * (factor[axis] - 1.0) / 2.0);
            factor[axis] *= options.factors[axis];
        }
        datasets.push_back({{"path", std::to_string(level)},
                            {"coordinateTransformations",
                             Json::array({Json{{"type", "scale"}, {"scale", std::move(scale)}},
                                          Json{{"type", "translation"}, {"translation", std::move(translation)}}})}});
    }
    return {{"version", "0.4"},
            {"name", options.name},
            {"axes", std::move(axes)},
            {"datasets", std::move(datasets)},
            {"metadata", {{"downsampling_method", "mean"}}}};
}

}  // namespace vcr::zarr
