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

// Each voxel of `dst` (dstShape, in C order) is `value` of the `factors` block of `src`
// under it, clipped to srcShape; voxels with no block keep their value.
template <class T, class Value>
void reduceBlocks(const T* src, const Index& srcShape, T* dst, const Index& dstShape, const std::array<int, 3>& f, Value&& value)
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
                out[x] = value([&](auto&& visit) {
                    for (std::int64_t k = z0; k < z1; ++k)
                        for (std::int64_t j = y0; j < y1; ++j)
                            for (std::int64_t i = x0; i < x1; ++i)
                                visit(src[k * planeStride + j * rowStride + i]);
                }, src[z0 * planeStride + y0 * rowStride + x0], std::uint64_t((z1 - z0) * (y1 - y0) * (x1 - x0)));
            }
        }
    }
}

template <class T>
void reduceMean(const T* src, const Index& srcShape, T* dst, const Index& dstShape, const std::array<int, 3>& f)
{
    reduceBlocks(src, srcShape, dst, dstShape, f, [](auto&& each, T, std::uint64_t count) {
        if constexpr (std::is_floating_point_v<T>) {
            float sum = 0.0f;
            each([&](T v) { sum += v; });
            return T(sum / float(count));
        } else {
            std::uint64_t sum = 0;
            each([&](T v) { sum += v; });
            return T(divide_round_half_even(sum, count));
        }
    });
}

template <class T>
void reduceNearest(const T* src, const Index& srcShape, T* dst, const Index& dstShape, const std::array<int, 3>& f)
{
    reduceBlocks(src, srcShape, dst, dstShape, f, [](auto&&, T first, std::uint64_t) { return first; });
}

template <class T>
void reduceMode(const T* src, const Index& srcShape, T* dst, const Index& dstShape, const std::array<int, 3>& f)
{
    std::vector<T> values;
    reduceBlocks(src, srcShape, dst, dstShape, f, [&](auto&& each, T, std::uint64_t) {
        values.clear();
        each([&](T v) {
            if (v != T(0))
                values.push_back(v);
        });
        std::sort(values.begin(), values.end());
        T best{0};
        std::size_t bestCount = 0;
        for (std::size_t i = 0; i < values.size();) {
            std::size_t next = i + 1;
            while (next < values.size() && values[next] == values[i])
                ++next;
            if (next - i > bestCount) {
                best = values[i];
                bestCount = next - i;
            }
            i = next;
        }
        return best;
    });
}

// The reduction, over elements of `itemsize` bytes taken as unsigned integers where it
// only moves or compares them.
void reduceChunk(Reduction reduction, const ArrayMeta& meta, const std::uint8_t* src, const Index& srcShape, std::uint8_t* dst,
                 const Index& dstShape, const std::array<int, 3>& f)
{
    const auto apply = [&]<class T>(T) {
        const auto* in = reinterpret_cast<const T*>(src);
        auto* out = reinterpret_cast<T*>(dst);
        if (reduction == Reduction::Mean)
            reduceMean(in, srcShape, out, dstShape, f);
        else if (reduction == Reduction::Nearest)
            reduceNearest(in, srcShape, out, dstShape, f);
        else
            reduceMode(in, srcShape, out, dstShape, f);
    };
    if (reduction == Reduction::Mean) {
        if (meta.dtype == "|u1")
            apply(std::uint8_t{});
        else if (meta.dtype == "<u2")
            apply(std::uint16_t{});
        else
            apply(float{});
        return;
    }
    switch (meta.itemsize()) {
        case 1: apply(std::uint8_t{}); break;
        case 2: apply(std::uint16_t{}); break;
        case 4: apply(std::uint32_t{}); break;
        default: apply(std::uint64_t{}); break;
    }
}

void checkReducible(Reduction reduction, const ArrayMeta& meta)
{
    if (reduction == Reduction::Mean && meta.dtype != "|u1" && meta.dtype != "<u2" && meta.dtype != "<f4")
        throw std::invalid_argument("mean pyramids are written for uint8, uint16 and float32: " + meta.dtype);
    const auto size = meta.itemsize();
    if (size != 1 && size != 2 && size != 4 && size != 8)
        throw std::invalid_argument("pyramids cannot be written for " + meta.dtype);
    if (reduction == Reduction::Mode && meta.type.kind == 'f')
        throw std::invalid_argument("mode pyramids are written for integers: " + meta.dtype);
}

}  // namespace

Reduction parseReduction(const std::string& name)
{
    if (name == "mean")
        return Reduction::Mean;
    if (name == "nearest")
        return Reduction::Nearest;
    if (name == "mode")
        return Reduction::Mode;
    throw std::invalid_argument("reduction must be mean, nearest or mode: " + name);
}

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

void createPyramidLevels(const fs::path& root, const PyramidOptions& options, bool overwrite)
{
    auto level0 = createOptionsLike(Writer::open(root / "0").meta());
    level0.overwrite = overwrite;
    createPyramidLevels(root, level0, options);
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
    checkReducible(options.reduction, meta);

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
        // Compute into a dstSize block, then place its rows in the chunk.
        std::vector<std::uint8_t> block(std::size_t(dstSize[0] * dstSize[1] * dstSize[2]) * itemsize);
        reduceChunk(options.reduction, meta, input.data(), srcSize, block.data(), dstSize, options.factors);
        for (std::int64_t z = 0; z < dstSize[0]; ++z)
            for (std::int64_t y = 0; y < dstSize[1]; ++y)
                std::memcpy(output.data() + std::size_t((z * meta.chunks[1] + y) * meta.chunks[2]) * itemsize,
                            block.data() + std::size_t((z * dstSize[1] + y) * dstSize[2]) * itemsize,
                            std::size_t(dstSize[2]) * itemsize);
        dst.writeChunkSkipEmpty(chunk, output.data(), output.size());
        if (progress)
            progress(++done, count);
    });
}

void writePyramid(const fs::path& root, const PyramidOptions& options, bool overwrite)
{
    createPyramidLevels(root, options, overwrite);
    for (int level = 1; level <= options.levels; ++level)
        writePyramidLevel(root, level, options);
}

Json omeMultiscales(const MultiscalesOptions& options)
{
    if (options.levels < 0)
        throw std::invalid_argument("multiscales need a nonnegative level count");
    Json axes = Json::array();
    for (const auto& name : options.axes) {
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
            // A reduced voxel is centred on its block; a nearest one is the block's first.
            translation.push_back(options.reduction == Reduction::Nearest ? 0.0 : options.scale[axis] * (factor[axis] - 1.0) / 2.0);
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
            {"metadata", {{"downsampling_method", options.reduction == Reduction::Mean      ? "mean"
                                                  : options.reduction == Reduction::Nearest ? "nearest"
                                                                                            : "mode"}}}};
}

}  // namespace vcr::zarr
