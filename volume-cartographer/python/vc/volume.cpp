#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <Python.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/Volume.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

std::string dtypeName(vc::render::ChunkDtype dtype)
{
    switch (dtype) {
    case vc::render::ChunkDtype::UInt8:
        return "uint8";
    case vc::render::ChunkDtype::UInt16:
        return "uint16";
    }
    throw std::runtime_error("unsupported chunk dtype");
}

Volume::MissingScaleLevelPolicy parseMissingPolicy(const std::string& value)
{
    if (value == "error")
        return Volume::MissingScaleLevelPolicy::Error;
    if (value == "all_fill")
        return Volume::MissingScaleLevelPolicy::AllFill;
    if (value == "empty")
        return Volume::MissingScaleLevelPolicy::Empty;
    if (value == "virtual_downsample")
        return Volume::MissingScaleLevelPolicy::VirtualDownsample;
    throw std::invalid_argument(
        "missing_policy must be one of: error, all_fill, empty, virtual_downsample");
}

nb::object jsonToPython(const utils::Json& json)
{
    nb::object loads = nb::module_::import_("json").attr("loads");
    return loads(json.dump());
}

nb::tuple tuple3(const std::array<int, 3>& value)
{
    return nb::make_tuple(value[0], value[1], value[2]);
}

vc::Sampling parseSampling(const std::string& value)
{
    if (value == "nearest")
        return vc::Sampling::Nearest;
    if (value == "trilinear")
        return vc::Sampling::Trilinear;
    throw std::invalid_argument("sampling must be one of: nearest, trilinear");
}

template <typename T>
nb::ndarray<T, nb::numpy, nb::c_contig> makeNumpyArray(std::vector<T>&& data,
                                                       std::array<size_t, 3> shape)
{
    auto* heap = new std::vector<T>(std::move(data));
    nb::capsule owner(heap, [](void* ptr) noexcept {
        delete static_cast<std::vector<T>*>(ptr);
    });
    return nb::ndarray<T, nb::numpy, nb::c_contig>(
        heap->data(),
        {shape[0], shape[1], shape[2]},
        owner);
}

template <typename T>
std::vector<T> toCOrder(const Array3D<T>& array)
{
    const auto shape = array.shape();
    std::vector<T> out(shape[0] * shape[1] * shape[2]);
    size_t dst = 0;
    for (size_t z = 0; z < shape[0]; ++z) {
        for (size_t y = 0; y < shape[1]; ++y) {
            for (size_t x = 0; x < shape[2]; ++x) {
                out[dst++] = array(z, y, x);
            }
        }
    }
    return out;
}

template <typename T>
T typedFill(double fillValue)
{
    const double maxValue = static_cast<double>(std::numeric_limits<T>::max());
    return static_cast<T>(std::clamp(fillValue, 0.0, maxValue));
}

template <typename T>
void copyChunkIntersection(std::vector<T>& out,
                           const std::array<size_t, 3>& outShape,
                           const std::array<int, 3>& requestOffset,
                           const std::array<int, 3>& chunkShape,
                           int level,
                           int cz,
                           int cy,
                           int cx,
                           vc::render::IChunkedArray& cache,
                           T fill)
{
    const int chunkBaseZ = cz * chunkShape[0];
    const int chunkBaseY = cy * chunkShape[1];
    const int chunkBaseX = cx * chunkShape[2];

    const int64_t reqZ0 = requestOffset[0];
    const int64_t reqY0 = requestOffset[1];
    const int64_t reqX0 = requestOffset[2];
    const int64_t reqZ1 = reqZ0 + static_cast<int64_t>(outShape[0]);
    const int64_t reqY1 = reqY0 + static_cast<int64_t>(outShape[1]);
    const int64_t reqX1 = reqX0 + static_cast<int64_t>(outShape[2]);

    const int z0 = static_cast<int>(std::max<int64_t>(reqZ0, chunkBaseZ));
    const int y0 = static_cast<int>(std::max<int64_t>(reqY0, chunkBaseY));
    const int x0 = static_cast<int>(std::max<int64_t>(reqX0, chunkBaseX));
    const int z1 = static_cast<int>(std::min<int64_t>(reqZ1, chunkBaseZ + chunkShape[0]));
    const int y1 = static_cast<int>(std::min<int64_t>(reqY1, chunkBaseY + chunkShape[1]));
    const int x1 = static_cast<int>(std::min<int64_t>(reqX1, chunkBaseX + chunkShape[2]));
    if (z0 >= z1 || y0 >= y1 || x0 >= x1)
        return;

    const size_t copyCount = static_cast<size_t>(x1 - x0);
    const size_t dstStrideY = outShape[2];
    const size_t dstStrideZ = outShape[1] * dstStrideY;

    auto result = cache.getChunkBlocking(level, cz, cy, cx);
    if (result.status == vc::render::ChunkStatus::Error)
        throw std::runtime_error(result.error.empty() ? "chunk fetch failed" : result.error);

    if (result.status == vc::render::ChunkStatus::AllFill ||
        result.status == vc::render::ChunkStatus::Missing ||
        !result.bytes) {
        for (int z = z0; z < z1; ++z) {
            const size_t dstZ = static_cast<size_t>(z - requestOffset[0]);
            for (int y = y0; y < y1; ++y) {
                const size_t dstY = static_cast<size_t>(y - requestOffset[1]);
                const size_t dstX = static_cast<size_t>(x0 - requestOffset[2]);
                const size_t dst = dstZ * dstStrideZ + dstY * dstStrideY + dstX;
                std::fill_n(out.data() + dst, copyCount, fill);
            }
        }
        return;
    }

    const size_t expectedBytes = static_cast<size_t>(chunkShape[0]) *
                                 static_cast<size_t>(chunkShape[1]) *
                                 static_cast<size_t>(chunkShape[2]) *
                                 sizeof(T);
    if (result.bytes->size() < expectedBytes)
        throw std::runtime_error("chunk payload is smaller than expected");

    const auto* srcData = reinterpret_cast<const T*>(result.bytes->data());
    const size_t srcStrideY = static_cast<size_t>(chunkShape[2]);
    const size_t srcStrideZ = static_cast<size_t>(chunkShape[1]) * srcStrideY;

    for (int z = z0; z < z1; ++z) {
        const size_t srcZ = static_cast<size_t>(z - chunkBaseZ);
        const size_t dstZ = static_cast<size_t>(z - requestOffset[0]);
        for (int y = y0; y < y1; ++y) {
            const size_t srcY = static_cast<size_t>(y - chunkBaseY);
            const size_t srcX = static_cast<size_t>(x0 - chunkBaseX);
            const size_t dstY = static_cast<size_t>(y - requestOffset[1]);
            const size_t dstX = static_cast<size_t>(x0 - requestOffset[2]);
            const size_t src = srcZ * srcStrideZ + srcY * srcStrideY + srcX;
            const size_t dst = dstZ * dstStrideZ + dstY * dstStrideY + dstX;
            std::memcpy(out.data() + dst, srcData + src, copyCount * sizeof(T));
        }
    }
}

template <typename T>
void sampleBlockChunkIntersection(std::vector<T>& out,
                                  std::vector<uint8_t>& valid,
                                  const std::array<size_t, 3>& outShape,
                                  const std::array<int, 3>& requestOffset,
                                  const std::array<int, 3>& chunkShape,
                                  int level,
                                  int cz,
                                  int cy,
                                  int cx,
                                  vc::render::IChunkedArray& cache,
                                  bool blocking,
                                  T fill,
                                  vc::render::ChunkedPlaneSampler::Stats& stats)
{
    const int chunkBaseZ = cz * chunkShape[0];
    const int chunkBaseY = cy * chunkShape[1];
    const int chunkBaseX = cx * chunkShape[2];

    const int64_t reqZ0 = requestOffset[0];
    const int64_t reqY0 = requestOffset[1];
    const int64_t reqX0 = requestOffset[2];
    const int64_t reqZ1 = reqZ0 + static_cast<int64_t>(outShape[0]);
    const int64_t reqY1 = reqY0 + static_cast<int64_t>(outShape[1]);
    const int64_t reqX1 = reqX0 + static_cast<int64_t>(outShape[2]);

    const int z0 = static_cast<int>(std::max<int64_t>(reqZ0, chunkBaseZ));
    const int y0 = static_cast<int>(std::max<int64_t>(reqY0, chunkBaseY));
    const int x0 = static_cast<int>(std::max<int64_t>(reqX0, chunkBaseX));
    const int z1 = static_cast<int>(std::min<int64_t>(reqZ1, chunkBaseZ + chunkShape[0]));
    const int y1 = static_cast<int>(std::min<int64_t>(reqY1, chunkBaseY + chunkShape[1]));
    const int x1 = static_cast<int>(std::min<int64_t>(reqX1, chunkBaseX + chunkShape[2]));
    if (z0 >= z1 || y0 >= y1 || x0 >= x1)
        return;

    auto result = blocking
        ? cache.getChunkBlocking(level, cz, cy, cx)
        : cache.tryGetChunk(level, cz, cy, cx);
    if (result.status == vc::render::ChunkStatus::Error) {
        ++stats.errorChunks;
        throw std::runtime_error(result.error.empty() ? "chunk fetch failed" : result.error);
    }
    if (blocking && result.status == vc::render::ChunkStatus::MissQueued) {
        ++stats.errorChunks;
        throw std::runtime_error(
            "blocking requested-level block sampling received unresolved chunk after getChunkBlocking");
    }
    if (result.status == vc::render::ChunkStatus::MissQueued)
        return;
    if (result.status == vc::render::ChunkStatus::Missing)
        ++stats.missingChunks;

    const size_t copyCount = static_cast<size_t>(x1 - x0);
    const size_t dstStrideY = outShape[2];
    const size_t dstStrideZ = outShape[1] * dstStrideY;

    const bool fillOnly = result.status == vc::render::ChunkStatus::AllFill ||
                          result.status == vc::render::ChunkStatus::Missing;
    const T fillValue = result.status == vc::render::ChunkStatus::Missing ? T{} : fill;
    const T* srcData = nullptr;
    size_t srcStrideY = 0;
    size_t srcStrideZ = 0;
    if (!fillOnly) {
        if (!result.bytes)
            throw std::runtime_error("chunk payload is missing for data chunk");
        const size_t expectedBytes = static_cast<size_t>(chunkShape[0]) *
                                     static_cast<size_t>(chunkShape[1]) *
                                     static_cast<size_t>(chunkShape[2]) *
                                     sizeof(T);
        if (result.bytes->size() < expectedBytes)
            throw std::runtime_error("chunk payload is smaller than expected");
        srcData = reinterpret_cast<const T*>(result.bytes->data());
        srcStrideY = static_cast<size_t>(chunkShape[2]);
        srcStrideZ = static_cast<size_t>(chunkShape[1]) * srcStrideY;
    }

    for (int z = z0; z < z1; ++z) {
        const size_t srcZ = static_cast<size_t>(z - chunkBaseZ);
        const size_t dstZ = static_cast<size_t>(z - requestOffset[0]);
        for (int y = y0; y < y1; ++y) {
            const size_t srcY = static_cast<size_t>(y - chunkBaseY);
            const size_t srcX = static_cast<size_t>(x0 - chunkBaseX);
            const size_t dstY = static_cast<size_t>(y - requestOffset[1]);
            const size_t dstX = static_cast<size_t>(x0 - requestOffset[2]);
            const size_t dst = dstZ * dstStrideZ + dstY * dstStrideY + dstX;
            if (fillOnly) {
                std::fill_n(out.data() + dst, copyCount, fillValue);
            } else {
                const size_t src = srcZ * srcStrideZ + srcY * srcStrideY + srcX;
                std::memcpy(out.data() + dst, srcData + src, copyCount * sizeof(T));
            }
            std::fill_n(valid.data() + dst, copyCount, uint8_t{1});
            stats.coveredPixels += static_cast<int>(copyCount);
        }
    }
}

template <typename T>
nb::object readZYXTypedSlow(Volume& volume,
                            const std::array<int, 3>& offset,
                            const std::array<size_t, 3>& shape,
                            int level,
                            Volume::MissingScaleLevelPolicy missingPolicy)
{
    Array3D<T> out(shape);
    bool ok = false;
    {
        nb::gil_scoped_release release;
        ok = volume.readZYX(out, offset, level, missingPolicy);
    }
    if (!ok)
        return nb::none();
    return nb::cast(makeNumpyArray(toCOrder(out), shape));
}

template <typename T>
nb::object readZYXTyped(Volume& volume,
                        const std::array<int, 3>& offset,
                        const std::array<size_t, 3>& shape,
                        int level,
                        const std::string& missingPolicy)
{
    const auto policy = parseMissingPolicy(missingPolicy);
    if (level < 0)
        throw std::out_of_range("level must be non-negative");
    if (!volume.hasScaleLevel(level)) {
        return readZYXTypedSlow<T>(volume, offset, shape, level, policy);
    }

    std::vector<T> out(shape[0] * shape[1] * shape[2], T{});
    if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0)
        return nb::cast(makeNumpyArray(std::move(out), shape));

    {
        nb::gil_scoped_release release;
        auto* cache = volume.chunkedCache();
        const auto volumeShape = cache->shape(level);
        const auto chunkShape = cache->chunkShape(level);

        const int64_t reqZ0 = offset[0];
        const int64_t reqY0 = offset[1];
        const int64_t reqX0 = offset[2];
        const int64_t reqZ1 = reqZ0 + static_cast<int64_t>(shape[0]) - 1;
        const int64_t reqY1 = reqY0 + static_cast<int64_t>(shape[1]) - 1;
        const int64_t reqX1 = reqX0 + static_cast<int64_t>(shape[2]) - 1;

        const int readZ0 = static_cast<int>(std::max<int64_t>(0, reqZ0));
        const int readY0 = static_cast<int>(std::max<int64_t>(0, reqY0));
        const int readX0 = static_cast<int>(std::max<int64_t>(0, reqX0));
        const int readZ1 = static_cast<int>(std::min<int64_t>(volumeShape[0] - 1, reqZ1));
        const int readY1 = static_cast<int>(std::min<int64_t>(volumeShape[1] - 1, reqY1));
        const int readX1 = static_cast<int>(std::min<int64_t>(volumeShape[2] - 1, reqX1));

        if (readZ0 <= readZ1 && readY0 <= readY1 && readX0 <= readX1) {
            const int cZ0 = readZ0 / chunkShape[0];
            const int cY0 = readY0 / chunkShape[1];
            const int cX0 = readX0 / chunkShape[2];
            const int cZ1 = readZ1 / chunkShape[0];
            const int cY1 = readY1 / chunkShape[1];
            const int cX1 = readX1 / chunkShape[2];
            const T fill = typedFill<T>(cache->fillValue());
            for (int cz = cZ0; cz <= cZ1; ++cz) {
                for (int cy = cY0; cy <= cY1; ++cy) {
                    for (int cx = cX0; cx <= cX1; ++cx) {
                        copyChunkIntersection(
                            out, shape, offset, chunkShape, level, cz, cy, cx, *cache, fill);
                    }
                }
            }
        }
    }
    return nb::cast(makeNumpyArray(std::move(out), shape));
}

template <typename T>
nb::object readXYZTyped(Volume& volume,
                        const std::array<int, 3>& offset,
                        const std::array<size_t, 3>& shapeXYZ,
                        int level,
                        const std::string& missingPolicy)
{
    const std::array<int, 3> offsetZYX{offset[2], offset[1], offset[0]};
    const std::array<size_t, 3> shapeZYX{shapeXYZ[2], shapeXYZ[1], shapeXYZ[0]};
    return readZYXTyped<T>(volume, offsetZYX, shapeZYX, level, missingPolicy);
}

nb::object readZYX(Volume& volume,
                   const std::array<int, 3>& offset,
                   const std::array<size_t, 3>& shape,
                   int level,
                   const std::string& missingPolicy)
{
    if (volume.dtype() == vc::render::ChunkDtype::UInt8)
        return readZYXTyped<uint8_t>(volume, offset, shape, level, missingPolicy);
    return readZYXTyped<uint16_t>(volume, offset, shape, level, missingPolicy);
}

nb::object readXYZ(Volume& volume,
                   const std::array<int, 3>& offset,
                   const std::array<size_t, 3>& shape,
                   int level,
                   const std::string& missingPolicy)
{
    if (volume.dtype() == vc::render::ChunkDtype::UInt8)
        return readXYZTyped<uint8_t>(volume, offset, shape, level, missingPolicy);
    return readXYZTyped<uint16_t>(volume, offset, shape, level, missingPolicy);
}

std::vector<vc::render::ChunkKey> collectChunkKeys(Volume& volume,
                                                   const std::array<int, 3>& offset,
                                                   const std::array<size_t, 3>& shape,
                                                   int level);

nb::dict statsToDict(const vc::render::ChunkedPlaneSampler::Stats& stats);

template <typename T>
nb::tuple sampleZYXBlockTyped(Volume& volume,
                              const std::array<int, 3>& offset,
                              const std::array<size_t, 3>& shape,
                              int level,
                              bool blocking)
{
    if (level < 0)
        throw std::out_of_range("level must be non-negative");
    if (!volume.hasScaleLevel(level))
        throw std::out_of_range(
            "requested missing zarr scale level " + std::to_string(level));

    vc::render::ChunkedPlaneSampler::Stats stats;
    stats.requestedLevelOnly = true;
    stats.fallbackLevels = 0;

    std::vector<T> out(shape[0] * shape[1] * shape[2], T{});
    std::vector<uint8_t> valid(out.size(), uint8_t{0});

    const auto keys = collectChunkKeys(volume, offset, shape, level);
    stats.requestedChunks = static_cast<int>(keys.size());
    if (keys.empty()) {
        nb::dict statsDict = statsToDict(stats);
        statsDict["blocking_prefetch_chunks"] = 0;
        return nb::make_tuple(
            makeNumpyArray(std::move(out), shape),
            makeNumpyArray(std::move(valid), shape),
            statsDict);
    }

    {
        nb::gil_scoped_release release;
        auto* cache = volume.chunkedCache();
        if (blocking)
            cache->prefetchChunks(keys, false);
        const auto chunkShape = cache->chunkShape(level);
        const T fill = typedFill<T>(cache->fillValue());
        for (const auto& key : keys) {
            sampleBlockChunkIntersection(
                out,
                valid,
                shape,
                offset,
                chunkShape,
                level,
                key.iz,
                key.iy,
                key.ix,
                *cache,
                blocking,
                fill,
                stats);
        }
    }

    nb::dict statsDict = statsToDict(stats);
    statsDict["blocking_prefetch_chunks"] = blocking ? stats.requestedChunks : 0;
    return nb::make_tuple(
        makeNumpyArray(std::move(out), shape),
        makeNumpyArray(std::move(valid), shape),
        statsDict);
}

nb::tuple sampleZYXBlock(Volume& volume,
                         const std::array<int, 3>& offset,
                         const std::array<size_t, 3>& shape,
                         int level,
                         bool blocking)
{
    if (volume.dtype() == vc::render::ChunkDtype::UInt8)
        return sampleZYXBlockTyped<uint8_t>(volume, offset, shape, level, blocking);
    return sampleZYXBlockTyped<uint16_t>(volume, offset, shape, level, blocking);
}

std::array<size_t, 3> checkedSizeArray(const std::array<int, 3>& value, const char* name)
{
    std::array<size_t, 3> out{};
    for (size_t i = 0; i < 3; ++i) {
        if (value[i] < 0)
            throw std::out_of_range(std::string(name) + " must be non-negative");
        out[i] = static_cast<size_t>(value[i]);
    }
    return out;
}

struct SliceRegion {
    std::array<int, 3> offset{};
    std::array<size_t, 3> shape{};
};

SliceRegion parseSliceKey(const nb::object& key, const std::array<int, 3>& volumeShape)
{
    nb::tuple tuple;
    if (PyTuple_Check(key.ptr())) {
        tuple = nb::cast<nb::tuple>(key);
    } else {
        tuple = nb::make_tuple(key);
    }

    if (tuple.size() > 3)
        throw nb::index_error("Volume slicing expects at most 3 indices");

    SliceRegion region;
    for (size_t dim = 0; dim < 3; ++dim) {
        Py_ssize_t start = 0;
        Py_ssize_t stop = volumeShape[dim];
        Py_ssize_t step = 1;
        Py_ssize_t length = volumeShape[dim];

        if (dim < tuple.size()) {
            nb::handle item = tuple[dim];
            if (PySlice_Check(item.ptr())) {
                if (PySlice_GetIndicesEx(
                        item.ptr(),
                        static_cast<Py_ssize_t>(volumeShape[dim]),
                        &start,
                        &stop,
                        &step,
                        &length) < 0) {
                    throw nb::python_error();
                }
                if (step != 1)
                    throw nb::index_error("Volume slicing currently supports step=1 only");
            } else if (PyLong_Check(item.ptr())) {
                start = PyLong_AsSsize_t(item.ptr());
                if (PyErr_Occurred())
                    throw nb::python_error();
                if (start < 0)
                    start += volumeShape[dim];
                if (start < 0 || start >= volumeShape[dim])
                    throw nb::index_error("Volume index out of bounds");
                length = 1;
            } else {
                throw nb::type_error("Volume indices must be slices or integers");
            }
        }

        region.offset[dim] = static_cast<int>(start);
        region.shape[dim] = static_cast<size_t>(std::max<Py_ssize_t>(0, length));
    }
    return region;
}

std::vector<vc::render::ChunkKey> collectChunkKeys(Volume& volume,
                                                   const std::array<int, 3>& offset,
                                                   const std::array<size_t, 3>& shape,
                                                   int level)
{
    if (level < 0)
        throw std::out_of_range("level must be non-negative");
    if (!volume.hasScaleLevel(level))
        throw std::out_of_range("requested missing zarr scale level " + std::to_string(level));

    const auto volumeShape = volume.shape(level);
    const auto chunkShape = volume.chunkShape(level);
    if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0)
        return {};

    const int z0 = std::max(0, offset[0]);
    const int y0 = std::max(0, offset[1]);
    const int x0 = std::max(0, offset[2]);
    const int z1 = std::min(volumeShape[0] - 1, offset[0] + static_cast<int>(shape[0]) - 1);
    const int y1 = std::min(volumeShape[1] - 1, offset[1] + static_cast<int>(shape[1]) - 1);
    const int x1 = std::min(volumeShape[2] - 1, offset[2] + static_cast<int>(shape[2]) - 1);
    if (z0 > z1 || y0 > y1 || x0 > x1)
        return {};

    std::vector<vc::render::ChunkKey> keys;
    for (int cz = z0 / chunkShape[0]; cz <= z1 / chunkShape[0]; ++cz) {
        for (int cy = y0 / chunkShape[1]; cy <= y1 / chunkShape[1]; ++cy) {
            for (int cx = x0 / chunkShape[2]; cx <= x1 / chunkShape[2]; ++cx) {
                keys.push_back({level, cz, cy, cx});
            }
        }
    }
    return keys;
}

size_t prefetchZYX(Volume& volume,
                   const std::array<int, 3>& offset,
                   const std::array<size_t, 3>& shape,
                   int level,
                   bool wait)
{
    auto keys = collectChunkKeys(volume, offset, shape, level);
    if (keys.empty())
        return 0;
    auto* cache = volume.chunkedCache();
    {
        nb::gil_scoped_release release;
        cache->prefetchChunks(keys, wait);
    }
    return keys.size();
}

template <typename T>
nb::object chunkResultToArray(const vc::render::ChunkResult& result, double fillValue)
{
    const auto shape = checkedSizeArray(result.shape, "chunk shape");
    std::vector<T> out(shape[0] * shape[1] * shape[2]);
    if (result.status == vc::render::ChunkStatus::AllFill) {
        const double maxValue = static_cast<double>(std::numeric_limits<T>::max());
        const T fill = static_cast<T>(std::clamp(fillValue, 0.0, maxValue));
        std::fill(out.begin(), out.end(), fill);
    } else if (result.status == vc::render::ChunkStatus::Data && result.bytes) {
        const size_t bytes = out.size() * sizeof(T);
        if (result.bytes->size() < bytes)
            throw std::runtime_error("chunk payload is smaller than expected");
        std::memcpy(out.data(), result.bytes->data(), bytes);
    } else {
        return nb::none();
    }
    return nb::cast(makeNumpyArray(std::move(out), shape));
}

nb::object readChunk(Volume& volume,
                     int level,
                     const std::array<int, 3>& chunkZYX,
                     bool blocking)
{
    auto* cache = volume.chunkedCache();
    vc::render::ChunkResult result;
    {
        nb::gil_scoped_release release;
        result = blocking
            ? cache->getChunkBlocking(level, chunkZYX[0], chunkZYX[1], chunkZYX[2])
            : cache->tryGetChunk(level, chunkZYX[0], chunkZYX[1], chunkZYX[2]);
    }
    if (result.status == vc::render::ChunkStatus::Error)
        throw std::runtime_error(result.error.empty() ? "chunk fetch failed" : result.error);
    if (result.dtype == vc::render::ChunkDtype::UInt8)
        return chunkResultToArray<uint8_t>(result, volume.fillValue());
    return chunkResultToArray<uint16_t>(result, volume.fillValue());
}

using FloatCoords = nb::ndarray<float, nb::numpy, nb::c_contig>;
using BoolMask = nb::ndarray<bool, nb::numpy, nb::c_contig>;

void validatePlaneVectors(const FloatCoords& values,
                          size_t planeCount,
                          const char* name)
{
    if (values.ndim() != 2 || values.shape(0) != planeCount || values.shape(1) != 3) {
        throw nb::value_error(
            (std::string(name) + " must have shape [N, 3]").c_str());
    }
}

cv::Mat_<cv::Vec3f> coordsArrayToMat(const FloatCoords& coords)
{
    if (coords.ndim() != 3 || coords.shape(2) != 3)
        throw nb::value_error("coords_xyz must have shape [H, W, 3]");
    const int h = static_cast<int>(coords.shape(0));
    const int w = static_cast<int>(coords.shape(1));
    cv::Mat_<cv::Vec3f> mat(h, w);
    const float* src = coords.data();
    for (int y = 0; y < h; ++y) {
        auto* row = mat.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; ++x) {
            const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 3;
            row[x] = cv::Vec3f(src[idx + 0], src[idx + 1], src[idx + 2]);
        }
    }
    return mat;
}

cv::Mat_<uint8_t> skipCoverageFromValidMask(const BoolMask& validMask, int h, int w)
{
    if (validMask.ndim() != 2 || validMask.shape(0) != static_cast<size_t>(h) ||
        validMask.shape(1) != static_cast<size_t>(w)) {
        throw nb::value_error("valid_mask must have shape [H, W] matching coords");
    }
    cv::Mat_<uint8_t> coverage(h, w);
    const bool* src = validMask.data();
    for (int y = 0; y < h; ++y) {
        auto* row = coverage.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
            row[x] = src[idx] ? uint8_t{0} : uint8_t{1};
        }
    }
    return coverage;
}

nb::dict statsToDict(const vc::render::ChunkedPlaneSampler::Stats& stats)
{
    nb::dict out;
    out["covered_pixels"] = stats.coveredPixels;
    out["requested_chunks"] = stats.requestedChunks;
    out["error_chunks"] = stats.errorChunks;
    out["missing_chunks"] = stats.missingChunks;
    out["fallback_levels"] = stats.fallbackLevels;
    out["requested_level_only"] = stats.requestedLevelOnly;
    return out;
}

nb::tuple sampleCoords(Volume& volume,
                       const FloatCoords& coordsXyz,
                       const BoolMask& validMask,
                       int level,
                       const std::string& sampling,
                       int tileSize,
                       bool blocking)
{
    auto coords = coordsArrayToMat(coordsXyz);
    auto coverage = skipCoverageFromValidMask(validMask, coords.rows, coords.cols);
    cv::Mat_<uint8_t> out(coords.rows, coords.cols, uint8_t{0});
    vc::render::ChunkedPlaneSampler::Stats stats;
    {
        nb::gil_scoped_release release;
        const vc::render::ChunkedPlaneSampler::Options options(
            parseSampling(sampling), tileSize);
        if (blocking) {
            stats = vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
                *volume.chunkedCache(),
                level,
                coords,
                out,
                coverage,
                options);
        } else {
            stats = vc::render::ChunkedPlaneSampler::sampleCoordsFineToCoarse(
                *volume.chunkedCache(),
                level,
                coords,
                out,
                coverage,
                options);
        }
    }

    std::vector<uint8_t> image(static_cast<size_t>(coords.rows) * static_cast<size_t>(coords.cols));
    std::vector<uint8_t> sampledValid(image.size());
    const bool* validSrc = validMask.data();
    for (int y = 0; y < coords.rows; ++y) {
        const auto* outRow = out.ptr<uint8_t>(y);
        const auto* covRow = coverage.ptr<uint8_t>(y);
        for (int x = 0; x < coords.cols; ++x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(coords.cols) + static_cast<size_t>(x);
            image[idx] = outRow[x];
            sampledValid[idx] = (validSrc[idx] && covRow[x]) ? uint8_t{1} : uint8_t{0};
        }
    }

    auto imageArr = makeNumpyArray<uint8_t>(std::move(image), {
        static_cast<size_t>(coords.rows), static_cast<size_t>(coords.cols), size_t{1}});
    auto validArr = makeNumpyArray<uint8_t>(std::move(sampledValid), {
        static_cast<size_t>(coords.rows), static_cast<size_t>(coords.cols), size_t{1}});
    nb::dict statsDict = statsToDict(stats);
    statsDict["blocking_prefetch_chunks"] = blocking ? stats.requestedChunks : 0;
    return nb::make_tuple(imageArr, validArr, statsDict);
}

nb::tuple samplePlanes(Volume& volume,
                       const FloatCoords& originsXyz,
                       const FloatCoords& xStepsXyz,
                       const FloatCoords& yStepsXyz,
                       const std::array<int, 2>& shape,
                       int level,
                       const std::string& sampling,
                       int tileSize)
{
    if (originsXyz.ndim() != 2 || originsXyz.shape(1) != 3)
        throw nb::value_error("origins_xyz must have shape [N, 3]");
    const size_t planeCount = originsXyz.shape(0);
    if (planeCount == 0)
        throw nb::value_error("at least one plane is required");
    validatePlaneVectors(xStepsXyz, planeCount, "x_steps_xyz");
    validatePlaneVectors(yStepsXyz, planeCount, "y_steps_xyz");
    if (shape[0] <= 0 || shape[1] <= 0)
        throw nb::value_error("shape must contain positive [height, width]");
    if (level < 0)
        throw std::out_of_range("level must be non-negative");
    if (!volume.hasScaleLevel(level))
        throw std::out_of_range("requested missing zarr scale level " + std::to_string(level));

    const size_t height = static_cast<size_t>(shape[0]);
    const size_t width = static_cast<size_t>(shape[1]);
    if (planeCount > static_cast<size_t>(std::numeric_limits<int>::max()) / height)
        throw std::overflow_error("stacked plane height exceeds OpenCV limits");

    // Stack every plane into one coordinate image. The chunk sampler can then
    // deduplicate and pin the union of dependencies once, which is substantially
    // cheaper than one Python/native transition and one dependency walk per plane.
    cv::Mat_<cv::Vec3f> coords(static_cast<int>(planeCount * height), shape[1]);
    const float* origins = originsXyz.data();
    const float* xSteps = xStepsXyz.data();
    const float* ySteps = yStepsXyz.data();
    for (size_t plane = 0; plane < planeCount; ++plane) {
        const cv::Vec3f origin(
            origins[3 * plane], origins[3 * plane + 1], origins[3 * plane + 2]);
        const cv::Vec3f xStep(
            xSteps[3 * plane], xSteps[3 * plane + 1], xSteps[3 * plane + 2]);
        const cv::Vec3f yStep(
            ySteps[3 * plane], ySteps[3 * plane + 1], ySteps[3 * plane + 2]);
        for (size_t y = 0; y < height; ++y) {
            auto* row = coords.ptr<cv::Vec3f>(static_cast<int>(plane * height + y));
            const cv::Vec3f rowOrigin = origin + static_cast<float>(y) * yStep;
            for (size_t x = 0; x < width; ++x)
                row[x] = rowOrigin + static_cast<float>(x) * xStep;
        }
    }

    const size_t pixelCount = planeCount * height * width;
    std::vector<uint8_t> images(pixelCount, uint8_t{0});
    std::vector<uint8_t> valid(pixelCount, uint8_t{0});
    cv::Mat_<uint8_t> out(coords.rows, coords.cols, images.data());
    cv::Mat_<uint8_t> coverage(coords.rows, coords.cols, valid.data());
    vc::render::ChunkedPlaneSampler::Stats stats;
    {
        nb::gil_scoped_release release;
        stats = vc::render::ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(
            *volume.chunkedCache(),
            level,
            coords,
            out,
            coverage,
            vc::render::ChunkedPlaneSampler::Options(parseSampling(sampling), tileSize));
    }

    nb::dict statsDict = statsToDict(stats);
    statsDict["blocking_prefetch_chunks"] = stats.requestedChunks;
    const std::array<size_t, 3> outShape{planeCount, height, width};
    return nb::make_tuple(
        makeNumpyArray(std::move(images), outShape),
        makeNumpyArray(std::move(valid), outShape),
        statsDict);
}

std::string joinUrl(std::string base, const std::string& key)
{
    while (!base.empty() && base.back() == '/')
        base.pop_back();
    return base.empty() ? std::string{} : base + "/" + key;
}

nb::list chunkDependenciesToPython(
    Volume& volume,
    const std::vector<vc::render::ChunkKey>& keys)
{
    nb::list out;
    const std::string remoteUrl = volume.remoteUrl();
    auto* cache = dynamic_cast<vc::render::ChunkCache*>(volume.chunkedCache());
    if (!cache)
        throw std::runtime_error("VC3D dependency metadata requires a ChunkCache-backed volume");
    for (const auto& key : keys) {
        const auto dependency = cache->persistentChunkDependency(
            key.level,
            key.iz,
            key.iy,
            key.ix);
        nb::dict item;
        item["level"] = key.level;
        item["iz"] = key.iz;
        item["iy"] = key.iy;
        item["ix"] = key.ix;
        item["key"] = std::to_string(key.level) + "/" +
                      std::to_string(key.iz) + "/" +
                      std::to_string(key.iy) + "/" +
                      std::to_string(key.ix);
        item["valid"] = dependency.valid;
        item["remote_chunk_key"] = dependency.sourceChunkKey
            ? *dependency.sourceChunkKey
            : std::string{};
        item["remote_url"] = dependency.sourceChunkKey
            ? joinUrl(remoteUrl, *dependency.sourceChunkKey)
            : std::string{};
        item["cache_path"] = dependency.persistentPath.string();
        item["empty_path"] = dependency.persistentEmptyPath.string();
        item["persistent_extension"] = dependency.persistentExtension;
        item["cache_payload_format"] = dependency.sourcePayloadMatchesPersistentCache
            ? std::string{"source_bytes"}
            : std::string{"unsupported"};
        item["source_payload_matches_cache"] = dependency.sourcePayloadMatchesPersistentCache;
        out.append(std::move(item));
    }
    return out;
}

nb::list collectCoordsDependencies(
    Volume& volume,
    const FloatCoords& coordsXyz,
    const BoolMask& validMask,
    int level,
    const std::string& sampling,
    int tileSize)
{
    auto coords = coordsArrayToMat(coordsXyz);
    auto coverage = skipCoverageFromValidMask(validMask, coords.rows, coords.cols);
    std::vector<vc::render::ChunkKey> keys;
    {
        nb::gil_scoped_release release;
        keys = vc::render::ChunkedPlaneSampler::collectCoordsDependencies(
            *volume.chunkedCache(),
            level,
            coords,
            coverage,
            vc::render::ChunkedPlaneSampler::Options(parseSampling(sampling), tileSize));
    }
    return chunkDependenciesToPython(volume, keys);
}

nb::list collectBBoxDependencies(
    Volume& volume,
    const std::array<int, 3>& offset,
    const std::array<size_t, 3>& shape,
    int level)
{
    return chunkDependenciesToPython(
        volume,
        collectChunkKeys(volume, offset, shape, level));
}

} // namespace

NB_MODULE(volume, m)
{
    m.doc() = "Python bindings for Volume Cartographer zarr volume access";

    nb::class_<Volume>(m, "Volume")
        .def_static("open",
            [](const std::string& path) {
                return Volume::New(path);
            },
            "path"_a)
        .def_static("open_url",
            [](const std::string& url, const std::filesystem::path& cacheRoot) {
                return Volume::NewFromUrl(url, cacheRoot);
            },
            "url"_a,
            "cache_root"_a = std::filesystem::path{})
        .def_prop_ro("is_remote", &Volume::isRemote)
        .def_prop_ro("path", [](const Volume& self) { return self.path().string(); })
        .def_prop_ro("remote_url", [](const Volume& self) { return self.remoteUrl(); })
        .def_prop_ro("remote_locator", [](const Volume& self) { return self.remoteLocator(); })
        .def_prop_ro("base_scale_level", &Volume::baseScaleLevel)
        .def_prop_ro("remote_cache_root", [](const Volume& self) { return self.remoteCacheRoot().string(); })
        .def_prop_ro("remote_cache_path", [](const Volume& self) { return self.remotePersistentCachePath().string(); })
        .def_prop_ro("id", &Volume::id)
        .def_prop_ro("name", &Volume::name)
        .def_prop_ro("metadata", [](const Volume& self) { return jsonToPython(self.metadata()); })
        .def_prop_ro("root_attrs", [](const Volume& self) { return jsonToPython(self.rootAttributes()); })
        .def_prop_ro("shape", [](const Volume& self) { return tuple3(self.shape()); })
        .def_prop_ro("shape_xyz", [](const Volume& self) { return tuple3(self.shapeXyz()); })
        .def_prop_ro("dtype", [](const Volume& self) { return dtypeName(self.dtype()); })
        .def_prop_ro("dtype_size", &Volume::dtypeSize)
        .def_prop_ro("fill_value", &Volume::fillValue)
        .def_prop_ro("num_scales", &Volume::numScales)
        .def("shape_at", [](const Volume& self, int level) { return tuple3(self.shape(level)); }, "level"_a)
        .def("chunk_shape", [](const Volume& self, int level) { return tuple3(self.chunkShape(level)); }, "level"_a = 0)
        .def("chunk_grid_shape", [](const Volume& self, int level) { return tuple3(self.chunkGridShape(level)); }, "level"_a = 0)
        .def("chunk_count", &Volume::chunkCount, "level"_a = 0)
        .def("has_scale_level", &Volume::hasScaleLevel, "level"_a)
        .def("present_scale_levels", &Volume::presentScaleLevels)
        .def("set_cache_budget",
             [](Volume& volume, size_t bytes) {
                 volume.setCacheBudget(bytes);
             },
             "bytes"_a)
        .def("invalidate_cache", &Volume::invalidateCache)
        .def("read_zyx", &readZYX,
            "offset"_a,
            "shape"_a,
            "level"_a = 0,
            "missing_policy"_a = "error")
        .def("sample_zyx_block", &sampleZYXBlock,
            "offset"_a,
            "shape"_a,
            "level"_a = 0,
            "blocking"_a = true)
        .def("read_xyz", &readXYZ,
            "offset"_a,
            "shape"_a,
            "level"_a = 0,
            "missing_policy"_a = "error")
        .def("prefetch_zyx", &prefetchZYX,
            "offset"_a,
            "shape"_a,
            "level"_a = 0,
            "wait"_a = false)
        .def("read_chunk", &readChunk,
            "level"_a,
            "chunk_zyx"_a,
            "blocking"_a = true)
        .def("sample_coords", &sampleCoords,
            "coords_xyz"_a,
            "valid_mask"_a,
            "level"_a = 0,
            "sampling"_a = "trilinear",
            "tile_size"_a = 32,
            "blocking"_a = true)
        .def("sample_planes", &samplePlanes,
            "origins_xyz"_a,
            "x_steps_xyz"_a,
            "y_steps_xyz"_a,
            "shape"_a,
            "level"_a = 0,
            "sampling"_a = "trilinear",
            "tile_size"_a = 32,
            "Sample several arbitrary affine planes in one blocking, chunk-aware call. "
            "Origins and step vectors use logical level-0 XYZ voxel coordinates; "
            "returns (images, valid, stats) with image shape [N, H, W].")
        .def("collect_coords_dependencies", &collectCoordsDependencies,
            "coords_xyz"_a,
            "valid_mask"_a,
            "level"_a = 0,
            "sampling"_a = "trilinear",
            "tile_size"_a = 32)
        .def("collect_bbox_dependencies", &collectBBoxDependencies,
            "offset"_a,
            "shape"_a,
            "level"_a = 0)
        .def("__getitem__",
            [](Volume& self, const nb::object& key) {
                const auto region = parseSliceKey(key, self.shape());
                return readZYX(self, region.offset, region.shape, 0, "error");
            },
            "key"_a);
}
