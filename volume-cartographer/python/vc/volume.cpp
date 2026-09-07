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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
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

template <typename T>
nb::ndarray<T, nb::numpy, nb::c_contig> makeNumpyArray4(
    std::vector<T>&& data,
    const std::array<size_t, 4>& shape)
{
    auto* heap = new std::vector<T>(std::move(data));
    nb::capsule owner(heap, [](void* ptr) noexcept {
        delete static_cast<std::vector<T>*>(ptr);
    });
    return nb::ndarray<T, nb::numpy, nb::c_contig>(
        heap->data(), {shape[0], shape[1], shape[2], shape[3]}, owner);
}

void requireFloatShape(const FloatCoords& value,
                       const std::vector<size_t>& expected,
                       const char* name)
{
    if (value.ndim() != expected.size())
        throw nb::value_error((std::string(name) + " has the wrong rank").c_str());
    for (size_t axis = 0; axis < expected.size(); ++axis) {
        if (value.shape(axis) != expected[axis]) {
            throw nb::value_error(
                (std::string(name) + " has the wrong shape").c_str());
        }
    }
}

void requireFinite(const FloatCoords& value, const char* name)
{
    size_t count = 1;
    for (size_t axis = 0; axis < value.ndim(); ++axis)
        count *= value.shape(axis);
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(value.data()[index])) {
            throw nb::value_error(
                (std::string(name) + " must contain only finite values").c_str());
        }
    }
}

float sampleControl1D(const FloatCoords& control,
                      float coordinate,
                      float origin,
                      float spacing)
{
    const size_t count = control.shape(0);
    if (count == 1)
        return control.data()[0];
    const float u = std::clamp(
        (coordinate - origin) / spacing, 0.0f, static_cast<float>(count - 1));
    const size_t lo = static_cast<size_t>(std::floor(u));
    const size_t hi = std::min(lo + 1, count - 1);
    const float fraction = u - static_cast<float>(lo);
    return control.data()[lo] * (1.0f - fraction) +
           control.data()[hi] * fraction;
}

float sampleControl2D(const FloatCoords& control,
                      float coordinateA,
                      float coordinateB,
                      float originA,
                      float originB,
                      float spacingA,
                      float spacingB)
{
    const size_t height = control.shape(0);
    const size_t width = control.shape(1);
    const float ua = std::clamp(
        (coordinateA - originA) / spacingA,
        0.0f,
        static_cast<float>(height - 1));
    const float ub = std::clamp(
        (coordinateB - originB) / spacingB,
        0.0f,
        static_cast<float>(width - 1));
    const size_t a0 = static_cast<size_t>(std::floor(ua));
    const size_t b0 = static_cast<size_t>(std::floor(ub));
    const size_t a1 = std::min(a0 + 1, height - 1);
    const size_t b1 = std::min(b0 + 1, width - 1);
    const float wa = ua - static_cast<float>(a0);
    const float wb = ub - static_cast<float>(b0);
    const float* data = control.data();
    const float c00 = data[a0 * width + b0];
    const float c01 = data[a0 * width + b1];
    const float c10 = data[a1 * width + b0];
    const float c11 = data[a1 * width + b1];
    const float c0 = c00 * (1.0f - wb) + c01 * wb;
    const float c1 = c10 * (1.0f - wb) + c11 * wb;
    return c0 * (1.0f - wa) + c1 * wa;
}

nb::tuple generateFiber3DCoords(
    const std::array<int, 3>& outputShapeZyx,
    const FloatCoords& cpLocalZyx,
    const FloatCoords& cpVolumeZyx,
    const FloatCoords& outputToSourceZyx,
    const std::string& smoothMode,
    const nb::list& smoothControls,
    const FloatCoords& smoothOriginZyx,
    const FloatCoords& smoothSpacingZyx,
    float volumeSpacingBase,
    const std::array<int, 3>& baseShapeZyx)
{
    requireFloatShape(cpLocalZyx, {3}, "cp_local_zyx");
    requireFloatShape(cpVolumeZyx, {3}, "cp_volume_zyx");
    requireFloatShape(outputToSourceZyx, {3, 3}, "output_to_source_zyx");
    requireFloatShape(smoothOriginZyx, {3}, "smooth_origin_zyx");
    requireFloatShape(smoothSpacingZyx, {3}, "smooth_spacing_zyx");
    requireFinite(cpLocalZyx, "cp_local_zyx");
    requireFinite(cpVolumeZyx, "cp_volume_zyx");
    requireFinite(outputToSourceZyx, "output_to_source_zyx");
    requireFinite(smoothOriginZyx, "smooth_origin_zyx");
    requireFinite(smoothSpacingZyx, "smooth_spacing_zyx");
    if (!std::isfinite(volumeSpacingBase) || volumeSpacingBase <= 0.0f)
        throw nb::value_error("volume_spacing_base must be finite and positive");

    size_t pointCount = 1;
    for (size_t axis = 0; axis < 3; ++axis) {
        if (outputShapeZyx[axis] <= 0)
            throw nb::value_error("output_shape_zyx values must be positive");
        if (baseShapeZyx[axis] <= 0)
            throw nb::value_error("base_shape_zyx values must be positive");
        if (!(smoothSpacingZyx.data()[axis] > 0.0f))
            throw nb::value_error("smooth_spacing_zyx values must be positive");
        const size_t dim = static_cast<size_t>(outputShapeZyx[axis]);
        if (pointCount > std::numeric_limits<size_t>::max() / dim)
            throw nb::value_error("output_shape_zyx allocation overflows size_t");
        pointCount *= dim;
    }
    if (pointCount > std::numeric_limits<size_t>::max() / 3 / sizeof(float))
        throw nb::value_error("coordinate allocation overflows size_t");

    size_t expectedControls = 0;
    if (smoothMode == "none")
        expectedControls = 0;
    else if (smoothMode == "1d" || smoothMode == "2d")
        expectedControls = 1;
    else if (smoothMode == "3d")
        expectedControls = 3;
    else
        throw nb::value_error("smooth_mode must be one of: none, 1d, 2d, 3d");
    if (smoothControls.size() != expectedControls)
        throw nb::value_error("smooth_controls count does not match smooth_mode");

    std::vector<FloatCoords> controls;
    controls.reserve(expectedControls);
    for (size_t index = 0; index < expectedControls; ++index) {
        auto control = nb::cast<FloatCoords>(smoothControls[index]);
        const size_t expectedRank = smoothMode == "1d" ? 1 : 2;
        if (control.ndim() != expectedRank)
            throw nb::value_error("smooth control has the wrong rank");
        for (size_t axis = 0; axis < control.ndim(); ++axis) {
            if (control.shape(axis) == 0)
                throw nb::value_error("smooth controls must not be empty");
        }
        requireFinite(control, "smooth control");
        controls.emplace_back(std::move(control));
    }

    std::vector<float> coords(pointCount * 3);
    std::vector<uint8_t> valid(pointCount);
    const float* cpLocal = cpLocalZyx.data();
    const float* cpVolume = cpVolumeZyx.data();
    const float* matrix = outputToSourceZyx.data();
    const float* origin = smoothOriginZyx.data();
    const float* spacing = smoothSpacingZyx.data();
    const size_t outY = static_cast<size_t>(outputShapeZyx[1]);
    const size_t outX = static_cast<size_t>(outputShapeZyx[2]);
    {
        nb::gil_scoped_release release;
        size_t point = 0;
        for (int z = 0; z < outputShapeZyx[0]; ++z) {
            const float relativeZ = static_cast<float>(z) - cpLocal[0];
            for (int y = 0; y < outputShapeZyx[1]; ++y) {
                const float relativeY = static_cast<float>(y) - cpLocal[1];
                for (int x = 0; x < outputShapeZyx[2]; ++x, ++point) {
                    const float relativeX = static_cast<float>(x) - cpLocal[2];
                    float sourceZ = matrix[0] * relativeZ + matrix[1] * relativeY +
                                    matrix[2] * relativeX;
                    float sourceY = matrix[3] * relativeZ + matrix[4] * relativeY +
                                    matrix[5] * relativeX;
                    float sourceX = matrix[6] * relativeZ + matrix[7] * relativeY +
                                    matrix[8] * relativeX;
                    if (smoothMode == "1d") {
                        sourceZ -= sampleControl1D(
                            controls[0], sourceX, origin[2], spacing[2]);
                    } else if (smoothMode == "2d") {
                        sourceZ -= sampleControl2D(
                            controls[0], sourceY, sourceX,
                            origin[1], origin[2], spacing[1], spacing[2]);
                    } else if (smoothMode == "3d") {
                        sourceX -= sampleControl2D(
                            controls[2], sourceZ, sourceY,
                            origin[0], origin[1], spacing[0], spacing[1]);
                        sourceY -= sampleControl2D(
                            controls[1], sourceZ, sourceX,
                            origin[0], origin[2], spacing[0], spacing[2]);
                        sourceZ -= sampleControl2D(
                            controls[0], sourceY, sourceX,
                            origin[1], origin[2], spacing[1], spacing[2]);
                    }

                    const float baseZ = (cpVolume[0] + sourceZ) * volumeSpacingBase;
                    const float baseY = (cpVolume[1] + sourceY) * volumeSpacingBase;
                    const float baseX = (cpVolume[2] + sourceX) * volumeSpacingBase;
                    const size_t offset = point * 3;
                    coords[offset] = baseZ;
                    coords[offset + 1] = baseY;
                    coords[offset + 2] = baseX;
                    valid[point] = static_cast<uint8_t>(
                        std::isfinite(baseZ) && std::isfinite(baseY) &&
                        std::isfinite(baseX) && baseZ >= 0.0f && baseY >= 0.0f &&
                        baseX >= 0.0f &&
                        baseZ <= static_cast<float>(baseShapeZyx[0] - 1) &&
                        baseY <= static_cast<float>(baseShapeZyx[1] - 1) &&
                        baseX <= static_cast<float>(baseShapeZyx[2] - 1));
                }
            }
        }
    }

    auto coordsArray = makeNumpyArray4<float>(
        std::move(coords),
        {static_cast<size_t>(outputShapeZyx[0]), outY, outX, size_t{3}});
    auto validArray = makeNumpyArray<uint8_t>(
        std::move(valid),
        {static_cast<size_t>(outputShapeZyx[0]), outY, outX});
    return nb::make_tuple(coordsArray, validArray);
}

struct BlurStencilEntry {
    int z;
    int y;
    int x;
    float weight;
};

nb::ndarray<float, nb::numpy, nb::c_contig> fiber3DValueTransform(
    const FloatCoords& volume,
    const BoolMask& validMask,
    const std::string& normalization,
    float contrast,
    float brightness,
    float gamma)
{
    if (volume.ndim() != 3 || validMask.ndim() != 3)
        throw nb::value_error("volume and valid_mask must have shape (Z,Y,X)");
    std::array<size_t, 3> shape{};
    size_t pointCount = 1;
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        shape[dimension] = volume.shape(dimension);
        if (shape[dimension] == 0)
            throw nb::value_error("volume dimensions must be positive");
        if (validMask.shape(dimension) != shape[dimension])
            throw nb::value_error("valid_mask shape must match volume");
        if (pointCount > std::numeric_limits<size_t>::max() / shape[dimension])
            throw nb::value_error("volume allocation overflows size_t");
        pointCount *= shape[dimension];
    }
    if (pointCount > std::numeric_limits<size_t>::max() / sizeof(float))
        throw nb::value_error("volume allocation overflows size_t");
    if (normalization != "none" && normalization != "raw" &&
        normalization != "identity" && normalization != "zscore" &&
        normalization != "minmax") {
        throw nb::value_error("unsupported image normalization");
    }
    if (!std::isfinite(contrast) || contrast <= 0.0f ||
        !std::isfinite(brightness) ||
        !std::isfinite(gamma) || gamma <= 0.0f) {
        throw nb::value_error(
            "contrast and gamma must be positive and all parameters finite");
    }

    std::vector<float> output(pointCount);
    const float* source = volume.data();
    const bool* mask = validMask.data();
    {
        nb::gil_scoped_release release;
        size_t validCount = 0;
        double sum = 0.0;
        float sourceMin = std::numeric_limits<float>::infinity();
        float sourceMax = -std::numeric_limits<float>::infinity();
        for (size_t index = 0; index < pointCount; ++index) {
            if (!mask[index])
                continue;
            const float value = source[index];
            sum += static_cast<double>(value);
            sourceMin = std::min(sourceMin, value);
            sourceMax = std::max(sourceMax, value);
            ++validCount;
        }
        const bool anyValid = validCount != 0;
        if (!anyValid) {
            for (size_t index = 0; index < pointCount; ++index) {
                const float value = source[index];
                sum += static_cast<double>(value);
                sourceMin = std::min(sourceMin, value);
                sourceMax = std::max(sourceMax, value);
            }
            validCount = pointCount;
        }

        const float sourceMean = static_cast<float>(
            sum / static_cast<double>(validCount));
        float normalizeDivisor = 1.0f;
        const bool useZscore = anyValid && normalization == "zscore";
        const bool useMinmax = anyValid && normalization == "minmax";
        if (useZscore) {
            double squared = 0.0;
            size_t varianceCount = 0;
            for (size_t index = 0; index < pointCount; ++index) {
                if (mask[index]) {
                    const double delta = static_cast<double>(source[index]) -
                                         static_cast<double>(sourceMean);
                    squared += delta * delta;
                    ++varianceCount;
                }
            }
            const float standardDeviation = std::max(
                1.0e-6f,
                static_cast<float>(std::sqrt(
                    squared / static_cast<double>(varianceCount))));
            normalizeDivisor = standardDeviation;
        } else if (useMinmax) {
            normalizeDivisor = std::max(1.0e-6f, sourceMax - sourceMin);
        }

        const auto normalizeValue = [&](float value) {
            if (useZscore)
                return (value - sourceMean) / normalizeDivisor;
            if (useMinmax)
                return (value - sourceMin) / normalizeDivisor;
            return value;
        };
        const float normalizedMean = normalizeValue(sourceMean);
        const auto transformValue = [&](float value) {
            const float normalized = normalizeValue(value);
            return (normalized - normalizedMean) * contrast +
                   normalizedMean + brightness;
        };
        const float transformedMin = transformValue(sourceMin);
        const float transformedMax = transformValue(sourceMax);
        const float transformedRange = std::max(
            1.0e-6f, transformedMax - transformedMin);
        for (size_t index = 0; index < pointCount; ++index) {
            float value = transformValue(source[index]);
            if (gamma != 1.0f) {
                const float unit = std::clamp(
                    (value - transformedMin) / transformedRange, 0.0f, 1.0f);
                value = std::pow(unit, gamma) *
                            (transformedMax - transformedMin) +
                        transformedMin;
            }
            output[index] = value;
        }
    }
    return makeNumpyArray<float>(std::move(output), shape);
}

std::array<float, 3> normalizeVector(const std::array<float, 3>& value,
                                     const char* name)
{
    const float norm = std::sqrt(
        value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
    if (!std::isfinite(norm) || norm <= 0.0f)
        throw nb::value_error(
            (std::string(name) + " must have positive finite norm").c_str());
    return {value[0] / norm, value[1] / norm, value[2] / norm};
}

std::array<float, 3> crossVector(const std::array<float, 3>& a,
                                 const std::array<float, 3>& b)
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

std::vector<BlurStencilEntry> makeBlurStencil(const std::array<float, 3>& axis,
                                              float sigma)
{
    if (sigma <= 0.0f)
        return {{0, 0, 0, 1.0f}};
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
    float kernelSum = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset) {
        const float scaled = static_cast<float>(offset) / sigma;
        const float weight = std::exp(-0.5f * scaled * scaled);
        kernel[static_cast<size_t>(offset + radius)] = weight;
        kernelSum += weight;
    }
    for (float& weight : kernel)
        weight /= kernelSum;

    std::vector<BlurStencilEntry> stencil;
    stencil.reserve(kernel.size() * 8);
    for (int tap = -radius; tap <= radius; ++tap) {
        const std::array<float, 3> delta{
            static_cast<float>(tap) * axis[0],
            static_cast<float>(tap) * axis[1],
            static_cast<float>(tap) * axis[2],
        };
        const std::array<int, 3> low{
            static_cast<int>(std::floor(delta[0])),
            static_cast<int>(std::floor(delta[1])),
            static_cast<int>(std::floor(delta[2])),
        };
        const std::array<float, 3> fraction{
            delta[0] - static_cast<float>(low[0]),
            delta[1] - static_cast<float>(low[1]),
            delta[2] - static_cast<float>(low[2]),
        };
        const float tapWeight = kernel[static_cast<size_t>(tap + radius)];
        for (int cornerZ = 0; cornerZ < 2; ++cornerZ) {
            const float weightZ = cornerZ != 0 ? fraction[0] : 1.0f - fraction[0];
            for (int cornerY = 0; cornerY < 2; ++cornerY) {
                const float weightY = cornerY != 0 ? fraction[1] : 1.0f - fraction[1];
                for (int cornerX = 0; cornerX < 2; ++cornerX) {
                    const float weightX = cornerX != 0 ? fraction[2] : 1.0f - fraction[2];
                    const BlurStencilEntry candidate{
                        low[0] + cornerZ,
                        low[1] + cornerY,
                        low[2] + cornerX,
                        tapWeight * weightZ * weightY * weightX,
                    };
                    if (candidate.weight == 0.0f)
                        continue;
                    auto existing = std::find_if(
                        stencil.begin(), stencil.end(), [&](const BlurStencilEntry& entry) {
                            return entry.z == candidate.z && entry.y == candidate.y &&
                                   entry.x == candidate.x;
                        });
                    if (existing == stencil.end())
                        stencil.push_back(candidate);
                    else
                        existing->weight += candidate.weight;
                }
            }
        }
    }
    return stencil;
}

std::optional<size_t> axisAlignedDimension(const std::array<float, 3>& axis)
{
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        const size_t a = (dimension + 1) % 3;
        const size_t b = (dimension + 2) % 3;
        if (std::abs(std::abs(axis[dimension]) - 1.0f) <= 1.0e-6f &&
            std::abs(axis[a]) <= 1.0e-6f && std::abs(axis[b]) <= 1.0e-6f) {
            return dimension;
        }
    }
    return std::nullopt;
}

std::vector<float> makeGaussianKernel(float sigma)
{
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
    float sum = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset) {
        const float scaled = static_cast<float>(offset) / sigma;
        const float weight = std::exp(-0.5f * scaled * scaled);
        kernel[static_cast<size_t>(offset + radius)] = weight;
        sum += weight;
    }
    for (float& weight : kernel)
        weight /= sum;
    return kernel;
}

void applyAxisAlignedBlur(const float* source,
                          float* destination,
                          const std::array<size_t, 3>& shape,
                          size_t dimension,
                          float sigma)
{
    const auto kernel = makeGaussianKernel(sigma);
    const int radius = static_cast<int>((kernel.size() - 1) / 2);
    const int depth = static_cast<int>(shape[0]);
    const int height = static_cast<int>(shape[1]);
    const int width = static_cast<int>(shape[2]);
    const size_t plane = shape[1] * shape[2];
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = 0.0f;
                for (int offset = -radius; offset <= radius; ++offset) {
                    int sourceZ = z;
                    int sourceY = y;
                    int sourceX = x;
                    if (dimension == 0)
                        sourceZ = std::clamp(z + offset, 0, depth - 1);
                    else if (dimension == 1)
                        sourceY = std::clamp(y + offset, 0, height - 1);
                    else
                        sourceX = std::clamp(x + offset, 0, width - 1);
                    const size_t index = static_cast<size_t>(sourceZ) * plane +
                                         static_cast<size_t>(sourceY) * shape[2] +
                                         static_cast<size_t>(sourceX);
                    value += source[index] *
                             kernel[static_cast<size_t>(offset + radius)];
                }
                destination[static_cast<size_t>(z) * plane +
                            static_cast<size_t>(y) * shape[2] +
                            static_cast<size_t>(x)] = value;
            }
        }
    }
}

void applyBlurStencil(const float* source,
                      float* destination,
                      const std::array<size_t, 3>& shape,
                      const std::vector<BlurStencilEntry>& stencil)
{
    const int depth = static_cast<int>(shape[0]);
    const int height = static_cast<int>(shape[1]);
    const int width = static_cast<int>(shape[2]);
    const size_t plane = shape[1] * shape[2];
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = 0.0f;
                for (const auto& entry : stencil) {
                    const int sourceZ = std::clamp(z + entry.z, 0, depth - 1);
                    const int sourceY = std::clamp(y + entry.y, 0, height - 1);
                    const int sourceX = std::clamp(x + entry.x, 0, width - 1);
                    const size_t index = static_cast<size_t>(sourceZ) * plane +
                                         static_cast<size_t>(sourceY) * shape[2] +
                                         static_cast<size_t>(sourceX);
                    value += source[index] * entry.weight;
                }
                destination[static_cast<size_t>(z) * plane +
                            static_cast<size_t>(y) * shape[2] +
                            static_cast<size_t>(x)] = value;
            }
        }
    }
}

nb::ndarray<float, nb::numpy, nb::c_contig> anisotropicBlur3D(
    const FloatCoords& volume,
    const FloatCoords& axisZyx,
    float sigmaAlong,
    float sigmaAcross)
{
    if (volume.ndim() != 3)
        throw nb::value_error("volume must have shape (Z,Y,X)");
    requireFloatShape(axisZyx, {3}, "axis_zyx");
    requireFinite(axisZyx, "axis_zyx");
    if (!std::isfinite(sigmaAlong) || sigmaAlong < 0.0f ||
        !std::isfinite(sigmaAcross) || sigmaAcross < 0.0f) {
        throw nb::value_error("blur sigmas must be finite and nonnegative");
    }
    std::array<size_t, 3> shape{};
    size_t pointCount = 1;
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        shape[dimension] = volume.shape(dimension);
        if (shape[dimension] == 0)
            throw nb::value_error("volume dimensions must be positive");
        if (shape[dimension] > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw nb::value_error("volume dimensions exceed native index range");
        if (pointCount > std::numeric_limits<size_t>::max() / shape[dimension])
            throw nb::value_error("volume allocation overflows size_t");
        pointCount *= shape[dimension];
    }
    if (pointCount > std::numeric_limits<size_t>::max() / sizeof(float))
        throw nb::value_error("volume allocation overflows size_t");

    const float* rawAxis = axisZyx.data();
    const auto along = normalizeVector(
        {rawAxis[0], rawAxis[1], rawAxis[2]}, "axis_zyx");
    const std::array<float, 3> reference =
        std::abs(along[0]) > 0.9f
            ? std::array<float, 3>{0.0f, 1.0f, 0.0f}
            : std::array<float, 3>{1.0f, 0.0f, 0.0f};
    const auto acrossA = normalizeVector(
        crossVector(along, reference), "perpendicular axis");
    const auto acrossB = normalizeVector(
        crossVector(along, acrossA), "perpendicular axis");
    const std::array<std::array<float, 3>, 3> axes{along, acrossA, acrossB};
    const std::array<float, 3> sigmas{sigmaAlong, sigmaAcross, sigmaAcross};

    std::vector<float> first(pointCount);
    std::vector<float> second(pointCount);
    const float* source = volume.data();
    float* destination = first.data();
    size_t completedPasses = 0;
    {
        nb::gil_scoped_release release;
        for (size_t pass = 0; pass < axes.size(); ++pass) {
            if (sigmas[pass] <= 0.0f)
                continue;
            const auto alignedDimension = axisAlignedDimension(axes[pass]);
            if (alignedDimension) {
                applyAxisAlignedBlur(
                    source, destination, shape, *alignedDimension, sigmas[pass]);
            } else {
                const auto stencil = makeBlurStencil(axes[pass], sigmas[pass]);
                applyBlurStencil(source, destination, shape, stencil);
            }
            source = destination;
            destination = destination == first.data() ? second.data() : first.data();
            ++completedPasses;
        }
        if (completedPasses == 0)
            std::copy(volume.data(), volume.data() + pointCount, first.data());
        else if (source != first.data())
            first.swap(second);
    }
    return makeNumpyArray<float>(std::move(first), shape);
}

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
    out["sample_workers"] = stats.sampleWorkers;
    out["dependency_seconds"] = stats.dependencySeconds;
    out["chunk_resolve_seconds"] = stats.chunkResolveSeconds;
    out["sample_seconds"] = stats.sampleSeconds;
    out["dependency_points"] = stats.dependencyPoints;
    out["dependency_footprint_reuses"] = stats.dependencyFootprintReuses;
    out["dependency_footprint_changes"] = stats.dependencyFootprintChanges;
    out["dependency_key_lookups"] = stats.dependencyKeyLookups;
    out["dependency_hash_probes"] = stats.dependencyHashProbes;
    out["dependency_chunk_run_reuses"] = stats.dependencyChunkRunReuses;
    out["sample_direct_gather_points"] = stats.sampleDirectGatherPoints;
    out["sample_boundary_points"] = stats.sampleBoundaryPoints;
    out["sample_pinned_map_lookups"] = stats.samplePinnedMapLookups;
    out["sample_chunk_run_reuses"] = stats.sampleChunkRunReuses;
    return out;
}

nb::tuple sampleCoords(Volume& volume,
                       const FloatCoords& coordsXyz,
                       const BoolMask& validMask,
                       int level,
                       const std::string& sampling,
                       int tileSize,
                       bool blocking,
                       int samplerThreads,
                       bool chunkLocalityFastPath,
                       bool collectLocalityStats)
{
    if (samplerThreads < 0)
        throw nb::value_error("sampler_threads must be non-negative");
    auto coords = coordsArrayToMat(coordsXyz);
    auto coverage = skipCoverageFromValidMask(validMask, coords.rows, coords.cols);
    cv::Mat_<uint8_t> out(coords.rows, coords.cols, uint8_t{0});
    vc::render::ChunkedPlaneSampler::Stats stats;
    auto* chunkCache = dynamic_cast<vc::render::ChunkCache*>(volume.chunkedCache());
    std::optional<vc::render::ChunkCache::Stats> cacheBefore;
    std::optional<vc::render::ChunkCache::Stats> cacheAfter;
    {
        nb::gil_scoped_release release;
        vc::render::ChunkedPlaneSampler::Options options(
            parseSampling(sampling), tileSize);
        options.parallelThreads = samplerThreads;
        options.chunkLocalityFastPath = chunkLocalityFastPath;
        options.collectLocalityStats = collectLocalityStats;
        if (chunkCache)
            cacheBefore = chunkCache->stats();
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
        if (chunkCache)
            cacheAfter = chunkCache->stats();
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
    statsDict["sampler_threads"] = samplerThreads;
    if (cacheBefore && cacheAfter) {
        statsDict["decode_attempts"] = cacheAfter->decodeAttempts - cacheBefore->decodeAttempts;
        statsDict["decoded_chunks"] = cacheAfter->decodedChunks - cacheBefore->decodedChunks;
        statsDict["decode_input_bytes"] = cacheAfter->decodeInputBytes - cacheBefore->decodeInputBytes;
        statsDict["decoded_chunk_bytes"] = cacheAfter->decodedChunkBytes - cacheBefore->decodedChunkBytes;
        statsDict["decode_seconds"] = static_cast<double>(
            cacheAfter->decodeNanoseconds - cacheBefore->decodeNanoseconds) / 1.0e9;
    }
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

    m.def("set_chunk_cache_budget",
          [](std::size_t bytes) {
              if (bytes == 0)
                  throw std::invalid_argument("bytes must be positive");
              vc::render::processChunkCacheService()
                  ->configureDecodedByteCapacity(bytes);
          },
          "bytes"_a,
          "Set the process-wide decoded regular-chunk cache capacity.");
    m.def("set_chunk_cache_io_threads",
          [](std::size_t count) {
              if (count == 0)
                  throw std::invalid_argument("count must be positive");
              vc::render::processChunkCacheService()
                  ->configureFetchConcurrency(count, false);
          },
          "count"_a,
          "Set fixed process-wide regular chunk source-read concurrency.");
    m.def("set_chunk_cache_decode_threads",
          [](std::size_t count) {
              if (count == 0)
                  throw std::invalid_argument("count must be positive");
              vc::render::processChunkCacheService()
                  ->configureDecodeConcurrency(count);
          },
          "count"_a,
          "Set fixed process-wide regular chunk decode concurrency.");
    m.def("generate_fiber3d_coords",
          &generateFiber3DCoords,
          "output_shape_zyx"_a,
          "cp_local_zyx"_a,
          "cp_volume_zyx"_a,
          "output_to_source_zyx"_a,
          "smooth_mode"_a,
          "smooth_controls"_a,
          "smooth_origin_zyx"_a,
          "smooth_spacing_zyx"_a,
          "volume_spacing_base"_a,
          "base_shape_zyx"_a,
          "Generate Fiber3D base coordinates and validity in one native pass.");
    m.def("anisotropic_blur_3d",
          &anisotropicBlur3D,
          "volume"_a,
          "axis_zyx"_a,
          "sigma_along"_a,
          "sigma_across"_a,
          "Apply the Fiber3D three-pass oriented Gaussian blur on one CPU thread.");
    m.def("fiber3d_value_transform",
          &fiber3DValueTransform,
          "volume"_a,
          "valid_mask"_a,
          "normalization"_a,
          "contrast"_a,
          "brightness"_a,
          "gamma"_a,
          "Apply Fiber3D normalization, contrast, brightness, and gamma in one CPU pass.");

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
            "blocking"_a = true,
            "sampler_threads"_a = 0,
            "chunk_locality_fast_path"_a = true,
            "collect_locality_stats"_a = false)
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
