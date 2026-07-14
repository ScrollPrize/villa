#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
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
#include "vc/core/types/Array3D.hpp"
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
        .def("set_cache_budget", &Volume::setCacheBudget, "bytes"_a)
        .def("set_io_threads", &Volume::setIOThreads, "count"_a)
        .def("invalidate_cache", &Volume::invalidateCache)
        .def("read_zyx", &readZYX,
            "offset"_a,
            "shape"_a,
            "level"_a = 0,
            "missing_policy"_a = "error")
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
        .def("__getitem__",
            [](Volume& self, const nb::object& key) {
                const auto region = parseSliceKey(key, self.shape());
                return readZYX(self, region.offset, region.shape, 0, "error");
            },
            "key"_a);
}
