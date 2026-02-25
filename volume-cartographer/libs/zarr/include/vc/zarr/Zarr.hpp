#pragma once

// vc::zarr — Minimal zarr v2 library for volume-cartographer.
// Replaces z5 with only what we need: filesystem backend, blosc/zstd, uint8/uint16/float32.
// Designed to accommodate zarr v3 in the future (format version stored in metadata).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xstrided_view.hpp>

namespace vc::zarr {

// ─────────────────────────────────────────────────────────────────────────────
//  Types
// ─────────────────────────────────────────────────────────────────────────────

using ShapeType = std::vector<std::size_t>;

enum class Datatype { uint8, uint16, float32 };

/// Zarr dtype string ("|u1", "<u2", "<f4") → Datatype enum
Datatype dtypeFromZarr(const std::string& s);

/// Datatype enum → zarr dtype string
const std::string& dtypeToZarr(Datatype dt);

/// Element size in bytes
std::size_t dtypeSize(Datatype dt);

/// Human-readable name ("uint8", "uint16", "float32") → Datatype
Datatype dtypeFromName(const std::string& name);

/// Datatype → human-readable name
const std::string& dtypeToName(Datatype dt);

// ─────────────────────────────────────────────────────────────────────────────
//  Group  (zarr group node — container with .zgroup marker and .zattrs)
// ─────────────────────────────────────────────────────────────────────────────

class Group {
public:
    /// Open an existing group rooted at `root`.
    explicit Group(std::filesystem::path root);

    /// Create a child group handle (sub-group).
    Group(const Group& parent, const std::string& child);

    const std::filesystem::path& path() const { return path_; }
    bool exists() const;

    /// List immediate subdirectory names.
    std::vector<std::string> keys() const;

    /// Read .zattrs (returns empty json if absent).
    nlohmann::json readAttrs() const;

    /// Write .zattrs (merges into existing).
    void writeAttrs(const nlohmann::json& j) const;

    /// Create a new group directory + .zgroup marker. Returns the group.
    static Group create(const std::filesystem::path& path, bool overwrite = false);

    /// Create a child group with .zgroup marker.
    Group createGroup(const std::string& name) const;

private:
    std::filesystem::path path_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  DatasetMeta  (parsed .zarray)
// ─────────────────────────────────────────────────────────────────────────────

struct DatasetMeta {
    ShapeType shape;
    ShapeType chunks;
    Datatype dtype = Datatype::uint8;
    std::string dimSeparator = ".";
    std::string bloscCodec = "zstd";
    int bloscLevel = 1;
    int bloscShuffle = 0;
    double fillValue = 0;

    /// Parse from .zarray JSON.
    void fromJson(const nlohmann::json& j);

    /// Serialize to .zarray JSON (zarr v2).
    nlohmann::json toJson() const;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Dataset  (replaces z5::Dataset)
// ─────────────────────────────────────────────────────────────────────────────

class Dataset {
public:
    ~Dataset();
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;
    Dataset(Dataset&&) noexcept;
    Dataset& operator=(Dataset&&) noexcept;

    // --- Open existing ---

    /// Open by parent group + name (auto-detects dimSeparator from .zarray).
    static std::unique_ptr<Dataset> open(const Group& parent, const std::string& name);

    /// Open by absolute path (auto-detects dimSeparator from .zarray).
    static std::unique_ptr<Dataset> open(const std::filesystem::path& dsPath);

    /// Open with explicit dimSeparator (skips re-reading .zarray for separator).
    static std::unique_ptr<Dataset> open(const Group& parent, const std::string& name,
                                         const std::string& dimSeparator);

    // --- Create new ---

    /// Create a new dataset. `compOpts` is {"cname":"zstd","clevel":1,"shuffle":0}.
    static std::unique_ptr<Dataset> create(
        const Group& parent, const std::string& name,
        const std::string& dtype, const ShapeType& shape, const ShapeType& chunks,
        const std::string& compressor, const nlohmann::json& compOpts,
        double fillValue = 0, const std::string& dimSeparator = ".");

    // --- Accessors (names match z5::Dataset for minimal migration churn) ---

    const ShapeType& shape() const { return meta_.shape; }
    const ShapeType& defaultChunkShape() const { return meta_.chunks; }
    std::size_t defaultChunkSize() const { return chunkSize_; }
    Datatype getDtype() const { return meta_.dtype; }
    const std::filesystem::path& path() const { return path_; }
    const DatasetMeta& metadata() const { return meta_; }
    unsigned dimension() const { return static_cast<unsigned>(meta_.shape.size()); }

    // --- Chunk I/O ---

    bool chunkExists(const ShapeType& id) const;
    void chunkPath(const ShapeType& id, std::filesystem::path& out) const;

    /// Read & decompress a full chunk into `buf` (must be >= defaultChunkSize() * dtypeSize elements).
    void readChunk(const ShapeType& id, void* buf) const;

    /// Read raw compressed bytes from chunk file.
    void readRawChunk(const ShapeType& id, std::vector<char>& out) const;

    /// Compress & write a full chunk from `data`.
    void writeChunk(const ShapeType& id, const void* data) const;

    /// Decompress raw bytes into `out`. `numElements` = number of typed elements.
    void decompress(const std::vector<char>& compressed, void* out, std::size_t numElements) const;

    /// Compress typed data into raw bytes.
    void compress(const void* data, std::vector<char>& out, std::size_t numElements) const;

    // --- Chunk geometry helpers ---

    /// Compute the actual (clipped) shape of a chunk at grid position `id`.
    void getChunkShape(const ShapeType& id, ShapeType& out) const;

    /// Validate that offset+shape is within dataset bounds.
    void checkRequestShape(const ShapeType& offset, const ShapeType& shape) const;

    // --- Blocking helpers (for readSubarray/writeSubarray) ---

    /// Return all chunk grid coordinates overlapping the given ROI.
    void getBlocksOverlappingRoi(const ShapeType& roiBegin, const ShapeType& roiShape,
                                 std::vector<ShapeType>& blockList) const;

    /// Compute the overlap between a chunk and a ROI.
    /// Returns true if the chunk is completely inside the ROI.
    bool getCoordinatesInRoi(const ShapeType& chunkCoord,
                             const ShapeType& roiBegin, const ShapeType& roiShape,
                             ShapeType& offsetInRoi, ShapeType& shapeInRoi,
                             ShapeType& offsetInChunk) const;

private:
    Dataset() = default;
    static std::unique_ptr<Dataset> openImpl(const std::filesystem::path& dsPath,
                                             const std::string& dimSeparator);
    void initChunkSize();

    std::filesystem::path path_;
    DatasetMeta meta_;
    std::size_t chunkSize_ = 0;   // product of chunk dims (element count)
    ShapeType chunksPerDim_;      // number of chunks along each dimension
};

// ─────────────────────────────────────────────────────────────────────────────
//  readSubarray / writeSubarray  (header-only templates, replace z5::multiarray)
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

// Convert ROI (offset, shape) to xtensor slice vector
inline void sliceFromRoi(xt::xstrided_slice_vector& sl,
                         const ShapeType& offset, const ShapeType& shape) {
    for (std::size_t d = 0; d < offset.size(); ++d)
        sl.push_back(xt::range(offset[d], offset[d] + shape[d]));
}

// Copy C-order buffer → strided view (N-dimensional, row-major)
template<typename T, typename VIEW, typename STRIDES>
void copyBufferToView(const std::vector<T>& buffer,
                      xt::xexpression<VIEW>& viewExpr,
                      const STRIDES& arrayStrides) {
    auto& view = viewExpr.derived_cast();
    if (view.dimension() == 1) {
        auto bv = xt::adapt(buffer, view.shape());
        view = bv;
        return;
    }

    const std::size_t dim = view.dimension();
    const auto& viewShape = view.shape();
    std::size_t bufferOffset = 0, viewOffset = view.data_offset();
    ShapeType dimPos(dim, 0);
    const std::size_t memLen = viewShape[dim - 1];

    for (int d = static_cast<int>(dim) - 2; d >= 0;) {
        std::copy(buffer.begin() + bufferOffset,
                  buffer.begin() + bufferOffset + memLen,
                  &view.data()[0] + viewOffset);
        bufferOffset += memLen;
        viewOffset += arrayStrides[dim - 2];

        for (d = static_cast<int>(dim) - 2; d >= 0; --d) {
            dimPos[d] += 1;
            if (dimPos[d] < viewShape[d]) break;
            dimPos[d] = 0;
            if (d > 0 && dimPos[d - 1] + 1 == viewShape[d - 1]) continue;
            if (d > 0) {
                std::size_t correction = 0;
                for (int dd = static_cast<int>(dim) - 2; dd >= d; --dd)
                    correction += arrayStrides[dd] * (viewShape[dd] - 1);
                correction += arrayStrides[dim - 2];
                viewOffset += (arrayStrides[d - 1] - correction);
            }
        }
    }
}

// Copy strided view → C-order buffer (N-dimensional, row-major)
template<typename T, typename VIEW, typename STRIDES>
void copyViewToBuffer(const xt::xexpression<VIEW>& viewExpr,
                      std::vector<T>& buffer,
                      const STRIDES& arrayStrides) {
    const auto& view = viewExpr.derived_cast();
    if (view.dimension() == 1) {
        auto bv = xt::adapt(buffer, view.shape());
        bv = view;
        return;
    }

    const std::size_t dim = view.dimension();
    const auto& viewShape = view.shape();
    std::size_t bufferOffset = 0, viewOffset = view.data_offset();
    ShapeType dimPos(dim, 0);
    const std::size_t memLen = viewShape[dim - 1];

    for (int d = static_cast<int>(dim) - 2; d >= 0;) {
        std::copy(&view.data()[0] + viewOffset,
                  &view.data()[0] + viewOffset + memLen,
                  buffer.begin() + bufferOffset);
        bufferOffset += memLen;
        viewOffset += arrayStrides[dim - 2];

        for (d = static_cast<int>(dim) - 2; d >= 0; --d) {
            dimPos[d] += 1;
            if (dimPos[d] < viewShape[d]) break;
            dimPos[d] = 0;
            if (d > 0 && dimPos[d - 1] + 1 == viewShape[d - 1]) continue;
            if (d > 0) {
                std::size_t correction = 0;
                for (int dd = static_cast<int>(dim) - 2; dd >= d; --dd)
                    correction += arrayStrides[dd] * (viewShape[dd] - 1);
                correction += arrayStrides[dim - 2];
                viewOffset += (arrayStrides[d - 1] - correction);
            }
        }
    }
}

// Enumerate all grid points in a rectangular range [min, max] (inclusive).
inline void makeRegularGrid(const ShapeType& min, const ShapeType& max,
                            std::vector<ShapeType>& out) {
    const std::size_t ndim = min.size();
    ShapeType current = min;
    for (;;) {
        out.push_back(current);
        // increment the last dimension first (row-major order)
        int d = static_cast<int>(ndim) - 1;
        for (; d >= 0; --d) {
            current[d]++;
            if (current[d] <= max[d]) break;
            current[d] = min[d];
        }
        if (d < 0) break;
    }
}

} // namespace detail

/// Read an arbitrary rectangular subarray from a Dataset into an xtensor expression.
/// T must match the dataset's dtype. Offset is given by an iterator.
template<typename T, typename ARRAY, typename ITER>
void readSubarray(const Dataset& ds, xt::xexpression<ARRAY>& outExpr, ITER offsetBegin) {
    auto& out = outExpr.derived_cast();
    const unsigned ndim = ds.dimension();

    ShapeType offset(offsetBegin, offsetBegin + ndim);
    ShapeType shape(out.shape().begin(), out.shape().end());
    ds.checkRequestShape(offset, shape);

    std::vector<ShapeType> chunkRequests;
    ds.getBlocksOverlappingRoi(offset, shape, chunkRequests);

    const std::size_t maxChunkSize = ds.defaultChunkSize();
    const auto& maxChunkShape = ds.defaultChunkShape();
    std::vector<T> buffer(maxChunkSize);

    T fillValue = static_cast<T>(ds.metadata().fillValue);

    for (const auto& chunkId : chunkRequests) {
        ShapeType offsetInRequest, requestShape, offsetInChunk;
        bool completeOvlp = ds.getCoordinatesInRoi(chunkId, offset, shape,
                                                   offsetInRequest, requestShape, offsetInChunk);

        xt::xstrided_slice_vector sl;
        detail::sliceFromRoi(sl, offsetInRequest, requestShape);
        auto view = xt::strided_view(out, sl);

        if (!ds.chunkExists(chunkId)) {
            view = fillValue;
            continue;
        }

        // Zarr always stores full chunk shape, even for edge chunks.
        // The actual data shape may differ from the stored shape.
        ShapeType chunkShape;
        ds.getChunkShape(chunkId, chunkShape);
        std::size_t chunkSize = std::accumulate(chunkShape.begin(), chunkShape.end(),
                                                std::size_t{1}, std::multiplies<>());

        // Read raw compressed data
        std::vector<char> dataBuffer;
        ds.readRawChunk(chunkId, dataBuffer);

        // Zarr edge chunks: stored at full chunk size, actual data may be smaller
        std::size_t chunkStoreSize = maxChunkSize;
        if (chunkStoreSize != chunkSize) {
            completeOvlp = false;
            chunkSize = maxChunkSize;
            chunkShape = maxChunkShape;
        }

        if (chunkSize != buffer.size())
            buffer.resize(chunkSize);

        ds.decompress(dataBuffer, buffer.data(), chunkSize);

        if (completeOvlp) {
            detail::copyBufferToView(buffer, view, out.strides());
        } else {
            auto fullBufView = xt::adapt(buffer, chunkShape);
            xt::xstrided_slice_vector bufSlice;
            detail::sliceFromRoi(bufSlice, offsetInChunk, requestShape);
            auto bufView = xt::strided_view(fullBufView, bufSlice);
            view = bufView;
        }
    }
}

/// Write an arbitrary rectangular subarray into a Dataset from an xtensor expression.
template<typename T, typename ARRAY, typename ITER>
void writeSubarray(Dataset& ds, const xt::xexpression<ARRAY>& inExpr, ITER offsetBegin) {
    const auto& in = inExpr.derived_cast();
    const unsigned ndim = ds.dimension();

    ShapeType offset(offsetBegin, offsetBegin + ndim);
    ShapeType shape(in.shape().begin(), in.shape().end());
    ds.checkRequestShape(offset, shape);

    std::vector<ShapeType> chunkRequests;
    ds.getBlocksOverlappingRoi(offset, shape, chunkRequests);

    T fillValue = static_cast<T>(ds.metadata().fillValue);

    const std::size_t maxChunkSize = ds.defaultChunkSize();
    std::size_t chunkSize = maxChunkSize;
    std::vector<T> buffer(chunkSize, fillValue);

    for (const auto& chunkId : chunkRequests) {
        ShapeType offsetInRequest, requestShape, offsetInChunk;
        bool completeOvlp = ds.getCoordinatesInRoi(chunkId, offset, shape,
                                                   offsetInRequest, requestShape, offsetInChunk);

        ShapeType chunkShape;
        ds.getChunkShape(chunkId, chunkShape);
        chunkSize = std::accumulate(chunkShape.begin(), chunkShape.end(),
                                    std::size_t{1}, std::multiplies<>());

        xt::xstrided_slice_vector sl;
        detail::sliceFromRoi(sl, offsetInRequest, requestShape);
        const auto view = xt::strided_view(in, sl);

        // Zarr edge chunks: always write full chunk size
        if (chunkSize != maxChunkSize) {
            completeOvlp = false;
            chunkShape = ds.defaultChunkShape();
            chunkSize = maxChunkSize;
            std::fill(buffer.begin(), buffer.end(), fillValue);
        }

        if (chunkSize != buffer.size())
            buffer.resize(chunkSize);

        if (completeOvlp) {
            detail::copyViewToBuffer(view, buffer, in.strides());
            ds.writeChunk(chunkId, buffer.data());
        } else {
            // Partial overlap: read existing chunk data first
            if (ds.chunkExists(chunkId)) {
                ds.readChunk(chunkId, buffer.data());
            } else {
                std::fill(buffer.begin(), buffer.end(), fillValue);
            }

            auto fullBufView = xt::adapt(buffer, chunkShape);
            xt::xstrided_slice_vector bufSlice;
            detail::sliceFromRoi(bufSlice, offsetInChunk, requestShape);
            auto bufView = xt::strided_view(fullBufView, bufSlice);
            bufView = view;

            ds.writeChunk(chunkId, buffer.data());
        }
    }
}

// Convenience overloads taking unique_ptr (matches z5 API)
template<typename T, typename ARRAY, typename ITER>
void readSubarray(const std::unique_ptr<Dataset>& ds, xt::xexpression<ARRAY>& out, ITER offsetBegin) {
    readSubarray<T>(*ds, out, offsetBegin);
}

template<typename T, typename ARRAY, typename ITER>
void writeSubarray(std::unique_ptr<Dataset>& ds, const xt::xexpression<ARRAY>& in, ITER offsetBegin) {
    writeSubarray<T>(*ds, in, offsetBegin);
}

} // namespace vc::zarr
