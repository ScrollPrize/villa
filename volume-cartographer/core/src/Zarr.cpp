#include "vc/core/util/Zarr.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "utils/Json.hpp"
#include <omp.h>

using Json = utils::Json;

template <typename T>
void downsampleTileIntoPreserveZ(const T* src, size_t srcZ, size_t srcY, size_t srcX,
                                T* dst, size_t dstZ, size_t dstY, size_t dstX,
                                size_t srcActualZ, size_t srcActualY, size_t srcActualX,
                                size_t dstOffY, size_t dstOffX)
{
    size_t halfY = (srcActualY + 1) / 2;
    size_t halfX = (srcActualX + 1) / 2;
    for (size_t zz = 0; zz < srcActualZ && zz < dstZ; zz++)
        for (size_t yy = 0; yy < halfY && (dstOffY + yy) < dstY; yy++)
            for (size_t xx = 0; xx < halfX && (dstOffX + xx) < dstX; xx++) {
                uint32_t sum = 0; int cnt = 0;
                for (int d1 = 0; d1 < 2 && 2 * yy + d1 < int(srcActualY); d1++)
                    for (int d2 = 0; d2 < 2 && 2 * xx + d2 < int(srcActualX); d2++) {
                        sum += src[zz * srcY * srcX + (2 * yy + d1) * srcX + (2 * xx + d2)];
                        cnt++;
                    }
                dst[zz * dstY * dstX + (dstOffY + yy) * dstX + (dstOffX + xx)] = T((sum + cnt / 2) / std::max(1, cnt));
            }
}

template void downsampleTileIntoPreserveZ<uint8_t>(const uint8_t*, size_t, size_t, size_t,
    uint8_t*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
template void downsampleTileIntoPreserveZ<uint16_t>(const uint16_t*, size_t, size_t, size_t,
    uint16_t*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

// ============================================================
// buildPyramidLevel
// ============================================================

template <typename T>
void buildPyramidLevel(const std::filesystem::path& outDir, int level,
                       size_t CH, size_t CW,
                       int numParts, int partId)
{
    auto src = std::make_unique<vc::VcDataset>(outDir / std::to_string(level - 1));
    auto dst = std::make_unique<vc::VcDataset>(outDir / std::to_string(level));
    const auto& ss = src->shape();
    const auto& sc = src->defaultChunkShape();
    const auto& dc = dst->defaultChunkShape();
    const auto& ds = dst->shape();

    size_t srcChunksY = (ss[1] + sc[1] - 1) / sc[1];
    size_t srcChunksX = (ss[2] + sc[2] - 1) / sc[2];
    size_t dstChunksY = (ds[1] + dc[1] - 1) / dc[1];
    size_t dstChunksX = (ds[2] + dc[2] - 1) / dc[2];

    size_t rowsPerPart = (dstChunksY + size_t(numParts) - 1) / size_t(numParts);
    size_t rowStart = size_t(partId) * rowsPerPart;
    size_t rowEnd = std::min(rowStart + rowsPerPart, dstChunksY);
    size_t myTiles = (rowEnd - rowStart) * dstChunksX;
    std::atomic<size_t> done{0};

    size_t srcElems = sc[0] * sc[1] * sc[2];
    size_t dstElems = dc[0] * dc[1] * dc[2];

    #pragma omp parallel for schedule(dynamic, 2)
    for (long long ti = 0; ti < (long long)myTiles; ti++) {
        size_t dcy = rowStart + size_t(ti) / dstChunksX;
        size_t dcx = size_t(ti) % dstChunksX;

        std::vector<T> dstBuf(dstElems, T(0));

        for (int sy = 0; sy < 2; sy++) {
            for (int sx = 0; sx < 2; sx++) {
                size_t scy = dcy * 2 + sy;
                size_t scx = dcx * 2 + sx;
                if (scy >= srcChunksY || scx >= srcChunksX) continue;

                std::vector<T> srcBuf(srcElems, T(0));
                if (!src->readChunk(0, scy, scx, srcBuf.data()))
                    continue;

                size_t saZ = std::min(sc[0], ss[0]);
                size_t saY = std::min(sc[1], ss[1] - scy * sc[1]);
                size_t saX = std::min(sc[2], ss[2] - scx * sc[2]);

                size_t halfY = sc[1] / 2, halfX = sc[2] / 2;
                size_t offY = sy * halfY;
                size_t offX = sx * halfX;

                downsampleTileIntoPreserveZ(
                    srcBuf.data(), sc[0], sc[1], sc[2],
                    dstBuf.data(), dc[0], dc[1], dc[2],
                    saZ, saY, saX, offY, offX);
            }
        }

        dst->writeChunkSkipEmpty(0, dcy, dcx,
                                 dstBuf.data(), dstBuf.size() * sizeof(T));

        size_t d = ++done;
        #pragma omp critical(pp)
        { std::cout << "\r[pyramid L" << level << "] " << d << "/" << myTiles
                    << " (" << int(100.0*d/myTiles) << "%)" << std::flush; }
    }
    if (myTiles > 0) std::cout << std::endl;
}

template void buildPyramidLevel<uint8_t>(const std::filesystem::path&, int, size_t, size_t, int, int);
template void buildPyramidLevel<uint16_t>(const std::filesystem::path&, int, size_t, size_t, int, int);

// ============================================================
// createPyramidDatasets
// ============================================================

void createPyramidDatasets(const std::filesystem::path& outDir,
                           const std::vector<size_t>& shape0,
                           size_t CH, size_t CW, bool isU16,
                           const std::string& compressor,
                           int compressionLevel,
                           const std::string& dimensionSeparator)
{
    auto dtype = isU16 ? vc::VcDtype::uint16 : vc::VcDtype::uint8;

    std::vector<size_t> prevShape = shape0;
    for (int level = 1; level <= 5; level++) {
        // Keep Z fixed and halve only Y/X at each level (anisotropic scaling).
        std::vector<size_t> shape = {prevShape[0], (prevShape[1]+1)/2, (prevShape[2]+1)/2};
        size_t chZ = std::min(shape[0], shape0[0]);
        std::vector<size_t> chunks = {chZ, std::min(CH, shape[1]), std::min(CW, shape[2])};
        vc::createZarrDataset(outDir, std::to_string(level), shape, chunks, dtype,
                              compressor, dimensionSeparator, 0, compressionLevel);
        prevShape = shape;
    }
}

// ============================================================
// buildMultiscales
// ============================================================

// The OME-NGFF "multiscales" attribute value.
//
// Always written: it is the discovery metadata for the image -- axes, the
// ordered list of resolution levels, and each level's
// coordinateTransformations. A reader that cannot find it does not recognise the
// output as a multiscale image at all.
//
// The physical size and the level-to-level scaling are separate concerns and are
// encoded separately:
//
//   * `axes[*].unit` carries the physical unit. It is written only when the
//     physical size is known -- a unit without a measurement is the thing we must
//     never invent.
//   * each dataset's `scale` is a coordinate transformation. With a known size it
//     is that size in physical units per axis. Without one it is the *relative*
//     factor between that level and level 0, which is exactly what OME-NGFF 0.4
//     asks for when a physical scale is unavailable:
//
//       "If scaling information is not available or applicable for one of the
//        axes, the value MUST express the scaling factor between the current
//        resolution and the first resolution for the given axis, defaulting to
//        1.0 if there is no downsampling along the axis."
//       -- https://ngff.openmicroscopy.org/0.4/#multiscale-md
//
//     So level 0 is [1, 1, 1] and is the reference; there is no absolute length
//     anywhere in the document. A reader can still place the levels relative to
//     each other, and cannot mistake the numbers for micrometres.
Json buildMultiscales(double baseVoxelSize, const std::string& voxelUnit,
                      double sliceStep, double pixelsPerVoxel)
{
    const bool physicalSizeKnown = std::isfinite(baseVoxelSize) && baseVoxelSize > 0.0;

    Json ms;
    ms["version"] = "0.4"; ms["name"] = "render";
    auto makeAxis = [&](const char* name) -> Json {
        Json ax = Json{{"name", name}, {"type", "space"}};
        // Only a known physical size may carry a unit: see above.
        if (physicalSizeKnown && !voxelUnit.empty()) ax["unit"] = voxelUnit;
        return ax;
    };
    Json axes = Json::array();
    axes.push_back(makeAxis("z"));
    axes.push_back(makeAxis("y"));
    axes.push_back(makeAxis("x"));
    ms["axes"] = std::move(axes);
    ms["datasets"] = Json::array();
    // When the size is known the two axes scale independently of each other:
    // in-plane, one output pixel spans 1/pixelsPerVoxel base voxels;
    // through-plane, adjacent output layers sit sliceStep base voxels apart.
    // Pyramid levels halve only YX (see createPyramidDatasets).
    const double px = (std::isfinite(pixelsPerVoxel) && pixelsPerVoxel > 0.0)
        ? pixelsPerVoxel : 1.0;
    const double step = (std::isfinite(sliceStep) && sliceStep > 0.0)
        ? sliceStep : 1.0;
    for (int l = 0; l <= 5; l++) {
        // Relative pyramid factor: levels are built by halving Y/X each step and
        // keeping Z unchanged, so Y/X double relative to level 0 while Z stays at
        // its level-0 spacing.
        const double relativeYX = std::pow(2.0, l);
        const double sYX = physicalSizeKnown ? baseVoxelSize / px * relativeYX
                                             : relativeYX;
        const double sZ = physicalSizeKnown ? baseVoxelSize * step : 1.0;
        Json scale_arr = Json::array();
        scale_arr.push_back(sZ); scale_arr.push_back(sYX); scale_arr.push_back(sYX);
        Json trans_arr = Json::array();
        trans_arr.push_back(0.0); trans_arr.push_back(0.0); trans_arr.push_back(0.0);
        Json transforms = Json::array();
        transforms.push_back(Json{{"type","scale"},{"scale",std::move(scale_arr)}});
        transforms.push_back(Json{{"type","translation"},{"translation",std::move(trans_arr)}});
        ms["datasets"].push_back(Json{
            {"path", std::to_string(l)},
            {"coordinateTransformations", std::move(transforms)}
        });
    }
    ms["metadata"] = Json{{"downsampling_method","mean"}};
    if (!physicalSizeKnown) {
        // Say why the scales are unitless, in the document itself, so a reader
        // (or a person) does not have to infer it from the missing "unit".
        ms["metadata"]["physical_size"] = "unknown";
    }
    return ms;
}

// ============================================================
// writeZarrAttrs
// ============================================================

void writeZarrAttrs(const std::filesystem::path& outDir,
                    const std::filesystem::path& volPath, int groupIdx,
                    size_t baseZ, double sliceStep, double accumStep,
                    const std::string& accumTypeStr, size_t accumSamples,
                    const cv::Size& canvasSize, size_t CZ, size_t CH, size_t CW,
                    double baseVoxelSize, const std::string& voxelUnit,
                    double pixelsPerVoxel)
{
    Json attrs;
    attrs["source_zarr"] = volPath.string();
    attrs["source_group"] = groupIdx;
    attrs["num_slices"] = baseZ;
    attrs["slice_step"] = sliceStep;
    if (accumSamples > 0) {
        attrs["accum_step"] = accumStep;
        attrs["accum_type"] = accumTypeStr;
        attrs["accum_samples"] = int(accumSamples);
    }
    {
        Json cs = Json::array(); cs.push_back(canvasSize.width); cs.push_back(canvasSize.height);
        attrs["canvas_size"] = std::move(cs);
    }
    {
        Json ck = Json::array(); ck.push_back(int(CZ)); ck.push_back(int(CH)); ck.push_back(int(CW));
        attrs["chunk_size"] = std::move(ck);
    }
    attrs["note_axes_order"] = "ZYX (slice, row, col)";

    Json multiscales = Json::array();
    multiscales.push_back(buildMultiscales(baseVoxelSize, voxelUnit, sliceStep, pixelsPerVoxel));
    attrs["multiscales"] = std::move(multiscales);

    vc::writeZarrAttributes(outDir, attrs);
}
