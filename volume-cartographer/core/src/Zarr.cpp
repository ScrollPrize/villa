#include "vc/core/util/Zarr.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>
#include <omp.h>

using json = nlohmann::json;

// ============================================================
// writeZarrBand
// ============================================================

template <typename T>
void writeZarrBand(vc::VcDataset* dsOut, const std::vector<cv::Mat>& slices,
                   uint32_t bandIdx, const std::vector<size_t>& chunks0,
                   size_t tilesXSrc, size_t tilesYSrc,
                   int rotQuad, int flipAxis)
{
    int outH = slices[0].rows, outW = slices[0].cols;
    size_t numZ = slices.size();
    size_t chunkZ = chunks0[0], chunkY = chunks0[1], chunkX = chunks0[2];

    for (size_t tx = 0; tx < tilesXSrc; tx++) {
        size_t x0s = tx * chunkX;
        size_t dxc = std::min(chunkX, size_t(outW) - x0s);

        int dstTx = int(tx), dstTy = int(bandIdx), dTX, dTY;
        if (rotQuad >= 0 || flipAxis >= 0)
            mapTileIndex(int(tx), int(bandIdx), int(tilesXSrc), int(tilesYSrc),
                         std::max(rotQuad, 0), flipAxis, dstTx, dstTy, dTX, dTY);

        std::vector<T> chunkBuf(chunkZ * chunkY * chunkX, T(0));
        size_t dy_actual = std::min(chunkY, size_t(outH));
        size_t dx_actual = std::min(chunkX, dxc);
        for (size_t zi = 0; zi < numZ; zi++) {
            size_t sliceOff = zi * chunkY * chunkX;
            for (size_t yy = 0; yy < dy_actual; yy++) {
                const T* row = slices[zi].ptr<T>(int(yy));
                std::memcpy(&chunkBuf[sliceOff + yy * chunkX], &row[x0s], dx_actual * sizeof(T));
            }
        }
        dsOut->writeChunk(0, size_t(dstTy), size_t(dstTx),
                          chunkBuf.data(), chunkBuf.size() * sizeof(T));
    }
}

template void writeZarrBand<uint8_t>(vc::VcDataset*, const std::vector<cv::Mat>&,
    uint32_t, const std::vector<size_t>&, size_t, size_t, int, int);
template void writeZarrBand<uint16_t>(vc::VcDataset*, const std::vector<cv::Mat>&,
    uint32_t, const std::vector<size_t>&, size_t, size_t, int, int);

// ============================================================
// downsampleChunk
// ============================================================

template <typename T>
void downsampleChunk(const T* src, size_t srcZ, size_t srcY, size_t srcX,
                     T* dst, size_t dstZ, size_t dstY, size_t dstX,
                     size_t srcActualZ, size_t srcActualY, size_t srcActualX)
{
    for (size_t zz = 0; zz < dstZ; zz++)
        for (size_t yy = 0; yy < dstY; yy++)
            for (size_t xx = 0; xx < dstX; xx++) {
                uint32_t sum = 0; int cnt = 0;
                for (int d0 = 0; d0 < 2 && 2*zz+d0 < srcActualZ; d0++)
                    for (int d1 = 0; d1 < 2 && 2*yy+d1 < srcActualY; d1++)
                        for (int d2 = 0; d2 < 2 && 2*xx+d2 < srcActualX; d2++) {
                            sum += src[(2*zz+d0)*srcY*srcX + (2*yy+d1)*srcX + (2*xx+d2)];
                            cnt++;
                        }
                dst[zz*dstY*dstX + yy*dstX + xx] = T((sum + cnt/2) / std::max(1, cnt));
            }
}

template void downsampleChunk<uint8_t>(const uint8_t*, size_t, size_t, size_t,
    uint8_t*, size_t, size_t, size_t, size_t, size_t, size_t);
template void downsampleChunk<uint16_t>(const uint16_t*, size_t, size_t, size_t,
    uint16_t*, size_t, size_t, size_t, size_t, size_t, size_t);

// ============================================================
// downsampleTileInto
// ============================================================

template <typename T>
void downsampleTileInto(const T* src, size_t srcZ, size_t srcY, size_t srcX,
                        T* dst, size_t dstZ, size_t dstY, size_t dstX,
                        size_t srcActualZ, size_t srcActualY, size_t srcActualX,
                        size_t dstOffY, size_t dstOffX)
{
    size_t halfZ = (srcActualZ + 1) / 2;
    size_t halfY = (srcActualY + 1) / 2;
    size_t halfX = (srcActualX + 1) / 2;
    for (size_t zz = 0; zz < halfZ && zz < dstZ; zz++)
        for (size_t yy = 0; yy < halfY && (dstOffY + yy) < dstY; yy++)
            for (size_t xx = 0; xx < halfX && (dstOffX + xx) < dstX; xx++) {
                uint32_t sum = 0; int cnt = 0;
                for (int d0 = 0; d0 < 2 && 2*zz+d0 < srcActualZ; d0++)
                    for (int d1 = 0; d1 < 2 && 2*yy+d1 < srcActualY; d1++)
                        for (int d2 = 0; d2 < 2 && 2*xx+d2 < srcActualX; d2++) {
                            sum += src[(2*zz+d0)*srcY*srcX + (2*yy+d1)*srcX + (2*xx+d2)];
                            cnt++;
                        }
                dst[zz*dstY*dstX + (dstOffY+yy)*dstX + (dstOffX+xx)] = T((sum + cnt/2) / std::max(1, cnt));
            }
}

template void downsampleTileInto<uint8_t>(const uint8_t*, size_t, size_t, size_t,
    uint8_t*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
template void downsampleTileInto<uint16_t>(const uint16_t*, size_t, size_t, size_t,
    uint16_t*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

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

        dst->writeChunk(0, dcy, dcx,
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
                           size_t CH, size_t CW, bool isU16)
{
    auto dtype = isU16 ? vc::VcDtype::uint16 : vc::VcDtype::uint8;

    std::vector<size_t> prevShape = shape0;
    for (int level = 1; level <= 5; level++) {
        std::vector<size_t> shape = {(prevShape[0]+1)/2, (prevShape[1]+1)/2, (prevShape[2]+1)/2};
        size_t chZ = std::min(shape[0], shape0[0]);
        std::vector<size_t> chunks = {chZ, std::min(CH, shape[1]), std::min(CW, shape[2])};
        vc::createZarrDataset(outDir, std::to_string(level), shape, chunks, dtype, "blosc");
        prevShape = shape;
    }
}

// ============================================================
// writeZarrAttrs
// ============================================================

void writeZarrAttrs(const std::filesystem::path& outDir,
                    const std::filesystem::path& volPath, int groupIdx,
                    size_t baseZ, double sliceStep, double accumStep,
                    const std::string& accumTypeStr, size_t accumSamples,
                    const cv::Size& canvasSize, size_t CZ, size_t CH, size_t CW)
{
    json attrs;
    attrs["source_zarr"] = volPath.string();
    attrs["source_group"] = groupIdx;
    attrs["num_slices"] = baseZ;
    attrs["slice_step"] = sliceStep;
    if (accumSamples > 0) {
        attrs["accum_step"] = accumStep;
        attrs["accum_type"] = accumTypeStr;
        attrs["accum_samples"] = int(accumSamples);
    }
    attrs["canvas_size"] = {canvasSize.width, canvasSize.height};
    attrs["chunk_size"] = {int(CZ), int(CH), int(CW)};
    attrs["note_axes_order"] = "ZYX (slice, row, col)";

    json ms;
    ms["version"] = "0.4"; ms["name"] = "render";
    ms["axes"] = json::array({
        json{{"name","z"},{"type","space"}},
        json{{"name","y"},{"type","space"}},
        json{{"name","x"},{"type","space"}}
    });
    ms["datasets"] = json::array();
    for (int l = 0; l <= 5; l++) {
        double s = std::pow(2.0, l);
        const double sz = 1.0;
        ms["datasets"].push_back({
            {"path", std::to_string(l)},
            {"coordinateTransformations", json::array({
                json{{"type","scale"},{"scale",json::array({sz,s,s})}},
                json{{"type","translation"},{"translation",json::array({0.0,0.0,0.0})}}
            })}
        });
    }
    ms["metadata"] = json{{"downsampling_method","mean"}};
    attrs["multiscales"] = json::array({ms});

    vc::writeZarrAttributes(outDir, attrs);
}
