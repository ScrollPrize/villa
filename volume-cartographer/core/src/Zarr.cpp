#include "vc/core/util/Zarr.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>
#include <omp.h>
#include <z5/factory.hxx>

using json = nlohmann::json;

// ============================================================
// writeZarrBand
// ============================================================

template <typename T>
void writeZarrBand(z5::Dataset* dsOut, const std::vector<cv::Mat>& slices,
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
        z5::types::ShapeType chunkId = {0, size_t(dstTy), size_t(dstTx)};
        dsOut->writeChunk(chunkId, chunkBuf.data());
    }
}

template void writeZarrBand<uint8_t>(z5::Dataset*, const std::vector<cv::Mat>&,
    uint32_t, const std::vector<size_t>&, size_t, size_t, int, int);
template void writeZarrBand<uint16_t>(z5::Dataset*, const std::vector<cv::Mat>&,
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

// ============================================================
// buildPyramidLevel
// ============================================================

template <typename T>
void buildPyramidLevel(z5::filesystem::handle::File& outFile, int level,
                       size_t CH, size_t CW,
                       int numParts, int partId)
{
    auto src = z5::openDataset(outFile, std::to_string(level - 1));
    auto dst = z5::openDataset(outFile, std::to_string(level));
    const auto& ss = src->shape();
    const auto& sc = src->defaultChunkShape();  // source chunk shape (fixed, e.g. 128×128)
    const auto& dc = dst->defaultChunkShape();  // dst chunk shape (same Y×X as source)
    const auto& ds = dst->shape();

    // Source chunk grid
    size_t srcChunksY = (ss[1] + sc[1] - 1) / sc[1];
    size_t srcChunksX = (ss[2] + sc[2] - 1) / sc[2];

    // Dest chunk grid (half as many chunks since shape halves but chunk size stays)
    size_t dstChunksY = (ds[1] + dc[1] - 1) / dc[1];
    size_t dstChunksX = (ds[2] + dc[2] - 1) / dc[2];

    // Contiguous block assignment: each part gets a contiguous range of dest tile rows
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

        // Each dest chunk assembles from a 2×2 grid of source chunks
        for (int sy = 0; sy < 2; sy++) {
            for (int sx = 0; sx < 2; sx++) {
                size_t scy = dcy * 2 + sy;
                size_t scx = dcx * 2 + sx;
                if (scy >= srcChunksY || scx >= srcChunksX) continue;

                z5::types::ShapeType srcId = {0, scy, scx};
                std::vector<T> srcBuf(srcElems, T(0));
                if (src->chunkExists(srcId))
                    src->readChunk(srcId, srcBuf.data());
                else
                    continue;

                // Actual valid extent in this source chunk
                size_t saZ = std::min(sc[0], ss[0]);
                size_t saY = std::min(sc[1], ss[1] - scy * sc[1]);
                size_t saX = std::min(sc[2], ss[2] - scx * sc[2]);

                // Offset within dest chunk: each source chunk downsamples to half-chunk
                size_t halfY = sc[1] / 2, halfX = sc[2] / 2;
                size_t offY = sy * halfY;
                size_t offX = sx * halfX;

                downsampleTileInto(
                    srcBuf.data(), sc[0], sc[1], sc[2],
                    dstBuf.data(), dc[0], dc[1], dc[2],
                    saZ, saY, saX, offY, offX);
            }
        }

        z5::types::ShapeType dstId = {0, dcy, dcx};
        dst->writeChunk(dstId, dstBuf.data());

        size_t d = ++done;
        #pragma omp critical(pp)
        { std::cout << "\r[pyramid L" << level << "] " << d << "/" << myTiles
                    << " (" << int(100.0*d/myTiles) << "%)" << std::flush; }
    }
    if (myTiles > 0) std::cout << std::endl;
}

template void buildPyramidLevel<uint8_t>(z5::filesystem::handle::File&, int, size_t, size_t, int, int);
template void buildPyramidLevel<uint16_t>(z5::filesystem::handle::File&, int, size_t, size_t, int, int);

// ============================================================
// createPyramidDatasets
// ============================================================

void createPyramidDatasets(z5::filesystem::handle::File& outFile,
                           const std::vector<size_t>& shape0,
                           size_t CH, size_t CW, bool isU16)
{
    json compOpts = {{"cname","zstd"},{"clevel",1},{"shuffle",0}};
    std::string dtype = isU16 ? "uint16" : "uint8";

    // All pyramid levels use the same chunk Y×X as the input volume (typically 128×128).
    // Only shape halves at each level; chunks stay fixed for fewer, larger files.
    std::vector<size_t> prevShape = shape0;
    for (int level = 1; level <= 5; level++) {
        std::vector<size_t> ds = {(prevShape[0]+1)/2, (prevShape[1]+1)/2, (prevShape[2]+1)/2};
        size_t chZ = std::min(ds[0], shape0[0]);  // clamp to level shape Z
        std::vector<size_t> dc = {chZ, std::min(CH, ds[1]), std::min(CW, ds[2])};
        z5::createDataset(outFile, std::to_string(level), dtype, ds, dc, std::string("blosc"), compOpts);
        prevShape = ds;
    }
}

// ============================================================
// writeZarrAttrs
// ============================================================

void writeZarrAttrs(z5::filesystem::handle::File& outFile,
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
        ms["datasets"].push_back({
            {"path", std::to_string(l)},
            {"coordinateTransformations", json::array({
                json{{"type","scale"},{"scale",json::array({s,s,s})}},
                json{{"type","translation"},{"translation",json::array({0.0,0.0,0.0})}}
            })}
        });
    }
    ms["metadata"] = json{{"downsampling_method","mean"}};
    attrs["multiscales"] = json::array({ms});
    z5::filesystem::writeAttributes(outFile, attrs);
}

