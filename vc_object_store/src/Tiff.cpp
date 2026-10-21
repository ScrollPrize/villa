#include "vc_object_store/Tiff.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <limits>
#include <stdexcept>
#include <vector>
#include <tiffio.h>

namespace {

// Get TIFF parameters for a given OpenCV type
struct TiffParams {
    int bits;
    int sampleFormat;
    int elemSize;
};

TiffParams getTiffParams(int cvType) {
    switch (cvType) {
        case CV_8UC1:  return {8,  SAMPLEFORMAT_UINT,   1};
        case CV_16UC1: return {16, SAMPLEFORMAT_UINT,   2};
        case CV_32FC1: return {32, SAMPLEFORMAT_IEEEFP, 4};
        default:
            throw std::runtime_error("Unsupported cv::Mat type for TIFF writing");
    }
}

void fillTileBuffer(std::vector<uint8_t>& buf, int cvType, float padValue) {
    if (cvType == CV_32FC1)
        std::fill(reinterpret_cast<float*>(buf.data()),
                  reinterpret_cast<float*>(buf.data() + buf.size()), padValue);
    else
        std::fill(buf.begin(), buf.end(), uint8_t(0));
}

// Convert image to target type with value scaling
cv::Mat convertWithScaling(const cv::Mat& img, int targetType) {
    if (img.type() == targetType)
        return img;

    cv::Mat result;
    const int srcType = img.type();

    double scale;
    if (srcType == CV_8UC1 && targetType == CV_16UC1) scale = 257.0;
    else if (srcType == CV_8UC1 && targetType == CV_32FC1) scale = 1.0 / 255.0;
    else if (srcType == CV_16UC1 && targetType == CV_8UC1) scale = 1.0 / 257.0;
    else if (srcType == CV_16UC1 && targetType == CV_32FC1) scale = 1.0 / 65535.0;
    else if (srcType == CV_32FC1 && targetType == CV_8UC1) scale = 255.0;
    else if (srcType == CV_32FC1 && targetType == CV_16UC1) scale = 65535.0;
    else throw std::runtime_error("Unsupported type conversion");
    img.convertTo(result, targetType, scale);

    return result;
}

} // anonymous namespace

void writeTiff(const std::filesystem::path& outPath, const cv::Mat& img, int cvType, uint32_t tileW, uint32_t tileH, float padValue, uint16_t compression, float dpi)
{
    if (img.empty())
        throw std::runtime_error("Empty image for " + outPath.string());
    if (img.channels() != 1)
        throw std::runtime_error("Expected single-channel image for " + outPath.string());

    const int outType = (cvType < 0) ? img.type() : cvType;
    const cv::Mat outImg = convertWithScaling(img, outType);

    const auto params = getTiffParams(outType);

    const uint32_t W = static_cast<uint32_t>(outImg.cols);
    const uint32_t H = static_cast<uint32_t>(outImg.rows);
    const std::uint64_t pixelBytes =
        static_cast<std::uint64_t>(W) * H * params.elemSize;
    const char* mode = pixelBytes > 0xffff0000ULL ? "w8" : "w";

    std::unique_ptr<TIFF, decltype(&TIFFClose)> file(TIFFOpen(outPath.string().c_str(), mode), TIFFClose);
    TIFF* tf = file.get();
    if (!tf)
        throw std::runtime_error("Failed to open TIFF for writing: " + outPath.string());

    TIFFSetField(tf, TIFFTAG_IMAGEWIDTH,      W);
    TIFFSetField(tf, TIFFTAG_IMAGELENGTH,     H);
    TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE,   params.bits);
    TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT,    params.sampleFormat);
    TIFFSetField(tf, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tf, TIFFTAG_COMPRESSION,     compression);
    if (compression == COMPRESSION_LZW || compression == COMPRESSION_DEFLATE ||
        compression == COMPRESSION_ADOBE_DEFLATE)
        TIFFSetField(tf, TIFFTAG_PREDICTOR,   PREDICTOR_HORIZONTAL);
    if (dpi > 0.f) {
        TIFFSetField(tf, TIFFTAG_RESOLUTIONUNIT, RESUNIT_INCH);
        TIFFSetField(tf, TIFFTAG_XRESOLUTION, dpi);
        TIFFSetField(tf, TIFFTAG_YRESOLUTION, dpi);
    }

    if (tileW == 0 || tileH == 0) {
        TIFFSetField(tf, TIFFTAG_ROWSPERSTRIP, H);
        for (uint32_t y = 0; y < H; ++y) {
            const uint8_t* row = outImg.ptr<uint8_t>(static_cast<int>(y));
            if (TIFFWriteScanline(tf, const_cast<uint8_t*>(row), y, 0) < 0)
                throw std::runtime_error("TIFFWriteScanline failed at row " +
                                        std::to_string(y) + " in " + outPath.string());
        }
    } else {
        TIFFSetField(tf, TIFFTAG_TILEWIDTH,       tileW);
        TIFFSetField(tf, TIFFTAG_TILELENGTH,      tileH);

        const size_t tileBytes = static_cast<size_t>(tileW) * tileH * params.elemSize;
        std::vector<uint8_t> tileBuf(tileBytes);

        for (uint32_t y0 = 0; y0 < H; y0 += tileH) {
            const uint32_t dy = std::min(tileH, H - y0);
            for (uint32_t x0 = 0; x0 < W; x0 += tileW) {
                const uint32_t dx = std::min(tileW, W - x0);

                fillTileBuffer(tileBuf, outType, padValue);

                for (uint32_t ty = 0; ty < dy; ++ty) {
                    const uint8_t* src = outImg.ptr<uint8_t>(static_cast<int>(y0 + ty)) + x0 * params.elemSize;
                    std::memcpy(tileBuf.data() + ty * tileW * params.elemSize,
                               src,
                               dx * params.elemSize);
                }

                const ttile_t tileIndex = TIFFComputeTile(tf, x0, y0, 0, 0);
                if (TIFFWriteEncodedTile(tf, tileIndex, tileBuf.data(), static_cast<tmsize_t>(tileBytes)) < 0) {
                    throw std::runtime_error("TIFFWriteEncodedTile failed at tile (" +
                                            std::to_string(x0) + "," + std::to_string(y0) +
                                            ") in " + outPath.string());
                }
            }
        }
    }

    if (!TIFFWriteDirectory(tf))
        throw std::runtime_error("TIFFWriteDirectory failed for " + outPath.string());
}

namespace {

// Decode every tile (or strip) of the first directory into a scratch buffer.
// A payload that is truncated, points past the end of the file or fails to
// decompress makes libtiff return -1 or a short count.
bool decodesCompletely(TIFF* tf)
{
    if (TIFFIsTiled(tf)) {
        const tmsize_t tileBytes = TIFFTileSize(tf);
        if (tileBytes <= 0)
            return false;
        std::vector<uint8_t> buf(static_cast<size_t>(tileBytes));
        const ttile_t n = TIFFNumberOfTiles(tf);
        for (ttile_t t = 0; t < n; ++t)
            if (TIFFReadEncodedTile(tf, t, buf.data(), tileBytes) != tileBytes)
                return false;
        return true;
    }

    uint32_t height = 0, rowsPerStrip = 0;
    if (!TIFFGetField(tf, TIFFTAG_IMAGELENGTH, &height) || height == 0)
        return false;
    TIFFGetFieldDefaulted(tf, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
    if (rowsPerStrip == 0 || rowsPerStrip > height)
        rowsPerStrip = height;
    const tmsize_t stripBytes = TIFFStripSize(tf);
    if (stripBytes <= 0)
        return false;
    std::vector<uint8_t> buf(static_cast<size_t>(stripBytes));
    const tstrip_t n = TIFFNumberOfStrips(tf);
    for (tstrip_t s = 0; s < n; ++s) {
        // The last strip may hold fewer rows than ROWSPERSTRIP.
        const uint32_t rows = std::min<uint32_t>(rowsPerStrip, height - s * rowsPerStrip);
        const tmsize_t expected = TIFFVStripSize(tf, rows);
        if (expected <= 0 || TIFFReadEncodedStrip(tf, s, buf.data(), expected) != expected)
            return false;
    }
    return true;
}

} // namespace

bool isReadableTiff(const std::filesystem::path& path)
{
    // TIFFOpen("r") reads the first directory and fails without one, which
    // is what a writer killed before close() leaves behind. That alone does
    // not prove the tile data is intact, so every tile is decoded as well.
    // A torn file makes libtiff complain through its global handlers, so
    // mute them for the probe and hand back a plain yes/no.
    TIFFErrorHandler prevError = TIFFSetErrorHandler(nullptr);
    TIFFErrorHandler prevWarning = TIFFSetWarningHandler(nullptr);
    TIFF* tf = TIFFOpen(path.string().c_str(), "r");
    const bool ok = tf && decodesCompletely(tf);
    if (tf)
        TIFFClose(tf);
    TIFFSetErrorHandler(prevError);
    TIFFSetWarningHandler(prevWarning);
    return ok;
}

TiffSamples readTiffSamples(const std::filesystem::path& path, bool strictLayout)
{
    std::unique_ptr<TIFF, decltype(&TIFFClose)> tif(TIFFOpen(path.string().c_str(), "r"), TIFFClose);
    auto fail = [&](const char* reason) { throw std::runtime_error(std::string(reason) + ": " + path.string()); };
    if (!tif) fail("cannot read TIFF");
    TiffSamples result;
    uint16_t planar = 0, orientation = 0, photometric = 0;
    uint32_t depth = 1;
    TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &result.width);
    TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &result.height);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &result.bits);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &result.sampleFormat);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &result.components);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_ORIENTATION, &orientation);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PHOTOMETRIC, &photometric);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_IMAGEDEPTH, &depth);
    if (!result.width || !result.height || result.width > INT_MAX || result.height > INT_MAX || !result.components ||
        (planar != PLANARCONFIG_CONTIG && planar != PLANARCONFIG_SEPARATE) ||
        !((result.sampleFormat == SAMPLEFORMAT_IEEEFP && (result.bits == 32 || result.bits == 64)) ||
          ((result.sampleFormat == SAMPLEFORMAT_UINT || result.sampleFormat == SAMPLEFORMAT_INT) &&
           (result.bits == 8 || result.bits == 16 || result.bits == 32 || result.bits == 64))))
        fail("unsupported TIFF samples");
    if (strictLayout && (orientation != ORIENTATION_TOPLEFT || depth != 1 || TIFFNumberOfDirectories(tif.get()) != 1 ||
        (photometric != PHOTOMETRIC_MINISBLACK && photometric != PHOTOMETRIC_MINISWHITE && photometric != PHOTOMETRIC_RGB)))
        fail("unsupported TIFF layout (orientation, depth, pages or photometric)");
    const size_t scalar = result.bits / 8, pixel = scalar * result.components;
    const uint64_t pixels = uint64_t(result.width) * result.height;
    if (pixels > result.bytes.max_size() / pixel) fail("TIFF dimensions overflow");
    const auto size = pixels * pixel;
    result.bytes.resize(size);
    const size_t stride = planar == PLANARCONFIG_SEPARATE ? scalar : pixel;
    const auto planes = planar == PLANARCONFIG_SEPARATE ? result.components : 1;
    auto copy = [&](uint32_t y, uint32_t x, uint16_t plane, const uint8_t* input) {
        std::memcpy(result.bytes.data() + (size_t(y)*result.width+x)*pixel + plane*scalar, input, stride);
    };
    if (TIFFIsTiled(tif.get())) {
        uint32_t tw = 0, th = 0, td = 1;
        TIFFGetField(tif.get(), TIFFTAG_TILEWIDTH, &tw); TIFFGetField(tif.get(), TIFFTAG_TILELENGTH, &th);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_TILEDEPTH, &td);
        const auto bytes = TIFFTileSize(tif.get());
        if (!tw || !th || td != 1 || bytes <= 0 || uint64_t(tw)*th*stride != uint64_t(bytes)) fail("invalid TIFF tile");
        std::vector<uint8_t> buffer(size_t(bytes), 0);
        for (uint16_t plane = 0; plane < planes; ++plane)
            for (uint32_t y = 0; y < result.height; y += th)
                for (uint32_t x = 0; x < result.width; x += tw) {
                    if (TIFFReadTile(tif.get(), buffer.data(), x, y, 0, plane) != bytes) fail("truncated TIFF tile");
                    for (uint32_t dy = 0; dy < std::min(th, result.height-y); ++dy)
                        for (uint32_t dx = 0; dx < std::min(tw, result.width-x); ++dx)
                            copy(y+dy, x+dx, plane, buffer.data()+(size_t(dy)*tw+dx)*stride);
                }
    } else {
        const auto bytes = TIFFScanlineSize(tif.get());
        if (bytes <= 0 || uint64_t(result.width)*stride != uint64_t(bytes)) fail("invalid TIFF row");
        std::vector<uint8_t> buffer(size_t(bytes), 0);
        for (uint16_t plane = 0; plane < planes; ++plane)
            for (uint32_t y = 0; y < result.height; ++y) {
                if (TIFFReadScanline(tif.get(), buffer.data(), y, plane) != 1) fail("truncated TIFF row");
                for (uint32_t x = 0; x < result.width; ++x) copy(y, x, plane, buffer.data()+x*stride);
            }
    }
    return result;
}
