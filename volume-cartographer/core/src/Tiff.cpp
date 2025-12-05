#include "vc/core/util/Tiff.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
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

// Fill tile buffer with padding value
void fillTileBuffer(std::vector<uint8_t>& buf, int cvType, float padValue) {
    switch (cvType) {
        case CV_8UC1:
            std::fill(buf.begin(), buf.end(), static_cast<uint8_t>(0));
            break;
        case CV_16UC1:
            std::fill(reinterpret_cast<uint16_t*>(buf.data()),
                     reinterpret_cast<uint16_t*>(buf.data() + buf.size()),
                     static_cast<uint16_t>(0));
            break;
        case CV_32FC1:
            std::fill(reinterpret_cast<float*>(buf.data()),
                     reinterpret_cast<float*>(buf.data() + buf.size()),
                     padValue);
            break;
    }
}

// Convert image to target type with value scaling
cv::Mat convertWithScaling(const cv::Mat& img, int targetType) {
    if (img.type() == targetType)
        return img;

    cv::Mat result;
    const int srcType = img.type();

    // Scaling factors for full-range conversion
    if (srcType == CV_8UC1 && targetType == CV_16UC1) {
        img.convertTo(result, CV_16UC1, 257.0);  // 255 * 257 = 65535
    } else if (srcType == CV_8UC1 && targetType == CV_32FC1) {
        img.convertTo(result, CV_32FC1, 1.0 / 255.0);
    } else if (srcType == CV_16UC1 && targetType == CV_8UC1) {
        img.convertTo(result, CV_8UC1, 1.0 / 257.0);
    } else if (srcType == CV_16UC1 && targetType == CV_32FC1) {
        img.convertTo(result, CV_32FC1, 1.0 / 65535.0);
    } else if (srcType == CV_32FC1 && targetType == CV_8UC1) {
        img.convertTo(result, CV_8UC1, 255.0);
    } else if (srcType == CV_32FC1 && targetType == CV_16UC1) {
        img.convertTo(result, CV_16UC1, 65535.0);
    } else {
        throw std::runtime_error("Unsupported type conversion");
    }

    return result;
}

} // anonymous namespace

// ============================================================================
// writeTiff implementation
// ============================================================================

void writeTiff(const std::filesystem::path& outPath,
               const cv::Mat& img,
               int cvType,
               uint32_t tileW,
               uint32_t tileH,
               float padValue)
{
    if (img.empty())
        throw std::runtime_error("Empty image for " + outPath.string());
    if (img.channels() != 1)
        throw std::runtime_error("Expected single-channel image for " + outPath.string());

    // Determine output type and convert if necessary
    const int outType = (cvType < 0) ? img.type() : cvType;
    const cv::Mat outImg = convertWithScaling(img, outType);

    const auto params = getTiffParams(outType);

    TIFF* tf = TIFFOpen(outPath.string().c_str(), "w");
    if (!tf)
        throw std::runtime_error("Failed to open TIFF for writing: " + outPath.string());

    const uint32_t W = static_cast<uint32_t>(outImg.cols);
    const uint32_t H = static_cast<uint32_t>(outImg.rows);

    TIFFSetField(tf, TIFFTAG_IMAGEWIDTH,      W);
    TIFFSetField(tf, TIFFTAG_IMAGELENGTH,     H);
    TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE,   params.bits);
    TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT,    params.sampleFormat);
    TIFFSetField(tf, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tf, TIFFTAG_COMPRESSION,     COMPRESSION_LZW);
    TIFFSetField(tf, TIFFTAG_PREDICTOR,       PREDICTOR_HORIZONTAL);
    TIFFSetField(tf, TIFFTAG_TILEWIDTH,       tileW);
    TIFFSetField(tf, TIFFTAG_TILELENGTH,      tileH);

    const size_t tileBytes = static_cast<size_t>(tileW) * tileH * params.elemSize;
    std::vector<uint8_t> tileBuf(tileBytes);

    for (uint32_t y0 = 0; y0 < H; y0 += tileH) {
        const uint32_t dy = std::min(tileH, H - y0);
        for (uint32_t x0 = 0; x0 < W; x0 += tileW) {
            const uint32_t dx = std::min(tileW, W - x0);

            // Fill with padding
            fillTileBuffer(tileBuf, outType, padValue);

            // Copy actual data
            for (uint32_t ty = 0; ty < dy; ++ty) {
                const uint8_t* src = outImg.ptr<uint8_t>(static_cast<int>(y0 + ty)) + x0 * params.elemSize;
                std::memcpy(tileBuf.data() + ty * tileW * params.elemSize,
                           src,
                           dx * params.elemSize);
            }

            const ttile_t tileIndex = TIFFComputeTile(tf, x0, y0, 0, 0);
            if (TIFFWriteEncodedTile(tf, tileIndex, tileBuf.data(), static_cast<tmsize_t>(tileBytes)) < 0) {
                TIFFClose(tf);
                throw std::runtime_error("TIFFWriteEncodedTile failed at tile (" +
                                        std::to_string(x0) + "," + std::to_string(y0) +
                                        ") in " + outPath.string());
            }
        }
    }

    if (!TIFFWriteDirectory(tf)) {
        TIFFClose(tf);
        throw std::runtime_error("TIFFWriteDirectory failed for " + outPath.string());
    }
    TIFFClose(tf);
}

// ============================================================================
// TiffWriter implementation
// ============================================================================

TiffWriter::TiffWriter(const std::filesystem::path& path,
                       uint32_t width, uint32_t height,
                       int cvType,
                       uint32_t tileW,
                       uint32_t tileH,
                       float padValue)
    : _width(width), _height(height), _tileW(tileW), _tileH(tileH),
      _cvType(cvType), _padValue(padValue), _path(path)
{
    const auto params = getTiffParams(cvType);
    _elemSize = params.elemSize;

    _tiff = TIFFOpen(path.string().c_str(), "w");
    if (!_tiff)
        throw std::runtime_error("Failed to open TIFF for writing: " + path.string());

    TIFFSetField(_tiff, TIFFTAG_IMAGEWIDTH,      width);
    TIFFSetField(_tiff, TIFFTAG_IMAGELENGTH,     height);
    TIFFSetField(_tiff, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(_tiff, TIFFTAG_BITSPERSAMPLE,   params.bits);
    TIFFSetField(_tiff, TIFFTAG_SAMPLEFORMAT,    params.sampleFormat);
    TIFFSetField(_tiff, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
    TIFFSetField(_tiff, TIFFTAG_COMPRESSION,     COMPRESSION_LZW);
    TIFFSetField(_tiff, TIFFTAG_PREDICTOR,       PREDICTOR_HORIZONTAL);
    TIFFSetField(_tiff, TIFFTAG_TILEWIDTH,       tileW);
    TIFFSetField(_tiff, TIFFTAG_TILELENGTH,      tileH);

    // Allocate reusable tile buffer
    _tileBuf.resize(static_cast<size_t>(tileW) * tileH * _elemSize);
}

TiffWriter::~TiffWriter() {
    close();
}

TiffWriter::TiffWriter(TiffWriter&& other) noexcept
    : _tiff(other._tiff), _width(other._width), _height(other._height),
      _tileW(other._tileW), _tileH(other._tileH), _cvType(other._cvType),
      _elemSize(other._elemSize), _padValue(other._padValue),
      _tileBuf(std::move(other._tileBuf)), _path(std::move(other._path))
{
    other._tiff = nullptr;
}

TiffWriter& TiffWriter::operator=(TiffWriter&& other) noexcept {
    if (this != &other) {
        close();
        _tiff = other._tiff;
        _width = other._width;
        _height = other._height;
        _tileW = other._tileW;
        _tileH = other._tileH;
        _cvType = other._cvType;
        _elemSize = other._elemSize;
        _padValue = other._padValue;
        _tileBuf = std::move(other._tileBuf);
        _path = std::move(other._path);
        other._tiff = nullptr;
    }
    return *this;
}

void TiffWriter::writeTile(uint32_t x0, uint32_t y0, const cv::Mat& tile) {
    if (!_tiff)
        throw std::runtime_error("TiffWriter: file not open");
    if (tile.type() != _cvType)
        throw std::runtime_error("TiffWriter: tile type mismatch");

    const uint32_t dx = static_cast<uint32_t>(tile.cols);
    const uint32_t dy = static_cast<uint32_t>(tile.rows);

    // Fill with padding
    fillTileBuffer(_tileBuf, _cvType, _padValue);

    // Copy actual data
    for (uint32_t ty = 0; ty < dy; ++ty) {
        const uint8_t* src = tile.ptr<uint8_t>(static_cast<int>(ty));
        std::memcpy(_tileBuf.data() + ty * _tileW * _elemSize,
                   src,
                   dx * _elemSize);
    }

    const ttile_t tileIndex = TIFFComputeTile(_tiff, x0, y0, 0, 0);
    const tmsize_t tileBytes = static_cast<tmsize_t>(_tileBuf.size());
    if (TIFFWriteEncodedTile(_tiff, tileIndex, _tileBuf.data(), tileBytes) < 0) {
        throw std::runtime_error("TIFFWriteEncodedTile failed at tile (" +
                                std::to_string(x0) + "," + std::to_string(y0) +
                                ") in " + _path.string());
    }
}

void TiffWriter::close() {
    if (_tiff) {
        TIFFWriteDirectory(_tiff);
        TIFFClose(_tiff);
        _tiff = nullptr;
    }
}
