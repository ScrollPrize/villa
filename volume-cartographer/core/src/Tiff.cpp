#include "vc/core/util/Tiff.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
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

} // anonymous namespace

// ============================================================================
// writeTiff implementation
// ============================================================================

// ============================================================================
// TiffWriter implementation
// ============================================================================

TiffWriter::TiffWriter(const std::filesystem::path& path, uint32_t width, uint32_t height, int cvType, uint32_t tileW, uint32_t tileH, float padValue, uint16_t compression, float dpi)
    : _width(width), _height(height), _tileW(tileW), _tileH(tileH), _cvType(cvType), _padValue(padValue), _path(path)
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
    TIFFSetField(_tiff, TIFFTAG_COMPRESSION,     compression);
    if (compression == COMPRESSION_LZW || compression == COMPRESSION_DEFLATE ||
        compression == COMPRESSION_ADOBE_DEFLATE)
        TIFFSetField(_tiff, TIFFTAG_PREDICTOR,   PREDICTOR_HORIZONTAL);
    TIFFSetField(_tiff, TIFFTAG_TILEWIDTH,       tileW);
    TIFFSetField(_tiff, TIFFTAG_TILELENGTH,      tileH);
    if (dpi > 0.f) {
        TIFFSetField(_tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_INCH);
        TIFFSetField(_tiff, TIFFTAG_XRESOLUTION, dpi);
        TIFFSetField(_tiff, TIFFTAG_YRESOLUTION, dpi);
    }

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

// ============================================================================
// mergeTiffParts implementation
// ============================================================================

bool mergeTiffParts(const std::string& outputPath, int numParts)
{
    std::filesystem::path outDir(outputPath);
    if (!std::filesystem::is_directory(outDir)) outDir = outDir.parent_path();
    if (outDir.empty()) outDir = ".";

    std::map<std::filesystem::path, std::vector<std::filesystem::path>> groups;
    for (auto& entry : std::filesystem::directory_iterator(outDir)) {
        std::string fname = entry.path().filename().string();
        auto pos = fname.find(".part");
        if (pos == std::string::npos) continue;
        auto dotTif = fname.find(".tif", pos);
        if (dotTif == std::string::npos) continue;
        groups[outDir / (fname.substr(0, pos) + ".tif")].push_back(entry.path());
    }
    if (groups.empty()) {
        std::cerr << "No .partN.tif files found in " << outDir << "\n";
        return false;
    }

    std::cout << "Merging " << groups.size() << " TIFF(s) from " << numParts << " parts..." << std::endl;
    size_t failures = 0;
    for (auto& [finalPath, partFiles] : groups) {
        std::sort(partFiles.begin(), partFiles.end());
        TIFF* first = TIFFOpen(partFiles[0].string().c_str(), "r");
        if (!first) { std::cerr << "Cannot open " << partFiles[0] << "\n"; failures++; continue; }
        // Initialize and check TIFFGetField: a part file from a killed/corrupt
        // render job may be missing tags. Reading an unset tag leaves the
        // variable uninitialized, and a missing TILEWIDTH/TILELENGTH makes tw/th
        // zero -> the (w + tw - 1) / tw tiling below is a divide-by-zero. Skip
        // such a group cleanly instead of crashing.
        uint32_t w = 0, h = 0, tw = 0, th = 0; uint16_t bps = 0, spp = 0, sf = 0, comp = 0;
        const bool haveGeom =
            TIFFGetField(first, TIFFTAG_IMAGEWIDTH, &w)
            && TIFFGetField(first, TIFFTAG_IMAGELENGTH, &h)
            && TIFFGetField(first, TIFFTAG_TILEWIDTH, &tw)
            && TIFFGetField(first, TIFFTAG_TILELENGTH, &th);
        TIFFGetField(first, TIFFTAG_BITSPERSAMPLE, &bps);
        TIFFGetField(first, TIFFTAG_SAMPLESPERPIXEL, &spp);
        TIFFGetField(first, TIFFTAG_SAMPLEFORMAT, &sf);
        TIFFGetField(first, TIFFTAG_COMPRESSION, &comp);
        if (!haveGeom || w == 0 || h == 0 || tw == 0 || th == 0) {
            std::cerr << "Skipping " << finalPath
                      << ": first part missing/zero geometry tags\n";
            TIFFClose(first);
            failures++;
            continue;
        }
        uint16_t resUnit = 0;
        float xRes = 0, yRes = 0;
        bool hasRes = TIFFGetField(first, TIFFTAG_RESOLUTIONUNIT, &resUnit) && TIFFGetField(first, TIFFTAG_XRESOLUTION, &xRes) &&
                      TIFFGetField(first, TIFFTAG_YRESOLUTION, &yRes);
        TIFFClose(first);

        TIFF* out = TIFFOpen(finalPath.string().c_str(), "w");
        if (!out) { std::cerr << "Cannot create " << finalPath << "\n"; failures++; continue; }
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, w); TIFFSetField(out, TIFFTAG_IMAGELENGTH, h);
        TIFFSetField(out, TIFFTAG_TILEWIDTH, tw); TIFFSetField(out, TIFFTAG_TILELENGTH, th);
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, bps); TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, spp);
        TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, sf); TIFFSetField(out, TIFFTAG_COMPRESSION, comp);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        if (hasRes) {
            TIFFSetField(out, TIFFTAG_RESOLUTIONUNIT, resUnit);
            TIFFSetField(out, TIFFTAG_XRESOLUTION, xRes);
            TIFFSetField(out, TIFFTAG_YRESOLUTION, yRes);
        }

        tmsize_t tileBytes = TIFFTileSize(out);
        std::vector<uint8_t> buf(tileBytes, 0), zero(tileBytes, 0);
        uint32_t tilesX = (w + tw - 1) / tw, tilesY = (h + th - 1) / th;
        size_t merged = 0;
        for (uint32_t ty = 0; ty < tilesY; ty++)
            for (uint32_t tx = 0; tx < tilesX; tx++) {
                bool found = false;
                for (auto& pf : partFiles) {
                    TIFF* pt = TIFFOpen(pf.string().c_str(), "r");
                    if (!pt) continue;
                    tmsize_t r = TIFFReadEncodedTile(pt, TIFFComputeTile(pt, tx*tw, ty*th, 0, 0), buf.data(), tileBytes);
                    TIFFClose(pt);
                    if (r > 0 && buf != zero) { found = true; break; }
                }
                TIFFWriteEncodedTile(out, TIFFComputeTile(out, tx*tw, ty*th, 0, 0),
                                     found ? buf.data() : zero.data(), tileBytes);
                if (found) merged++;
            }
        TIFFWriteDirectory(out); TIFFClose(out);
        for (auto& pf : partFiles) std::filesystem::remove(pf);
        std::cout << "  " << finalPath.filename().string() << ": " << merged << " tiles from " << partFiles.size() << " parts\n";
    }
    if (failures > 0) {
        std::cerr << "Merge failed: " << failures << " of " << groups.size() << " TIFF(s) could not be created.\n";
        return false;
    }
    std::cout << "Merge complete.\n";
    return true;
}

