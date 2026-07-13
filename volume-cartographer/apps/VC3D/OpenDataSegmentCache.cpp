#include "OpenDataSegmentCache.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/AffineTransform.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <nlohmann/json.hpp>

#include <tiffio.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>

namespace vc3d::opendata {
namespace {

std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string trimTrailingSlashes(std::string value)
{
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

std::string artifactUrl(const OpenDataArtifact& artifact)
{
    return trimTrailingSlashes(artifact.resolvedUrl.empty()
                                  ? artifact.sourcePath
                                  : artifact.resolvedUrl);
}

std::string urlPathWithoutQuery(std::string value)
{
    if (const auto pos = value.find('#'); pos != std::string::npos) {
        value.erase(pos);
    }
    if (const auto pos = value.find('?'); pos != std::string::npos) {
        value.erase(pos);
    }
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

std::string imageExtensionForUrl(const std::string& url)
{
    const std::string path = lowerCopy(urlPathWithoutQuery(url));
    for (const auto* ext : {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}) {
        if (path.size() >= std::strlen(ext) &&
            path.compare(path.size() - std::strlen(ext), std::strlen(ext), ext) == 0) {
            return ext;
        }
    }
    return {};
}

bool isJpegExtension(const std::string& ext)
{
    return ext == ".jpg" || ext == ".jpeg";
}

bool isSupportedInkImageArtifact(const OpenDataArtifact& artifact)
{
    const std::string type = lowerCopy(artifact.type);
    if (type.find("ink") == std::string::npos) {
        return false;
    }
    if (type.find("zarr") != std::string::npos || type.find("3d") != std::string::npos) {
        return false;
    }
    return !imageExtensionForUrl(artifactUrl(artifact)).empty();
}

std::string safePathComponent(std::string value)
{
    for (char& c : value) {
        const auto uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '-' && c != '_' && c != '.') {
            c = '_';
        }
    }
    while (!value.empty() && (value.front() == '.' || value.front() == '_')) {
        value.erase(value.begin());
    }
    return value.empty() ? std::string("unnamed") : value;
}

std::string segmentLabel(const OpenDataSegment& segment)
{
    return segment.id.empty() ? std::string("<unnamed segment>") : segment.id;
}

std::string segmentStableId(const OpenDataSegment& segment)
{
    if (!segment.longId.empty()) {
        return segment.longId;
    }
    if (!segment.id.empty()) {
        return segment.id;
    }
    return safePathComponent(segment.suffix);
}

std::string sourceVolumeIdForSegment(const OpenDataSegment& segment);

bool hasSegmentEntry(const VolumePkg& pkg, const std::string& location)
{
    const auto& entries = pkg.segmentEntries();
    return std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.location == location;
    });
}

bool isNonEmptyFile(const std::filesystem::path& path)
{
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec) &&
           std::filesystem::file_size(path, ec) > 0 &&
           !ec;
}

std::string readTextFile(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void writeBytesAtomic(const std::filesystem::path& path,
                      const std::vector<std::byte>& bytes)
{
    std::filesystem::create_directories(path.parent_path());
    const auto tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!bytes.empty()) {
            out.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
        }
        if (!out) {
            throw std::runtime_error("failed to write " + tmp);
        }
    }
    std::filesystem::rename(tmp, path);
}

void writeStringAtomic(const std::filesystem::path& path,
                       const std::string& text)
{
    const auto bytes = std::vector<std::byte>(
        reinterpret_cast<const std::byte*>(text.data()),
        reinterpret_cast<const std::byte*>(text.data() + text.size()));
    writeBytesAtomic(path, bytes);
}

struct TiffInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    uint16_t bits = 0;
    uint16_t sampleFormat = SAMPLEFORMAT_UINT;
    uint16_t samplesPerPixel = 1;
    uint16_t compression = COMPRESSION_NONE;
    uint32_t rowsPerStrip = 0;
    tstrip_t strips = 0;
    bool tiled = false;
    bool byteSwapped = false;
};

TiffInfo readTiffInfo(const std::filesystem::path& path)
{
    TIFF* tif = TIFFOpen(path.string().c_str(), "r");
    if (!tif) {
        throw std::runtime_error("failed to open TIFF: " + path.string());
    }
    TiffInfo info;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &info.width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &info.height);
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &info.bits);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &info.sampleFormat);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &info.samplesPerPixel);
    TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &info.compression);
    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &info.rowsPerStrip);
    info.strips = TIFFNumberOfStrips(tif);
    info.tiled = TIFFIsTiled(tif);
    info.byteSwapped = TIFFIsByteSwapped(tif);
    TIFFClose(tif);
    return info;
}

bool isMmapCompatibleTiff(const TiffInfo& info)
{
    return !info.tiled &&
           !info.byteSwapped &&
           info.samplesPerPixel == 1 &&
           info.bits == 32 &&
           info.sampleFormat == SAMPLEFORMAT_IEEEFP &&
           info.compression == COMPRESSION_NONE &&
           info.strips == 1 &&
           info.rowsPerStrip >= info.height &&
           info.width > 0 &&
           info.height > 0;
}

float tiffSampleToFloat(const uint8_t* p, uint16_t sampleFormat, uint16_t bits)
{
    switch (sampleFormat) {
        case SAMPLEFORMAT_IEEEFP:
            if (bits == 32) {
                float v = 0.0f;
                std::memcpy(&v, p, sizeof(v));
                return v;
            }
            if (bits == 64) {
                double v = 0.0;
                std::memcpy(&v, p, sizeof(v));
                return static_cast<float>(v);
            }
            break;
        case SAMPLEFORMAT_UINT:
            if (bits == 8) return static_cast<float>(*p);
            if (bits == 16) {
                uint16_t v = 0;
                std::memcpy(&v, p, sizeof(v));
                return static_cast<float>(v);
            }
            if (bits == 32) {
                uint32_t v = 0;
                std::memcpy(&v, p, sizeof(v));
                return static_cast<float>(v);
            }
            break;
        case SAMPLEFORMAT_INT:
            if (bits == 8) return static_cast<float>(*reinterpret_cast<const int8_t*>(p));
            if (bits == 16) {
                int16_t v = 0;
                std::memcpy(&v, p, sizeof(v));
                return static_cast<float>(v);
            }
            if (bits == 32) {
                int32_t v = 0;
                std::memcpy(&v, p, sizeof(v));
                return static_cast<float>(v);
            }
            break;
        default:
            break;
    }
    throw std::runtime_error("unsupported TIFF sample type");
}

std::vector<float> readTiffAsFloat(const std::filesystem::path& path, TiffInfo* outInfo)
{
    TIFF* tif = TIFFOpen(path.string().c_str(), "r");
    if (!tif) {
        throw std::runtime_error("failed to open TIFF: " + path.string());
    }

    const TiffInfo info = readTiffInfo(path);
    if (info.width == 0 || info.height == 0 || info.samplesPerPixel != 1) {
        TIFFClose(tif);
        throw std::runtime_error("unsupported TIFF geometry: " + path.string());
    }
    if (!(info.bits == 8 || info.bits == 16 || info.bits == 32 || info.bits == 64)) {
        TIFFClose(tif);
        throw std::runtime_error("unsupported TIFF bits per sample: " + path.string());
    }

    const int bytesPer = (info.bits + 7) / 8;
    std::vector<float> pixels(static_cast<std::size_t>(info.width) * info.height);
    if (TIFFIsTiled(tif)) {
        uint32_t tileW = 0;
        uint32_t tileH = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
        if (tileW == 0 || tileH == 0) {
            TIFFClose(tif);
            throw std::runtime_error("invalid TIFF tile geometry: " + path.string());
        }
        const tmsize_t tileBytes = TIFFTileSize(tif);
        std::vector<uint8_t> tileBuf(static_cast<std::size_t>(tileBytes));
        for (uint32_t y0 = 0; y0 < info.height; y0 += tileH) {
            const uint32_t dy = std::min(tileH, info.height - y0);
            for (uint32_t x0 = 0; x0 < info.width; x0 += tileW) {
                const uint32_t dx = std::min(tileW, info.width - x0);
                const ttile_t tidx = TIFFComputeTile(tif, x0, y0, 0, 0);
                if (TIFFReadEncodedTile(tif, tidx, tileBuf.data(), tileBytes) < 0) {
                    TIFFClose(tif);
                    throw std::runtime_error("failed reading tile: " + path.string());
                }
                for (uint32_t y = 0; y < dy; ++y) {
                    const uint8_t* row = tileBuf.data() + static_cast<std::size_t>(y) * tileW * bytesPer;
                    for (uint32_t x = 0; x < dx; ++x) {
                        pixels[static_cast<std::size_t>(y0 + y) * info.width + x0 + x] =
                            tiffSampleToFloat(row + static_cast<std::size_t>(x) * bytesPer,
                                              info.sampleFormat,
                                              info.bits);
                    }
                }
            }
        }
    } else {
        const tmsize_t scanBytes = TIFFScanlineSize(tif);
        std::vector<uint8_t> scanBuf(static_cast<std::size_t>(scanBytes));
        for (uint32_t y = 0; y < info.height; ++y) {
            if (TIFFReadScanline(tif, scanBuf.data(), y, 0) != 1) {
                TIFFClose(tif);
                throw std::runtime_error("failed reading scanline: " + path.string());
            }
            for (uint32_t x = 0; x < info.width; ++x) {
                pixels[static_cast<std::size_t>(y) * info.width + x] =
                    tiffSampleToFloat(scanBuf.data() + static_cast<std::size_t>(x) * bytesPer,
                                      info.sampleFormat,
                                      info.bits);
            }
        }
    }

    TIFFClose(tif);
    if (outInfo) {
        *outInfo = info;
    }
    return pixels;
}

void writeMmapCompatibleTiff(const std::filesystem::path& path,
                             const TiffInfo& info,
                             const std::vector<float>& pixels)
{
    const auto tmp = path.string() + ".mmap_tmp";
    const std::uint64_t pixelBytes =
        static_cast<std::uint64_t>(pixels.size()) * sizeof(float);
    const char* mode = pixelBytes > 0xffff0000ULL ? "w8" : "w";
    TIFF* out = TIFFOpen(tmp.c_str(), mode);
    if (!out) {
        throw std::runtime_error("failed to create TIFF: " + tmp);
    }

    TIFFSetField(out, TIFFTAG_IMAGEWIDTH, info.width);
    TIFFSetField(out, TIFFTAG_IMAGELENGTH, info.height);
    TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, info.height);

    const tsize_t bytes = static_cast<tsize_t>(pixelBytes);
    if (TIFFWriteEncodedStrip(out, 0, const_cast<float*>(pixels.data()), bytes) < 0) {
        TIFFClose(out);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error("failed writing TIFF: " + tmp);
    }
    TIFFClose(out);
    std::filesystem::rename(tmp, path);
}

void normalizeTiffForMmap(const std::filesystem::path& path)
{
    const TiffInfo info = readTiffInfo(path);
    if (isMmapCompatibleTiff(info)) {
        return;
    }
    TiffInfo readInfo;
    auto pixels = readTiffAsFloat(path, &readInfo);
    writeMmapCompatibleTiff(path, readInfo, pixels);
}

std::optional<double> jsonNumberLike(const nlohmann::json& obj, const char* key)
{
    if (!obj.is_object()) {
        return std::nullopt;
    }
    const auto it = obj.find(key);
    if (it == obj.end()) {
        return std::nullopt;
    }
    if (it->is_number()) {
        return it->get<double>();
    }
    if (it->is_string()) {
        try {
            return std::stod(it->get<std::string>());
        } catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

double originalVolumeDownscale(const OpenDataSegment& segment)
{
    for (const auto* key : {"original_volume_downscale",
                            "originalVolumeDownscale",
                            "volume_downscale",
                            "volumeDownscale"}) {
        if (const auto value = jsonNumberLike(segment.properties, key);
            value.has_value() && std::isfinite(*value) && *value > 0.0) {
            return *value;
        }
    }
    return 1.0;
}

std::string derivedVolumeId(const OpenDataSegment& segment)
{
    if (segment.raw.is_object()) {
        const auto creation = segment.raw.find("creation");
        if (creation != segment.raw.end() && creation->is_object()) {
            const auto derived = creation->find("derived_from");
            if (derived != creation->end() && derived->is_object()) {
                const auto type = derived->find("type");
                const auto id = derived->find("id");
                if (type != derived->end() && type->is_string() &&
                    id != derived->end() && id->is_string() &&
                    lowerCopy(type->get<std::string>()) == "volume") {
                    return id->get<std::string>();
                }
            }
        }
    }
    return segment.originalVolumeId;
}

std::string sourceVolumeIdForSegment(const OpenDataSegment& segment)
{
    const auto derived = derivedVolumeId(segment);
    return derived.empty() ? segment.originalVolumeId : derived;
}

void applyOpenDataMetadata(nlohmann::json& meta,
                           const std::string& baseUrl,
                           const OpenDataSegment& segment)
{
    meta["vc_open_data_tifxyz_url"] = baseUrl;
    meta["vc_open_data_segment_id"] = segment.id;
    meta["vc_open_data_segment_long_id"] = segment.longId;
    meta["vc_open_data_original_volume_id"] = segment.originalVolumeId;
    if (const auto derivedId = derivedVolumeId(segment); !derivedId.empty()) {
        meta["vc_open_data_derived_volume_id"] = derivedId;
    }
    meta["vc_open_data_original_volume_downscale"] = originalVolumeDownscale(segment);
}

bool coordinatesAlreadyScaled(const nlohmann::json& meta, double expectedFactor)
{
    const auto factor = jsonNumberLike(
        meta, "vc_open_data_coordinates_scaled_to_original_volume");
    if (!factor)
        return false;
    if (!std::isfinite(*factor) ||
        std::abs(*factor - expectedFactor) >
            1.0e-9 * std::max({1.0, std::abs(*factor), std::abs(expectedFactor)})) {
        throw std::runtime_error(
            "legacy catalog segment coordinate-scale marker does not match manifest downscale");
    }
    return true;
}

bool artifactAlreadyInOriginalVolumeScale(const OpenDataArtifact& artifact)
{
    const auto type = lowerCopy(artifact.type);
    return type.find("transformed") != std::string::npos;
}

std::optional<cv::Matx44d> parseOpenDataMatrix(const nlohmann::json& matrixJson)
{
    if (!matrixJson.is_array() || (matrixJson.size() != 3 && matrixJson.size() != 4)) {
        return std::nullopt;
    }

    cv::Matx44d matrix = cv::Matx44d::eye();
    for (int row = 0; row < static_cast<int>(matrixJson.size()); ++row) {
        const auto& rowJson = matrixJson.at(static_cast<std::size_t>(row));
        if (!rowJson.is_array() || rowJson.size() != 4) {
            return std::nullopt;
        }
        for (int col = 0; col < 4; ++col) {
            if (!rowJson.at(static_cast<std::size_t>(col)).is_number()) {
                return std::nullopt;
            }
            matrix(row, col) = rowJson.at(static_cast<std::size_t>(col)).get<double>();
        }
    }
    if (matrixJson.size() == 4 &&
        (std::abs(matrix(3, 0)) > 1e-12 ||
         std::abs(matrix(3, 1)) > 1e-12 ||
         std::abs(matrix(3, 2)) > 1e-12 ||
         std::abs(matrix(3, 3) - 1.0) > 1e-12)) {
        return std::nullopt;
    }
    return matrix;
}

std::optional<cv::Matx44d> findVolumeTransformMatrix(const OpenDataSample& sample,
                                                     const std::string& sourceVolumeId,
                                                     const std::string& targetVolumeId)
{
    if (sourceVolumeId.empty() || targetVolumeId.empty() || sourceVolumeId == targetVolumeId) {
        return std::nullopt;
    }

    const nlohmann::json* volumeTransforms = nullptr;
    auto maybeSetTransforms = [&](const nlohmann::json& owner) {
        if (volumeTransforms || !owner.is_object()) {
            return;
        }
        const auto it = owner.find("volume_transforms");
        if (it != owner.end()) {
            volumeTransforms = &*it;
        }
    };

    maybeSetTransforms(sample.properties);
    if (sample.properties.is_object()) {
        if (const auto nested = sample.properties.find("properties"); nested != sample.properties.end()) {
            maybeSetTransforms(*nested);
        }
    }
    if (sample.raw.is_object()) {
        if (const auto sampleIt = sample.raw.find("sample"); sampleIt != sample.raw.end()) {
            maybeSetTransforms(*sampleIt);
            if (sampleIt->is_object()) {
                if (const auto propsIt = sampleIt->find("properties"); propsIt != sampleIt->end()) {
                    maybeSetTransforms(*propsIt);
                }
            }
        }
        if (const auto propsIt = sample.raw.find("properties"); propsIt != sample.raw.end()) {
            maybeSetTransforms(*propsIt);
        }
    }

    if (!volumeTransforms) {
        return std::nullopt;
    }

    auto inspectTransformList = [&](const nlohmann::json& transforms) -> std::optional<cv::Matx44d> {
        if (!transforms.is_array()) {
            return std::nullopt;
        }
        for (const auto& transform : transforms) {
            if (!transform.is_object()) {
                continue;
            }
            const auto toIt = transform.find("to_volume_id");
            if (toIt == transform.end() || !toIt->is_string() ||
                toIt->get<std::string>() != targetVolumeId) {
                continue;
            }
            const auto matrixIt = transform.find("matrix");
            if (matrixIt == transform.end()) {
                continue;
            }
            if (auto matrix = parseOpenDataMatrix(*matrixIt)) {
                return matrix;
            }
        }
        return std::nullopt;
    };

    if (volumeTransforms->is_array()) {
        for (const auto& fromEntry : *volumeTransforms) {
            if (!fromEntry.is_object()) {
                continue;
            }
            const auto fromIt = fromEntry.find("from_volume_id");
            if (fromIt == fromEntry.end() || !fromIt->is_string() ||
                fromIt->get<std::string>() != sourceVolumeId) {
                continue;
            }
            const auto transformsIt = fromEntry.find("transforms");
            if (transformsIt != fromEntry.end()) {
                if (auto matrix = inspectTransformList(*transformsIt)) {
                    return matrix;
                }
            }
        }
        return std::nullopt;
    }

    if (volumeTransforms->is_object()) {
        const auto fromIt = volumeTransforms->find(sourceVolumeId);
        if (fromIt == volumeTransforms->end()) {
            return std::nullopt;
        }
        if (fromIt->is_object()) {
            const auto transformsIt = fromIt->find("transforms");
            if (transformsIt != fromIt->end()) {
                if (auto matrix = inspectTransformList(*transformsIt)) {
                    return matrix;
                }
            }
            const auto targetIt = fromIt->find(targetVolumeId);
            if (targetIt != fromIt->end()) {
                if (targetIt->is_object()) {
                    if (const auto matrixIt = targetIt->find("matrix"); matrixIt != targetIt->end()) {
                        return parseOpenDataMatrix(*matrixIt);
                    }
                }
                return parseOpenDataMatrix(*targetIt);
            }
        }
    }

    return std::nullopt;
}

nlohmann::json matrixToJson(const cv::Matx44d& matrix)
{
    nlohmann::json out = nlohmann::json::array();
    for (int row = 0; row < 3; ++row) {
        nlohmann::json rowJson = nlohmann::json::array();
        for (int col = 0; col < 4; ++col) {
            rowJson.push_back(matrix(row, col));
        }
        out.push_back(std::move(rowJson));
    }
    return out;
}

utils::Json matrixToUtilsJson(const cv::Matx44d& matrix)
{
    auto out = utils::Json::array();
    for (int row = 0; row < 3; ++row) {
        auto rowJson = utils::Json::array();
        for (int col = 0; col < 4; ++col) {
            rowJson.push_back(matrix(row, col));
        }
        out.push_back(std::move(rowJson));
    }
    return out;
}

void applyOriginalVolumeDownscale(const std::filesystem::path& segmentDir,
                                  const OpenDataSegment& segment,
                                  const std::string& representationId)
{
    const double downscale = originalVolumeDownscale(segment);
    const auto metaPath = segmentDir / "meta.json";
    auto meta = nlohmann::json::parse(readTextFile(metaPath));
    if (!std::isfinite(downscale) || downscale <= 0.0 || downscale == 1.0 ||
        coordinatesAlreadyScaled(meta, downscale)) {
        return;
    }

    auto surface = load_quad_from_tifxyz(segmentDir.string());
    if (!surface) {
        throw std::runtime_error("failed to load cached catalog tifxyz segment: " +
                                 segmentDir.string());
    }

    vc::core::util::transformSurfacePoints(surface.get(), downscale, std::nullopt, 1.0);
    vc::core::util::refreshTransformedSurfaceState(surface.get());
    surface->meta["vc_open_data_coordinates_scaled_to_original_volume"] = downscale;
    surface->save(segmentDir.string(), representationId, true);
}

bool cachedMetadataNeedsNormalization(const std::filesystem::path& metaPath,
                                      const OpenDataSegment& segment,
                                      bool applyDownscale,
                                      const std::string& representationId)
{
    try {
        const auto meta = nlohmann::json::parse(readTextFile(metaPath));
        if (!meta.is_object()) {
            return true;
        }
        auto stringField = [&](const char* key) -> std::string {
            const auto it = meta.find(key);
            return it != meta.end() && it->is_string() ? it->get<std::string>() : std::string{};
        };
        if (stringField("uuid") != representationId ||
            stringField("type") != "seg" ||
            stringField("format") != "tifxyz" ||
            stringField("vc_open_data_segment_id") != segment.id ||
            stringField("vc_open_data_segment_long_id") != segment.longId ||
            stringField("vc_open_data_source_path").empty() ||
            !meta.contains("vc_open_data_source_coordinate_scale_factor") ||
            !meta.contains("vc_open_data_source_original_resolution")) {
            return true;
        }
        return applyDownscale &&
               originalVolumeDownscale(segment) != 1.0 &&
               !coordinatesAlreadyScaled(meta, originalVolumeDownscale(segment));
    } catch (...) {
        return true;
    }
}

const OpenDataVolume* findSampleVolume(const OpenDataSample& sample,
                                       const std::string& volumeId)
{
    const auto it = std::find_if(
        sample.volumes.begin(), sample.volumes.end(), [&](const auto& volume) {
            return volume.id == volumeId;
        });
    return it == sample.volumes.end() ? nullptr : &*it;
}

void applySourceCoordinateMetadata(nlohmann::json& meta,
                                   const OpenDataSample& sample,
                                   const std::string& volumeId,
                                   int coordinateLevel)
{
    const auto* volume = findSampleVolume(sample, volumeId);
    if (!volume || !volume->pixelSizeUm ||
        !std::isfinite(*volume->pixelSizeUm) || *volume->pixelSizeUm <= 0.0) {
        return;
    }
    const auto* source = preferredVolumeArtifact(*volume);
    if (!source)
        return;
    const std::string sourcePath = artifactUrl(*source);
    if (sourcePath.empty())
        return;
    meta["vc_open_data_source_path"] = sourcePath;
    meta["vc_open_data_source_coordinate_scale_factor"] =
        std::uint64_t{1} << coordinateLevel;
    meta["vc_open_data_source_original_resolution"] = *volume->pixelSizeUm;
}

void applySourceCoordinateMetadata(utils::Json& meta,
                                   const OpenDataSample& sample,
                                   const std::string& volumeId,
                                   int coordinateLevel)
{
    const auto* volume = findSampleVolume(sample, volumeId);
    if (!volume || !volume->pixelSizeUm ||
        !std::isfinite(*volume->pixelSizeUm) || *volume->pixelSizeUm <= 0.0) {
        return;
    }
    const auto* source = preferredVolumeArtifact(*volume);
    if (!source)
        return;
    const std::string sourcePath = artifactUrl(*source);
    if (sourcePath.empty())
        return;
    meta["vc_open_data_source_path"] = sourcePath;
    meta["vc_open_data_source_coordinate_scale_factor"] =
        std::uint64_t{1} << coordinateLevel;
    meta["vc_open_data_source_original_resolution"] = *volume->pixelSizeUm;
}

void writeCachedMetadata(const std::string& baseUrl,
                         const OpenDataSample& sample,
                         const OpenDataSegment& segment,
                         const std::filesystem::path& target,
                         const std::string& representationId,
                         const std::string& representation,
                         const std::string& coordinateSpace,
                         int coordinateLevel,
                         const std::string& coordinateVolumeId)
{
    const auto url = joinOpenDataUrl(baseUrl, "meta.json");
    auto bytes = vc::httpGetBytes(url);
    if (bytes.empty()) {
        throw std::runtime_error("missing or empty meta.json at " + url);
    }
    std::string body(reinterpret_cast<const char*>(bytes.data()), bytes.size());

    auto meta = nlohmann::json::parse(body);
    if (!meta.is_object()) {
        throw std::runtime_error("meta.json is not an object");
    }
    if (!meta.contains("type") || !meta["type"].is_string() || meta["type"].get<std::string>().empty()) {
        meta["type"] = "seg";
    }
    meta["uuid"] = representationId;
    if (!meta.contains("name") || !meta["name"].is_string() || meta["name"].get<std::string>().empty()) {
        meta["name"] = segment.suffix.empty() ? segmentLabel(segment) : segment.suffix;
    }
    if (!meta.contains("format") || !meta["format"].is_string() || meta["format"].get<std::string>().empty()) {
        meta["format"] = "tifxyz";
    }
    applyOpenDataMetadata(meta, baseUrl, segment);
    meta["vc_open_data_catalog_segment_lineage_id"] = segmentStableId(segment);
    meta["vc_open_data_representation"] = representation;
    meta["vc_open_data_source_coordinate_level"] = coordinateLevel;
    if (!coordinateSpace.empty())
        meta["vc_open_data_coordinate_space"] = coordinateSpace;
    applySourceCoordinateMetadata(
        meta, sample, coordinateVolumeId, coordinateLevel);

    writeStringAtomic(target, meta.dump(2));
}

void normalizeCachedMetadata(const std::string& baseUrl,
                             const OpenDataSample& sample,
                             const OpenDataSegment& segment,
                             const std::filesystem::path& target,
                             bool applyDownscale,
                             const std::string& representationId,
                             const std::string& representation,
                             const std::string& coordinateSpace,
                             int coordinateLevel,
                             const std::string& coordinateVolumeId)
{
    auto meta = nlohmann::json::parse(readTextFile(target));
    if (!meta.is_object()) {
        throw std::runtime_error("meta.json is not an object");
    }
    if (!meta.contains("type") || !meta["type"].is_string() || meta["type"].get<std::string>().empty()) {
        meta["type"] = "seg";
    }
    meta["uuid"] = representationId;
    if (!meta.contains("name") || !meta["name"].is_string() || meta["name"].get<std::string>().empty()) {
        meta["name"] = segment.suffix.empty() ? segmentLabel(segment) : segment.suffix;
    }
    if (!meta.contains("format") || !meta["format"].is_string() || meta["format"].get<std::string>().empty()) {
        meta["format"] = "tifxyz";
    }
    applyOpenDataMetadata(meta, baseUrl, segment);
    meta["vc_open_data_catalog_segment_lineage_id"] = segmentStableId(segment);
    meta["vc_open_data_representation"] = representation;
    meta["vc_open_data_source_coordinate_level"] = coordinateLevel;
    if (!coordinateSpace.empty())
        meta["vc_open_data_coordinate_space"] = coordinateSpace;
    applySourceCoordinateMetadata(
        meta, sample, coordinateVolumeId, coordinateLevel);
    writeStringAtomic(target, meta.dump(2));
    if (applyDownscale) {
        applyOriginalVolumeDownscale(target.parent_path(), segment, representationId);
    }
}

void writeCachedTifxyzBand(const std::string& baseUrl,
                           const char* fileName,
                           const std::filesystem::path& target)
{
    const auto url = joinOpenDataUrl(baseUrl, fileName);
    auto bytes = vc::httpGetBytes(url);
    if (bytes.empty()) {
        throw std::runtime_error("missing or empty " + std::string(fileName) +
                                 " at " + url);
    }
    writeBytesAtomic(target, bytes);
    normalizeTiffForMmap(target);
}

bool cacheOptionalFile(const std::string& baseUrl,
                       const char* fileName,
                       const std::filesystem::path& target)
{
    try {
        const auto url = joinOpenDataUrl(baseUrl, fileName);
        auto bytes = vc::httpGetBytes(url);
        if (!bytes.empty()) {
            writeBytesAtomic(target, bytes);
            return true;
        }
    } catch (...) {
    }
    return false;
}

std::optional<nlohmann::json> readCatalogOrigin(const std::filesystem::path& dir);

std::string inkDetectionLabel(const OpenDataSegment& segment,
                              const OpenDataArtifact& artifact,
                              std::size_t index)
{
    std::string label = segment.suffix.empty() ? segmentLabel(segment) : segment.suffix;
    if (!artifact.type.empty()) {
        label += " - " + artifact.type;
    } else {
        label += " - ink detection";
    }
    if (index > 0) {
        label += " " + std::to_string(index + 1);
    }
    return label;
}

std::vector<nlohmann::json> readInkDetectionRecords(const std::filesystem::path& segmentDir)
{
    auto readArrayFile = [&](const std::filesystem::path& path) -> std::vector<nlohmann::json> {
        if (!std::filesystem::is_regular_file(path)) {
            return {};
        }
        try {
            auto root = nlohmann::json::parse(readTextFile(path));
            if (!root.is_array()) {
                return {};
            }
            std::vector<nlohmann::json> out;
            out.reserve(root.size());
            for (const auto& item : root) {
                if (item.is_object()) {
                    out.push_back(item);
                }
            }
            return out;
        } catch (...) {
            return {};
        }
    };

    auto records = readArrayFile(segmentDir / "ink-detections.json");
    if (!records.empty()) {
        return records;
    }
    if (auto origin = readCatalogOrigin(segmentDir)) {
        const auto it = origin->find("ink_detections");
        if (it != origin->end() && it->is_array()) {
            std::vector<nlohmann::json> out;
            out.reserve(it->size());
            for (const auto& item : *it) {
                if (item.is_object()) {
                    out.push_back(item);
                }
            }
            return out;
        }
    }
    return {};
}

void writeInkDetectionRecords(const std::filesystem::path& segmentDir,
                              const std::vector<nlohmann::json>& records)
{
    nlohmann::json array = nlohmann::json::array();
    for (const auto& record : records) {
        array.push_back(record);
    }
    writeStringAtomic(segmentDir / "ink-detections.json", array.dump(2));

    if (auto origin = readCatalogOrigin(segmentDir)) {
        (*origin)["ink_detections"] = array;
        writeStringAtomic(segmentDir / "catalog-origin.json", origin->dump(2));
    }
}

std::size_t cacheInkDetectionImages(const OpenDataSample& sample,
                                    const OpenDataSegment& segment,
                                    const std::filesystem::path& segmentDir)
{
    std::vector<nlohmann::json> records = readInkDetectionRecords(segmentDir);
    std::set<std::string> localFiles;
    for (const auto& record : records) {
        const auto it = record.find("local_file");
        if (it != record.end() && it->is_string()) {
            localFiles.insert(it->get<std::string>());
        }
    }

    std::vector<const OpenDataArtifact*> supportedArtifacts;
    supportedArtifacts.reserve(segment.artifacts.size());
    bool hasJpegArtifact = false;
    for (const auto& artifact : segment.artifacts) {
        if (!isSupportedInkImageArtifact(artifact)) {
            continue;
        }
        const std::string ext = imageExtensionForUrl(artifactUrl(artifact));
        if (isJpegExtension(ext)) {
            hasJpegArtifact = true;
        }
        supportedArtifacts.push_back(&artifact);
    }

    std::size_t supportedIndex = 0;
    for (const auto* artifactPtr : supportedArtifacts) {
        const auto& artifact = *artifactPtr;
        const std::string url = artifactUrl(artifact);
        const std::string ext = imageExtensionForUrl(url);
        if (url.empty() || ext.empty()) {
            continue;
        }
        if (hasJpegArtifact && !isJpegExtension(ext)) {
            continue;
        }

        std::string baseName = artifact.type.empty() ? "ink_detection" : artifact.type;
        if (supportedIndex > 0) {
            baseName += "_" + std::to_string(supportedIndex + 1);
        }
        const std::filesystem::path relative = std::filesystem::path("ink-detections") /
            (safePathComponent(baseName) + ext);
        const std::filesystem::path target = segmentDir / relative;
        const std::string relativeString = relative.generic_string();

        bool hasFile = isNonEmptyFile(target);
        if (!hasFile) {
            try {
                auto bytes = vc::httpGetBytes(url);
                if (!bytes.empty()) {
                    writeBytesAtomic(target, bytes);
                    hasFile = true;
                }
            } catch (...) {
                hasFile = false;
            }
        }

        if (hasFile && localFiles.insert(relativeString).second) {
            nlohmann::json record;
            record["label"] = inkDetectionLabel(segment, artifact, supportedIndex);
            record["sample_id"] = sample.id;
            record["segment_id"] = segment.id;
            record["segment_long_id"] = segment.longId;
            record["artifact_type"] = artifact.type;
            record["original_source_uri"] = artifact.sourcePath;
            record["resolved_http_url"] = artifact.resolvedUrl.empty() ? artifact.sourcePath : artifact.resolvedUrl;
            record["local_file"] = relativeString;
            records.push_back(std::move(record));
        }
        ++supportedIndex;
    }

    if (!records.empty()) {
        writeInkDetectionRecords(segmentDir, records);
    }
    return records.size();
}

std::string nowUtcIso()
{
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    char buf[32] = {};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

nlohmann::json catalogOriginJson(const OpenDataSample& sample,
                                 const OpenDataSegment& segment,
                                 const OpenDataArtifact& artifact,
                                 const std::vector<std::string>& downloadedFiles)
{
    nlohmann::json j;
    j["manifest_url"] = std::string(kDefaultManifestUrl);
    j["sample_id"] = sample.id;
    j["segment_id"] = segment.id;
    j["segment_long_id"] = segment.longId;
    j["artifact_type"] = artifact.type;
    j["original_source_uri"] = artifact.sourcePath;
    j["resolved_http_url"] = artifact.resolvedUrl;
    j["downloaded_at_utc"] = nowUtcIso();
    j["downloaded_files"] = downloadedFiles;
    j["source_file_metadata"] = nlohmann::json::array();
    j["cache_state"] = cacheStateName(OpenDataSegmentCacheState::Current);
    return j;
}

bool requiredFilesPresent(const std::filesystem::path& dir)
{
    return isNonEmptyFile(dir / "meta.json") &&
           isNonEmptyFile(dir / "x.tif") &&
           isNonEmptyFile(dir / "y.tif") &&
           isNonEmptyFile(dir / "z.tif");
}

bool originMatches(const nlohmann::json& origin,
                   const OpenDataSample& sample,
                   const OpenDataSegment& segment,
                   const OpenDataArtifact& artifact)
{
    if (!origin.is_object()) {
        return false;
    }
    auto stringField = [&](const char* key) -> std::string {
        const auto it = origin.find(key);
        return it != origin.end() && it->is_string() ? it->get<std::string>() : std::string{};
    };
    const std::string expectedUrl = artifact.resolvedUrl.empty() ? artifact.sourcePath : artifact.resolvedUrl;
    const std::string actualUrl = stringField("resolved_http_url").empty()
        ? stringField("original_source_uri")
        : stringField("resolved_http_url");
    return stringField("sample_id") == sample.id &&
           stringField("segment_id") == segment.id &&
           (actualUrl.empty() || expectedUrl.empty() || actualUrl == expectedUrl);
}

bool transformedOriginMatches(const nlohmann::json& origin,
                              const OpenDataSample& sample,
                              const OpenDataSegment& segment,
                              const std::string& sourceVolumeId,
                              const std::string& targetVolumeId,
                              const cv::Matx44d& matrix)
{
    if (!origin.is_object()) {
        return false;
    }
    auto stringField = [&](const char* key) -> std::string {
        const auto it = origin.find(key);
        return it != origin.end() && it->is_string() ? it->get<std::string>() : std::string{};
    };
    if (stringField("sample_id") != sample.id ||
        stringField("segment_id") != segment.id ||
        stringField("source_volume_id") != sourceVolumeId ||
        stringField("target_volume_id") != targetVolumeId) {
        return false;
    }
    const auto matrixIt = origin.find("matrix");
    const auto cachedMatrix = matrixIt == origin.end() ? std::optional<cv::Matx44d>{}
                                                       : parseOpenDataMatrix(*matrixIt);
    if (!cachedMatrix) {
        return false;
    }
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (std::abs((*cachedMatrix)(row, col) - matrix(row, col)) > 1e-12) {
                return false;
            }
        }
    }
    return true;
}

std::optional<nlohmann::json> readCatalogOrigin(const std::filesystem::path& dir)
{
    const auto path = dir / "catalog-origin.json";
    if (!std::filesystem::is_regular_file(path)) {
        return std::nullopt;
    }
    try {
        return nlohmann::json::parse(readTextFile(path));
    } catch (...) {
        return std::nullopt;
    }
}

bool originStateAllowsFastOpen(const nlohmann::json& origin)
{
    const auto it = origin.find("cache_state");
    if (it == origin.end()) {
        return true;
    }
    if (!it->is_string()) {
        return false;
    }
    const auto state = lowerCopy(it->get<std::string>());
    return state.empty() || state == cacheStateName(OpenDataSegmentCacheState::Current);
}

void writeCatalogOriginState(const std::filesystem::path& dir,
                             OpenDataSegmentCacheState state)
{
    auto origin = readCatalogOrigin(dir).value_or(nlohmann::json::object());
    origin["cache_state"] = cacheStateName(state);
    writeStringAtomic(dir / "catalog-origin.json", origin.dump(2));
}

std::filesystem::path makeTempSegmentDir(const std::filesystem::path& finalDir)
{
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    return finalDir.parent_path() /
           (finalDir.filename().string() + ".tmp-" + std::to_string(tick));
}

void publishSegmentDirectory(const std::filesystem::path& tempDir,
                             const std::filesystem::path& finalDir)
{
    std::error_code ec;
    std::filesystem::create_directories(finalDir.parent_path(), ec);
    if (ec) {
        throw std::runtime_error("failed to create cache parent: " + ec.message());
    }

    const auto backupDir = finalDir.parent_path() /
        (finalDir.filename().string() + ".previous");
    std::filesystem::remove_all(backupDir, ec);
    ec.clear();
    if (std::filesystem::exists(finalDir, ec)) {
        std::filesystem::rename(finalDir, backupDir, ec);
        if (ec) {
            throw std::runtime_error("failed to move existing cache aside: " + ec.message());
        }
    }

    std::filesystem::rename(tempDir, finalDir, ec);
    if (ec) {
        std::error_code restoreEc;
        if (std::filesystem::exists(backupDir, restoreEc) &&
            !std::filesystem::exists(finalDir, restoreEc)) {
            std::filesystem::rename(backupDir, finalDir, restoreEc);
        }
        throw std::runtime_error("failed to publish segment cache: " + ec.message());
    }

    std::filesystem::remove_all(backupDir, ec);
}

bool cacheTifxyzRepresentation(
                        const OpenDataSample& sample,
                        const OpenDataSegment& segment,
                        const OpenDataArtifact& artifact,
                        const std::filesystem::path& segmentDir,
                        bool applyDownscale,
                        const std::string& representationId,
                        const std::string& representation,
                        const std::string& coordinateVolumeId,
                        const std::string& coordinateSpace,
                        int coordinateLevel,
                        std::string* errorOut,
                        const std::function<void(const char*, const char*)>& fileProgress = {},
                        bool forceRefresh = false)
{
    const auto url = artifactUrl(artifact);
    if (url.empty()) {
        if (errorOut) *errorOut = "tifxyz artifact has no URL.";
        return false;
    }

    if (!forceRefresh && requiredFilesPresent(segmentDir)) {
        const auto origin = readCatalogOrigin(segmentDir);
        if (origin && originMatches(*origin, sample, segment, artifact) &&
            originStateAllowsFastOpen(*origin)) {
            if (cachedMetadataNeedsNormalization(segmentDir / "meta.json",
                                                 segment,
                                                 applyDownscale,
                                                 representationId)) {
                normalizeCachedMetadata(url, sample,
                                        segment,
                                        segmentDir / "meta.json",
                                        applyDownscale,
                                        representationId,
                                        representation,
                                        coordinateSpace,
                                        coordinateLevel,
                                        coordinateVolumeId);
            }
            return true;
        }
        if (!origin || originMatches(*origin, sample, segment, artifact)) {
            normalizeCachedMetadata(url, sample,
                                    segment,
                                    segmentDir / "meta.json",
                                    applyDownscale,
                                    representationId,
                                    representation,
                                    coordinateSpace,
                                    coordinateLevel,
                                    coordinateVolumeId);
            if (origin) {
                writeCatalogOriginState(segmentDir, OpenDataSegmentCacheState::Current);
            } else {
                writeStringAtomic(
                    segmentDir / "catalog-origin.json",
                    catalogOriginJson(sample,
                                      segment,
                                      artifact,
                                      {"meta.json", "x.tif", "y.tif", "z.tif"}).dump(2));
            }
            try {
                cacheInkDetectionImages(sample, segment, segmentDir);
            } catch (...) {
            }
            return true;
        }
    }

    const auto tempDir = makeTempSegmentDir(segmentDir);
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir);
    std::vector<std::string> downloadedFiles;

    try {
        auto runFile = [&](const char* fileName, const auto& fn) {
            if (fileProgress) fileProgress(fileName, "start");
            if (fn()) {
                downloadedFiles.push_back(fileName);
            }
            if (fileProgress) fileProgress(fileName, "done");
        };
        runFile("meta.json", [&]() {
            writeCachedMetadata(url, sample, segment, tempDir / "meta.json",
                                representationId, representation,
                                coordinateSpace, coordinateLevel,
                                coordinateVolumeId);
            return true;
        });
        runFile("x.tif", [&]() {
            writeCachedTifxyzBand(url, "x.tif", tempDir / "x.tif");
            return true;
        });
        runFile("y.tif", [&]() {
            writeCachedTifxyzBand(url, "y.tif", tempDir / "y.tif");
            return true;
        });
        runFile("z.tif", [&]() {
            writeCachedTifxyzBand(url, "z.tif", tempDir / "z.tif");
            return true;
        });
        runFile("mask.tif", [&]() {
            return cacheOptionalFile(url, "mask.tif", tempDir / "mask.tif");
        });
        runFile("overlapping.json", [&]() {
            return cacheOptionalFile(url, "overlapping.json", tempDir / "overlapping.json");
        });

        if (!requiredFilesPresent(tempDir)) {
            throw std::runtime_error("downloaded segment is missing required tifxyz files");
        }
        normalizeCachedMetadata(url, sample,
                                segment,
                                tempDir / "meta.json",
                                applyDownscale,
                                representationId,
                                representation,
                                coordinateSpace,
                                coordinateLevel,
                                coordinateVolumeId);
        writeStringAtomic(tempDir / "catalog-origin.json",
                          catalogOriginJson(sample, segment, artifact, downloadedFiles).dump(2));
        try {
            cacheInkDetectionImages(sample, segment, tempDir);
        } catch (...) {
        }
        publishSegmentDirectory(tempDir, segmentDir);
        return true;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(tempDir, ec);
        if (errorOut) *errorOut = e.what();
    } catch (...) {
        std::filesystem::remove_all(tempDir, ec);
        if (errorOut) *errorOut = "unknown error.";
    }
    return false;
}

bool cacheTifxyzSegment(const OpenDataSample& sample,
                        const OpenDataSegment& segment,
                        const std::filesystem::path& segmentDir,
                        std::string* errorOut,
                        const std::function<void(const char*, const char*)>& fileProgress = {},
                        bool forceRefresh = false)
{
    const auto* artifact = preferredTifxyzArtifact(segment);
    if (!artifact) {
        if (errorOut) *errorOut = "no tifxyz artifact.";
        return false;
    }
    const bool applyDownscale = !artifactAlreadyInOriginalVolumeScale(*artifact);
    const auto sourceId = sourceVolumeIdForSegment(segment);
    return cacheTifxyzRepresentation(
        sample, segment, *artifact, segmentDir, applyDownscale,
        segmentStableId(segment),
        applyDownscale ? "derived-native" : "published-native",
        sourceId,
        sample.id + "/" + sourceId + "@L0", 0,
        errorOut, fileProgress, forceRefresh);
}

bool cacheTransformedTifxyzSegment(const OpenDataSample& sample,
                                   const OpenDataSegment& segment,
                                   const std::filesystem::path& sourceSegmentDir,
                                   const std::filesystem::path& targetSegmentDir,
                                   const std::string& targetVolumeId,
                                   std::string* errorOut)
{
    if (!requiredFilesPresent(sourceSegmentDir)) {
        if (errorOut) *errorOut = "source cached segment is incomplete.";
        return false;
    }

    const std::string sourceVolumeId = sourceVolumeIdForSegment(segment);
    const auto matrix = findVolumeTransformMatrix(sample, sourceVolumeId, targetVolumeId);
    if (!matrix) {
        if (errorOut) *errorOut = "no transform from " + sourceVolumeId + " to " + targetVolumeId + ".";
        return false;
    }

    if (requiredFilesPresent(targetSegmentDir)) {
        const auto origin = readCatalogOrigin(targetSegmentDir);
        if (origin && transformedOriginMatches(*origin, sample, segment, sourceVolumeId, targetVolumeId, *matrix)) {
            return true;
        }
    }

    const auto tempDir = makeTempSegmentDir(targetSegmentDir);
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir);

    try {
        auto surface = load_quad_from_tifxyz(sourceSegmentDir.string());
        if (!surface) {
            throw std::runtime_error("failed to load source cached tifxyz segment");
        }

        vc::core::util::transformSurfacePoints(surface.get(), 1.0, *matrix, 1.0);
        vc::core::util::refreshTransformedSurfaceState(surface.get());
        surface->meta["vc_open_data_transform_source_volume_id"] = sourceVolumeId;
        surface->meta["vc_open_data_transform_target_volume_id"] = targetVolumeId;
        surface->meta["vc_open_data_volume_transform_matrix"] = matrixToUtilsJson(*matrix);
        surface->meta["vc_open_data_catalog_segment_lineage_id"] =
            segmentStableId(segment);
        surface->meta["vc_open_data_representation"] = "generated-native-transform";
        surface->meta["vc_open_data_source_coordinate_level"] = 0;
        surface->meta["vc_open_data_coordinate_space"] =
            sample.id + "/" + targetVolumeId + "@L0";
        applySourceCoordinateMetadata(
            surface->meta, sample, targetVolumeId, 0);
        surface->save(
            tempDir.string(),
            segmentStableId(segment) + "-generated-" +
                safePathComponent(targetVolumeId) + "-L0",
            true);

        nlohmann::json origin;
        origin["manifest_url"] = std::string(kDefaultManifestUrl);
        origin["sample_id"] = sample.id;
        origin["segment_id"] = segment.id;
        origin["segment_long_id"] = segment.longId;
        origin["source_volume_id"] = sourceVolumeId;
        origin["target_volume_id"] = targetVolumeId;
        origin["matrix"] = matrixToJson(*matrix);
        origin["transformed_at_utc"] = nowUtcIso();
        origin["cache_state"] = cacheStateName(OpenDataSegmentCacheState::Current);
        writeStringAtomic(tempDir / "catalog-origin.json", origin.dump(2));

        publishSegmentDirectory(tempDir, targetSegmentDir);
        return true;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(tempDir, ec);
        if (errorOut) *errorOut = e.what();
    } catch (...) {
        std::filesystem::remove_all(tempDir, ec);
        if (errorOut) *errorOut = "unknown error.";
    }
    return false;
}

void markOrphanedEntries(const std::filesystem::path& segmentsRoot,
                         const std::set<std::string>& expectedDirNames)
{
    std::error_code ec;
    if (!std::filesystem::is_directory(segmentsRoot, ec)) {
        return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(segmentsRoot, ec)) {
        if (ec || !entry.is_directory()) {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (expectedDirNames.find(name) != expectedDirNames.end()) {
            continue;
        }
        if (isOpenDataCatalogSegmentDirectory(entry.path())) {
            try {
                writeCatalogOriginState(entry.path(), OpenDataSegmentCacheState::Orphaned);
            } catch (...) {
            }
        }
    }
}

void reportProgress(const OpenDataSampleProgressCallback& callback,
                    const OpenDataSampleDownloadProgress& progress) noexcept
{
    if (!callback) {
        return;
    }
    try {
        callback(progress);
    } catch (...) {
    }
}

} // namespace

std::vector<OpenDataSegmentRepresentation>
classifyOpenDataSegmentRepresentations(const OpenDataSample& sample,
                                       const OpenDataSegment& segment)
{
    std::vector<OpenDataSegmentRepresentation> result;
    std::set<std::string> volumeIds;
    for (const auto& volume : sample.volumes) {
        if (!volume.id.empty())
            volumeIds.insert(volume.id);
    }

    const auto hashUrl = [](std::string_view value) {
        std::uint64_t hash = 14695981039346656037ULL;
        for (const unsigned char c : value) {
            hash ^= c;
            hash *= 1099511628211ULL;
        }
        std::ostringstream out;
        out << std::hex << std::setfill('0') << std::setw(16) << hash;
        return out.str();
    };
    const auto levelForDownscale = [&]() -> std::optional<int> {
        const double downscale = originalVolumeDownscale(segment);
        for (int level = 0; level <= 5; ++level) {
            if (std::abs(downscale - std::ldexp(1.0, level)) <=
                1.0e-9 * std::max(1.0, std::abs(downscale))) {
                return level;
            }
        }
        return std::nullopt;
    }();

    for (const auto& artifact : segment.artifacts) {
        const auto type = lowerCopy(artifact.type);
        if (type.find("tifxyz") == std::string::npos || artifactUrl(artifact).empty())
            continue;
        const std::string hash = hashUrl(artifactUrl(artifact));
        const std::string lineage = segmentStableId(segment);

        if (type == "tifxyz-transformed") {
            if (!artifact.targetVolumeId || artifact.targetVolumeId->empty() ||
                volumeIds.find(*artifact.targetVolumeId) == volumeIds.end()) {
                continue;
            }
            OpenDataSegmentRepresentation published;
            published.artifact = &artifact;
            published.kind = OpenDataSegmentRepresentationKind::PublishedTransformed;
            published.coordinateVolumeId = *artifact.targetVolumeId;
            published.sourceCoordinateLevel = 0;
            published.coordinateSpace = sample.id + "/" +
                published.coordinateVolumeId + "@L0";
            published.representationId = lineage + "-published-" +
                safePathComponent(published.coordinateVolumeId) + "-L0-" + hash;
            result.push_back(std::move(published));
            continue;
        }

        if (type != "tifxyz" && type != "tifxyz-flattened" &&
            type != "tifxyz-normalized") {
            continue;
        }

        // Flattened, normalized, and ordinary tifxyz artifacts are authored
        // in the original-volume coordinate level described by the exact
        // power-of-two downscale. Unknown factors remain manual-only.
        if (!levelForDownscale)
            continue;
        const std::string sourceId = segment.originalVolumeId.empty()
            ? sourceVolumeIdForSegment(segment)
            : segment.originalVolumeId;
        if (sourceId.empty() || volumeIds.find(sourceId) == volumeIds.end())
            continue;

        OpenDataSegmentRepresentation authored;
        authored.artifact = &artifact;
        authored.kind = OpenDataSegmentRepresentationKind::Authored;
        authored.coordinateVolumeId = sourceId;
        authored.sourceCoordinateLevel = *levelForDownscale;
        authored.coordinateSpace = sample.id + "/" + sourceId + "@L" +
            std::to_string(*levelForDownscale);
        authored.representationId = lineage + "-authored-L" +
            std::to_string(*levelForDownscale) + "-" + hash;
        result.push_back(authored);

        if (*levelForDownscale > 0) {
            auto native = authored;
            native.kind = OpenDataSegmentRepresentationKind::DerivedNative;
            native.sourceCoordinateLevel = 0;
            native.coordinateSpace = sample.id + "/" + sourceId + "@L0";
            native.representationId = lineage + "-derived-native-L0-" + hash;
            result.push_back(std::move(native));
        }
    }
    return result;
}

std::filesystem::path openDataSegmentRepresentationCacheRoot(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegmentRepresentation& representation)
{
    auto root = openDataSegmentCacheRoot(
        remoteCacheRoot, sample, representation.coordinateVolumeId);
    switch (representation.kind) {
        case OpenDataSegmentRepresentationKind::Authored:
            return root / ("authored-L" +
                           std::to_string(representation.sourceCoordinateLevel));
        case OpenDataSegmentRepresentationKind::DerivedNative:
            return root / "derived-native-L0";
        case OpenDataSegmentRepresentationKind::PublishedTransformed:
            return root / "published-L0";
    }
    return root / "unknown";
}

std::filesystem::path openDataSegmentRepresentationCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment&,
    const OpenDataSegmentRepresentation& representation)
{
    return openDataSegmentRepresentationCacheRoot(remoteCacheRoot, sample, representation) /
           safePathComponent(representation.representationId);
}

const char* cacheStateName(OpenDataSegmentCacheState state) noexcept
{
    switch (state) {
        case OpenDataSegmentCacheState::Missing: return "missing";
        case OpenDataSegmentCacheState::Current: return "current";
        case OpenDataSegmentCacheState::Incomplete: return "incomplete";
        case OpenDataSegmentCacheState::Stale: return "stale";
        case OpenDataSegmentCacheState::Orphaned: return "orphaned";
    }
    return "unknown";
}

std::filesystem::path openDataSegmentCacheRoot(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const std::string& sourceVolumeId)
{
    const auto sampleComponent = safePathComponent(sample.id.empty() ? "sample" : sample.id);
    const auto sourceComponent = safePathComponent(sourceVolumeId.empty() ? "volume" : sourceVolumeId);
    return remoteCacheRoot / "open_data" / "segments" /
           sampleComponent / sourceComponent;
}

std::filesystem::path openDataSegmentCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment)
{
    return openDataSegmentCacheRoot(remoteCacheRoot, sample, sourceVolumeIdForSegment(segment)) /
           safePathComponent(segment.id);
}

std::filesystem::path openDataTransformedSegmentCacheRoot(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const std::string& sourceVolumeId,
    const std::string& targetVolumeId)
{
    return openDataSegmentCacheRoot(remoteCacheRoot, sample, targetVolumeId) /
           ("generated-from-" +
            safePathComponent(sourceVolumeId.empty() ? "volume" : sourceVolumeId) +
            "-L0");
}

std::filesystem::path openDataTransformedSegmentCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment,
    const std::string& targetVolumeId)
{
    return openDataTransformedSegmentCacheRoot(
               remoteCacheRoot, sample, sourceVolumeIdForSegment(segment), targetVolumeId) /
           safePathComponent(segment.id);
}

std::filesystem::path openDataEditableSegmentRoot(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId)
{
    return remoteCacheRoot / "open_data" / "editable" /
           safePathComponent(sampleId.empty() ? "sample" : sampleId);
}

std::filesystem::path openDataPatchesRoot(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId)
{
    return remoteCacheRoot / "open_data" / "segments" /
           safePathComponent(sampleId.empty() ? "sample" : sampleId) / "patches";
}

std::size_t manualOpenDataSegmentCount(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId)
{
    const auto sampleRoot = remoteCacheRoot / "open_data" / "segments" /
                            safePathComponent(sampleId.empty() ? "sample" : sampleId);
    std::error_code ec;
    std::filesystem::recursive_directory_iterator entries(
        sampleRoot,
        std::filesystem::directory_options::skip_permission_denied,
        ec);
    if (ec) {
        return 0;
    }

    std::size_t count = 0;
    const std::filesystem::recursive_directory_iterator end;
    while (entries != end) {
        const auto& entry = *entries;
        std::error_code entryEc;
        if (entry.is_directory(entryEc) && !entryEc) {
            const auto name = entry.path().filename().string();
            if (name == "backups" || (!name.empty() && name.front() == '.')) {
                entries.disable_recursion_pending();
            } else {
                const bool isSegment =
                    std::filesystem::is_regular_file(entry.path() / "meta.json", entryEc);
                entryEc.clear();
                const bool isCatalogSegment = std::filesystem::is_regular_file(
                    entry.path() / "catalog-origin.json", entryEc);
                if (isSegment) {
                    if (!isCatalogSegment) {
                        ++count;
                    }
                    entries.disable_recursion_pending();
                }
            }
        }
        entries.increment(ec);
        if (ec) {
            break;
        }
    }
    return count;
}

OpenDataSegmentCacheState cacheStateForSegment(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment)
{
    const auto dir = openDataSegmentCacheDirectory(remoteCacheRoot, sample, segment);
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
        return OpenDataSegmentCacheState::Missing;
    }
    if (!requiredFilesPresent(dir)) {
        return OpenDataSegmentCacheState::Incomplete;
    }

    const auto* artifact = preferredTifxyzArtifact(segment);
    if (!artifact) {
        return OpenDataSegmentCacheState::Orphaned;
    }

    const auto origin = readCatalogOrigin(dir);
    if (!origin) {
        return OpenDataSegmentCacheState::Current;
    }
    if (!originMatches(*origin, sample, segment, *artifact)) {
        return OpenDataSegmentCacheState::Stale;
    }
    const auto it = origin->find("cache_state");
    if (it != origin->end() && it->is_string()) {
        const auto state = lowerCopy(it->get<std::string>());
        if (state == "orphaned") return OpenDataSegmentCacheState::Orphaned;
        if (state == "stale") return OpenDataSegmentCacheState::Stale;
        if (state == "incomplete") return OpenDataSegmentCacheState::Incomplete;
    }
    return OpenDataSegmentCacheState::Current;
}

OpenDataSegmentCacheReconcileResult attachExistingOpenDataSegmentCaches(
    VolumePkg& pkg,
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot)
{
    OpenDataSegmentCacheReconcileResult result;
    if (sample.tifxyzSegmentCount() == 0) {
        return result;
    }

    result.supportedTifxyzSegments = static_cast<int>(sample.tifxyzSegmentCount());
    if (remoteCacheRoot.empty()) {
        result.skippedTifxyzSegments = result.supportedTifxyzSegments;
        result.messages.push_back("Skipped cached tifxyz segment discovery: no remote cache directory configured.");
        return result;
    }

    std::vector<const OpenDataSegment*> tifxyzSegments;
    tifxyzSegments.reserve(sample.segments.size());
    for (const auto& segment : sample.segments) {
        if (segment.hasTifxyz()) {
            tifxyzSegments.push_back(&segment);
        }
    }

    std::map<std::string, int> cachedSegmentsBySource;
    for (const auto* segment : tifxyzSegments) {
        const auto* preferred = preferredTifxyzArtifact(*segment);
        if (preferred && lowerCopy(preferred->type) == "tifxyz-transformed")
            continue;
        const auto sourceVolumeId = sourceVolumeIdForSegment(*segment);
        if (cacheStateForSegment(remoteCacheRoot, sample, *segment) ==
            OpenDataSegmentCacheState::Current) {
            ++result.cachedTifxyzSegments;
            ++cachedSegmentsBySource[sourceVolumeId];
        }
    }

    auto attachEntry = [&](const std::string& location,
                           std::vector<std::string> tags,
                           const std::string& label,
                           const std::string& cacheDescription,
                           int& failedCount) {
        const std::string failurePrefix =
            "Failed to attach " + cacheDescription + " for " + label;
        try {
            if (pkg.addSegmentsEntry(location, tags)) {
                ++result.attachedSegmentEntries;
            } else if (hasSegmentEntry(pkg, location)) {
                pkg.reconcileSegmentsEntryTags(
                    location, tags,
                    {"vc-open-data-source-coordinate-level:",
                     "vc-open-data-coordinate-space:",
                     "vc-open-data-source-volume-id:",
                     "vc-open-data-target-volume-id:",
                     "vc-open-data-segment-representation:"});
            } else {
                ++failedCount;
                result.messages.push_back(failurePrefix + ".");
            }
        } catch (const std::exception& e) {
            ++failedCount;
            result.messages.push_back(failurePrefix + ": " + e.what());
        } catch (...) {
            ++failedCount;
            result.messages.push_back(failurePrefix + ": unknown error.");
        }
    };

    for (const auto& [sourceVolumeId, cachedCount] : cachedSegmentsBySource) {
        if (cachedCount <= 0) {
            continue;
        }
        bool hasDerivedRepresentationCache = false;
        for (const auto* segment : tifxyzSegments) {
            for (const auto& representation :
                 classifyOpenDataSegmentRepresentations(sample, *segment)) {
                if (representation.kind !=
                        OpenDataSegmentRepresentationKind::DerivedNative ||
                    representation.coordinateVolumeId != sourceVolumeId) {
                    continue;
                }
                if (requiredFilesPresent(openDataSegmentRepresentationCacheDirectory(
                        remoteCacheRoot, sample, *segment, representation))) {
                    hasDerivedRepresentationCache = true;
                    break;
                }
            }
            if (hasDerivedRepresentationCache)
                break;
        }
        if (hasDerivedRepresentationCache)
            continue;
        std::vector<std::string> tags = {
            "open-data", "immutable",
            "vc-open-data-segment-representation:derived-native"};
        if (!sourceVolumeId.empty()) {
            tags.push_back("vc-open-data-source-volume-id:" + sourceVolumeId);
            tags.push_back("vc-open-data-source-coordinate-level:0");
            tags.push_back("vc-open-data-coordinate-space:" + sample.id + "/" +
                           sourceVolumeId + "@L0");
        }
        attachEntry(openDataSegmentCacheRoot(remoteCacheRoot, sample, sourceVolumeId).string(),
                    std::move(tags),
                    sourceVolumeId.empty() ? std::string("source volume") : sourceVolumeId,
                    "cached tifxyz segment directory",
                    result.failedTifxyzSegments);
    }

    std::map<std::filesystem::path, OpenDataSegmentRepresentation>
        cachedRepresentationRoots;
    for (const auto* segment : tifxyzSegments) {
        for (const auto& representation :
             classifyOpenDataSegmentRepresentations(sample, *segment)) {
            const auto dir = openDataSegmentRepresentationCacheDirectory(
                remoteCacheRoot, sample, *segment, representation);
            if (!requiredFilesPresent(dir))
                continue;
            cachedRepresentationRoots.emplace(
                openDataSegmentRepresentationCacheRoot(
                    remoteCacheRoot, sample, representation),
                representation);
        }
    }
    for (const auto& [root, representation] : cachedRepresentationRoots) {
        std::string kindTag;
        switch (representation.kind) {
            case OpenDataSegmentRepresentationKind::Authored:
                kindTag = "authored";
                break;
            case OpenDataSegmentRepresentationKind::DerivedNative:
                kindTag = "derived-native";
                break;
            case OpenDataSegmentRepresentationKind::PublishedTransformed:
                kindTag = "published-transformed";
                break;
        }
        std::vector<std::string> tags{
            "open-data",
            "immutable",
            "vc-open-data-segment-representation:" + kindTag,
            "vc-open-data-source-coordinate-level:" +
                std::to_string(representation.sourceCoordinateLevel),
            "vc-open-data-coordinate-space:" + representation.coordinateSpace,
        };
        if (representation.kind == OpenDataSegmentRepresentationKind::PublishedTransformed)
            tags.push_back("vc-open-data-target-volume-id:" +
                           representation.coordinateVolumeId);
        else
            tags.push_back("vc-open-data-source-volume-id:" +
                           representation.coordinateVolumeId);
        if (representation.kind ==
            OpenDataSegmentRepresentationKind::DerivedNative) {
            pkg.relocateSegmentsEntry(
                openDataSegmentCacheRoot(
                    remoteCacheRoot, sample,
                    representation.coordinateVolumeId).string(),
                root.string());
        }
        attachEntry(root.string(), std::move(tags), representation.coordinateSpace,
                    "coordinate-specific tifxyz representation",
                    result.failedTifxyzSegments);
    }

    std::set<std::string> targetVolumeIds;
    for (const auto& volume : sample.volumes) {
        if (!volume.id.empty()) {
            targetVolumeIds.insert(volume.id);
        }
    }

    std::map<std::pair<std::string, std::string>, int> transformedByRoute;
    for (const auto& targetVolumeId : targetVolumeIds) {
        for (const auto* segment : tifxyzSegments) {
            const auto sourceVolumeId = sourceVolumeIdForSegment(*segment);
            if (sourceVolumeId.empty() || sourceVolumeId == targetVolumeId) {
                continue;
            }
            const auto matrix = findVolumeTransformMatrix(sample, sourceVolumeId, targetVolumeId);
            if (!matrix) {
                continue;
            }
            const auto transformedDir = openDataTransformedSegmentCacheDirectory(
                remoteCacheRoot, sample, *segment, targetVolumeId);
            if (!requiredFilesPresent(transformedDir)) {
                continue;
            }
            const auto origin = readCatalogOrigin(transformedDir);
            if (origin && !transformedOriginMatches(
                              *origin, sample, *segment, sourceVolumeId, targetVolumeId, *matrix)) {
                continue;
            }
            ++result.transformedTifxyzSegments;
            ++transformedByRoute[{sourceVolumeId, targetVolumeId}];
        }
    }

    for (const auto& [route, transformedCount] : transformedByRoute) {
        if (transformedCount <= 0) {
            continue;
        }
        const auto& sourceVolumeId = route.first;
        const auto& targetVolumeId = route.second;
        attachEntry(openDataTransformedSegmentCacheRoot(
                        remoteCacheRoot, sample, sourceVolumeId, targetVolumeId).string(),
                    {"open-data",
                     "immutable",
                     "open-data-transformed",
                     "vc-open-data-segment-representation:generated-native-transform",
                     "vc-open-data-source-volume-id:" + sourceVolumeId,
                     "vc-open-data-target-volume-id:" + targetVolumeId,
                     "vc-open-data-source-coordinate-level:0",
                     "vc-open-data-coordinate-space:" + sample.id + "/" +
                         targetVolumeId + "@L0"},
                    sourceVolumeId + " to " + targetVolumeId,
                    "transformed tifxyz segment directory",
                    result.failedTransformedTifxyzSegments);
    }

    return result;
}

bool isOpenDataCatalogSegmentDirectory(const std::filesystem::path& segmentDir)
{
    return std::filesystem::is_regular_file(segmentDir / "catalog-origin.json");
}

std::vector<OpenDataInkDetectionEntry> cachedInkDetectionsForSegmentDirectory(
    const std::filesystem::path& segmentDir)
{
    std::vector<OpenDataInkDetectionEntry> out;
    if (!std::filesystem::is_directory(segmentDir)) {
        return out;
    }

    for (const auto& record : readInkDetectionRecords(segmentDir)) {
        auto stringField = [&](const char* key) -> std::string {
            const auto it = record.find(key);
            return it != record.end() && it->is_string() ? it->get<std::string>() : std::string{};
        };
        std::string localFile = stringField("local_file");
        if (localFile.empty()) {
            continue;
        }
        std::filesystem::path localPath = std::filesystem::path(localFile);
        if (localPath.is_relative()) {
            localPath = segmentDir / localPath;
        }
        if (!isNonEmptyFile(localPath)) {
            continue;
        }
        OpenDataInkDetectionEntry entry;
        entry.label = stringField("label");
        entry.sampleId = stringField("sample_id");
        entry.segmentId = stringField("segment_id");
        entry.segmentLongId = stringField("segment_long_id");
        entry.artifactType = stringField("artifact_type");
        entry.sourceUrl = stringField("resolved_http_url");
        if (entry.sourceUrl.empty()) {
            entry.sourceUrl = stringField("original_source_uri");
        }
        entry.localPath = std::move(localPath);
        if (entry.label.empty()) {
            entry.label = entry.segmentId.empty() ? entry.localPath.filename().string()
                                                  : entry.segmentId;
        }
        out.push_back(std::move(entry));
    }
    return out;
}

std::filesystem::path defaultEditableCopyPathForCatalogSegment(
    const std::filesystem::path& catalogSegmentDir,
    const std::filesystem::path& activeSegmentsRoot)
{
    const auto currentSegmentsRoot = activeSegmentsRoot.empty()
        ? catalogSegmentDir.parent_path()
        : activeSegmentsRoot;
    auto editableRootName = currentSegmentsRoot.filename().string();
    if (editableRootName.empty()) {
        editableRootName = "segments";
    }
    editableRootName += "_editable";

    auto baseName = catalogSegmentDir.filename().string();
    if (baseName.empty()) {
        baseName = "segment";
    }
    return currentSegmentsRoot.parent_path() / editableRootName /
           baseName;
}

void copyCatalogSegmentToEditableDirectory(
    const std::filesystem::path& catalogSegmentDir,
    const std::filesystem::path& editableSegmentDir)
{
    if (!isOpenDataCatalogSegmentDirectory(catalogSegmentDir) ||
        !requiredFilesPresent(catalogSegmentDir)) {
        throw std::runtime_error("catalog segment is not a complete open-data tifxyz directory");
    }
    std::error_code ec;
    if (std::filesystem::exists(editableSegmentDir, ec)) {
        if (!std::filesystem::is_directory(editableSegmentDir, ec)) {
            throw std::runtime_error("editable destination exists and is not a directory");
        }
        if (!std::filesystem::is_empty(editableSegmentDir, ec) &&
            !requiredFilesPresent(editableSegmentDir)) {
            throw std::runtime_error("editable destination is not empty and is not a tifxyz segment");
        }
    } else {
        std::filesystem::create_directories(editableSegmentDir);
    }

    if (std::filesystem::is_empty(editableSegmentDir, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(catalogSegmentDir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            if (entry.path().filename() == "catalog-origin.json") {
                continue;
            }
            std::filesystem::copy_file(
                entry.path(),
                editableSegmentDir / entry.path().filename(),
                std::filesystem::copy_options::overwrite_existing);
        }
    }
    if (!requiredFilesPresent(editableSegmentDir)) {
        throw std::runtime_error("editable destination is missing required tifxyz files");
    }
}

OpenDataSegmentCacheReconcileResult reconcileOpenDataSampleSegments(
    VolumePkg& pkg,
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSampleProgressCallback& progressCallback,
    bool forceRefresh)
{
    OpenDataSegmentCacheReconcileResult result;
    if (sample.tifxyzSegmentCount() == 0) {
        return result;
    }

    result.supportedTifxyzSegments = static_cast<int>(sample.tifxyzSegmentCount());

    if (remoteCacheRoot.empty()) {
        result.skippedTifxyzSegments += result.supportedTifxyzSegments;
        result.messages.push_back("Skipped tifxyz segments: no remote cache directory configured.");
        return result;
    }

    std::vector<const OpenDataSegment*> tifxyzSegments;
    tifxyzSegments.reserve(sample.segments.size());
    std::map<std::string, std::set<std::string>> expectedDirNamesBySource;
    for (const auto& segment : sample.segments) {
        if (!segment.hasTifxyz()) {
            continue;
        }
        tifxyzSegments.push_back(&segment);
        expectedDirNamesBySource[sourceVolumeIdForSegment(segment)].insert(safePathComponent(segment.id));
    }

    std::atomic_size_t next{0};
    std::atomic_int completedFiles{0};
    std::atomic_int completedSegments{0};
    std::atomic_int failedSegments{0};
    std::atomic_int activeWorkers{0};
    std::mutex resultMutex;
    std::mutex progressMutex;
    const auto hardware = std::thread::hardware_concurrency();
    const std::size_t desiredWorkers = hardware == 0 ? 4 : hardware;
    const std::size_t workerCount = std::min<std::size_t>(
        tifxyzSegments.size(),
        tifxyzSegments.size() <= 1
            ? 1
            : std::max<std::size_t>(2, std::min<std::size_t>(desiredWorkers, 8)));
    constexpr int kProgressFilesPerSegment = 6;
    const int totalFiles =
        static_cast<int>(tifxyzSegments.size()) * kProgressFilesPerSegment;

    auto makeProgress = [&](const OpenDataSegment* segment,
                            const char* fileName,
                            std::string status) {
        OpenDataSampleDownloadProgress progress;
        progress.totalSegments = static_cast<int>(tifxyzSegments.size());
        progress.completedSegments = completedSegments.load(std::memory_order_relaxed);
        progress.failedSegments = failedSegments.load(std::memory_order_relaxed);
        progress.totalFiles = totalFiles;
        progress.completedFiles = completedFiles.load(std::memory_order_relaxed);
        progress.activeWorkers = activeWorkers.load(std::memory_order_relaxed);
        progress.totalWorkers = static_cast<int>(workerCount);
        if (segment) {
            progress.segmentId = segmentLabel(*segment);
        }
        if (fileName) {
            progress.fileName = fileName;
        }
        progress.status = std::move(status);
        return progress;
    };
    auto emitProgress = [&](const OpenDataSegment* segment,
                            const char* fileName,
                            std::string status) {
        auto progress = makeProgress(segment, fileName, std::move(status));
        std::lock_guard<std::mutex> lk(progressMutex);
        reportProgress(progressCallback, progress);
    };
    emitProgress(nullptr, nullptr, "starting");

    auto worker = [&]() {
        for (;;) {
            const std::size_t idx = next.fetch_add(1);
            if (idx >= tifxyzSegments.size()) {
                return;
            }
            const auto& segment = *tifxyzSegments[idx];
            const auto* preferred = preferredTifxyzArtifact(segment);
            if (preferred && lowerCopy(preferred->type) == "tifxyz-transformed") {
                completedSegments.fetch_add(1, std::memory_order_relaxed);
                emitProgress(&segment, nullptr, "segment-representation-only");
                continue;
            }
            const auto segmentDir = openDataSegmentCacheDirectory(remoteCacheRoot, sample, segment);
            std::string error;
            activeWorkers.fetch_add(1, std::memory_order_relaxed);
            emitProgress(&segment, nullptr, "segment-start");
            auto fileProgress = [&](const char* fileName, const char* status) {
                if (std::string_view(status) == "done") {
                    completedFiles.fetch_add(1, std::memory_order_relaxed);
                }
                emitProgress(&segment, fileName, status);
            };
            const bool cached = cacheTifxyzSegment(
                sample,
                segment,
                segmentDir,
                &error,
                fileProgress,
                forceRefresh);
            activeWorkers.fetch_sub(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(resultMutex);
            if (cached) {
                ++result.cachedTifxyzSegments;
                completedSegments.fetch_add(1, std::memory_order_relaxed);
                emitProgress(&segment, nullptr, "segment-done");
            } else {
                ++result.failedTifxyzSegments;
                failedSegments.fetch_add(1, std::memory_order_relaxed);
                result.messages.push_back("Failed to cache " + segmentLabel(segment) +
                                          ": " + error);
                emitProgress(&segment, nullptr, "segment-failed");
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (std::size_t i = 0; i < workerCount; ++i) {
        workers.emplace_back(worker);
    }
    for (auto& t : workers) {
        t.join();
    }
    for (const auto& [sourceVolumeId, expectedDirNames] : expectedDirNamesBySource) {
        markOrphanedEntries(
            openDataSegmentCacheRoot(remoteCacheRoot, sample, sourceVolumeId),
            expectedDirNames);
    }
    emitProgress(nullptr, nullptr, "finished");

    // Preserve every manifest representation independently. The legacy cache
    // above remains the derived-native migration target; authored bytes and
    // published target-volume bytes live in representation-specific roots and
    // therefore can never be overwritten by coordinate conversion.
    std::map<std::filesystem::path, OpenDataSegmentRepresentation> preparedRoots;
    int preparedRepresentations = 0;
    for (const auto* segment : tifxyzSegments) {
        for (const auto& representation :
             classifyOpenDataSegmentRepresentations(sample, *segment)) {
            const auto dir = openDataSegmentRepresentationCacheDirectory(
                remoteCacheRoot, sample, *segment, representation);
            std::string error;
            const bool applyDownscale =
                representation.kind == OpenDataSegmentRepresentationKind::DerivedNative;
            const char* representationName =
                representation.kind == OpenDataSegmentRepresentationKind::Authored
                    ? "authored"
                    : representation.kind == OpenDataSegmentRepresentationKind::DerivedNative
                        ? "derived-native"
                        : "published-transformed";
            if (representation.kind ==
                    OpenDataSegmentRepresentationKind::DerivedNative &&
                !requiredFilesPresent(dir)) {
                const auto legacyDir = openDataSegmentCacheDirectory(
                    remoteCacheRoot, sample, *segment);
                try {
                    const auto legacyMeta = nlohmann::json::parse(
                        readTextFile(legacyDir / "meta.json"));
                    if (requiredFilesPresent(legacyDir) &&
                        coordinatesAlreadyScaled(
                            legacyMeta, originalVolumeDownscale(*segment))) {
                        const auto tempDir = makeTempSegmentDir(dir);
                        std::error_code ec;
                        std::filesystem::remove_all(tempDir, ec);
                        std::filesystem::create_directories(tempDir.parent_path());
                        std::filesystem::copy(
                            legacyDir, tempDir,
                            std::filesystem::copy_options::recursive |
                                std::filesystem::copy_options::overwrite_existing);
                        normalizeCachedMetadata(
                            artifactUrl(*representation.artifact), sample, *segment,
                            tempDir / "meta.json", true,
                            representation.representationId, representationName,
                            representation.coordinateSpace,
                            representation.sourceCoordinateLevel,
                            representation.coordinateVolumeId);
                        publishSegmentDirectory(tempDir, dir);
                    }
                } catch (const std::exception& e) {
                    result.messages.push_back(
                        "Could not migrate legacy derived-native cache for " +
                        segmentLabel(*segment) + ": " + e.what());
                }
            }
            if (!cacheTifxyzRepresentation(
                    sample, *segment, *representation.artifact, dir,
                    applyDownscale, representation.representationId,
                    representationName, representation.coordinateVolumeId,
                    representation.coordinateSpace,
                    representation.sourceCoordinateLevel, &error, {}, forceRefresh)) {
                ++result.failedTifxyzSegments;
                result.messages.push_back(
                    "Failed to preserve " + std::string(representationName) +
                    " representation for " + segmentLabel(*segment) + ": " + error);
                continue;
            }
            ++preparedRepresentations;
            preparedRoots.emplace(
                openDataSegmentRepresentationCacheRoot(
                    remoteCacheRoot, sample, representation),
                representation);
        }
    }

    for (const auto& [root, representation] : preparedRoots) {
        std::string kindTag;
        switch (representation.kind) {
            case OpenDataSegmentRepresentationKind::Authored:
                kindTag = "authored";
                break;
            case OpenDataSegmentRepresentationKind::DerivedNative:
                kindTag = "derived-native";
                break;
            case OpenDataSegmentRepresentationKind::PublishedTransformed:
                kindTag = "published-transformed";
                break;
        }
        std::vector<std::string> tags{
            "open-data",
            "immutable",
            "vc-open-data-segment-representation:" + kindTag,
            "vc-open-data-source-coordinate-level:" +
                std::to_string(representation.sourceCoordinateLevel),
            "vc-open-data-coordinate-space:" + representation.coordinateSpace,
        };
        if (representation.kind == OpenDataSegmentRepresentationKind::PublishedTransformed) {
            tags.push_back("vc-open-data-target-volume-id:" +
                           representation.coordinateVolumeId);
        } else {
            tags.push_back("vc-open-data-source-volume-id:" +
                           representation.coordinateVolumeId);
        }
        if (representation.kind ==
            OpenDataSegmentRepresentationKind::DerivedNative) {
            pkg.relocateSegmentsEntry(
                openDataSegmentCacheRoot(
                    remoteCacheRoot, sample,
                    representation.coordinateVolumeId).string(),
                root.string());
        }
        if (pkg.addSegmentsEntry(root.string(), tags)) {
            ++result.attachedSegmentEntries;
        } else if (hasSegmentEntry(pkg, root.string())) {
            pkg.reconcileSegmentsEntryTags(
                root.string(), tags,
                {"vc-open-data-source-coordinate-level:",
                 "vc-open-data-coordinate-space:",
                 "vc-open-data-source-volume-id:",
                 "vc-open-data-target-volume-id:",
                 "vc-open-data-segment-representation:"});
        }
    }

    const int prepared = result.cachedTifxyzSegments + preparedRepresentations;
    if (prepared <= 0) {
        return result;
    }

    std::map<std::string, int> cachedSegmentsBySource;
    for (const auto* segment : tifxyzSegments) {
        const auto* preferred = preferredTifxyzArtifact(*segment);
        if (preferred && lowerCopy(preferred->type) == "tifxyz-transformed")
            continue;
        const auto sourceVolumeId = sourceVolumeIdForSegment(*segment);
        const auto sourceRoot = openDataSegmentCacheRoot(remoteCacheRoot, sample, sourceVolumeId);
        if (requiredFilesPresent(sourceRoot / safePathComponent(segment->id))) {
            ++cachedSegmentsBySource[sourceVolumeId];
        }
    }
    for (const auto& [sourceVolumeId, cachedCount] : cachedSegmentsBySource) {
        if (cachedCount <= 0) {
            continue;
        }
        const bool hasDerivedRepresentation = std::any_of(
            preparedRoots.begin(), preparedRoots.end(),
            [&](const auto& entry) {
                return entry.second.kind ==
                           OpenDataSegmentRepresentationKind::DerivedNative &&
                       entry.second.coordinateVolumeId == sourceVolumeId;
            });
        if (hasDerivedRepresentation)
            continue;
        const auto location = openDataSegmentCacheRoot(remoteCacheRoot, sample, sourceVolumeId).string();
        std::vector<std::string> sourceTags = {
            "open-data", "immutable",
            "vc-open-data-segment-representation:derived-native"};
        if (!sourceVolumeId.empty()) {
            sourceTags.push_back("vc-open-data-source-volume-id:" + sourceVolumeId);
            sourceTags.push_back("vc-open-data-source-coordinate-level:0");
            sourceTags.push_back("vc-open-data-coordinate-space:" + sample.id + "/" +
                                 sourceVolumeId + "@L0");
        }
        try {
            if (pkg.addSegmentsEntry(location, sourceTags)) {
                ++result.attachedSegmentEntries;
            } else if (hasSegmentEntry(pkg, location)) {
                pkg.reconcileSegmentsEntryTags(
                    location, sourceTags,
                    {"vc-open-data-source-coordinate-level:",
                     "vc-open-data-coordinate-space:",
                     "vc-open-data-source-volume-id:",
                     "vc-open-data-target-volume-id:",
                     "vc-open-data-segment-representation:"});
            } else {
                result.messages.push_back("Failed to attach cached tifxyz segment directory.");
            }
        } catch (const std::exception& e) {
            ++result.failedTifxyzSegments;
            result.messages.push_back("Failed to attach cached tifxyz segment directory: " +
                                      std::string(e.what()));
        } catch (...) {
            ++result.failedTifxyzSegments;
            result.messages.push_back("Failed to attach cached tifxyz segment directory: unknown error.");
        }
    }

    std::set<std::string> targetVolumeIds;
    for (const auto& volume : sample.volumes) {
        if (!volume.id.empty()) {
            targetVolumeIds.insert(volume.id);
        }
    }
    for (const auto& [sourceVolumeId, cachedCount] : cachedSegmentsBySource) {
        (void)cachedCount;
        targetVolumeIds.erase(sourceVolumeId);
    }

    struct TransformTask {
        const OpenDataSegment* segment = nullptr;
        std::string targetVolumeId;
    };
    std::vector<TransformTask> transformTasks;
    for (const auto& targetVolumeId : targetVolumeIds) {
        for (const auto* segment : tifxyzSegments) {
            const auto sourceVolumeId = sourceVolumeIdForSegment(*segment);
            if (sourceVolumeId.empty() || sourceVolumeId == targetVolumeId ||
                !findVolumeTransformMatrix(sample, sourceVolumeId, targetVolumeId)) {
                continue;
            }
            transformTasks.push_back({segment, targetVolumeId});
        }
    }

    std::map<std::pair<std::string, std::string>, int> transformedByRoute;
    if (!transformTasks.empty()) {
        std::atomic_size_t nextTransform{0};
        std::atomic_int completedTransforms{0};
        std::atomic_int failedTransforms{0};
        std::atomic_int activeTransformWorkers{0};
        std::mutex transformResultMutex;
        std::mutex transformProgressMutex;
        const auto hardware = std::thread::hardware_concurrency();
        const std::size_t desiredWorkers = hardware == 0 ? 4 : hardware;
        const std::size_t transformWorkerCount = std::min<std::size_t>(
            transformTasks.size(),
            transformTasks.size() <= 1
                ? 1
                : std::max<std::size_t>(2, std::min<std::size_t>(desiredWorkers, 8)));

        auto emitTransformProgress = [&](const TransformTask* task, std::string status) {
            OpenDataSampleDownloadProgress progress;
            progress.totalSegments = static_cast<int>(transformTasks.size());
            progress.completedSegments = completedTransforms.load(std::memory_order_relaxed);
            progress.failedSegments = failedTransforms.load(std::memory_order_relaxed);
            progress.totalFiles = static_cast<int>(transformTasks.size());
            progress.completedFiles = progress.completedSegments + progress.failedSegments;
            progress.activeWorkers = activeTransformWorkers.load(std::memory_order_relaxed);
            progress.totalWorkers = static_cast<int>(transformWorkerCount);
            if (task && task->segment) {
                progress.segmentId = segmentLabel(*task->segment);
                progress.fileName = task->targetVolumeId;
            }
            progress.status = std::move(status);
            std::lock_guard<std::mutex> lk(transformProgressMutex);
            reportProgress(progressCallback, progress);
        };
        emitTransformProgress(nullptr, "transform-starting");

        auto transformWorker = [&]() {
            for (;;) {
                const std::size_t idx = nextTransform.fetch_add(1);
                if (idx >= transformTasks.size()) {
                    return;
                }
                const auto& task = transformTasks[idx];
                activeTransformWorkers.fetch_add(1, std::memory_order_relaxed);
                emitTransformProgress(&task, "transform-segment-start");
                const auto sourceSegmentDir = openDataSegmentCacheDirectory(
                    remoteCacheRoot, sample, *task.segment);
                const auto targetSegmentDir = openDataTransformedSegmentCacheDirectory(
                    remoteCacheRoot, sample, *task.segment, task.targetVolumeId);
                std::string error;
                const bool transformed = cacheTransformedTifxyzSegment(sample,
                                                                       *task.segment,
                                                                       sourceSegmentDir,
                                                                       targetSegmentDir,
                                                                       task.targetVolumeId,
                                                                       &error);
                activeTransformWorkers.fetch_sub(1, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(transformResultMutex);
                if (transformed) {
                    ++result.transformedTifxyzSegments;
                    ++transformedByRoute[
                        {sourceVolumeIdForSegment(*task.segment), task.targetVolumeId}];
                    completedTransforms.fetch_add(1, std::memory_order_relaxed);
                    emitTransformProgress(&task, "transform-segment-done");
                } else {
                    ++result.failedTransformedTifxyzSegments;
                    failedTransforms.fetch_add(1, std::memory_order_relaxed);
                    result.messages.push_back("Failed to transform " + segmentLabel(*task.segment) +
                                              " to " + task.targetVolumeId + ": " + error);
                    emitTransformProgress(&task, "transform-segment-failed");
                }
            }
        };

        std::vector<std::thread> transformWorkers;
        transformWorkers.reserve(transformWorkerCount);
        for (std::size_t i = 0; i < transformWorkerCount; ++i) {
            transformWorkers.emplace_back(transformWorker);
        }
        for (auto& t : transformWorkers) {
            t.join();
        }
        emitTransformProgress(nullptr, "transform-finished");
    }

    for (const auto& [route, transformedForTarget] : transformedByRoute) {
        if (transformedForTarget <= 0) {
            continue;
        }
        const auto& sourceVolumeId = route.first;
        const auto& targetVolumeId = route.second;

        const auto transformedRoot = openDataTransformedSegmentCacheRoot(
            remoteCacheRoot, sample, sourceVolumeId, targetVolumeId);
        const auto transformedLocation = transformedRoot.string();
        try {
            if (pkg.addSegmentsEntry(transformedLocation,
                                     {"open-data",
                                      "immutable",
                                      "open-data-transformed",
                                      "vc-open-data-segment-representation:generated-native-transform",
                                      "vc-open-data-source-volume-id:" + sourceVolumeId,
                                      "vc-open-data-target-volume-id:" + targetVolumeId,
                                      "vc-open-data-source-coordinate-level:0",
                                      "vc-open-data-coordinate-space:" + sample.id + "/" +
                                          targetVolumeId + "@L0"})) {
                ++result.attachedSegmentEntries;
            } else if (hasSegmentEntry(pkg, transformedLocation)) {
                pkg.reconcileSegmentsEntryTags(
                    transformedLocation,
                    {"open-data",
                     "immutable",
                     "open-data-transformed",
                     "vc-open-data-segment-representation:generated-native-transform",
                     "vc-open-data-source-volume-id:" + sourceVolumeId,
                     "vc-open-data-target-volume-id:" + targetVolumeId,
                     "vc-open-data-source-coordinate-level:0",
                     "vc-open-data-coordinate-space:" + sample.id + "/" +
                         targetVolumeId + "@L0"},
                    {"vc-open-data-source-coordinate-level:",
                     "vc-open-data-coordinate-space:",
                     "vc-open-data-source-volume-id:",
                     "vc-open-data-target-volume-id:",
                     "vc-open-data-segment-representation:"});
            } else {
                result.messages.push_back("Failed to attach transformed tifxyz segment directory for " +
                                          targetVolumeId + ".");
            }
        } catch (const std::exception& e) {
            ++result.failedTransformedTifxyzSegments;
            result.messages.push_back("Failed to attach transformed tifxyz segment directory for " +
                                      targetVolumeId + ": " + e.what());
        } catch (...) {
            ++result.failedTransformedTifxyzSegments;
            result.messages.push_back("Failed to attach transformed tifxyz segment directory for " +
                                      targetVolumeId + ": unknown error.");
        }
    }

    return result;
}

} // namespace vc3d::opendata
