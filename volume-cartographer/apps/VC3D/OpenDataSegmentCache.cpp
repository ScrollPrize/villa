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

bool coordinatesAlreadyScaled(const nlohmann::json& meta, double)
{
    return jsonNumberLike(meta, "vc_open_data_coordinates_scaled_to_original_volume").has_value();
}

bool artifactAlreadyInOriginalVolumeScale(const OpenDataArtifact& artifact)
{
    const auto type = lowerCopy(artifact.type);
    return type.find("transformed") != std::string::npos;
}

void applyOriginalVolumeDownscale(const std::filesystem::path& segmentDir,
                                  const OpenDataSegment& segment)
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
    surface->save(segmentDir.string(), segmentStableId(segment), true);
}

bool cachedMetadataNeedsNormalization(const std::filesystem::path& metaPath,
                                      const OpenDataSegment& segment,
                                      bool applyDownscale)
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
        if (stringField("uuid") != segmentStableId(segment) ||
            stringField("type") != "seg" ||
            stringField("format") != "tifxyz" ||
            stringField("vc_open_data_segment_id") != segment.id ||
            stringField("vc_open_data_segment_long_id") != segment.longId) {
            return true;
        }
        return applyDownscale &&
               originalVolumeDownscale(segment) != 1.0 &&
               !coordinatesAlreadyScaled(meta, originalVolumeDownscale(segment));
    } catch (...) {
        return true;
    }
}

void writeCachedMetadata(const std::string& baseUrl,
                         const OpenDataSegment& segment,
                         const std::filesystem::path& target)
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
    meta["uuid"] = segmentStableId(segment);
    if (!meta.contains("name") || !meta["name"].is_string() || meta["name"].get<std::string>().empty()) {
        meta["name"] = segment.suffix.empty() ? segmentLabel(segment) : segment.suffix;
    }
    if (!meta.contains("format") || !meta["format"].is_string() || meta["format"].get<std::string>().empty()) {
        meta["format"] = "tifxyz";
    }
    applyOpenDataMetadata(meta, baseUrl, segment);

    writeStringAtomic(target, meta.dump(2));
}

void normalizeCachedMetadata(const std::string& baseUrl,
                             const OpenDataSegment& segment,
                             const std::filesystem::path& target,
                             bool applyDownscale)
{
    auto meta = nlohmann::json::parse(readTextFile(target));
    if (!meta.is_object()) {
        throw std::runtime_error("meta.json is not an object");
    }
    if (!meta.contains("type") || !meta["type"].is_string() || meta["type"].get<std::string>().empty()) {
        meta["type"] = "seg";
    }
    meta["uuid"] = segmentStableId(segment);
    if (!meta.contains("name") || !meta["name"].is_string() || meta["name"].get<std::string>().empty()) {
        meta["name"] = segment.suffix.empty() ? segmentLabel(segment) : segment.suffix;
    }
    if (!meta.contains("format") || !meta["format"].is_string() || meta["format"].get<std::string>().empty()) {
        meta["format"] = "tifxyz";
    }
    applyOpenDataMetadata(meta, baseUrl, segment);
    writeStringAtomic(target, meta.dump(2));
    if (applyDownscale) {
        applyOriginalVolumeDownscale(target.parent_path(), segment);
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

    const auto url = artifactUrl(*artifact);
    if (url.empty()) {
        if (errorOut) *errorOut = "tifxyz artifact has no URL.";
        return false;
    }

    const bool applyDownscale = !artifactAlreadyInOriginalVolumeScale(*artifact);
    if (!forceRefresh && requiredFilesPresent(segmentDir)) {
        const auto origin = readCatalogOrigin(segmentDir);
        if (origin && originMatches(*origin, sample, segment, *artifact) &&
            originStateAllowsFastOpen(*origin)) {
            if (cachedMetadataNeedsNormalization(segmentDir / "meta.json",
                                                 segment,
                                                 applyDownscale)) {
                normalizeCachedMetadata(url,
                                        segment,
                                        segmentDir / "meta.json",
                                        applyDownscale);
            }
            return true;
        }
        if (!origin || originMatches(*origin, sample, segment, *artifact)) {
            normalizeCachedMetadata(url,
                                    segment,
                                    segmentDir / "meta.json",
                                    applyDownscale);
            if (origin) {
                writeCatalogOriginState(segmentDir, OpenDataSegmentCacheState::Current);
            } else {
                writeStringAtomic(
                    segmentDir / "catalog-origin.json",
                    catalogOriginJson(sample,
                                      segment,
                                      *artifact,
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
            writeCachedMetadata(url, segment, tempDir / "meta.json");
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
        normalizeCachedMetadata(url,
                                segment,
                                tempDir / "meta.json",
                                applyDownscale);
        writeStringAtomic(tempDir / "catalog-origin.json",
                          catalogOriginJson(sample, segment, *artifact, downloadedFiles).dump(2));
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
    const OpenDataSample& sample)
{
    return remoteCacheRoot / "open_data" / "segments" /
           safePathComponent(sample.id.empty() ? "sample" : sample.id);
}

std::filesystem::path openDataSegmentCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment)
{
    return openDataSegmentCacheRoot(remoteCacheRoot, sample) /
           safePathComponent(segment.id);
}

std::filesystem::path openDataEditableSegmentRoot(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId)
{
    return remoteCacheRoot / "open_data" / "editable" /
           safePathComponent(sampleId.empty() ? "sample" : sampleId);
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
    const std::filesystem::path& remoteCacheRoot)
{
    std::string sampleId = "sample";
    if (const auto origin = readCatalogOrigin(catalogSegmentDir)) {
        const auto it = origin->find("sample_id");
        if (it != origin->end() && it->is_string() && !it->get<std::string>().empty()) {
            sampleId = it->get<std::string>();
        }
    }
    auto baseName = catalogSegmentDir.filename().string();
    if (baseName.empty()) {
        baseName = "segment";
    }
    return openDataEditableSegmentRoot(remoteCacheRoot, sampleId) /
           (safePathComponent(baseName) + "-edit");
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

    const auto segmentsRoot = openDataSegmentCacheRoot(remoteCacheRoot, sample);
    std::vector<const OpenDataSegment*> tifxyzSegments;
    tifxyzSegments.reserve(sample.segments.size());
    std::set<std::string> expectedDirNames;
    for (const auto& segment : sample.segments) {
        if (!segment.hasTifxyz()) {
            continue;
        }
        tifxyzSegments.push_back(&segment);
        expectedDirNames.insert(safePathComponent(segment.id));
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
            : std::max<std::size_t>(2, std::min<std::size_t>(desiredWorkers, 4)));
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
    markOrphanedEntries(segmentsRoot, expectedDirNames);
    emitProgress(nullptr, nullptr, "finished");

    const int prepared = result.cachedTifxyzSegments;
    if (prepared <= 0) {
        return result;
    }

    const auto location = segmentsRoot.string();
    try {
        if (pkg.addSegmentsEntry(location, {"open-data", "immutable"})) {
            ++result.attachedSegmentEntries;
        } else if (hasSegmentEntry(pkg, location)) {
            pkg.refreshSegmentations();
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

    return result;
}

} // namespace vc3d::opendata
