#include "OpenDataSampleProject.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/HttpFetch.hpp"

#include <atomic>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>

#include <tiffio.h>

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

bool containsInsensitive(const std::string& haystack, const std::string& needle)
{
    return lowerCopy(haystack).find(lowerCopy(needle)) != std::string::npos;
}

std::string artifactUrl(const OpenDataArtifact& artifact)
{
    return trimTrailingSlashes(artifact.resolvedUrl.empty()
                                  ? artifact.sourcePath
                                  : artifact.resolvedUrl);
}

bool isSupportedRemoteZarr(const OpenDataVolume& volume,
                           const OpenDataArtifact& artifact,
                           const std::string& url)
{
    if (url.empty()) {
        return false;
    }
    const std::string loweredUrl = lowerCopy(url);
    if (loweredUrl.rfind("http://", 0) != 0 &&
        loweredUrl.rfind("https://", 0) != 0 &&
        loweredUrl.rfind("s3://", 0) != 0) {
        return false;
    }
    if (loweredUrl.size() >= 5 &&
        loweredUrl.substr(loweredUrl.size() - 5) == ".zarr") {
        return true;
    }
    return containsInsensitive(artifact.type, "zarr") ||
           containsInsensitive(volume.dataFormat, "zarr");
}

bool jsonStringEqualsInsensitive(const nlohmann::json& obj,
                                 const char* key,
                                 const std::string& expected)
{
    if (!obj.is_object()) {
        return false;
    }
    const auto it = obj.find(key);
    if (it == obj.end() || !it->is_string()) {
        return false;
    }
    return lowerCopy(it->get<std::string>()) == expected;
}

std::vector<std::string> volumeTags(const OpenDataVolume& volume,
                                    const OpenDataArtifact& artifact)
{
    std::vector<std::string> tags;
    auto addUnique = [&tags](std::string tag) {
        if (!tag.empty() &&
            std::find(tags.begin(), tags.end(), tag) == tags.end()) {
            tags.push_back(std::move(tag));
        }
    };

    if (jsonStringEqualsInsensitive(artifact.properties, "representation", "normal3d") ||
        jsonStringEqualsInsensitive(volume.properties, "representation", "normal3d") ||
        containsInsensitive(artifact.type, "normal3d") ||
        containsInsensitive(volume.suffix, "normal3d")) {
        addUnique("normal3d");
    }

    return tags;
}

bool hasVolumeEntry(const VolumePkg& pkg, const std::string& location)
{
    const auto& entries = pkg.volumeEntries();
    return std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.location == location;
    });
}

bool hasSegmentEntry(const VolumePkg& pkg, const std::string& location)
{
    const auto& entries = pkg.segmentEntries();
    return std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.location == location;
    });
}

std::string volumeLabel(const OpenDataVolume& volume)
{
    return volume.id.empty() ? std::string("<unnamed volume>") : volume.id;
}

std::string segmentLabel(const OpenDataSegment& segment)
{
    return segment.id.empty() ? std::string("<unnamed segment>") : segment.id;
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

std::filesystem::path sampleSegmentCacheRoot(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample)
{
    return remoteCacheRoot / "open_data" / "segments" /
           safePathComponent(sample.id.empty() ? "sample" : sample.id);
}

std::filesystem::path sampleProjectCachePath(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample)
{
    return remoteCacheRoot / "open_data" / "projects" /
           (safePathComponent(sample.id.empty() ? "sample" : sample.id) + ".volpkg.json");
}

bool isNonEmptyFile(const std::filesystem::path& path)
{
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec) &&
           std::filesystem::file_size(path, ec) > 0 &&
           !ec;
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

std::string readTextFile(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void ensureCachedMetadata(const std::string& baseUrl,
                          const OpenDataSegment& segment,
                          const std::filesystem::path& target)
{
    std::string body;
    if (isNonEmptyFile(target)) {
        body = readTextFile(target);
    } else {
        const auto url = joinOpenDataUrl(baseUrl, "meta.json");
        auto bytes = vc::httpGetBytes(url);
        if (bytes.empty()) {
            throw std::runtime_error("missing or empty meta.json at " + url);
        }
        body.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }

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
    meta["vc_open_data_tifxyz_url"] = baseUrl;
    meta["vc_open_data_segment_id"] = segment.id;
    meta["vc_open_data_segment_long_id"] = segment.longId;
    meta["vc_open_data_original_volume_id"] = segment.originalVolumeId;

    writeStringAtomic(target, meta.dump(2));
}

void ensureCachedTifxyzBand(const std::string& baseUrl,
                            const char* fileName,
                            const std::filesystem::path& target)
{
    if (!isNonEmptyFile(target)) {
        const auto url = joinOpenDataUrl(baseUrl, fileName);
        auto bytes = vc::httpGetBytes(url);
        if (bytes.empty()) {
            throw std::runtime_error("missing or empty " + std::string(fileName) +
                                     " at " + url);
        }
        writeBytesAtomic(target, bytes);
    }
    normalizeTiffForMmap(target);
}

void cacheOptionalFile(const std::string& baseUrl,
                       const char* fileName,
                       const std::filesystem::path& target)
{
    if (isNonEmptyFile(target)) {
        return;
    }
    try {
        const auto url = joinOpenDataUrl(baseUrl, fileName);
        auto bytes = vc::httpGetBytes(url);
        if (!bytes.empty()) {
            writeBytesAtomic(target, bytes);
        }
    } catch (...) {
        // Optional sidecars should not prevent the tifxyz surface from loading.
    }
}

bool cacheTifxyzSegment(const OpenDataSegment& segment,
                        const std::filesystem::path& segmentDir,
                        std::string* errorOut,
                        const std::function<void(const char*, const char*)>& fileProgress = {})
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

    try {
        auto runFile = [&](const char* fileName, const auto& fn) {
            if (fileProgress) fileProgress(fileName, "start");
            fn();
            if (fileProgress) fileProgress(fileName, "done");
        };
        runFile("meta.json", [&]() {
            ensureCachedMetadata(url, segment, segmentDir / "meta.json");
        });
        runFile("x.tif", [&]() {
            ensureCachedTifxyzBand(url, "x.tif", segmentDir / "x.tif");
        });
        runFile("y.tif", [&]() {
            ensureCachedTifxyzBand(url, "y.tif", segmentDir / "y.tif");
        });
        runFile("z.tif", [&]() {
            ensureCachedTifxyzBand(url, "z.tif", segmentDir / "z.tif");
        });
        runFile("mask.tif", [&]() {
            cacheOptionalFile(url, "mask.tif", segmentDir / "mask.tif");
        });
        runFile("overlapping.json", [&]() {
            cacheOptionalFile(url, "overlapping.json", segmentDir / "overlapping.json");
        });
        return true;
    } catch (const std::exception& e) {
        if (errorOut) *errorOut = e.what();
    } catch (...) {
        if (errorOut) *errorOut = "unknown error.";
    }
    return false;
}

} // namespace

std::shared_ptr<VolumePkg> createOpenDataSampleProject(
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult* resultOut,
    const OpenDataSampleProgressCallback& progressCallback)
{
    auto result = OpenDataSampleProjectResult{};
    std::shared_ptr<VolumePkg> pkg;
    const auto cachedProjectPath = remoteCacheRoot.empty()
        ? std::filesystem::path{}
        : sampleProjectCachePath(remoteCacheRoot, sample);

    if (!cachedProjectPath.empty() && isNonEmptyFile(cachedProjectPath)) {
        try {
            vc::project::LoadOptions opts;
            opts.remoteCacheRoot = remoteCacheRoot;
            pkg = VolumePkg::load(
                cachedProjectPath,
                opts);
            result.messages.push_back("Loaded cached sample project: " +
                                      cachedProjectPath.string());
        } catch (const std::exception& e) {
            result.messages.push_back("Ignored cached sample project " +
                                      cachedProjectPath.string() + ": " + e.what());
        } catch (...) {
            result.messages.push_back("Ignored cached sample project " +
                                      cachedProjectPath.string() + ": unknown error.");
        }
    }

    if (!pkg) {
        pkg = VolumePkg::newEmpty();
    }
    pkg->setName(sample.id.empty() ? "Open Data Sample" : sample.id);
    if (!remoteCacheRoot.empty()) {
        pkg->setRemoteCacheRoot(remoteCacheRoot);
    }

    auto attachResult = attachOpenDataSampleVolumes(*pkg, sample);
    result.supportedVolumes = attachResult.supportedVolumes;
    result.attachedVolumeEntries = attachResult.attachedVolumeEntries;
    result.skippedVolumes = attachResult.skippedVolumes;
    result.failedVolumes = attachResult.failedVolumes;
    result.preferredVolumeId = attachResult.preferredVolumeId;
    result.messages.insert(result.messages.end(),
                           attachResult.messages.begin(),
                           attachResult.messages.end());
    attachOpenDataSampleSegments(*pkg, sample, remoteCacheRoot, result, progressCallback);

    if (!cachedProjectPath.empty()) {
        try {
            pkg->save(cachedProjectPath);
        } catch (const std::exception& e) {
            result.messages.push_back("Failed to save cached sample project " +
                                      cachedProjectPath.string() + ": " + e.what());
        } catch (...) {
            result.messages.push_back("Failed to save cached sample project " +
                                      cachedProjectPath.string() + ": unknown error.");
        }
    }

    if (resultOut) {
        *resultOut = std::move(result);
    }
    return pkg;
}

OpenDataSampleProjectResult attachOpenDataSampleVolumes(
    VolumePkg& pkg,
    const OpenDataSample& sample)
{
    OpenDataSampleProjectResult result;

    for (const auto& volume : sample.volumes) {
        const auto* artifact = preferredVolumeArtifact(volume);
        if (!artifact) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": no volume artifact.");
            continue;
        }

        const std::string url = artifactUrl(*artifact);
        if (!isSupportedRemoteZarr(volume, *artifact, url)) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": unsupported volume artifact.");
            continue;
        }

        ++result.supportedVolumes;
        if (result.preferredVolumeId.empty()) {
            result.preferredVolumeId = volume.id;
        }

        if (hasVolumeEntry(pkg, url)) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": already attached.");
            continue;
        }

        try {
            if (pkg.addVolumeEntry(url, volumeTags(volume, *artifact))) {
                ++result.attachedVolumeEntries;
            } else {
                ++result.failedVolumes;
                result.messages.push_back("Failed to attach " + volumeLabel(volume) + ".");
            }
        } catch (const std::exception& e) {
            ++result.failedVolumes;
            result.messages.push_back("Failed to attach " + volumeLabel(volume) + ": " + e.what());
        } catch (...) {
            ++result.failedVolumes;
            result.messages.push_back("Failed to attach " + volumeLabel(volume) + ": unknown error.");
        }
    }

    return result;
}

void attachOpenDataSampleSegments(
    VolumePkg& pkg,
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult& result,
    const OpenDataSampleProgressCallback& progressCallback)
{
    if (sample.tifxyzSegmentCount() == 0) {
        return;
    }

    result.supportedTifxyzSegments = static_cast<int>(sample.tifxyzSegmentCount());

    if (remoteCacheRoot.empty()) {
        result.skippedTifxyzSegments += result.supportedTifxyzSegments;
        result.messages.push_back("Skipped tifxyz segments: no remote cache directory configured.");
        return;
    }

    const auto segmentsRoot = sampleSegmentCacheRoot(remoteCacheRoot, sample);
    std::vector<const OpenDataSegment*> tifxyzSegments;
    tifxyzSegments.reserve(sample.segments.size());
    for (const auto& segment : sample.segments) {
        if (!segment.hasTifxyz()) {
            continue;
        }
        tifxyzSegments.push_back(&segment);
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
            const auto segmentDir = segmentsRoot / safePathComponent(segment.id);
            std::string error;
            activeWorkers.fetch_add(1, std::memory_order_relaxed);
            emitProgress(&segment, nullptr, "segment-start");
            auto fileProgress = [&](const char* fileName, const char* status) {
                if (std::string_view(status) == "done") {
                    completedFiles.fetch_add(1, std::memory_order_relaxed);
                }
                emitProgress(&segment, fileName, status);
            };
            const bool cached = cacheTifxyzSegment(segment, segmentDir, &error, fileProgress);
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
    emitProgress(nullptr, nullptr, "finished");

    const int prepared = result.cachedTifxyzSegments;
    if (prepared <= 0) {
        return;
    }

    const auto location = segmentsRoot.string();
    try {
        if (pkg.addSegmentsEntry(location, {"open-data"})) {
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
}

} // namespace vc3d::opendata
