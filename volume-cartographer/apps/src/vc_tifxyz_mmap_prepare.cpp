#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <tiffio.h>

namespace fs = std::filesystem;

namespace {

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

TiffInfo readInfo(const fs::path& path)
{
    TIFF* tif = TIFFOpen(path.string().c_str(), "r");
    if (!tif) {
        throw std::runtime_error("failed to open " + path.string());
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

bool isMmapCompatible(const TiffInfo& info)
{
    return !info.tiled &&
           !info.byteSwapped &&
           info.samplesPerPixel == 1 &&
           info.bits == 32 &&
           info.sampleFormat == SAMPLEFORMAT_IEEEFP &&
           info.compression == COMPRESSION_NONE &&
           info.strips > 0 &&
           info.rowsPerStrip > 0;
}

float sampleToFloat(const uint8_t* p, uint16_t sampleFormat, uint16_t bits)
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

std::vector<float> readBandAsFloat(const fs::path& path, TiffInfo* outInfo)
{
    TIFF* tif = TIFFOpen(path.string().c_str(), "r");
    if (!tif) {
        throw std::runtime_error("failed to open " + path.string());
    }

    TiffInfo info = readInfo(path);
    if (info.width == 0 || info.height == 0 || info.samplesPerPixel != 1) {
        TIFFClose(tif);
        throw std::runtime_error("unsupported TIFF geometry: " + path.string());
    }
    const int bytesPer = (info.bits + 7) / 8;
    if (!(info.bits == 8 || info.bits == 16 || info.bits == 32 || info.bits == 64)) {
        TIFFClose(tif);
        throw std::runtime_error("unsupported TIFF bits per sample: " + path.string());
    }

    std::vector<float> pixels(static_cast<std::size_t>(info.width) * info.height);
    if (TIFFIsTiled(tif)) {
        uint32_t tileW = 0;
        uint32_t tileH = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileW);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
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
                            sampleToFloat(row + static_cast<std::size_t>(x) * bytesPer,
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
                    sampleToFloat(scanBuf.data() + static_cast<std::size_t>(x) * bytesPer,
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

void writeMmapBand(const fs::path& path, const TiffInfo& info, const std::vector<float>& pixels)
{
    const fs::path tmp = path.string() + ".mmap_tmp";
    TIFF* out = TIFFOpen(tmp.string().c_str(), "w");
    if (!out) {
        throw std::runtime_error("failed to create " + tmp.string());
    }
    TIFFSetField(out, TIFFTAG_IMAGEWIDTH, info.width);
    TIFFSetField(out, TIFFTAG_IMAGELENGTH, info.height);
    TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, info.height);

    const tsize_t bytes = static_cast<tsize_t>(pixels.size() * sizeof(float));
    if (TIFFWriteEncodedStrip(out, 0, const_cast<float*>(pixels.data()), bytes) < 0) {
        TIFFClose(out);
        fs::remove(tmp);
        throw std::runtime_error("failed writing " + tmp.string());
    }
    TIFFClose(out);
    fs::rename(tmp, path);
}

bool isTifxyzDir(const fs::path& dir)
{
    return fs::is_directory(dir) &&
           fs::exists(dir / "x.tif") &&
           fs::exists(dir / "y.tif") &&
           fs::exists(dir / "z.tif");
}

bool prepareBand(const fs::path& path, bool dryRun)
{
    const TiffInfo info = readInfo(path);
    if (isMmapCompatible(info)) {
        return false;
    }
    if (!dryRun) {
        TiffInfo readBack;
        auto pixels = readBandAsFloat(path, &readBack);
        writeMmapBand(path, readBack, pixels);
    }
    return true;
}

bool prepareSurface(const fs::path& dir, bool dryRun)
{
    bool changed = false;
    changed = prepareBand(dir / "x.tif", dryRun) || changed;
    changed = prepareBand(dir / "y.tif", dryRun) || changed;
    changed = prepareBand(dir / "z.tif", dryRun) || changed;
    return changed;
}

void usage(const char* argv0)
{
    std::cerr << "Usage: " << argv0 << " <folder> [--recursive] [--dry-run] [--jobs N]\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    fs::path root;
    bool recursive = false;
    bool dryRun = false;
    unsigned jobs = std::max(1u, std::thread::hardware_concurrency());

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--recursive") {
            recursive = true;
        } else if (arg == "--dry-run") {
            dryRun = true;
        } else if (arg == "--jobs" && i + 1 < argc) {
            jobs = std::max(1, std::stoi(argv[++i]));
        } else if (root.empty()) {
            root = arg;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (root.empty() || !fs::is_directory(root)) {
        usage(argv[0]);
        return 2;
    }

    std::vector<fs::path> surfaces;
    if (isTifxyzDir(root)) {
        surfaces.push_back(root);
    } else if (recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(root)) {
            if (entry.is_directory() && isTifxyzDir(entry.path())) {
                surfaces.push_back(entry.path());
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(root)) {
            if (entry.is_directory() && isTifxyzDir(entry.path())) {
                surfaces.push_back(entry.path());
            }
        }
    }
    std::sort(surfaces.begin(), surfaces.end());

    std::atomic_size_t next{0};
    std::atomic_size_t checked{0};
    std::atomic_size_t changed{0};
    std::atomic_size_t failed{0};
    std::vector<std::future<void>> workers;
    jobs = std::min<unsigned>(jobs, std::max<std::size_t>(1, surfaces.size()));
    for (unsigned worker = 0; worker < jobs; ++worker) {
        workers.emplace_back(std::async(std::launch::async, [&]() {
            while (true) {
                const size_t i = next.fetch_add(1);
                if (i >= surfaces.size()) {
                    break;
                }
                try {
                    if (prepareSurface(surfaces[i], dryRun)) {
                        ++changed;
                        std::cout << (dryRun ? "would rewrite " : "rewrote ")
                                  << surfaces[i] << std::endl;
                    }
                } catch (const std::exception& e) {
                    ++failed;
                    std::cerr << "failed " << surfaces[i] << ": " << e.what() << std::endl;
                }
                const size_t done = checked.fetch_add(1) + 1;
                if (done == surfaces.size() || done % 1000 == 0) {
                    std::cout << "progress checked=" << done
                              << "/" << surfaces.size()
                              << " changed=" << changed.load()
                              << " failed=" << failed.load()
                              << std::endl;
                }
            }
        }));
    }
    for (auto& worker : workers) {
        worker.get();
    }

    std::cout << "checked=" << surfaces.size()
              << " changed=" << changed.load()
              << " failed=" << failed.load() << std::endl;
    return failed == 0 ? 0 : 1;
}
