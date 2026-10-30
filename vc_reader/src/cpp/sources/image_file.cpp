// PNG, JPEG and TIFF files as series of independently decoded pages.
#include "vc_reader/source_reader.hpp"

#include <jpeglib.h>
#include <png.h>
#include <tiffio.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <set>
#include <sys/stat.h>

namespace vcr::sources {
using nlohmann::json;
namespace {
[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }

std::string lowerSuffix(const std::string& location)
{
    auto path = location.substr(0, location.find_first_of("?#"));
    const auto slash = path.rfind('/');
    if (slash != std::string::npos) path = path.substr(slash + 1);
    const auto dot = path.rfind('.');
    if (dot == std::string::npos) return {};
    auto suffix = path.substr(dot);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) { return std::tolower(c); });
    return suffix;
}

std::int64_t product(const std::vector<std::int64_t>& values, std::size_t begin = 0, std::size_t end = std::string::npos)
{
    std::int64_t out = 1;
    for (std::size_t i = begin; i < std::min(end, values.size()); ++i) out *= values[i];
    return out;
}

// A FILE on its own descriptor, positioned at the start.
struct DupFile {
    std::FILE* file = nullptr;
    explicit DupFile(int fd)
    {
        const int copy = ::dup(fd);
        if (copy < 0 || !(file = ::fdopen(copy, "rb"))) throw ImageFormatError("cannot read image file");
        std::rewind(file);
    }
    ~DupFile() { std::fclose(file); }
};

// ---- PNG ----
struct Raster {
    std::vector<std::int64_t> shape;
    std::vector<std::string> axes;
    std::string dtype;
    json palette;
};

struct PngHandles {
    png_structp png = nullptr;
    png_infop info = nullptr;
    PngHandles()
    {
        png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, [](png_structp, png_const_charp) {}, [](png_structp, png_const_charp) {});
        if (!png || !(info = png_create_info_struct(png))) throw ImageFormatError("cannot start a PNG decoder");
    }
    ~PngHandles() { png_destroy_read_struct(&png, &info, nullptr); }
};

// Whether the stream holds an animation: chunks before the image data.
bool pngAnimated(int fd)
{
    std::uint8_t header[8];
    off_t at = 8;
    for (;;) {
        if (::pread(fd, header, 8, at) != 8) return false;
        if (!std::memcmp(header + 4, "IDAT", 4)) return false;
        if (!std::memcmp(header + 4, "acTL", 4)) return true;
        const std::uint32_t length = (std::uint32_t(header[0]) << 24) | (header[1] << 16) | (header[2] << 8) | header[3];
        at += off_t(length) + 12;
    }
}

Raster pngHeader(int fd)
{
    DupFile file(fd);
    PngHandles h;
    if (setjmp(png_jmpbuf(h.png))) throw ImageFormatError("cannot read this PNG");
    png_init_io(h.png, file.file);
    png_read_info(h.png, h.info);
    if (pngAnimated(fd)) bad("animated/multi-frame images require an explicit supported stack");
    const auto width = png_get_image_width(h.png, h.info), height = png_get_image_height(h.png, h.info);
    const int color = png_get_color_type(h.png, h.info), depth = png_get_bit_depth(h.png, h.info);
    int bands = 1;
    if (color == PNG_COLOR_TYPE_RGB) bands = 3;
    else if (color == PNG_COLOR_TYPE_RGB_ALPHA) bands = 4;
    else if (color == PNG_COLOR_TYPE_GRAY_ALPHA) bands = 2;
    Raster raster;
    raster.shape = {std::int64_t(height), std::int64_t(width)};
    raster.axes = {"y", "x"};
    if (bands > 1) {
        raster.shape.push_back(bands);
        raster.axes.emplace_back("c");
    }
    raster.dtype = color == PNG_COLOR_TYPE_GRAY && depth == 1 ? "|b1" : depth == 16 ? "<u2" : "|u1";
    if (color == PNG_COLOR_TYPE_PALETTE) {
        png_colorp entries = nullptr;
        int count = 0;
        json values = json::array();
        if (png_get_PLTE(h.png, h.info, &entries, &count)) {
            for (int i = 0; i < count; ++i) {
                values.push_back(entries[i].red);
                values.push_back(entries[i].green);
                values.push_back(entries[i].blue);
            }
        }
        json transparency(nullptr);
        png_bytep alpha = nullptr;
        int alphas = 0;
        if (png_get_tRNS(h.png, h.info, &alpha, &alphas, nullptr) && alpha) {
            // One fully transparent entry among opaque ones is that entry's index; otherwise one alpha per entry.
            int zero = -1, zeros = 0, others = 0;
            for (int i = 0; i < alphas; ++i) {
                if (alpha[i] == 0) {
                    zero = i;
                    ++zeros;
                } else if (alpha[i] != 255) {
                    ++others;
                }
            }
            if (zeros == 1 && !others) {
                transparency = zero;
            } else if (zeros || others) {
                transparency = json::array();
                for (int i = 0; i < alphas; ++i) transparency.push_back(alpha[i]);
            }
        }
        raster.palette = {{"mode", "RGB"}, {"values", std::move(values)}, {"transparency", std::move(transparency)}};
    }
    return raster;
}

Block pngPixels(int fd, const Raster& raster)
{
    DupFile file(fd);
    PngHandles h;
    Block block = Block::allocate(raster.shape, raster.dtype);
    std::vector<png_bytep> rows(std::size_t(raster.shape[0]));
    if (setjmp(png_jmpbuf(h.png))) throw ImageFormatError("cannot decode this PNG");
    png_init_io(h.png, file.file);
    png_read_info(h.png, h.info);
    const int color = png_get_color_type(h.png, h.info), depth = png_get_bit_depth(h.png, h.info);
    if (depth < 8) {
        if (raster.dtype == "|b1" || color == PNG_COLOR_TYPE_PALETTE) png_set_packing(h.png);
        else png_set_expand_gray_1_2_4_to_8(h.png);
    }
    if (depth == 16) png_set_swap(h.png);
    png_set_interlace_handling(h.png);
    png_read_update_info(h.png, h.info);
    const std::size_t stride = png_get_rowbytes(h.png, h.info);
    if (stride != std::size_t(block.nbytes / raster.shape[0])) throw ImageFormatError("decoded shape/dtype differs from image header");
    for (std::size_t y = 0; y < rows.size(); ++y) rows[y] = block.data.get() + y * stride;
    png_read_image(h.png, rows.data());
    png_read_end(h.png, nullptr);
    return block;
}

// ---- JPEG ----
struct JpegError : jpeg_error_mgr {
    std::jmp_buf jump;
};
struct JpegHandles {
    jpeg_decompress_struct cinfo{};
    JpegError error{};
    JpegHandles()
    {
        cinfo.err = jpeg_std_error(&error);
        error.error_exit = [](j_common_ptr c) { std::longjmp(static_cast<JpegError*>(c->err)->jump, 1); };
        error.output_message = [](j_common_ptr) {};
        jpeg_create_decompress(&cinfo);
    }
    ~JpegHandles() { jpeg_destroy_decompress(&cinfo); }
};

Raster jpegHeader(int fd)
{
    DupFile file(fd);
    JpegHandles h;
    if (setjmp(h.error.jump)) throw ImageFormatError("cannot read this JPEG");
    jpeg_stdio_src(&h.cinfo, file.file);
    jpeg_read_header(&h.cinfo, TRUE);
    int bands = 3;
    if (h.cinfo.jpeg_color_space == JCS_GRAYSCALE) bands = 1;
    else if (h.cinfo.jpeg_color_space == JCS_CMYK || h.cinfo.jpeg_color_space == JCS_YCCK) bands = 4;
    Raster raster;
    raster.shape = {std::int64_t(h.cinfo.image_height), std::int64_t(h.cinfo.image_width)};
    raster.axes = {"y", "x"};
    if (bands > 1) {
        raster.shape.push_back(bands);
        raster.axes.emplace_back("c");
    }
    raster.dtype = "|u1";
    return raster;
}

Block jpegPixels(int fd, const Raster& raster)
{
    DupFile file(fd);
    JpegHandles h;
    Block block = Block::allocate(raster.shape, raster.dtype);
    if (setjmp(h.error.jump)) throw ImageFormatError("cannot decode this JPEG");
    jpeg_stdio_src(&h.cinfo, file.file);
    jpeg_read_header(&h.cinfo, TRUE);
    const bool cmyk = h.cinfo.jpeg_color_space == JCS_CMYK || h.cinfo.jpeg_color_space == JCS_YCCK;
    const bool inverted = cmyk && h.cinfo.saw_Adobe_marker;
    if (cmyk) h.cinfo.out_color_space = JCS_CMYK;
    jpeg_start_decompress(&h.cinfo);
    const std::size_t stride = std::size_t(h.cinfo.output_width) * h.cinfo.output_components;
    if (stride * h.cinfo.output_height != block.nbytes) throw ImageFormatError("decoded shape/dtype differs from image header");
    while (h.cinfo.output_scanline < h.cinfo.output_height) {
        JSAMPROW row = block.data.get() + std::size_t(h.cinfo.output_scanline) * stride;
        jpeg_read_scanlines(&h.cinfo, &row, 1);
    }
    jpeg_finish_decompress(&h.cinfo);
    if (inverted)
        for (std::size_t i = 0; i < block.nbytes; ++i) block.data[i] = std::uint8_t(255 - block.data[i]);
    return block;
}

// ---- TIFF ----
using TiffHandle = std::unique_ptr<TIFF, void (*)(TIFF*)>;

TiffHandle openTiffHandle(int fd, const std::string& name)
{
    const int copy = ::dup(fd);
    if (copy < 0) throw ImageFormatError(name + ": cannot read TIFF");
    ::lseek(copy, 0, SEEK_SET);
#if TIFFLIB_VERSION >= 20221213
    TIFFOpenOptions* options = TIFFOpenOptionsAlloc();
    TIFFOpenOptionsSetErrorHandlerExtR(options, [](TIFF*, void*, const char*, const char*, va_list) { return 1; }, nullptr);
    TIFFOpenOptionsSetWarningHandlerExtR(options, [](TIFF*, void*, const char*, const char*, va_list) { return 1; }, nullptr);
    TIFF* tif = TIFFFdOpenExt(copy, name.c_str(), "r", options);
    TIFFOpenOptionsFree(options);
#else
    TIFF* tif = TIFFFdOpen(copy, name.c_str(), "r");
#endif
    if (!tif) {
        ::close(copy);
        throw ImageFormatError(name + ": cannot read this TIFF");
    }
    return TiffHandle(tif, TIFFClose);
}

struct TiffPage {
    std::uint64_t offset = 0;
    std::int64_t width = 0, height = 0, depth = 1, samples = 1;
    int bits = 8, format = SAMPLEFORMAT_UINT, planar = PLANARCONFIG_CONTIG, compression = COMPRESSION_NONE, photometric = 1;
    std::int64_t rowsPerStrip = 0, tileWidth = 0, tileHeight = 0, tileDepth = 1;
    bool tiled = false, reduced = false;
    std::string description;
    std::vector<std::uint64_t> subifds;
    std::optional<int> resolutionUnit;

    [[nodiscard]] std::vector<std::int64_t> shape() const
    {
        std::vector<std::int64_t> out;
        if (planar == PLANARCONFIG_SEPARATE && samples > 1) out.push_back(samples);
        if (depth > 1) out.push_back(depth);
        out.push_back(height);
        out.push_back(width);
        if (planar != PLANARCONFIG_SEPARATE && samples > 1) out.push_back(samples);
        return out;
    }
    [[nodiscard]] std::string axes() const
    {
        std::string out;
        if (planar == PLANARCONFIG_SEPARATE && samples > 1) out += 'S';
        if (depth > 1) out += 'Z';
        out += "YX";
        if (planar != PLANARCONFIG_SEPARATE && samples > 1) out += 'S';
        return out;
    }
    [[nodiscard]] std::string dtype() const
    {
        if (bits == 1 && samples == 1) return "|b1";
        if (bits != 8 && bits != 16 && bits != 32 && bits != 64) bad("TIFF samples of " + std::to_string(bits) + " bits are not supported");
        const char kind = format == SAMPLEFORMAT_INT ? 'i' : format == SAMPLEFORMAT_IEEEFP ? 'f' : 'u';
        if (kind == 'f' && bits < 32) bad("TIFF floating samples of " + std::to_string(bits) + " bits are not supported");
        std::string out = bits == 8 ? "|" : "<";
        out += kind;
        out += char('0' + bits / 8);
        return out;
    }
    [[nodiscard]] auto key() const
    {
        return std::tuple(width, height, depth, samples, bits, format, planar, compression, photometric, rowsPerStrip, tileWidth, tileHeight, tileDepth);
    }
};

TiffPage readPage(TIFF* tif)
{
    TiffPage page;
    page.offset = TIFFCurrentDirOffset(tif);
    std::uint32_t u32 = 0;
    std::uint16_t u16 = 0;
    if (TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &u32)) page.width = u32;
    if (TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &u32)) page.height = u32;
    if (TIFFGetField(tif, TIFFTAG_IMAGEDEPTH, &u32) && u32) page.depth = u32;
    if (TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &u16)) page.samples = u16;
    if (TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &u16)) page.bits = u16;
    if (TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &u16)) page.format = u16;
    if (TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &u16)) page.planar = u16;
    if (TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &u16)) page.compression = u16;
    if (TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &u16)) page.photometric = u16;
    page.tiled = TIFFIsTiled(tif);
    if (page.tiled) {
        if (TIFFGetField(tif, TIFFTAG_TILEWIDTH, &u32)) page.tileWidth = u32;
        if (TIFFGetField(tif, TIFFTAG_TILELENGTH, &u32)) page.tileHeight = u32;
        if (TIFFGetField(tif, TIFFTAG_TILEDEPTH, &u32) && u32) page.tileDepth = u32;
    } else {
        page.rowsPerStrip = page.height;
        if (TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &u32)) page.rowsPerStrip = std::min<std::int64_t>(u32, page.height);
    }
    if (TIFFGetField(tif, TIFFTAG_SUBFILETYPE, &u32)) page.reduced = (u32 & FILETYPE_REDUCEDIMAGE) != 0;
    const char* text = nullptr;
    if (TIFFGetField(tif, TIFFTAG_IMAGEDESCRIPTION, &text) && text) page.description = text;
    std::uint16_t count = 0;
    toff_t* offsets = nullptr;
    if (TIFFGetField(tif, TIFFTAG_SUBIFD, &count, &offsets) && count) page.subifds.assign(offsets, offsets + count);
    if (TIFFGetField(tif, TIFFTAG_RESOLUTIONUNIT, &u16)) page.resolutionUnit = u16;
    return page;
}

// The rational stored at a tag, exactly: libtiff hands out single-precision values.
std::optional<std::array<std::uint32_t, 2>> rationalTag(int fd, TIFF* tif, std::uint64_t directory, std::uint16_t wanted)
{
    const bool swapped = TIFFIsByteSwapped(tif), big = TIFFIsBigTIFF(tif);
    const auto read = [&](std::uint64_t at, void* out, std::size_t n) {
        if (::pread(fd, out, n, off_t(at)) != ssize_t(n)) throw ImageFormatError("truncated TIFF directory");
    };
    const auto swap = [&](auto value) {
        if (swapped) {
            auto* bytes = reinterpret_cast<std::uint8_t*>(&value);
            std::reverse(bytes, bytes + sizeof value);
        }
        return value;
    };
    std::uint64_t entries = 0;
    std::uint64_t at = directory;
    if (big) {
        read(at, &entries, 8);
        entries = swap(entries);
        at += 8;
    } else {
        std::uint16_t count = 0;
        read(at, &count, 2);
        entries = swap(count);
        at += 2;
    }
    for (std::uint64_t i = 0; i < entries; ++i, at += big ? 20 : 12) {
        std::uint16_t tag = 0, type = 0;
        read(at, &tag, 2);
        if (swap(tag) != wanted) continue;
        read(at + 2, &type, 2);
        if (swap(type) != TIFF_RATIONAL) return std::nullopt;
        std::uint64_t offset = 0;
        if (big) {
            read(at + 12, &offset, 8);
            offset = swap(offset);
        } else {
            std::uint32_t small = 0;
            read(at + 8, &small, 4);
            offset = swap(small);
        }
        std::array<std::uint32_t, 2> value{};
        read(offset, value.data(), 8);
        return std::array<std::uint32_t, 2>{swap(value[0]), swap(value[1])};
    }
    return std::nullopt;
}

struct TiffSeries {
    std::string name;
    std::string axes;  // TIFF letters; Q for an unknown dimension
    std::vector<std::int64_t> shape;
    std::vector<std::uint64_t> pages;   // IFD offsets, in order, at level 0
    std::vector<std::vector<std::uint64_t>> reducedLevels;  // levels from reduced pages instead of SubIFDs
    std::vector<std::string> channelNames;
    std::map<std::string, std::string> ome;  // OME Pixels attributes
};

// Attributes of XML tags, ignoring prefixes; enough for the flat attributes of OME metadata.
struct XmlTag {
    std::string name;
    std::map<std::string, std::string> attributes;
    bool closing = false;
};
std::vector<XmlTag> xmlTags(const std::string& text)
{
    std::vector<XmlTag> tags;
    std::size_t at = 0;
    while ((at = text.find('<', at)) != std::string::npos) {
        auto end = at + 1;
        char quote = 0;
        while (end < text.size() && (quote || text[end] != '>')) {
            if (quote) {
                if (text[end] == quote) quote = 0;
            } else if (text[end] == '"' || text[end] == '\'') {
                quote = text[end];
            }
            ++end;
        }
        if (end >= text.size()) break;
        std::string body = text.substr(at + 1, end - at - 1);
        at = end + 1;
        if (body.empty() || body[0] == '?' || body[0] == '!') continue;
        XmlTag tag;
        if (body[0] == '/') {
            tag.closing = true;
            body.erase(0, 1);
        }
        std::size_t i = 0;
        while (i < body.size() && !std::isspace(static_cast<unsigned char>(body[i])) && body[i] != '/') ++i;
        tag.name = body.substr(0, i);
        if (const auto colon = tag.name.rfind(':'); colon != std::string::npos) tag.name.erase(0, colon + 1);
        while (i < body.size()) {
            while (i < body.size() && (std::isspace(static_cast<unsigned char>(body[i])) || body[i] == '/')) ++i;
            const auto equals = body.find('=', i);
            if (equals == std::string::npos) break;
            auto key = body.substr(i, equals - i);
            while (!key.empty() && std::isspace(static_cast<unsigned char>(key.back()))) key.pop_back();
            auto q = equals + 1;
            while (q < body.size() && std::isspace(static_cast<unsigned char>(body[q]))) ++q;
            if (q >= body.size() || (body[q] != '"' && body[q] != '\'')) break;
            const auto close = body.find(body[q], q + 1);
            if (close == std::string::npos) break;
            tag.attributes[key] = body.substr(q + 1, close - q - 1);
            i = close + 1;
        }
        tags.push_back(std::move(tag));
    }
    return tags;
}

std::vector<TiffSeries> omeSeries(const std::string& xml, const std::vector<TiffPage>& pages)
{
    std::vector<TiffSeries> out;
    std::int64_t next = 0;
    TiffSeries* current = nullptr;
    std::map<std::string, std::string> pixels;
    std::vector<std::string> channels;
    std::int64_t samples = 1;
    std::optional<std::int64_t> firstIfd;
    std::string name;
    const auto finish = [&] {
        if (pixels.empty()) return;
        const auto size = [&](const char* key) { return pixels.count(key) ? std::stoll(pixels[key]) : std::int64_t(1); };
        const auto order = pixels.count("DimensionOrder") ? pixels["DimensionOrder"] : std::string("XYZCT");
        TiffSeries series;
        series.name = name;
        series.channelNames = channels;
        series.ome = pixels;
        std::vector<std::int64_t> shape;
        std::string axes;
        std::int64_t count = 1;
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            if (*it == 'X' || *it == 'Y') continue;
            std::int64_t n = size((std::string("Size") + *it).c_str());
            if (*it == 'C') n /= std::max<std::int64_t>(samples, 1);
            if (n > 1) {
                shape.push_back(n);
                axes += *it;
                count *= n;
            }
        }
        const auto first = firstIfd.value_or(next);
        if (first < 0 || first + count > std::int64_t(pages.size())) bad("TIFF series/page layout cannot be read as independent pages; use a chunked source");
        for (std::int64_t i = 0; i < count; ++i) series.pages.push_back(pages[std::size_t(first + i)].offset);
        const auto& page = pages[std::size_t(first)];
        const auto own = page.shape();
        shape.insert(shape.end(), own.begin(), own.end());
        axes += page.axes();
        series.shape = std::move(shape);
        series.axes = std::move(axes);
        next = first + count;
        out.push_back(std::move(series));
    };
    for (const auto& tag : xmlTags(xml)) {
        if (tag.closing) {
            if (tag.name == "Image") {
                finish();
                pixels.clear();
                channels.clear();
                samples = 1;
                firstIfd.reset();
            }
            continue;
        }
        if (tag.name == "Image") name = tag.attributes.count("Name") ? tag.attributes.at("Name") : std::string();
        else if (tag.name == "Pixels") pixels = tag.attributes;
        else if (tag.name == "Channel") {
            channels.push_back(tag.attributes.count("Name") ? tag.attributes.at("Name") : std::string());
            if (tag.attributes.count("SamplesPerPixel")) samples = std::max(samples, std::int64_t(std::stoll(tag.attributes.at("SamplesPerPixel"))));
        } else if (tag.name == "TiffData" && !firstIfd) {
            firstIfd = tag.attributes.count("IFD") ? std::stoll(tag.attributes.at("IFD")) : std::int64_t(0);
        }
    }
    (void)current;
    return out;
}

std::vector<TiffSeries> shapedSeries(const std::vector<TiffPage>& pages)
{
    std::vector<TiffSeries> out;
    std::size_t index = 0;
    while (index < pages.size()) {
        const auto& first = pages[index];
        const auto document = json::parse(first.description, nullptr, false);
        if (!document.is_object() || !document.contains("shape") || !document["shape"].is_array()) return {};
        TiffSeries series;
        for (const auto& n : document["shape"]) {
            if (!n.is_number_integer()) return {};
            series.shape.push_back(n.get<std::int64_t>());
        }
        const auto own = first.shape();
        const std::int64_t perPage = product(own);
        const std::int64_t count = perPage ? product(series.shape) / perPage : 0;
        series.axes = document.value("axes", std::string());
        if (series.axes.size() != series.shape.size()) {
            series.axes.clear();
            if (series.shape.size() >= own.size()) series.axes = std::string(series.shape.size() - own.size(), 'Q') + first.axes();
            else series.axes = std::string(series.shape.size(), 'Q');
        }
        series.name = document.value("name", std::string());
        const auto taken = std::size_t(std::max<std::int64_t>(count, 1));
        for (std::size_t i = index; i < std::min(pages.size(), index + taken); ++i) series.pages.push_back(pages[i].offset);
        index += taken;
        out.push_back(std::move(series));
    }
    return out;
}

std::vector<TiffSeries> imagejSeries(const std::vector<TiffPage>& pages)
{
    const auto& text = pages.front().description;
    const auto field = [&](const char* key) -> std::int64_t {
        const auto at = text.find(std::string("\n") + key + "=");
        if (at == std::string::npos) return 1;
        return std::atoll(text.c_str() + at + std::strlen(key) + 2);
    };
    TiffSeries series;
    const std::int64_t images = text.find("images=") == std::string::npos ? std::int64_t(pages.size()) : field("images");
    for (const auto& [letter, key] : {std::pair{'T', "frames"}, std::pair{'Z', "slices"}, std::pair{'C', "channels"}}) {
        const auto n = field(key);
        if (n > 1) {
            series.shape.push_back(n);
            series.axes += letter;
        }
    }
    if (series.shape.empty() && images > 1) {
        series.shape.push_back(images);
        series.axes = "Q";
    }
    const auto own = pages.front().shape();
    series.shape.insert(series.shape.end(), own.begin(), own.end());
    series.axes += pages.front().axes();
    for (const auto& page : pages) series.pages.push_back(page.offset);
    return {std::move(series)};
}

std::vector<TiffSeries> genericSeries(const std::vector<TiffPage>& pages)
{
    std::vector<TiffSeries> out;
    std::vector<std::vector<const TiffPage*>> groups;
    for (const auto& page : pages) {
        auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) { return g.front()->key() == page.key(); });
        if (it == groups.end()) groups.push_back({&page});
        else it->push_back(&page);
    }
    for (const auto& group : groups) {
        TiffSeries series;
        const auto own = group.front()->shape();
        series.axes = group.front()->axes();
        series.shape = own;
        if (group.size() > 1) {
            series.axes.insert(series.axes.begin(), 'Q');
            series.shape.insert(series.shape.begin(), std::int64_t(group.size()));
        }
        for (const auto* page : group) series.pages.push_back(page->offset);
        // Reduced pages following a series with their count and rank are its next levels.
        if (group.front()->reduced && !out.empty() && out.back().pages.size() == series.pages.size() && out.back().shape.size() == series.shape.size()) {
            out.back().reducedLevels.push_back(std::move(series.pages));
            continue;
        }
        out.push_back(std::move(series));
    }
    return out;
}

std::optional<std::string> axisName(char letter)
{
    switch (letter) {
    case 'X': return "x";
    case 'Y': return "y";
    case 'Z': return "z";
    case 'T': return "t";
    case 'C':
    case 'S': return "c";
    default: return std::nullopt;
    }
}
} // namespace

std::vector<std::string> resolveAxes(const std::optional<std::vector<std::string>>& declared, const std::vector<std::optional<std::string>>& observed, const std::vector<std::int64_t>& shape)
{
    const auto list = [](const auto& values) {
        std::string text = "(";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i) text += ", ";
            using V = std::decay_t<decltype(values[i])>;
            if constexpr (std::is_same_v<V, std::string>) text += "'" + values[i] + "'";
            else if constexpr (std::is_same_v<V, std::int64_t>) text += std::to_string(values[i]);
            else text += values[i] ? "'" + *values[i] + "'" : std::string("None");
        }
        return text + ")";
    };
    std::vector<std::string> axes;
    if (declared) {
        if (declared->size() != shape.size()) bad("declared axes " + list(*declared) + " do not match storage shape " + list(shape));
        for (std::size_t i = 0; i < declared->size() && i < observed.size(); ++i)
            if (observed[i] && *observed[i] != (*declared)[i]) bad("declared axes " + list(*declared) + " contradict storage axes " + list(observed));
        axes = *declared;
    } else {
        for (const auto& axis : observed) {
            if (!axis) bad("ambiguous storage axes " + list(observed) + "; declare Source.axes explicitly");
            axes.push_back(*axis);
        }
    }
    const std::set<std::string> unique(axes.begin(), axes.end()), allowed{"x", "y", "z", "c", "t"};
    if (unique.size() != axes.size() || !std::includes(allowed.begin(), allowed.end(), unique.begin(), unique.end()))
        bad("unsupported or repeated storage axes " + list(axes));
    if (!unique.count("x") || !unique.count("y")) bad("image storage must have x and y axes");
    return axes;
}

struct ImageFile::Tiff {
    std::vector<std::uint64_t> pages;  // IFD offsets at the selected level
    TiffPage layout;
};

ImageFile::ImageFile(std::string location, const std::optional<std::string>& format, const Selection& selection,
                     const std::optional<std::vector<std::string>>& axes, std::shared_ptr<EncodedFiles> files)
    : location_(std::move(location)), files_(std::move(files))
{
    if (format) {
        format_ = *format;
    } else {
        const auto suffix = lowerSuffix(location_);
        format_ = suffix == ".tif" || suffix == ".tiff" ? "tiff" : suffix == ".png" ? "png" : suffix == ".jpg" || suffix == ".jpeg" ? "jpeg" : "";
    }
    if (format_ != "tiff" && format_ != "png" && format_ != "jpeg")
        bad("image format " + (format_.empty() ? std::string("None") : "'" + format_ + "'") + " is not supported");
    if (selection.arrayPath || selection.corresponding) bad("image files select series/level, not Zarr array paths or implicit corresponding levels");
    const auto opened = files_->open(location_);
    token_ = opened.token;
    struct stat st{};
    ::fstat(opened.fd, &st);
    encodedSize_ = std::size_t(st.st_size);
    if (format_ == "tiff") {
        openTiff(opened.fd, selection, axes);
    } else {
        if (selection.series || selection.level) bad("PNG/JPEG sources do not have selectable series or levels");
        openRaster(opened.fd, axes);
    }
}
ImageFile::~ImageFile() = default;

std::vector<std::int64_t> ImageFile::prefixShape() const
{
    return std::vector<std::int64_t>(info_.shape.begin(), info_.shape.end() - std::ptrdiff_t(info_.pageShape.size()));
}

void ImageFile::openRaster(int fd, const std::optional<std::vector<std::string>>& axes)
{
    std::uint8_t magic[8]{};
    (void)!::pread(fd, magic, 8, 0);
    const char* found = !std::memcmp(magic, "\x89PNG\r\n\x1a\n", 8) ? "png" : (magic[0] == 0xFF && magic[1] == 0xD8) ? "jpeg" : nullptr;
    if (!found) throw ImageFormatError(location_ + ": cannot identify image file");
    if (format_ != found) bad("declared format " + format_ + " contradicts file format " + found);
    const auto raster = format_ == "png" ? pngHeader(fd) : jpegHeader(fd);
    std::vector<std::optional<std::string>> observed;
    for (const auto& axis : raster.axes) observed.emplace_back(axis);
    const auto resolved = resolveAxes(axes, observed, raster.shape);
    info_ = {raster.shape, resolved, raster.dtype, raster.shape, resolved};
    palette_ = raster.palette;
}

void ImageFile::openTiff(int fd, const Selection& selection, const std::optional<std::vector<std::string>>& axes)
{
    auto handle = openTiffHandle(fd, location_);
    TIFF* tif = handle.get();
    std::vector<TiffPage> pages;
    do {
        pages.push_back(readPage(tif));
    } while (TIFFReadDirectory(tif));
    std::vector<TiffSeries> series;
    const auto& description = pages.front().description;
    if (description.find("<OME") != std::string::npos) series = omeSeries(description, pages);
    if (series.empty() && description.starts_with("ImageJ=")) series = imagejSeries(pages);
    if (series.empty() && description.starts_with("{")) series = shapedSeries(pages);
    if (series.empty()) series = genericSeries(pages);

    std::int64_t index = 0;
    if (!selection.series) {
        if (series.size() != 1) bad("TIFF has " + std::to_string(series.size()) + " series; select series explicitly");
    } else if (const auto* named = std::get_if<std::string>(&*selection.series)) {
        std::vector<std::int64_t> candidates;
        for (std::size_t i = 0; i < series.size(); ++i)
            if (series[i].name == *named) candidates.push_back(std::int64_t(i));
        if (candidates.size() != 1) bad("TIFF series name '" + *named + "' is absent or ambiguous");
        index = candidates[0];
    } else {
        index = std::get<std::int64_t>(*selection.series);
    }
    if (index >= std::int64_t(series.size())) bad("TIFF series " + std::to_string(index) + " does not exist");
    const auto& chosen = series[std::size_t(index)];

    // Levels: reduced pages grouped with the series, else the SubIFDs of its pages.
    std::vector<std::vector<std::uint64_t>> levels{chosen.pages};
    for (const auto& reduced : chosen.reducedLevels) levels.push_back(reduced);
    if (chosen.reducedLevels.empty()) {
        std::map<std::uint64_t, const TiffPage*> byOffset;
        for (const auto& page : pages) byOffset[page.offset] = &page;
        std::size_t subs = std::numeric_limits<std::size_t>::max();
        for (const auto offset : chosen.pages) subs = std::min(subs, byOffset.at(offset)->subifds.size());
        for (std::size_t k = 0; k < subs; ++k) {
            std::vector<std::uint64_t> level;
            for (const auto offset : chosen.pages) level.push_back(byOffset.at(offset)->subifds[k]);
            levels.push_back(std::move(level));
        }
    }
    levelCount_ = std::int64_t(levels.size());
    std::int64_t level = 0;
    if (!selection.level) {
        if (levels.size() > 1) bad("pyramidal TIFF requires an explicit level selection");
    } else {
        level = *selection.level;
    }
    if (level >= std::int64_t(levels.size())) bad("TIFF level " + std::to_string(level) + " does not exist in series " + std::to_string(index));
    tiff_ = std::make_unique<Tiff>();
    tiff_->pages = levels[std::size_t(level)];
    if (!TIFFSetSubDirectory(tif, tiff_->pages.front())) throw ImageFormatError(location_ + ": cannot read a TIFF directory");
    const auto first = readPage(tif);
    tiff_->layout = first;

    // The level's own shape: its page's in place of the series' page dimensions.
    auto seriesShape = chosen.shape;
    auto seriesAxes = chosen.axes;
    const auto baseRank = TIFFSetSubDirectory(tif, chosen.pages.front()) ? readPage(tif).shape().size() : std::size_t(0);
    if (!TIFFSetSubDirectory(tif, tiff_->pages.front())) throw ImageFormatError(location_ + ": cannot read a TIFF directory");
    const auto own = first.shape();
    if (level && seriesShape.size() >= baseRank) {
        seriesShape.resize(seriesShape.size() - baseRank);
        seriesShape.insert(seriesShape.end(), own.begin(), own.end());
        seriesAxes.resize(seriesAxes.size() - baseRank);
        seriesAxes += first.axes();
    }
    channelNames_ = chosen.channelNames;

    const bool hasZ = seriesAxes.find('Z') != std::string::npos;
    if (first.resolutionUnit && !hasZ) {
        const double multiplier = *first.resolutionUnit == RESUNIT_INCH ? 25400. : *first.resolutionUnit == RESUNIT_CENTIMETER ? 10000. : 0.;
        if (multiplier > 0) {
            const auto x = rationalTag(fd, tif, first.offset, TIFFTAG_XRESOLUTION), y = rationalTag(fd, tif, first.offset, TIFFTAG_YRESOLUTION);
            if (x && y) {
                if (!(*x)[0] || !(*x)[1] || !(*y)[0] || !(*y)[1]) bad("TIFF physical resolution must be positive");
                Affine rows;
                rows(0, 0) = multiplier * (*x)[1] / (*x)[0];
                rows(1, 1) = multiplier * (*y)[1] / (*y)[0];
                storageFromArray_ = VolumeTransform{rows, {}};
                storageUnits_ = "micrometer";
            }
        }
    }
    if (!chosen.ome.empty() && level == 0) {
        const std::string spatial = hasZ ? "XYZ" : "XY";
        bool all = true;
        for (const char a : spatial) all = all && chosen.ome.count(std::string("PhysicalSize") + a);
        if (all) {
            Affine rows;
            int i = 0;
            for (const char a : std::string("XYZ")) {
                const auto key = std::string("PhysicalSize") + a;
                double value = 1.;
                if (chosen.ome.count(key)) {
                    const auto unitKey = key + "Unit";
                    const auto unit = chosen.ome.count(unitKey) ? chosen.ome.at(unitKey) : std::string("µm");
                    const auto factor = micrometersPer(unit);
                    if (!factor) bad("unknown OME spatial unit " + unit);
                    value = std::stod(chosen.ome.at(key)) * *factor;
                }
                if (!std::isfinite(value) || value <= 0) bad("OME physical spacing must be finite and positive");
                rows(i, i) = value;
                ++i;
            }
            storageFromArray_ = VolumeTransform{rows, {}};
            storageUnits_ = "micrometer";
        }
    }

    std::vector<std::optional<std::string>> observed;
    for (const char letter : seriesAxes) observed.push_back(axisName(letter));
    const auto resolved = resolveAxes(axes, observed, seriesShape);
    if (seriesShape.size() < own.size()) bad("TIFF series/page layout cannot be read as independent pages; use a chunked source");
    const auto prefix = seriesShape.size() - own.size();
    if (!std::equal(own.begin(), own.end(), seriesShape.begin() + std::ptrdiff_t(prefix)) || product(seriesShape, 0, prefix) != std::int64_t(tiff_->pages.size()))
        bad("TIFF series/page layout cannot be read as independent pages; use a chunked source");
    info_ = {seriesShape, resolved, first.dtype(), own, std::vector<std::string>(resolved.begin() + std::ptrdiff_t(prefix), resolved.end()), index, level};
}

namespace {
void swapElements(std::uint8_t* data, std::size_t bytes, std::size_t itemsize)
{
    if (itemsize < 2) return;
    for (std::size_t i = 0; i + itemsize <= bytes; i += itemsize) std::reverse(data + i, data + i + itemsize);
}
} // namespace

Block ImageFile::region(std::int64_t pageIndex, const std::vector<PageSlice>& slices, DecodedCache& cache, std::size_t scratchBytes)
{
    const auto opened = files_->open(location_);
    if (opened.token != token_) bad(location_ + ": source changed since opening; reopen the source");
    const std::size_t itemsize = std::size_t(info_.dtype[2] - '0');
    if (format_ != "tiff") {
        const std::size_t nbytes = std::size_t(product(info_.pageShape)) * itemsize;
        const auto load = [&] {
            auto block = format_ == "png" ? pngPixels(opened.fd, {info_.pageShape, info_.pageAxes, info_.dtype, {}}) : jpegPixels(opened.fd, {info_.pageShape, info_.pageAxes, info_.dtype, {}});
            bytesRead_ += encodedSize_;
            return block;
        };
        auto [page, lease] = cache.use(token_ + "|raster", 8 * nbytes + encodedSize_ + (1u << 20), scratchBytes, load);
        std::vector<std::int64_t> low, count, kept;
        for (std::size_t i = 0; i < slices.size(); ++i) {
            low.push_back(slices[i].low);
            count.push_back(slices[i].high - slices[i].low);
            if (!slices[i].single) kept.push_back(slices[i].high - slices[i].low);
        }
        Block out = Block::allocate(kept, info_.dtype);
        std::vector<std::int64_t> strides(slices.size());
        std::int64_t stride = std::int64_t(itemsize);
        for (std::size_t i = slices.size(); i-- > 0;) {
            strides[i] = stride;
            stride *= info_.pageShape[i];
        }
        // Rows of the last dimension copied whole.
        std::vector<std::int64_t> at(slices.size(), 0);
        const std::size_t run = slices.empty() ? itemsize : std::size_t(count.back()) * itemsize;
        std::uint8_t* to = out.data.get();
        const bool any = std::all_of(count.begin(), count.end(), [](auto n) { return n > 0; });
        while (any) {
            std::int64_t from = 0;
            for (std::size_t i = 0; i < slices.size(); ++i) from += (low[i] + (i + 1 < slices.size() ? at[i] : 0)) * strides[i];
            std::memcpy(to, page.data.get() + from, run);
            to += run;
            std::size_t d = slices.size() > 1 ? slices.size() - 1 : 0;
            bool done = true;
            while (d-- > 0) {
                if (++at[d] < count[d]) {
                    done = false;
                    break;
                }
                at[d] = 0;
            }
            if (done) break;
        }
        return out;
    }
    if (pageIndex < 0 || pageIndex >= std::int64_t(tiff_->pages.size())) bad("TIFF page " + std::to_string(pageIndex) + " does not exist");
    auto handle = openTiffHandle(opened.fd, location_);
    TIFF* tif = handle.get();
    if (!TIFFSetSubDirectory(tif, tiff_->pages[std::size_t(pageIndex)])) throw ImageFormatError(location_ + ": cannot read a TIFF directory");
    const auto layout = readPage(tif);
    if (layout.shape() != info_.pageShape || layout.dtype() != info_.dtype)
        bad("TIFF page " + std::to_string(pageIndex) + " shape/dtype differs from the selected series");

    // Decoder coordinates: separate sample, depth, Y, X, contiguous sample.
    const bool separate = layout.planar == PLANARCONFIG_SEPARATE && layout.samples > 1;
    const std::array<std::int64_t, 5> shape{separate ? layout.samples : 1, layout.depth, layout.height, layout.width, separate ? 1 : layout.samples};
    std::array<std::pair<std::int64_t, std::int64_t>, 5> bounds;
    for (int i = 0; i < 5; ++i) bounds[i] = {0, shape[i]};
    std::vector<std::int64_t> resultShape;
    const auto letters = layout.axes();
    if (letters.size() != slices.size()) bad("TIFF region rank differs from its page");
    for (std::size_t i = 0; i < slices.size(); ++i) {
        const int dimension = letters[i] == 'Z' ? 1 : letters[i] == 'Y' ? 2 : letters[i] == 'X' ? 3 : separate ? 0 : 4;
        bounds[std::size_t(dimension)] = {slices[i].low, slices[i].single ? slices[i].low + 1 : slices[i].high};
        if (!slices[i].single) resultShape.push_back(slices[i].high - slices[i].low);
    }
    const std::int64_t tileHeight = layout.tiled ? layout.tileHeight : std::max<std::int64_t>(layout.rowsPerStrip, 1);
    const std::int64_t tileWidth = layout.tiled ? layout.tileWidth : layout.width;
    const std::int64_t tileDepth = layout.tiled ? layout.tileDepth : 1;
    const auto ceilDiv = [](std::int64_t a, std::int64_t b) { return (a + b - 1) / b; };
    const std::int64_t nz = ceilDiv(shape[1], tileDepth), ny = ceilDiv(shape[2], tileHeight), nx = ceilDiv(shape[3], tileWidth);
    std::array<std::int64_t, 5> normalized{};
    std::size_t regionBytes = itemsize;
    for (int i = 0; i < 5; ++i) {
        normalized[i] = bounds[i].second - bounds[i].first;
        regionBytes *= std::size_t(normalized[i]);
    }
    const bool raw = layout.compression == COMPRESSION_NONE && layout.bits >= 8;
    if (!raw && !TIFFIsCODECConfigured(std::uint16_t(layout.compression)))
        bad("TIFF codec " + std::to_string(layout.compression) + " is not available in this build of libtiff");
    const std::size_t pageSamples = std::size_t(shape[4]);
    DecodedCache::Lease reserved = cache.working(raw ? scratchBytes : regionBytes + scratchBytes);
    Block out = Block::allocate(resultShape, info_.dtype);
    if (out.nbytes) std::memset(out.data.get(), 0, out.nbytes);
    std::array<std::int64_t, 5> outStrides{};
    std::int64_t stride = std::int64_t(itemsize);
    for (int i = 5; i-- > 0;) {
        outStrides[i] = stride;
        stride *= normalized[i];
    }
    const std::size_t segmentBytes = std::size_t(tileDepth * tileHeight * tileWidth) * pageSamples * itemsize;
    const bool swapped = TIFFIsByteSwapped(tif);
    if (layout.compression == COMPRESSION_JPEG && layout.photometric == PHOTOMETRIC_YCBCR) TIFFSetField(tif, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB);
    for (std::int64_t sample = bounds[0].first; sample < bounds[0].second; ++sample)
        for (std::int64_t z = bounds[1].first / tileDepth; z <= (bounds[1].second - 1) / tileDepth; ++z)
            for (std::int64_t y = bounds[2].first / tileHeight; y <= (bounds[2].second - 1) / tileHeight; ++y)
                for (std::int64_t x = bounds[3].first / tileWidth; x <= (bounds[3].second - 1) / tileWidth; ++x) {
                    const auto segment = std::uint32_t(((sample * nz + z) * ny + y) * nx + x);
                    const auto offset = TIFFGetStrileOffset(tif, segment);
                    const auto count = TIFFGetStrileByteCount(tif, segment);
                    if (!count || !offset) bad("TIFF page " + std::to_string(pageIndex) + " segment " + std::to_string(segment) + " is missing");
                    const std::int64_t rowsHere = layout.tiled ? tileHeight : std::min(tileHeight, layout.height - y * tileHeight);
                    const std::int64_t atZ = z * tileDepth, atY = y * tileHeight, atX = x * tileWidth;
                    // Overlap of the segment and the region.
                    const std::int64_t z0 = std::max(bounds[1].first, atZ), z1 = std::min(bounds[1].second, atZ + tileDepth);
                    const std::int64_t y0 = std::max(bounds[2].first, atY), y1 = std::min(bounds[2].second, atY + rowsHere);
                    const std::int64_t x0 = std::max(bounds[3].first, atX), x1 = std::min(bounds[3].second, atX + tileWidth);
                    const std::int64_t s0 = bounds[4].first, s1 = bounds[4].second;
                    if (z0 >= z1 || y0 >= y1 || x0 >= x1) continue;
                    const std::int64_t pixel = std::int64_t(pageSamples * itemsize);
                    const auto target = [&](std::int64_t zz, std::int64_t yy) {
                        return out.data.get() + (sample - bounds[0].first) * outStrides[0] + (zz - bounds[1].first) * outStrides[1] +
                               (yy - bounds[2].first) * outStrides[2] + (x0 - bounds[3].first) * outStrides[3];
                    };
                    const auto run = std::size_t(x1 - x0) * std::size_t(s1 - s0) * itemsize;
                    if (raw) {
                        for (std::int64_t zz = z0; zz < z1; ++zz)
                            for (std::int64_t yy = y0; yy < y1; ++yy) {
                                const off_t at = off_t(offset) + ((zz - atZ) * tileHeight + (yy - atY)) * tileWidth * pixel + (x0 - atX) * pixel + s0 * std::int64_t(itemsize);
                                auto* to = target(zz, yy);
                                if (::pread(opened.fd, to, run, at) != ssize_t(run)) bad("truncated TIFF segment");
                                if (swapped) swapElements(to, run, itemsize);
                                bytesRead_ += run;
                            }
                        continue;
                    }
                    const auto load = [&] {
                        const auto stored = TIFFIsTiled(tif) ? TIFFTileSize64(tif) : TIFFStripSize64(tif);
                        std::vector<std::uint8_t> packed;
                        const bool bits = layout.bits == 1;
                        Block block = Block::allocate({tileDepth, rowsHere, tileWidth, std::int64_t(pageSamples)}, info_.dtype);
                        void* into = block.data.get();
                        std::size_t capacity = block.nbytes;
                        if (bits) {
                            packed.resize(std::size_t(stored));
                            into = packed.data();
                            capacity = packed.size();
                        }
                        const auto decoded = TIFFIsTiled(tif) ? TIFFReadEncodedTile(tif, segment, into, tmsize_t(capacity)) : TIFFReadEncodedStrip(tif, segment, into, tmsize_t(capacity));
                        if (decoded < 0) bad("TIFF codec " + std::to_string(layout.compression) + ": cannot decode segment " + std::to_string(segment));
                        bytesRead_ += std::size_t(count);
                        if (bits) {
                            const auto rowBytes = std::size_t(tileWidth + 7) / 8;
                            const bool whiteZero = layout.photometric == PHOTOMETRIC_MINISWHITE;
                            for (std::int64_t r = 0; r < rowsHere; ++r)
                                for (std::int64_t c = 0; c < tileWidth; ++c) {
                                    const auto byte = packed[std::size_t(r) * rowBytes + std::size_t(c) / 8];
                                    const bool bit = (byte >> (7 - c % 8)) & 1;
                                    block.data[std::size_t(r * tileWidth + c)] = bit != whiteZero;
                                }
                        }
                        return block;
                    };
                    const auto key = token_ + "|tiff|" + std::to_string(info_.series) + "|" + std::to_string(info_.level) + "|" + std::to_string(pageIndex) + "|" + std::to_string(segment);
                    auto [tile, lease] = cache.use(key, 8 * segmentBytes + std::size_t(count) + (1u << 16), 0, load);
                    for (std::int64_t zz = z0; zz < z1; ++zz)
                        for (std::int64_t yy = y0; yy < y1; ++yy) {
                            const auto from = ((zz - atZ) * tileHeight + (yy - atY)) * tileWidth * pixel + (x0 - atX) * pixel + s0 * std::int64_t(itemsize);
                            std::memcpy(target(zz, yy), tile.data.get() + from, run);
                        }
                }
    return out;
}
} // namespace vcr::sources
