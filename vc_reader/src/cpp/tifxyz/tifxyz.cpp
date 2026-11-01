#include "vc_reader/tifxyz.hpp"

#include "vc_reader/image_write.hpp"
#include "vc_reader/sources.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_write.hpp"
#include "../element_type.hpp"
#include "../rounding.hpp"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>

namespace vcr::tifxyz {
using nlohmann::json;
namespace {
[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }
std::string trimmed(std::string text)
{
    while (text.size() > 1 && text.back() == '/') text.pop_back();
    return text;
}
std::string child(const std::string& location, const std::string& name) { return trimmed(location) + "/" + name; }
sources::OptionsFor resolved(const sources::OptionsFor& options) { return options ? options : sources::defaultOptions(); }
std::string readText(const std::string& location, const sources::OptionsFor& options)
{
    sources::EncodedFiles files(resolved(options));
    const auto opened = files.open(location);
    std::string text;
    char block[1 << 16];
    ssize_t n;
    off_t at = 0;
    while ((n = ::pread(opened.fd, block, sizeof block, at)) > 0) {
        text.append(block, std::size_t(n));
        at += n;
    }
    files.close();
    return text;
}
// Settings for one-off reads: no decoded cache, no decode limit.
sources::SourceReader::Settings settingsOf(const sources::OptionsFor& options)
{
    sources::SourceReader::Settings settings;
    settings.options = resolved(options);
    settings.decodedCacheBytes = 0;
    settings.maxDecodeBytes = std::numeric_limits<std::size_t>::max() / 2;
    return settings;
}
// One image of the directory open with the declared axes (checked against the file).
std::unique_ptr<sources::SourceReader> openImage(const std::string& path, const char* role, std::vector<std::string> axes, const sources::SourceReader::Settings& settings)
{
    sources::Source source;
    source.id = path;
    source.storage.location = path;
    source.storage.format = "tiff";
    source.role = role;
    source.axes = std::move(axes);
    auto reader = std::make_unique<sources::SourceReader>(sources::finished(std::move(source)), settings);
    (void)reader->shape();
    return reader;
}
// mask.tif as YX or as pages (ZYX, the first counts); null when there is none.
std::unique_ptr<sources::SourceReader> openMask(const std::string& location, const sources::SourceReader::Settings& settings)
{
    const auto path = child(location, "mask.tif");
    try {
        try {
            return openImage(path, "label", {"y", "x"}, settings);
        } catch (const std::invalid_argument&) {
            return openImage(path, "label", {"z", "y", "x"}, settings);
        }
    } catch (const zarr::NotFound&) {
        return nullptr;
    }
}
// How a mask's pixels cover a grid of rows x cols: an integer number per vertex each way.
struct MaskGrid {
    std::int64_t factorY = 1, factorX = 1;
    bool pages = false;
};
MaskGrid maskGridOf(sources::SourceReader& mask, std::int64_t rows, std::int64_t cols, const std::string& location)
{
    const auto& shape = mask.shape();
    const auto height = shape[shape.size() - 2], width = shape[shape.size() - 1];
    if (height % rows || width % cols)
        bad(child(location, "mask.tif") + ": tifxyz mask shape (" + std::to_string(height) + ", " + std::to_string(width) + ") does not match the coordinate grid");
    return {height / rows, width / cols, shape.size() == 3};
}
// Clears `valid` of the vertices in the window (rows x cols from grid row r0, column c0) whose
// mask block has a zero pixel.
void applyMask(sources::SourceReader& mask, const MaskGrid& g, std::int64_t r0, std::int64_t c0, std::int64_t rows, std::int64_t cols, std::uint8_t* valid)
{
    std::vector<std::int64_t> start{r0 * g.factorY, c0 * g.factorX}, size{rows * g.factorY, cols * g.factorX};
    if (g.pages) {
        start.insert(start.begin(), 0);
        size.insert(size.begin(), 1);
    }
    const auto stored = mask.read(start, size);
    const auto item = stored.itemsize();
    const auto height = rows * g.factorY, width = cols * g.factorX;
    for (std::int64_t y = 0; y < height; ++y)
        for (std::int64_t x = 0; x < width; ++x) {
            const auto* p = stored.data.get() + (std::size_t(y) * std::size_t(width) + std::size_t(x)) * item;
            if (std::all_of(p, p + item, [](std::uint8_t b) { return b == 0; })) valid[std::size_t(y / g.factorY) * std::size_t(cols) + std::size_t(x / g.factorX)] = 0;
        }
}
// A block's values as floats, every `stride`th output.
void toFloats(const sources::Block& block, float* out, std::size_t stride)
{
    const auto type = elementOf(block.dtype);
    if (type.kind != 'f' && type.kind != 'u' && type.kind != 'i') bad("tifxyz coordinates of dtype " + block.dtype + " are not supported");
    const auto n = block.nbytes / type.itemsize;
    const auto* data = block.data.get();
    if (block.dtype == "<f4") {
        for (std::size_t i = 0; i < n; ++i) std::memcpy(out + i * stride, data + i * 4, 4);
        return;
    }
    for (std::size_t i = 0; i < n; ++i) out[i * stride] = float(type.load(data + i * type.itemsize));
}
std::string newUuid()
{
    std::random_device device;
    std::uint8_t b[16];
    for (auto& value : b) value = std::uint8_t(device());
    b[6] = std::uint8_t((b[6] & 0x0f) | 0x40);
    b[8] = std::uint8_t((b[8] & 0x3f) | 0x80);
    char text[37];
    std::snprintf(text, sizeof text, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return text;
}
}  // namespace

std::int64_t fullExtent(std::int64_t stored, double scale)
{
    const float scale32 = float(scale);
    if (!std::isfinite(scale32) || scale32 <= 0) return std::max<std::int64_t>(1, stored);
    return std::max<std::int64_t>(1, nearest_index(double(stored) * (1.0 / double(scale32))));
}

bool hasMissingMarker(const Bounds& bbox)
{
    return std::any_of(bbox.begin(), bbox.end(), [](double v) { return v == -1.0; });
}

grid::Raster fullRaster(const std::array<double, 2>& scale, const std::array<std::int64_t, 2>& start, const std::array<std::int64_t, 2>& size)
{
    if (size[0] <= 0 || size[1] <= 0 || size[0] >= (1LL << 31) || size[1] >= (1LL << 31)) bad("full-resolution window must have a positive size");
    const auto [sy, sx] = scale;
    return grid::Raster{double(start[1]) * sx, double(start[0]) * sy, sx, sy, int(size[1]), int(size[0])};
}

// ---- Reader ----
struct Reader::Impl {
    std::string location;
    Metadata metadata;
    std::unique_ptr<sources::SourceReader> xyz[3], mask;
    MaskGrid maskGrid;
    std::array<std::int64_t, 2> shape{};
};

Reader::Reader(std::string location, bool useMask, sources::SourceReader::Settings settings) : impl_(std::make_unique<Impl>())
{
    auto& s = *impl_;
    s.location = trimmed(std::move(location));
    if (!settings.options) settings.options = sources::defaultOptions();
    s.metadata = readMetadata(s.location, settings.options);
    const char* names[3]{"x.tif", "y.tif", "z.tif"};
    for (std::size_t a = 0; a < 3; ++a) {
        s.xyz[a] = openImage(child(s.location, names[a]), "scalar", {"y", "x"}, settings);
        const auto& shape = s.xyz[a]->shape();
        if (!a) s.shape = {shape[0], shape[1]};
        else if (shape[0] != s.shape[0] || shape[1] != s.shape[1]) bad(s.location + ": tifxyz coordinates require matching YX grids");
    }
    if (useMask) {
        s.mask = openMask(s.location, settings);
        if (s.mask) s.maskGrid = maskGridOf(*s.mask, s.shape[0], s.shape[1], s.location);
    }
}

Reader::~Reader() = default;
const std::string& Reader::location() const { return impl_->location; }
const Metadata& Reader::metadata() const { return impl_->metadata; }
std::array<std::int64_t, 2> Reader::shape() const { return impl_->shape; }

Coordinates Reader::window(const std::array<std::int64_t, 4>& box)
{
    auto& s = *impl_;
    if (box[2] <= 0 || box[3] <= 0 || box[0] < 0 || box[1] < 0 || box[0] + box[2] > s.shape[0] || box[1] + box[3] > s.shape[1]) bad(s.location + ": window is outside the coordinate grid");
    Coordinates out;
    out.whole = s.shape;
    out.rows = box[2];
    out.cols = box[3];
    const auto cells = std::size_t(box[2]) * std::size_t(box[3]);
    out.xyz.resize(cells * 3);
    for (std::size_t a = 0; a < 3; ++a) {
        const auto block = s.xyz[a]->read({box[0], box[1]}, {box[2], box[3]});
        out.storedBytes += block.nbytes;
        toFloats(block, out.xyz.data() + a, 3);
    }
    out.valid.resize(cells);
    for (std::size_t i = 0; i < cells; ++i) out.valid[i] = vertexValid(out.xyz.data() + i * 3);
    if (s.mask) applyMask(*s.mask, s.maskGrid, box[0], box[1], box[2], box[3], out.valid.data());
    return out;
}

Coordinates Reader::whole() { return window({0, 0, impl_->shape[0], impl_->shape[1]}); }

// ---- Surface ----
struct Surface::Interpolators {
    std::once_flag once[2];
    std::unique_ptr<grid::Surface> surfaces[2];
};

Surface::Surface(std::int64_t rows, std::int64_t cols, std::vector<float> xyz, std::vector<std::uint8_t> mask, Metadata metadata, std::string location)
    : rows_(rows), cols_(cols), xyz_(std::move(xyz)), metadata_(std::move(metadata)), location_(std::move(location)), interpolators_(std::make_shared<Interpolators>())
{
    if (rows <= 0 || cols <= 0 || rows >= (1LL << 30) || cols >= (1LL << 30)) bad("tifxyz grid must have positive rows and columns");
    const auto cells = std::size_t(rows) * std::size_t(cols);
    if (xyz_.size() != cells * 3) bad("tifxyz coordinates must hold rows x cols x 3 floats");
    if (!mask.empty() && mask.size() != cells) bad("tifxyz mask shape does not match the coordinate grid");
    valid_.resize(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        float* p = xyz_.data() + i * 3;
        valid_[i] = vertexValid(p) && (mask.empty() || mask[i]);
        if (!valid_[i]) p[0] = p[1] = p[2] = -1.f;
    }
}

std::array<std::int64_t, 2> Surface::fullShape() const
{
    return {fullExtent(rows_, metadata_.scale[0]), fullExtent(cols_, metadata_.scale[1])};
}

std::optional<Bounds> Surface::bounds() const
{
    float low[3], high[3];
    bool any = false;
    for (std::size_t i = 0; i < valid_.size(); ++i) {
        if (!valid_[i]) continue;
        const float* p = xyz_.data() + i * 3;
        for (int a = 0; a < 3; ++a) {
            low[a] = any ? std::min(low[a], p[a]) : p[a];
            high[a] = any ? std::max(high[a], p[a]) : p[a];
        }
        any = true;
    }
    if (!any) return std::nullopt;
    return Bounds{low[0], low[1], low[2], high[0], high[1], high[2]};
}

Surface Surface::retargeted(double factor) const
{
    if (!std::isfinite(factor) || factor <= 0) bad("retarget factor must be positive and finite");
    auto xyz = xyz_;
    const float divisor = float(factor);
    for (std::size_t i = 0; i < valid_.size(); ++i)
        if (valid_[i])
            for (int a = 0; a < 3; ++a) xyz[i * 3 + std::size_t(a)] /= divisor;
    auto metadata = metadata_;
    metadata.scale = {metadata.scale[0] * factor, metadata.scale[1] * factor};
    metadata.area.reset();
    Surface out(rows_, cols_, std::move(xyz), valid_, std::move(metadata), location_);
    out.metadata_.bbox = out.bounds();
    return out;
}

const grid::Surface& Surface::interpolator(grid::Interpolation interpolation) const
{
    const auto k = std::size_t(interpolation == grid::Interpolation::Smooth);
    std::call_once(interpolators_->once[k], [&] {
        grid::SurfaceOptions options;
        options.interpolation = interpolation;
        interpolators_->surfaces[k] = std::make_unique<grid::Surface>(grid::Vertices{xyz_.data(), int(rows_), int(cols_), std::size_t(cols_) * 3, nullptr, valid_.data()}, options);
    });
    return *interpolators_->surfaces[k];
}

void Surface::pointsAt(const float* xy, std::size_t n, grid::Interpolation interpolation, float* xyz, float* normals) const
{
    interpolator(interpolation).pointsAt(xy, n, xyz, normals);
}

void Surface::resample(const grid::Raster& raster, grid::Interpolation interpolation, float* xyz, float* normals) const
{
    interpolator(interpolation).resample(raster, xyz, normals);
}

grid::Raster Surface::fullRaster(const std::array<std::int64_t, 2>& start, const std::array<std::int64_t, 2>& size) const
{
    return tifxyz::fullRaster(metadata_.scale, start, size);
}

// ---- files ----
std::vector<std::uint8_t> readMask(const std::string& location, std::int64_t rows, std::int64_t cols, const sources::OptionsFor& options)
{
    const auto mask = openMask(location, settingsOf(options));
    if (!mask) return {};
    const auto g = maskGridOf(*mask, rows, cols, location);
    std::vector<std::uint8_t> valid(std::size_t(rows) * std::size_t(cols), 1);
    applyMask(*mask, g, 0, 0, rows, cols, valid.data());
    return valid;
}

Metadata parseMetadata(const json& document, const std::string& location)
{
    Metadata out;
    if (!document.is_object()) bad(location + ": meta.json must be an object");
    const auto scale = document.value("scale", json::array());
    if (!scale.is_array() || scale.size() != 2 || !scale[0].is_number() || !scale[1].is_number() || !std::isfinite(scale[0].get<double>()) || !std::isfinite(scale[1].get<double>()) ||
        scale[0].get<double>() <= 0 || scale[1].get<double>() <= 0)
        bad("tifxyz scale must contain two positive finite UV scales");
    out.scale = {scale[1].get<double>(), scale[0].get<double>()};  // the file's order is x, y
    const auto slash = trimmed(location).find_last_of('/');
    const auto name = slash == std::string::npos ? trimmed(location) : trimmed(location).substr(slash + 1);
    out.uuid = document.contains("uuid") && document["uuid"].is_string() ? document["uuid"].get<std::string>() : name;
    if (document.contains("bbox") && document["bbox"].is_array() && document["bbox"].size() == 2) {
        const auto& low = document["bbox"][0];
        const auto& high = document["bbox"][1];
        if (low.is_array() && high.is_array() && low.size() == 3 && high.size() == 3)
            out.bbox = Bounds{low[0].get<double>(), low[1].get<double>(), low[2].get<double>(), high[0].get<double>(), high[1].get<double>(), high[2].get<double>()};
    }
    if (document.contains("area") && document["area"].is_number()) out.area = document["area"].get<double>();
    out.extra = document;
    for (const char* key : {"uuid", "scale", "bbox", "area"}) out.extra.erase(key);
    return out;
}

Metadata readMetadata(const std::string& location, const sources::OptionsFor& options)
{
    return parseMetadata(json::parse(readText(child(location, "meta.json"), options)), location);
}

Surface read(const std::string& location, bool useMask, const sources::OptionsFor& options)
{
    Reader reader(location, useMask, settingsOf(options));
    auto stored = reader.whole();
    Surface out(stored.rows, stored.cols, std::move(stored.xyz), std::move(stored.valid), reader.metadata(), reader.location());
    if (out.metadata().bbox && hasMissingMarker(*out.metadata().bbox)) out.metadata().bbox = out.bounds();
    return out;
}

std::vector<Entry> list(const std::string& folder, bool recursive, const std::optional<std::array<double, 2>>& zRange, const sources::OptionsFor& options)
{
    const auto optionsFor = resolved(options);
    std::vector<Entry> out;
    std::vector<std::string> pending{trimmed(folder)};
    bool top = true;
    while (!pending.empty()) {
        const auto location = std::move(pending.back());
        pending.pop_back();
        const zarr::Store store(location, optionsFor(location).access);
        if (!top && store.exists("meta.json") && store.exists("x.tif") && store.exists("y.tif") && store.exists("z.tif")) {
            Entry entry;
            entry.location = location;
            try {
                entry.metadata = readMetadata(location, optionsFor);
                if (zRange) {
                    auto box = entry.metadata.bbox;
                    if (box && hasMissingMarker(*box)) box = read(location, true, optionsFor).bounds();
                    if (!box || ((*box)[2] <= (*zRange)[1] && (*box)[5] >= (*zRange)[0])) out.push_back(std::move(entry));
                } else {
                    out.push_back(std::move(entry));
                }
            } catch (const std::exception& e) {
                entry.error = e.what();
                out.push_back(std::move(entry));
            }
        }
        if (top || recursive)
            for (const auto& name : store.children()) pending.push_back(child(location, name));
        top = false;
    }
    std::sort(out.begin(), out.end(), [](const Entry& a, const Entry& b) { return a.location < b.location; });
    return out;
}

void write(const std::filesystem::path& path, Surface& surface, const WriteOptions& options)
{
    namespace fs = std::filesystem;
    if (fs::exists(path) && !options.overwrite) throw fs::filesystem_error("tifxyz directory already exists", path, std::make_error_code(std::errc::file_exists));
    auto& metadata = surface.metadata();
    if (metadata.uuid.empty()) metadata.uuid = newUuid();
    const auto parent = path.has_parent_path() ? path.parent_path() : fs::path(".");
    fs::create_directories(parent);
    const auto temporary = parent / (".tifxyz_tmp_" + path.filename().string() + "." + std::to_string(::getpid()));
    fs::remove_all(temporary);
    fs::create_directories(temporary);
    try {
        sources::TiffOptions tiff;
        tiff.compression = options.compression;
        tiff.pixelUm = options.pixelUm;
        if (options.tile) tiff.tile = std::array<int, 2>{*options.tile, *options.tile};
        const auto rows = surface.rows(), cols = surface.cols();
        const auto cells = std::size_t(rows) * std::size_t(cols);
        std::vector<float> component(cells);
        const char* names[3]{"x.tif", "y.tif", "z.tif"};
        for (std::size_t a = 0; a < 3; ++a) {
            for (std::size_t i = 0; i < cells; ++i) component[i] = surface.xyz()[i * 3 + a];
            sources::writeTiff(temporary / names[a], component.data(), "<f4", rows, cols, 1, tiff);
        }
        const auto box = surface.bounds().value_or(Bounds{0, 0, 0, 0, 0, 0});
        json document = metadata.extra;
        document.update(json{{"uuid", metadata.uuid}, {"scale", {metadata.scale[1], metadata.scale[0]}}, {"bbox", {{box[0], box[1], box[2]}, {box[3], box[4], box[5]}}}});
        if (!document.contains("type")) document["type"] = "seg";
        if (!document.contains("format")) document["format"] = "tifxyz";
        if (metadata.area) document["area"] = *metadata.area;
        std::ofstream(temporary / "meta.json") << document.dump(2) << '\n';
        if (options.mask) {
            std::vector<std::uint8_t> mask(cells);
            for (std::size_t i = 0; i < cells; ++i) mask[i] = surface.valid()[i] ? 255 : 0;
            sources::writeTiff(temporary / "mask.tif", mask.data(), "|u1", rows, cols, 1, tiff);
        }
        fs::remove_all(path);
        fs::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(temporary, ignored);
        throw;
    }
}

}  // namespace vcr::tifxyz
