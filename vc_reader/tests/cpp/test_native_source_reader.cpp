// The source reader from C++: Zarr selections and encodings, TIFF and PNG decoding, budgets, stacks.
#include "vc_reader/source_reader.hpp"
#include "vc_reader/zarr_write.hpp"

#include <png.h>
#include <sys/wait.h>
#include <tiffio.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;
using namespace vcr::sources;
using nlohmann::json;
using Index = std::vector<std::int64_t>;

namespace {
void require(bool v, const std::string& message)
{
    if (!v) throw std::runtime_error(message);
}
template <class E, class F> void raises(F&& f, const std::string& message)
{
    try {
        f();
    } catch (const E&) {
        return;
    }
    throw std::runtime_error("did not raise: " + message);
}
template <class T> const T* values(const Block& b) { return reinterpret_cast<const T*>(b.data.get()); }

Source sourceAt(const std::string& id, const std::string& location, const std::string& format)
{
    Source s;
    s.id = id;
    s.storage.location = location;
    s.storage.format = format;
    return finished(std::move(s));
}

void writeZarr(const fs::path& path, const Index& shape, const std::string& dtype, const void* data, const json& attributes, const json& fill = 0)
{
    vcr::zarr::CreateOptions options;
    options.shape = shape;
    options.chunks = shape;
    options.dtype = dtype;
    options.fillValue = fill;
    auto writer = vcr::zarr::Writer::create(path, options);
    writer.writeRegion(Index(shape.size(), 0), shape, static_cast<const std::uint8_t*>(data));
    vcr::zarr::writeAttributes(path, attributes);
}

void writeTiff(const fs::path& path, int width, int height, const std::vector<std::uint16_t>& data, bool tiled, bool deflate, const char* description = nullptr, int pages = 1)
{
    TIFF* tif = TIFFOpen(path.c_str(), "w");
    require(tif != nullptr, "tiff open");
    for (int page = 0; page < pages; ++page) {
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 16);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, deflate ? COMPRESSION_ADOBE_DEFLATE : COMPRESSION_NONE);
        if (description && page == 0) TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, description);
        const auto* plane = data.data() + std::size_t(page) * std::size_t(width * height);
        if (tiled) {
            TIFFSetField(tif, TIFFTAG_TILEWIDTH, 16);
            TIFFSetField(tif, TIFFTAG_TILELENGTH, 16);
            std::vector<std::uint16_t> tile(16 * 16);
            for (int y = 0; y < height; y += 16)
                for (int x = 0; x < width; x += 16) {
                    std::fill(tile.begin(), tile.end(), 0);
                    for (int j = 0; j < 16 && y + j < height; ++j)
                        for (int i = 0; i < 16 && x + i < width; ++i) tile[std::size_t(j * 16 + i)] = plane[(y + j) * width + x + i];
                    TIFFWriteTile(tif, tile.data(), std::uint32_t(x), std::uint32_t(y), 0, 0);
                }
        } else {
            TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 5);
            for (int row = 0; row < height; ++row) TIFFWriteScanline(tif, const_cast<std::uint16_t*>(plane + row * width), std::uint32_t(row), 0);
        }
        TIFFWriteDirectory(tif);
    }
    TIFFClose(tif);
}

void writePng(const fs::path& path, int width, int height, const std::vector<std::uint8_t>& gray)
{
    std::FILE* file = std::fopen(path.c_str(), "wb");
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    png_init_io(png, file);
    png_set_IHDR(png, info, std::uint32_t(width), std::uint32_t(height), 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    for (int y = 0; y < height; ++y) png_write_row(png, gray.data() + y * width);
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    std::fclose(file);
}

void testZarr(const fs::path& root)
{
    std::vector<std::uint64_t> data(2 * 3 * 4 * 5);
    std::iota(data.begin(), data.end(), std::uint64_t(1) << 40);
    writeZarr(root / "image.zarr", {2, 3, 4, 5}, "<u8", data.data(), {{"_ARRAY_DIMENSIONS", {"t", "c", "y", "x"}}, {"channel_names", {"red", "green", "blue"}}, {"time_names", {"a", "b"}}});
    auto source = sourceAt("zarr", (root / "image.zarr").string(), "zarr");
    raises<std::invalid_argument>([&] { SourceReader reader(source, {}); }, "time selection is required");
    Selection selection;
    selection.channels = std::vector<Selector>{Selector("blue"), Selector("red")};
    selection.time = Selector("b");
    SourceReader reader(source.withSelection(selection), {});
    require(reader.shape() == Index{2, 4, 5} && reader.axes() == std::vector<std::string>{"c", "y", "x"}, "selected shape");
    const auto crop = reader.read({1, 2}, {3, 3});
    require(crop.shape == Index{2, 3, 3} && crop.dtype == "<u8", "crop shape");
    const auto at = [&](int t, int c, int y, int x) { return data[std::size_t(((t * 3 + c) * 4 + y) * 5 + x)]; };
    require(values<std::uint64_t>(crop)[0] == at(1, 2, 1, 2) && values<std::uint64_t>(crop)[9] == at(1, 0, 1, 2), "channels in selection order");

    // Padding with the array's fill, and the direct path.
    std::vector<std::uint8_t> small(12, 5);
    writeZarr(root / "labels.zarr", {3, 4}, "|u1", small.data(), {{"_ARRAY_DIMENSIONS", {"y", "x"}}}, 7);
    auto labels = sourceAt("labels", (root / "labels.zarr").string(), "zarr");
    SourceReader direct(labels, {});
    require(direct.directArray() != nullptr, "unencoded spatial zarr reads directly");
    const auto padded = direct.read({-1, 0}, {2, 2});
    require(padded.data[0] == 7 && padded.data[2] == 5, "fill outside the array");
    // Encoding: exact and with a fallback.
    Source encoded = labels;
    Encoding encoding;
    encoding.kind = "integer";
    encoding.dtype = "uint64";
    encoding.pairs.push_back({{{5, false}}, {std::uint64_t(1) << 40, false}});
    encoded.encoding = encoding;
    raises<std::invalid_argument>([&] { SourceReader r(encoded, {}); (void)r.read({0, 0}, {1, 1}); }, "unknown fill needs a fallback");
    encoding.fallback = WideInt{9, false};
    encoded.encoding = encoding;
    SourceReader lookup(encoded, {});
    require(lookup.directArray() == nullptr && lookup.dtype() == "<u8", "encoded output");
    const auto mapped = lookup.read({2, 3}, {2, 2});
    require(values<std::uint64_t>(mapped)[0] == std::uint64_t(1) << 40 && values<std::uint64_t>(mapped)[1] == 9, "encoded with padding");
    // Companions and declared metadata.
    Source mask = sourceAt("mask", (root / "labels.zarr").string(), "zarr");
    mask.role = "label";
    Source parent = labels;
    parent.companions.push_back({"supervision", Shared<Source>(finished(mask)), true});
    SourceReader withMask(finished(parent), {});
    require(withMask.companionNames() == std::vector<std::string>{"supervision"} && withMask.companion("supervision")->shape() == Index{3, 4}, "companion");
    Source wrong = labels;
    wrong.shape = Index{4, 3};
    raises<std::invalid_argument>([&] { SourceReader r(finished(wrong), {}); }, "declared shape");
    Source missing = sourceAt("absent", (root / "absent.zarr").string(), "zarr");
    raises<vcr::zarr::NotFound>([&] { SourceReader r(missing, {}); }, "missing array");
    // Registration on another grid through a catalog-style fallback.
    Selection zero;
    zero.level = 0;
    std::vector<std::uint8_t> level0(64, 1), level1(8, 2);
    writeZarr(root / "pyramid.zarr" / "0", {4, 4, 4}, "|u1", level0.data(), {{"_ARRAY_DIMENSIONS", {"z", "y", "x"}}});
    writeZarr(root / "pyramid.zarr" / "1", {2, 2, 2}, "|u1", level1.data(), {{"_ARRAY_DIMENSIONS", {"z", "y", "x"}}});
    vcr::zarr::writeAttributes(root / "pyramid.zarr", json::object());
    Source pyramid;
    pyramid.id = "pyramid";
    pyramid.storage.location = (root / "pyramid.zarr").string();
    pyramid.storage.format = "zarr";
    pyramid.storage.pyramid = true;
    pyramid.storage.selection = zero;
    pyramid.axes = std::vector<std::string>{"z", "y", "x"};
    VolumeRegistration registration;
    registration.sampleId = "s";
    registration.volumeId = "v";
    registration.affine = VolumeTransform{};
    registration.origin = "catalog-fallback";
    registration.declaredGrid = zero;
    pyramid.registration = registration;
    pyramid = finished(pyramid);
    SourceReader coarse(selectLevel(pyramid, 1), {});
    const auto placement = arrayPlacement(coarse.registration());
    require(placement.forward == Affine::blockReduced(2), "level placed by block reduction");
    require(coarse.resolved().dtype == "uint8" && coarse.read({0, 0, 0}, {1, 1, 1}).data[0] == 2, "level 1 read");
}

void testTiff(const fs::path& root)
{
    const int width = 51, height = 33;
    std::vector<std::uint16_t> data(std::size_t(width * height));
    std::iota(data.begin(), data.end(), std::uint16_t(0));
    for (const bool tiled : {false, true})
        for (const bool deflate : {false, true}) {
            const auto path = root / (std::string(tiled ? "tiled" : "strips") + (deflate ? "-deflate.tif" : ".tif"));
            writeTiff(path, width, height, data, tiled, deflate);
            SourceReader reader(sourceAt("tiff", path.string(), "tiff"), {});
            require(reader.shape() == Index{height, width} && reader.dtype() == "<u2", "tiff shape");
            const auto crop = reader.read({14, 13}, {7, 8});
            for (int y = 0; y < 7; ++y)
                for (int x = 0; x < 8; ++x)
                    require(values<std::uint16_t>(crop)[y * 8 + x] == data[std::size_t((14 + y) * width + 13 + x)], "tiff crop");
            const auto before = reader.stats().imagePayloadBytes;
            (void)reader.read({14, 13}, {7, 8});
            require(!deflate || reader.stats().imagePayloadBytes == before, "decoded segments are cached");
            if (deflate) {
                require(before < std::size_t(width * height) * 2, "only the needed segments are decoded");
                SourceReader::Settings tight;
                tight.maxDecodeBytes = 64;
                SourceReader budget(sourceAt("tiff", path.string(), "tiff"), std::move(tight));
                raises<DecodeBudgetError>([&] { (void)budget.read({0, 0}, {1, 1}); }, "decode budget");
            } else {
                SourceReader::Settings none;
                none.maxDecodeBytes = 0;
                SourceReader inPlace(sourceAt("tiff", path.string(), "tiff"), std::move(none));
                require(values<std::uint16_t>(inPlace.read({1, 1}, {1, 1}))[0] == data[std::size_t(width + 1)], "uncompressed reads need no decode budget");
            }
        }
    // Pages: ambiguous without axes, then selected.
    std::vector<std::uint16_t> stack(std::size_t(3 * 5 * 7));
    std::iota(stack.begin(), stack.end(), std::uint16_t(0));
    const auto generic = root / "pages.tif";
    writeTiff(generic, 7, 5, stack, false, false, nullptr, 3);
    raises<std::invalid_argument>([&] { SourceReader r(sourceAt("pages", generic.string(), "tiff"), {}); }, "anonymous page axis");
    Source declared = sourceAt("pages", generic.string(), "tiff");
    declared.axes = std::vector<std::string>{"z", "y", "x"};
    SourceReader pages(finished(declared), {});
    require(values<std::uint16_t>(pages.read({1, 2, 3}, {1, 1, 1}))[0] == stack[std::size_t((1 * 5 + 2) * 7 + 3)], "page selection");
    const auto shaped = root / "shaped.tif";
    writeTiff(shaped, 7, 5, stack, false, true, R"({"shape": [3, 5, 7], "axes": "CYX"})", 3);
    Selection channels;
    channels.channels = std::vector<Selector>{Selector(std::int64_t(2)), Selector(std::int64_t(0))};
    SourceReader chosen(sourceAt("shaped", shaped.string(), "tiff").withSelection(channels), {});
    require(chosen.axes() == std::vector<std::string>{"c", "y", "x"} && chosen.shape() == Index{2, 5, 7}, "declared channel axis");
    const auto picked = chosen.read({1, 2}, {3, 4});
    require(values<std::uint16_t>(picked)[0] == stack[std::size_t((2 * 5 + 1) * 7 + 2)] && values<std::uint16_t>(picked)[12] == stack[std::size_t((0 * 5 + 1) * 7 + 2)], "pages per channel");
}

void testPngAndStack(const fs::path& root)
{
    std::vector<std::uint8_t> gray(35);
    std::iota(gray.begin(), gray.end(), std::uint8_t(0));
    writePng(root / "a.png", 7, 5, gray);
    std::vector<std::uint8_t> other(35, 99);
    writePng(root / "b.png", 7, 5, other);
    writePng(root / "c.png", 4, 3, std::vector<std::uint8_t>(12, 1));
    SourceReader reader(sourceAt("png", (root / "a.png").string(), "png"), {});
    require(reader.shape() == Index{5, 7} && reader.dtype() == "|u1" && reader.stats().imagePayloadBytes == 0, "header only");
    const auto crop = reader.read({-2, 3}, {6, 6});
    require(crop.data[0] == 0 && crop.data[2 * 6] == gray[3], "crop with padding");
    require(reader.decodedCache()->stats().misses == 1, "decoded once");
    (void)reader.read({0, 0}, {1, 1});
    require(reader.decodedCache()->stats().hits >= 1, "decoded page reused");
    std::ofstream(root / "bad.png") << "not an image";
    raises<ImageFormatError>([&] { SourceReader r(sourceAt("bad", (root / "bad.png").string(), "png"), {}); }, "not an image");
    raises<std::invalid_argument>([&] { SourceReader r(sourceAt("wrong", (root / "a.png").string(), "jpeg"), {}); }, "format contradiction");

    Source stack;
    stack.id = "stack";
    stack.storage.format = "stack";
    stack.storage.locations = {(root / "b.png").string(), (root / "a.png").string(), (root / "absent.png").string(), (root / "c.png").string()};
    SourceReader planes(finished(stack), {});
    require(planes.shape() == Index{4, 5, 7} && planes.axes() == std::vector<std::string>{"z", "y", "x"}, "stack shape");
    const auto two = planes.read({0, 0, 1}, {2, 1, 1});
    require(two.data[0] == 99 && two.data[1] == gray[1], "ordered planes");
    require(!planes.loaded(2), "planes open when needed");
    raises<vcr::zarr::NotFound>([&] { (void)planes.read({2, 0, 0}, {1, 1, 1}); }, "absent plane");
    raises<std::invalid_argument>([&] { (void)planes.read({3, 0, 0}, {1, 1, 1}); }, "plane of another shape");

    // A forked process reopens what it inherited.
    SourceReader::Settings settings;
    SourceReader inherited(sourceAt("png", (root / "a.png").string(), "png"), std::move(settings));
    (void)inherited.read({0, 0}, {1, 1});
    const pid_t child = ::fork();
    if (child == 0) {
        const auto block = inherited.read({0, 1}, {1, 1});
        ::_exit(block.data[0] == gray[1] && inherited.decodedCache()->stats().misses == 1 ? 0 : 1);
    }
    int status = 1;
    ::waitpid(child, &status, 0);
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "forked read");
}

void testGrids()
{
    const auto grid = storageTransform({"z", "y", "x"}, {"micrometer", "micrometer", "micrometer"},
                                       {json::array({{{"type", "scale"}, {"scale", {4, 2, 2}}}, {{"type", "translation"}, {"translation", {1, 1, 1}}}})});
    require(grid.units == "micrometer" && grid.transform->forward(2, 2) == 4 && grid.transform->forward(0, 3) == 1, "ngff transform");
    raises<std::invalid_argument>([] { (void)storageTransform({"y", "x"}, {}, {json::array({{{"type", "rotation"}}})}); }, "unsupported transformation");
    raises<std::invalid_argument>([] { (void)arrayPlacement(std::nullopt); }, "no placement");
    const auto reduced = blockReducedTransforms(json::array({{{"type", "scale"}, {"scale", {2, 2}}}}), json::array({{{"type", "scale"}, {"scale", {1, 1}}}}));
    require(reduced.size() == 2 && reduced[1]["translation"][0] == .5, "scale-only level placed");
    require(canonicalDtype("uint16") == "<u2" && dtypeName("<u2") == "uint16" && dtypeName(">u2") == ">u2", "dtype names");
    require(micrometersPer("mm") == 1000. && !micrometersPer("parsec"), "units");
}
} // namespace

int main()
{
    const auto root = fs::temp_directory_path() / ("vc_reader_source_reader_" + std::to_string(::getpid()));
    fs::create_directories(root);
    int status = 0;
    try {
        testGrids();
        testZarr(root);
        testTiff(root);
        testPngAndStack(root);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        status = 1;
    }
    fs::remove_all(root);
    return status;
}
