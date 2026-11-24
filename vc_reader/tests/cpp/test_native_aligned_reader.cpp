// Aligned reads from C++: reference grids, categorical and field targets, flattened pairs,
// surfaces; the CUDA backend must match the CPU one where there is a device.
#include "vc_reader/aligned_reader.hpp"
#include "vc_reader/zarr_write.hpp"

#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>

#ifdef VC_READER_CUDA
#include "../../src/cpp/cuda_available.hpp"
#include "../../src/cpp/cuda_buffer.hpp"
#endif

namespace fs = std::filesystem;
using namespace vcr::sources;
using nlohmann::json;
using Index = std::vector<std::int64_t>;
using Triple = std::array<std::int64_t, 3>;
using Pair = std::array<std::int64_t, 2>;

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
void writeZarr(const fs::path& path, const Index& shape, const std::string& dtype, const void* data, const json& axes)
{
    vcr::zarr::CreateOptions options;
    options.shape = shape;
    options.chunks = shape;
    options.dtype = dtype;
    auto writer = vcr::zarr::Writer::create(path, options);
    writer.writeRegion(Index(shape.size(), 0), shape, static_cast<const std::uint8_t*>(data));
    vcr::zarr::writeAttributes(path, {{"_ARRAY_DIMENSIONS", axes}});
}
Source zarrSource(const std::string& id, const fs::path& path, const std::string& role, std::optional<SourceRegistration> registration = std::nullopt)
{
    Source s;
    s.id = id;
    s.storage.location = path.string();
    s.storage.format = "zarr";
    s.role = role;
    s.registration = std::move(registration);
    return finished(std::move(s));
}
VolumeRegistration placedIn(const std::string& volume, Affine affine = {})
{
    VolumeRegistration r;
    r.sampleId = "s";
    r.volumeId = volume;
    r.affine = VolumeTransform{affine, {}};
    r.volumeSpacingUm = Vec3{2, 2, 2};
    return r;
}
// Host bytes of an array, from wherever it is.
std::vector<std::uint8_t> hostBytes(const SampledArray& array)
{
    const auto bytes = std::size_t(array.count()) * std::size_t(array.dtype[2] - '0');
    std::vector<std::uint8_t> out(bytes);
#ifdef VC_READER_CUDA
    if (array.onDevice()) {
        auto storage = std::static_pointer_cast<vcr::device::Buffer>(array.device);
        vcr::device::Guard guard(storage->device);
        if (bytes) vcr::device::check(cudaMemcpyAsync(out.data(), storage->ptr, bytes, cudaMemcpyDeviceToHost, vcr::device::stream()));
        vcr::device::complete();
        return out;
    }
#endif
    std::memcpy(out.data(), array.host.get(), bytes);
    return out;
}
template <class T> std::vector<T> values(const SampledArray& array)
{
    const auto bytes = hostBytes(array);
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}
AlignedOptions optionsFor(const std::string& backend)
{
    AlignedOptions options;
    options.backend = backend;
    return options;
}

void testReference(const fs::path& root, const std::string& backend)
{
    std::vector<std::uint8_t> ct(20 * 22 * 24);
    std::iota(ct.begin(), ct.end(), std::uint8_t(0));
    writeZarr(root / "ct.zarr", {20, 22, 24}, "|u1", ct.data(), {"z", "y", "x"});
    std::vector<std::uint16_t> labels(10 * 11 * 12);
    std::iota(labels.begin(), labels.end(), std::uint16_t(0));
    writeZarr(root / "labels.zarr", {10, 11, 12}, "<u2", labels.data(), {"z", "y", "x"});
    std::vector<float> field(3 * 4 * 4 * 4, 1.f);
    writeZarr(root / "field.zarr", {3, 4, 4, 4}, "<f4", field.data(), {"c", "z", "y", "x"});

    const auto reference = zarrSource("ct", root / "ct.zarr", "image", placedIn("v"));
    std::vector<std::pair<std::string, Source>> targets;
    targets.emplace_back("label", zarrSource("labels", root / "labels.zarr", "label", placedIn("v", Affine::scale(2))));
    Source vectors = zarrSource("field", root / "field.zarr", "vector", placedIn("v", Affine::scale(2)));
    vectors.componentOrder = std::vector<std::string>{"x", "y", "z"};
    vectors.normalize = false;
    ValueFrame frame;
    frame.sampleId = "s";
    frame.volumeId = "v";
    frame.affine = VolumeTransform{Affine::scale(2), {}};
    vectors.valueFrame = frame;
    targets.emplace_back("vector", finished(vectors));

    SampleReader reader(reference, targets, std::nullopt, optionsFor(backend));
    require(reader.shape() == Triple{20, 22, 24}, "reference shape");
    {
        SourceReader opened(reference, optionsFor("cpu").reader);
        const auto grid = openedGrid(opened);
        require(grid.shapeZYX == std::array<std::int64_t, 3>{20, 22, 24}, "opened grid");
        const auto manifest = [](const std::string& volume) {
            return vcr::catalog::Catalog(vcr::catalog::parseOpenDataManifest(R"({"samples":{"s":{"sample":{"id":"s"},"volumes":{")" + volume + R"(":{"id":")" + volume + R"(","data":[]}}}}})"));
        };
        const auto mine = manifest("v");
        const vcr::catalog::VolumeLevel level(mine.scroll("s").volumes().at("v"), 0);
        raises<std::invalid_argument>([&] { SampleReader r(level, {}, manifest("w"), optionsFor(backend)); }, "a level of another catalog");
    }
    const auto result = reader.read({2, 4, 6}, {3, 4, 5});
    const auto image = values<std::uint8_t>(result["image"].data);
    require(result["image"].data.shape == Index{1, 3, 4, 5} && result["image"].data.dtype == "|u1", "image shape");
    require(image[0] == ct[std::size_t((2 * 22 + 4) * 24 + 6)], "identity read is the stored values");
    const auto valid = values<std::uint8_t>(result["image"].valid);
    require(std::all_of(valid.begin(), valid.end(), [](auto v) { return v == 1; }), "inside the volume");
    // Nearest labels on a twice coarser grid: index = round((reference) / 2).
    const auto label = values<std::uint16_t>(result["label"].data);
    require(result["label"].data.dtype == "<u2", "labels keep their dtype");
    require(label[0] == labels[std::size_t((1 * 11 + 2) * 12 + 3)], "categorical nearest");
    const auto vector = values<float>(result["vector"].data);
    require(std::abs(vector[0] - 2.f) < 1e-5f, "vectors moved through the value frame into reference voxels");
    require(!result.metadata.is_null() && result.metadata["value_frames"]["vector"]["kind"] == "volume", "value frame metadata");
    // Padding and validity outside the volume.
    const auto outside = reader.read({-1, 0, 0}, {2, 1, 1});
    const auto support = values<std::uint8_t>(outside["image"].valid);
    require(support[0] == 0 && support[1] == 1, "outside is invalid");
    // One target alone; empty regions; a closed reader.
    const auto [one, oneValid] = reader.sampleTarget("label", {2, 4, 6}, {1, 1, 1});
    (void)oneValid;
    require(values<std::uint16_t>(one)[0] == label[0], "named target");
    require(reader.read({0, 0, 0}, {0, 1, 1})["image"].data.shape == Index{1, 0, 1, 1}, "empty region");
    raises<std::invalid_argument>([&] { (void)reader.read({0, 0, 0}, {1 << 25, 1, 1}); }, "size limit");
    reader.close();
    raises<std::invalid_argument>([&] { (void)reader.read({0, 0, 0}, {1, 1, 1}); }, "closed");

    // Reserved and unrelated targets.
    std::vector<std::pair<std::string, Source>> reserved{{"image", reference}};
    raises<std::invalid_argument>([&] { SampleReader r(reference, reserved, std::nullopt, optionsFor(backend)); }, "image is reserved");
    std::vector<std::pair<std::string, Source>> other{{"t", zarrSource("other", root / "labels.zarr", "label", placedIn("elsewhere"))}};
    raises<std::invalid_argument>([&] { SampleReader r(reference, other, std::nullopt, optionsFor(backend)); }, "unrelated volume");
}

Surface coordinateSurface(const fs::path& root)
{
    // A flat sheet at z = 5, spanning x and y on a unit grid.
    const std::int64_t rows = 8, cols = 9;
    std::vector<float> xyz(std::size_t(3 * rows * cols));
    for (std::int64_t y = 0; y < rows; ++y)
        for (std::int64_t x = 0; x < cols; ++x) {
            xyz[std::size_t((y * cols + x))] = float(x) + 3;
            xyz[std::size_t((rows + y) * cols + x)] = float(y) + 2;
            xyz[std::size_t((2 * rows + y) * cols + x)] = 5;
        }
    writeZarr(root / "surface.zarr", {3, rows, cols}, "<f4", xyz.data(), {"c", "y", "x"});
    Surface surface;
    surface.id = "surface";
    surface.geometry.location = (root / "surface.zarr").string();
    surface.geometry.format = "zarr";
    surface.sampleId = "s";
    surface.volumeId = "v";
    surface.axes = std::vector<std::string>{"c", "y", "x"};
    surface.componentOrder = std::vector<std::string>{"x", "y", "z"};
    surface.xyzToVolume = VolumeTransform{};
    surface.pixelToUv = PixelAffine{{{1, 0, 0}, {0, 1, 0}}};
    surface.spacingUm = Vec3{2, 2, 2};
    validate(surface);
    return surface;
}

void testSurface(const fs::path& root, const std::string& backend)
{
    const auto surface = coordinateSurface(root);
    const auto reference = zarrSource("ct", root / "ct.zarr", "image", placedIn("v"));
    std::vector<std::uint8_t> ink(8 * 9);
    std::iota(ink.begin(), ink.end(), std::uint8_t(0));
    writeZarr(root / "ink.zarr", {8, 9}, "|u1", ink.data(), {"y", "x"});
    SurfaceRegistration registration;
    registration.surface = surface;
    registration.pixelToUv = PixelAffine{{{1, 0, 0}, {0, 1, 0}}};
    std::vector<std::pair<std::string, Source>> targets{{"ink", zarrSource("ink", root / "ink.zarr", "label", SourceRegistration(registration))}};
    SampleReader reader(reference, targets, std::nullopt, optionsFor(backend));
    raises<std::invalid_argument>([&] { (void)reader.read({0, 0, 0}, {1, 1, 1}); }, "surface targets need explicit reads");
    require(reader.surfaceNamed("surface") == surface, "surface by id");
    // Render: the sheet lies at z = 5.
    const auto rendered = reader.renderSurface(surface, {1, 1}, {2, 3}, {0});
    const auto image = values<std::uint8_t>(rendered["image"].data);
    require(rendered["image"].data.shape == Index{1, 1, 2, 3}, "layers, patch");
    require(image[0] == 0 || image[0] == std::uint8_t(((5 * 22 + 3) * 24 + 4) & 255), "sampled at the sheet");
    {   // A work budget that fits only the cells the patch reaches: the render still works, the context is not charged to it.
        auto tight = optionsFor(backend);
        tight.samplingWorkBytes = 12 * 80;
        SampleReader squeezed(reference, targets, std::nullopt, tight);
        const auto small = squeezed.renderSurface(surface, {1, 1}, {2, 3}, {0});
        require(values<std::uint8_t>(small["image"].data) == image, "a budget that fits the cells renders");
        squeezed.close();
    }
    const auto inkValues = values<std::uint8_t>(rendered["ink"].data);
    require(inkValues[0] == ink[std::size_t(1 * 9 + 1)], "surface label at its own pixels");
    raises<std::invalid_argument>([&] { (void)reader.renderSurface(surface, {0, 0}, {1, 1}, {}); }, "no offsets");
    // Project: the label patch selects a crop around the sheet, and labels reach its voxels.
    const auto projection = reader.projection(surface, {1, 1}, {4, 4}, "ink", 2.0, 0.0);
    const auto origin = projection->origin();
    const auto shape = projection->shape();
    require(shape[0] >= 1 && origin[0] <= 5 && origin[0] + shape[0] > 5, "crop around the sheet");
    const auto crop = projection->readCrop(origin, shape);
    const auto coverage = values<std::uint8_t>(crop["ink"].mask("coverage"));
    require(std::any_of(coverage.begin(), coverage.end(), [](auto v) { return v == 1; }), "geometry reaches the crop");
    const auto validity = values<std::uint8_t>(crop["ink"].valid);
    const auto projected = values<std::uint8_t>(crop["ink"].data);
    bool found = false;
    for (std::size_t i = 0; i < validity.size(); ++i)
        if (validity[i]) {
            found = true;
            require(std::find(ink.begin(), ink.end(), projected[i]) != ink.end(), "projected values come from the label");
        }
    require(found, "labelled voxels");
    raises<std::invalid_argument>([&] { (void)reader.projection(surface, {7, 8}, {4, 4}, "ink"); }, "patch outside its grid");
    const auto stats = reader.stats();
    require(stats.hasSurface && stats.geometryReadBytes > 0, "surface stats");
    reader.close();
}


// A curved sheet on a unit grid, z = 6 + 0.35 (x - 4)^2 / 4 + 0.3 sin(1.3 y); the CT value is 10 z, so a render reads the sheet height.
Surface curvedSurface(const fs::path& root)
{
    const std::int64_t rows = 14, cols = 16;
    std::vector<float> xyz(std::size_t(3 * rows * cols));
    for (std::int64_t y = 0; y < rows; ++y)
        for (std::int64_t x = 0; x < cols; ++x) {
            xyz[std::size_t((y * cols + x))] = float(x) + 3;
            xyz[std::size_t((rows + y) * cols + x)] = float(y) + 2;
            xyz[std::size_t((2 * rows + y) * cols + x)] = 6.f + 0.35f * float((x - 4) * (x - 4)) / 4.f + 0.3f * std::sin(1.3f * float(y));
        }
    writeZarr(root / "curved.zarr", {3, rows, cols}, "<f4", xyz.data(), {"c", "y", "x"});
    Surface surface;
    surface.id = "curved";
    surface.geometry.location = (root / "curved.zarr").string();
    surface.geometry.format = "zarr";
    surface.sampleId = "s";
    surface.volumeId = "v";
    surface.axes = std::vector<std::string>{"c", "y", "x"};
    surface.componentOrder = std::vector<std::string>{"x", "y", "z"};
    surface.xyzToVolume = VolumeTransform{};
    surface.pixelToUv = PixelAffine{{{2, 0, 0}, {0, 2, 0}}};  // two pixels per grid step: the pixels lie between the vertices
    surface.spacingUm = Vec3{2, 2, 2};
    validate(surface);
    return surface;
}

// The render of a patch does not depend on the work budget, and it is the same pixels of the render of a larger patch.
void testContext(const fs::path& root, const std::string& backend)
{
    std::vector<std::uint8_t> ct(24 * 24 * 24);
    for (std::size_t z = 0; z < 24; ++z)
        for (std::size_t i = 0; i < 24 * 24; ++i) ct[z * 24 * 24 + i] = std::uint8_t(10 * z);
    writeZarr(root / "ramp.zarr", {24, 24, 24}, "|u1", ct.data(), {"z", "y", "x"});
    const auto surface = curvedSurface(root);
    const auto reference = zarrSource("ct", root / "ramp.zarr", "image", placedIn("v"));
    const std::vector<std::pair<std::string, Source>> none;
    const auto render = [&](std::size_t workBytes, const Pair& start, const Pair& size) {
        auto options = optionsFor(backend);
        if (workBytes) options.samplingWorkBytes = workBytes;
        SampleReader reader(reference, none, std::nullopt, options);
        const auto rendered = reader.renderSurface(surface, start, size, {0});
        auto out = values<float>(rendered["image"].data);
        reader.close();
        return out;
    };
    const Pair start{8, 10}, size{10, 14};  // pixels at half grid steps: 5 x 7 grid steps
    const auto full = render(0, start, size);
    // cells reached: (5 + 7 wide) grid positions need 7 x 9 vertices, 63 cells
    const auto tight = render(63 * 80, start, size);
    const auto equal = [](float a, float b) { return (std::isnan(a) && std::isnan(b)) || std::fabs(a - b) < 1e-3f; };
    // the ring is charged to the budget with the cells: at the budget of the cells alone it does not fit and the render is made on the cells
    // (a read or budget error of the ring leaves them to render), so the pixels are close to the render with the ring, not equal to it
    require(full.size() == tight.size(), "a patch renders at the budget of its cells");
    float worst = 0;
    for (std::size_t k = 0; k < full.size(); ++k) {
        require(std::isnan(full[k]) == std::isnan(tight[k]), "the same pixels are valid with and without the ring");
        if (!std::isnan(full[k])) worst = std::max(worst, std::fabs(full[k] - tight[k]));
    }
    require(worst < 2.f, "a patch at the budget of its cells is close to the render with the ring");
    // an error in the cells themselves is not hidden by the retry on the cells: a budget below the cells still refuses the render
    bool refused = false;
    try { (void)render(40 * 80, start, size); } catch (const std::runtime_error&) { refused = true; }
    require(refused, "a work budget the cells do not fit still refuses the render");
    {   // the ring is charged to the work budget like any window, so a window that does not fit is never kept: a later linear render of the
        // window the ring made (rows 2..12, columns 3..15, found by the pixels below) is refused at that budget, not served from the cache
        auto options = optionsFor(backend);
        options.samplingWorkBytes = 63 * 80;
        SampleReader reader(reference, none, std::nullopt, options);
        (void)reader.renderSurface(surface, start, size, {0});
        raises<std::runtime_error>([&] { (void)reader.renderSurface(surface, {4, 6}, {18, 22}, {0}, vcr::grid::Interpolation::Linear); },
                                   "a window over the work budget is not served from a cache filled by a bicubic render");
        reader.close();
    }
    const auto large = render(0, {4, 6}, {18, 22});
    bool same = true;
    for (std::int64_t j = 0; j < size[0]; ++j)
        for (std::int64_t i = 0; i < size[1]; ++i) same = same && equal(large[std::size_t((j + 4) * 22 + (i + 4))], full[std::size_t(j * size[1] + i)]);
    require(same, "a patch is the same pixels of a larger patch");
}


// The bot's registered-normals case: a flat 32 x 64 grid, float32 XYZ normals registered half a pixel over in X, a one pixel render at 40 KiB.
void testNormalsBudget(const fs::path& root, const std::string& backend)
{
    const std::int64_t rows = 32, cols = 64;
    std::vector<float> xyz(std::size_t(3 * rows * cols)), normal(std::size_t(3 * rows * cols));
    for (std::int64_t y = 0; y < rows; ++y)
        for (std::int64_t x = 0; x < cols; ++x) {
            xyz[std::size_t(y * cols + x)] = float(x);
            xyz[std::size_t((rows + y) * cols + x)] = float(y);
            xyz[std::size_t((2 * rows + y) * cols + x)] = 5.f;
            normal[std::size_t((2 * rows + y) * cols + x)] = 1.f;
        }
    writeZarr(root / "flat.zarr", {3, rows, cols}, "<f4", xyz.data(), {"c", "y", "x"});
    writeZarr(root / "flat-normals.zarr", {3, rows, cols}, "<f4", normal.data(), {"c", "y", "x"});
    std::vector<std::uint8_t> ct(std::size_t(12 * rows * cols));
    for (std::size_t z = 0; z < 12; ++z)
        for (std::size_t i = 0; i < std::size_t(rows * cols); ++i) ct[z * std::size_t(rows * cols) + i] = std::uint8_t(10 * z);
    writeZarr(root / "flat-ct.zarr", {12, rows, cols}, "|u1", ct.data(), {"z", "y", "x"});
    Surface base;
    base.id = "flat";
    base.geometry.location = (root / "flat.zarr").string();
    base.geometry.format = "zarr";
    base.sampleId = "s";
    base.volumeId = "v";
    base.axes = std::vector<std::string>{"c", "y", "x"};
    base.componentOrder = std::vector<std::string>{"x", "y", "z"};
    base.xyzToVolume = VolumeTransform{};
    base.pixelToUv = PixelAffine{{{1, 0, 0}, {0, 1, 0}}};
    base.spacingUm = Vec3{2, 2, 2};
    validate(base);
    SurfaceRegistration registration;
    registration.surface = base;
    registration.pixelToUv = PixelAffine{{{1, 0, 0.5}, {0, 1, 0}}};  // registered half a pixel over in X
    auto surface = base;
    auto normals = zarrSource("normals", root / "flat-normals.zarr", "normal", SourceRegistration(registration));
    normals.componentOrder = std::vector<std::string>{"x", "y", "z"};
    normals.normalize = true;
    ValueFrame frame;
    frame.sampleId = "s";
    frame.volumeId = "v";
    frame.affine = VolumeTransform{Affine{}, {}};
    normals.valueFrame = frame;
    surface.normals = Shared<Source>(finished(normals));
    validate(surface);
    const auto reference = zarrSource("ct", root / "flat-ct.zarr", "image", placedIn("v"));
    const std::vector<std::pair<std::string, Source>> none;
    for (const auto interpolation : {vcr::grid::Interpolation::Bicubic, vcr::grid::Interpolation::Linear}) {
        const auto render = [&](std::size_t workBytes) {
            auto options = optionsFor(backend);
            options.samplingWorkBytes = workBytes;
            SampleReader reader(reference, none, std::nullopt, options);
            const auto rendered = reader.renderSurface(surface, {6, 29}, {1, 1}, {-2, 0, 2}, interpolation);
            auto out = values<float>(rendered["image"].data);
            reader.close();
            return out;
        };
        const auto roomy = render(80 * 1024);
        require(render(40 * 1024) == roomy, "a one pixel render with registered normals does not need more budget than its cells and normals");
        bool refused = false;
        try { (void)render(20 * 1024); } catch (const std::runtime_error&) { refused = true; }
        require(refused, "a budget the cells and normals do not fit still refuses the render");
    }
}

void testPaired(const fs::path& root, const std::string& backend)
{
    std::vector<std::uint16_t> image(3 * 7 * 9);
    std::iota(image.begin(), image.end(), std::uint16_t(0));
    writeZarr(root / "render.zarr", {3, 7, 9}, "<u2", image.data(), {"z", "y", "x"});
    std::vector<std::uint8_t> labels(7 * 9);
    std::iota(labels.begin(), labels.end(), std::uint8_t(0));
    writeZarr(root / "label.zarr", {7, 9}, "|u1", labels.data(), {"y", "x"});
    const auto render = zarrSource("render", root / "render.zarr", "image");
    const auto label = zarrSource("label", root / "label.zarr", "label");
    vcr::catalog::Pairing same{label.id, render.id, PixelAffine{{{1, 0, 0}, {0, 1, 0}}}, "explicit same grid"};
    PairedReader reader(label, render, same, 0, {}, optionsFor(backend));
    require(reader.shape() == Pair{7, 9} && reader.layerCount() == 3, "image shape");
    const auto result = reader.read({1, 2}, {3, 4});
    require(result["image"].data.shape == Index{1, 3, 3, 4} && result["label"].data.shape == Index{1, 3, 4}, "shapes");
    require(values<std::uint16_t>(result["image"].data)[0] == image[std::size_t(1 * 9 + 2)], "first layer");
    require(values<std::uint8_t>(result["label"].data)[0] == labels[std::size_t(1 * 9 + 2)], "same grid");
    require(result.metadata["pairing_evidence"] == "explicit same grid", "evidence");
    reader.close();
    // A pixel map, and one chosen layer.
    vcr::catalog::Pairing shifted{label.id, render.id, PixelAffine{{{1, 0, 2}, {0, 1, 0}}}, "explicit user pixel mapping"};
    PairedReader mapped(label, render, shifted, 0, [](std::int64_t) { return PairedReader::Layers{1, 1, true}; }, optionsFor(backend));
    const auto moved = mapped.read({0, 3}, {1, 2});
    require(moved["image"].data.shape == Index{1, 1, 2}, "dropped layer");
    require(values<std::uint16_t>(moved["image"].data)[0] == image[std::size_t(1 * 7 * 9 + 3)], "selected layer");
    require(values<std::uint8_t>(moved["label"].data)[0] == labels[1], "label pixels shifted by the map");
    raises<std::invalid_argument>([&] { PairedReader r(render, render, same, 0, {}, optionsFor(backend)); }, "roles");
}
} // namespace

int main()
{
    const auto root = fs::temp_directory_path() / ("vc_reader_aligned_" + std::to_string(::getpid()));
    fs::create_directories(root);
    int status = 0;
    std::vector<std::string> backends{"cpu"};
#ifdef VC_READER_CUDA
    if (vcr::cuda::available()) backends.emplace_back("cuda");
#endif
    try {
        for (const auto& backend : backends) {
            fs::remove_all(root);
            fs::create_directories(root);
            testReference(root, backend);
            testSurface(root, backend);
            testContext(root, backend);
            testNormalsBudget(root, backend);
            testPaired(root, backend);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        status = 1;
    }
    fs::remove_all(root);
    return status;
}
