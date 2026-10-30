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

    if (backend == "cpu") {
        SurfaceGeometry geometry(surface, AlignedOptions{});
        require(geometry.shape() == Pair{8, 9}, "grid shape");
        const auto [points, valid] = geometry.readPoints({1, 1}, {2, 2}, {0, 2});
        require(points.shape == Index{2, 2, 2, 3}, "points shape");
        const auto* xyz = reinterpret_cast<const float*>(points.data.get());
        require(xyz[0] == 4.f && xyz[1] == 3.f && xyz[2] == 5.f && valid.data[0] == 1, "points on the sheet");
        require(std::abs(std::abs(xyz[3 * 4 + 2] - 5.f) - 1.f) < 1e-5f, "two micrometers along the normal at 2 um spacing");
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
            testPaired(root, backend);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        status = 1;
    }
    fs::remove_all(root);
    return status;
}
