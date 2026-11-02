#include "formats/GeometryInput.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include "geometry/ArrayLayout.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "formats/SurfaceInput.hpp"
#include <vc_object_store/geometry/SurfaceGrid.hpp>
#include <cmath>
#include <bit>
#include <set>
namespace vc::objects::formats {
using Json = nlohmann::json;
namespace fs = std::filesystem;
GeometryArray surfaceChannel(const std::string& name, const fs::path& path)
{
    auto samples = readTiffSamples(path);
    std::string dtype;
    if (samples.sampleFormat == SAMPLEFORMAT_UINT && samples.bits == 8) dtype = "|u1";
    else if (samples.sampleFormat == SAMPLEFORMAT_UINT && samples.bits == 16) dtype = "<u2";
    else if (samples.sampleFormat == SAMPLEFORMAT_UINT && samples.bits == 64) dtype = "<u8";
    else if (samples.sampleFormat == SAMPLEFORMAT_INT && samples.bits == 32) dtype = "<i4";
    else if (samples.sampleFormat == SAMPLEFORMAT_INT && samples.bits == 64) dtype = "<i8";
    else if (samples.sampleFormat == SAMPLEFORMAT_IEEEFP && samples.bits == 32) dtype = "<f4";
    else if (samples.sampleFormat == SAMPLEFORMAT_IEEEFP && samples.bits == 64) dtype = "<f8";
    else throw std::runtime_error("unsupported surface channel dtype: " + path.string());
    if constexpr (std::endian::native != std::endian::little)
        for (size_t i = 0; i < samples.bytes.size(); i += samples.bits/8)
            std::reverse(samples.bytes.begin()+i, samples.bytes.begin()+i+samples.bits/8);
    std::vector<int64_t> shape{int64_t(samples.height), int64_t(samples.width)};
    if (samples.components != 1) shape.push_back(samples.components);
    return {name, dtype, shape, std::move(samples.bytes)};
}
Prepared surfaceGeometry(const fs::path& source)
{
    auto fingerprint = [&]() {
        std::map<std::string, std::pair<uintmax_t, fs::file_time_type>> result;
        for (const auto& entry : fs::directory_iterator(source))
            if (entry.is_regular_file() && (entry.path().extension() == ".tif" || entry.path().extension() == ".tiff" || entry.path().filename() == "meta.json"))
                result[entry.path().filename().string()] = {entry.file_size(), entry.last_write_time()};
        return result;
    };
    const auto before = fingerprint();
    const auto root = readJson(source / "meta.json");
    Prepared p;
    p.object.kind = "surface"; p.geometry.kind = "surface";
    detail::validateSurfaceFiles(source);
    auto surface = vc::object_links::readSurfaceGrid(source);
    const auto grid = surface.rawPoints();
    // TIFF loading produces a contiguous XYZ grid. Retain a fallback for a
    // strided view, but avoid building a second full-size scalar vector.
    const cv::Mat contiguous = grid.isContinuous() ? grid : grid.clone();
    p.geometry.arrays.push_back(detail::scalarArray<float>("points", "<f4", {grid.rows, grid.cols, 3},
        std::span<const float>(contiguous.ptr<float>(), contiguous.total() * 3)));
    if (auto valid = vc::object_links::readSurfaceValidity(source, grid.rows, grid.cols))
        p.geometry.arrays.push_back({detail::surfaceValidChannel, "|u1", {grid.rows, grid.cols}, std::move(*valid)});
    SurfaceDetail d;
    d.gridWidth = grid.cols; d.gridHeight = grid.rows;
    const auto scale = surface.scale(); d.scale = {scale[0], scale[1]};
    if (root.contains("bbox") && root.at("bbox").is_array() && root.at("bbox").size() == 2) {
        const auto low = root.at("bbox")[0].get<std::array<double, 3>>();
        const auto high = root.at("bbox")[1].get<std::array<double, 3>>();
        d.bbox = {low[0], low[1], low[2], high[0], high[1], high[2]};
    }
    p.object.surface = d;
    p.source = root;
    std::map<std::string, fs::path> files;
    for (const auto& entry : fs::directory_iterator(source)) {
        if (!entry.is_regular_file() || (entry.path().extension() != ".tif" && entry.path().extension() != ".tiff")) continue;
        const auto name = entry.path().stem().string();
        if (name == "x" || name == "y" || name == "z" || name == "mask") continue;
        if (name == "points" || name == detail::surfaceValidChannel || !files.emplace(name, entry.path()).second)
            throw std::runtime_error("duplicate/reserved surface channel name: " + name);
    }
    for (const auto& [name, path] : files) {
        auto channel = surfaceChannel(name, path);
        detail::surfaceChannelResolution(path.string(), channel.shape, {grid.rows, grid.cols});
        p.geometry.arrays.push_back(std::move(channel));
    }
    if (before != fingerprint()) throw std::runtime_error("surface changed while reading; retry ingestion");
    p.geometry.attributes = Json{{"scale", root.at("scale")}}.dump();
    return p;
}

Prepared surface(const fs::path& source)
{
    auto p = surfaceGeometry(source);
    const auto detail = p.object.surface;
    p.object = sourceObject(p.source, "surface", source);
    p.object.surface = detail;
    return p;
}
}
