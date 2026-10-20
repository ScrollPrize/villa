#include "formats/GeometryInput.hpp"
#include "geometry/ArrayBytes.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "formats/PatchInput.hpp"
#include "spatial/PatchGrid.hpp"
#include <cmath>
namespace vc::objects::formats {
using Json = nlohmann::json;
namespace fs = std::filesystem;
Prepared patch(const fs::path& source)
{
    auto fingerprint = [&]() {
        std::map<std::string, std::pair<uintmax_t, fs::file_time_type>> result;
        for (const auto& entry : fs::directory_iterator(source))
            if (entry.is_regular_file() && (entry.path().extension() == ".tif" || entry.path().filename() == "meta.json"))
                result[entry.path().filename().string()] = {entry.file_size(), entry.last_write_time()};
        return result;
    };
    const auto before = fingerprint();
    const auto root = readJson(source / "meta.json");
    Prepared p;
    p.object = sourceObject(root, "patch", source); p.geometry.kind = "patch";
    detail::validatePatchFiles(source);
    auto surface = vc::object_links::readPatchGrid(source);
    const auto grid = surface.rawPoints();
    // TIFF loading produces a contiguous XYZ grid. Retain a fallback for a
    // strided view, but avoid building a second full-size scalar vector.
    const cv::Mat contiguous = grid.isContinuous() ? grid : grid.clone();
    p.geometry.arrays.push_back(detail::scalarArray<float>("points", "<f4", {grid.rows, grid.cols, 3},
        std::span<const float>(contiguous.ptr<float>(), contiguous.total() * 3)));
    PatchDetail d;
    d.gridWidth = grid.cols; d.gridHeight = grid.rows;
    const auto scale = surface.scale(); d.scale = {scale[0], scale[1]};
    if (root.contains("bbox") && root.at("bbox").is_array() && root.at("bbox").size() == 2) {
        const auto low = root.at("bbox")[0].get<std::array<double, 3>>();
        const auto high = root.at("bbox")[1].get<std::array<double, 3>>();
        d.bbox = {low[0], low[1], low[2], high[0], high[1], high[2]};
    }
    p.object.patch = d;
    if (before != fingerprint()) throw std::runtime_error("patch changed while reading; retry ingestion");
    p.source = root;
    p.geometry.attributes = Json{{"scale", root.at("scale")}}.dump();
    return p;
}
}
