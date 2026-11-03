// JSON-lines interface: keep native datasets warm across a collection, and use
// exactly the normal sampler and display corrections used by VC3D.
#include <iostream>
#include <map>
#include <memory>
#include "vc/fiber_tracer/FiberJson.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

int main()
{
    using namespace vc::fiber_tracer;
    using nlohmann::json;
    try {
        std::map<std::string, std::unique_ptr<vc::lasagna::LasagnaDataset>> datasets;
        std::string line;
        while (std::getline(std::cin, line)) {
            const auto request = json::parse(line);
            const auto& document = request.at("fiber");
            const auto fiber = parseVc3dFiberJson(document, "normal baking");
            const auto indices = vc3dControlPointLineIndices(fiber, "normal baking");
            const auto manifest = request.at("manifest").get<std::string>();
            const auto cache = request.value("cache_root", std::string{});
            const auto key = manifest + '\n' + cache;
            if (!datasets.contains(key)) {
                vc::lasagna::LasagnaDatasetOpenOptions options;
                options.remoteCacheRoot = cache;
                options.workingToBaseScale = 1.0;
                datasets[key] = std::make_unique<vc::lasagna::LasagnaDataset>(
                    vc::lasagna::LasagnaDataset::openLocation(manifest, options));
            }
            vc::lasagna::LasagnaNormalSampler sampler(*datasets.at(key));
            std::vector<cv::Vec3f> points, normals;
            for (const auto& p : fiber.linePoints) points.emplace_back(p);
            std::vector<vc::lasagna::LasagnaNormalSampler::FloatNormalSample> samples;
            (void)sampler.sampleNormalBatch(points, request.value("threads", 1), samples);
            auto valid = json::array();
            for (size_t i = 0; i < samples.size(); ++i) {
                valid.push_back(samples[i].valid);
                auto normal = samples[i].valid ? samples[i].normal : cv::Vec3f{};
                if (i && normal.dot(normals.back()) < 0) normal *= -1;
                normals.push_back(normal);
            }
            std::vector<double> positions(indices.begin(), indices.end());
            std::vector<std::optional<cv::Vec3d>> corrected, directions;
            for (const auto& cp : document.at("control_points")) {
                corrected.push_back(displayNormalFromJson(cp));
                directions.push_back(cp.contains("direction") ?
                    std::optional<cv::Vec3d>(cv::Vec3d(cp["direction"][0].get<double>(),
                        cp["direction"][1].get<double>(), cp["direction"][2].get<double>())) : std::nullopt);
            }
            const auto field = fiberDisplayField(points, normals, positions, corrected, directions);
            if (!field.resetControls.empty())
                throw std::runtime_error("invalid control normal correction; refusing implicit reset");
            auto axes = json::array();
            for (size_t i = 0; i < field.normals.size(); ++i) {
                const auto& n = field.normals[i];
                if (!samples[i].valid) {
                    axes.push_back({0., 0., 0.});
                    continue;
                }
                const auto unit = displayUnit(cv::Vec3d(n));
                if (!unit) throw std::runtime_error("invalid baked normal");
                axes.push_back({(*unit)[0], (*unit)[1], (*unit)[2]});
            }
            std::cout << json{{"version", 1}, {"normals_xyz", axes},
                {"valid", valid},
                {"coordinate_space", document.at("vc_open_data_coordinate_space")},
                {"method", "vc3d-fiberDisplayField"}}.dump() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "vc_fiber_bake_annotations: " << e.what() << '\n';
        return 1;
    }
}
