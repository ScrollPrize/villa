#pragma once

// Export the production normal sampler for offline, label-independent features.
// The request contains coordinates only, not decisions or fitted graph state.
inline int sampleFiberReviewNormals(const char* requestPath, const char* outputPath) {
    using Json = nlohmann::json;
    if (std::filesystem::weakly_canonical(requestPath) == std::filesystem::weakly_canonical(outputPath))
        throw std::runtime_error("Normal output must not overwrite the request");
    std::ifstream input(requestPath);
    if (!input) throw std::runtime_error("Cannot open normal request");
    const auto request = Json::parse(input);
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.workingToBaseScale = request.at("normal_scale").get<double>();
    const auto dataset = vc::lasagna::LasagnaDataset::openLocation(
        request.at("normal_manifest").get<std::string>(), options);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    std::ofstream output(outputPath);
    if (!output) throw std::runtime_error("Cannot open normal output");
    size_t count = 0;
    for (const auto& group : request.at("groups")) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<cv::Vec3d> points;
        for (const auto& xyz : group.at("points")) {
            if (xyz.size() != 3) throw std::runtime_error("Expected XYZ coordinates");
            cv::Vec3d p{xyz[0].get<double>(), xyz[1].get<double>(), xyz[2].get<double>()};
            for (int k = 0; k < 3; ++k)
                if (!std::isfinite(p[k])) throw std::runtime_error("Nonfinite normal request");
            points.push_back(p);
        }
        Json values = Json::array();
        for (size_t i = 0; i < points.size(); ++i) {
            // Use the same scalar path as proposal construction. The optional
            // float batch path need not agree with this double-coordinate path.
            const auto s = sampler.sampleNormal(points[i]);
            const auto density = sampler.sampleWindingDensity(points[i]);
            values.push_back({{"valid", s.valid},
                {"normal", {s.normal[0], s.normal[1], s.normal[2]}},
                {"density", density ? Json(*density) : Json(nullptr)}});
        }
        output << Json({{"id", group.at("id")}, {"samples", std::move(values)},
            {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}}).dump() << '\n';
        output.flush();
        if (!output) throw std::runtime_error("Failed writing normal samples");
        if (++count % 100 == 0) std::cout << "normal groups=" << count << std::endl;
    }
    std::cout << "normal groups=" << count << " complete" << std::endl;
    return 0;
}
