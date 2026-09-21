#pragma once

// Sample the production rendered surfaces at common UV positions. This avoids
// imposing a different triangulation/overlap policy in the Python experiment.
inline int exportFiberReviewMaps(const char* reviewPath, const char* fitsPath, const char* outputPath) {
    using namespace vc::constraints;
    using Json = nlohmann::json;
    if (std::filesystem::weakly_canonical(outputPath) == std::filesystem::weakly_canonical(reviewPath) ||
        std::filesystem::weakly_canonical(outputPath) == std::filesystem::weakly_canonical(fitsPath))
        throw std::runtime_error("Map output must not overwrite its inputs");
    const auto review = loadReview(reviewPath);
    std::ifstream input(fitsPath);
    std::ofstream output(outputPath);
    if (!input || !output) throw std::runtime_error("Cannot open map input/output");
    std::map<std::string, FitResult> bases;
    std::string line;
    size_t count = 0;
    while (std::getline(input, line)) {
        auto record = Json::parse(line);
        if (record.contains("accepted")) throw std::runtime_error("Requires decision-independent fits");
        if (record.at("review_revision") != review.sessionRevision)
            throw std::runtime_error("Fit export belongs to a different review snapshot");
        const std::string fid = record.at("fiber"), pid = record.at("patch");
        const auto start = std::chrono::steady_clock::now();
        for (auto& attempt : record.at("attempts")) {
            if (!attempt.value("valid", false) || !attempt.value("layer_valid", false)) continue;
            auto graph = review.fibers.at(fid).base;
            graph.settings = review.settings;
            if (!bases.contains(fid)) bases.emplace(fid, renderPlacement(graph));
            const auto& base = bases.at(fid);
            if (!base.valid) throw std::runtime_error(base.message);
            const auto& fixed = graph.nodes.at(fid).uv;
            const auto& saved = attempt.at("fiber_uv");
            if (saved.size() != fixed.size()) throw std::runtime_error("Fixed fiber shape changed");
            for (size_t i = 0; i < fixed.size(); ++i)
                if (saved[i][0].get<double>() != fixed[i][0] || saved[i][1].get<double>() != fixed[i][1])
                    throw std::runtime_error("Fit export moved the fixed fiber");
            const auto source = loadPatchNode({pid, review.settings.patchFolder / pid.substr(6), {}, {}}, review.settings);
            std::vector<cv::Vec2d> uv;
            for (const auto& value : attempt.at("patch_uv")) uv.push_back({value[0], value[1]});
            if (uv.size() != source->xyz.size()) throw std::runtime_error("Patch source shape changed");
            graph.nodes.emplace(pid, Placement{source, uv});
            const auto patch = renderSourceLayer(graph, false);
            if (!patch.valid) throw std::runtime_error(patch.message);
            const auto& fiber = graph.nodes.at(fid);
            const int rows = fiber.source->rows, cols = fiber.source->cols;
            int first = cols, last = -1;
            for (int c = 0; c < cols; ++c) for (int r = 0; r < rows; ++r)
                if (patch.surface->sampleAtSurface(fiber.uv[r*cols+c]).valid()) {
                    first = std::min(first, c); last = std::max(last, c);
                }
            Json coordinates = Json::array(), masks = Json::array();
            for (int r = 0; r < 17; ++r) for (int c = 0; c < 128; ++c) {
                const double row = r * (rows-1) / 16.0;
                const double col = last >= first ? first + c * (last-first) / 127.0 : 0;
                const auto query = placedLocation(graph, {fid, {col, row}});
                const auto a = base.surface->sampleAtSurface(query), b = patch.surface->sampleAtSurface(query);
                const bool valid = last >= first && a.valid() && b.valid();
                coordinates.push_back({a.volume[0], a.volume[1], a.volume[2], b.volume[0], b.volume[1], b.volume[2]});
                masks.push_back(valid);
            }
            attempt["registered_xyz"] = std::move(coordinates);
            attempt["registered_valid"] = std::move(masks);
            attempt["registered_span_columns"] = std::max(0, last-first);
            attempt.erase("patch_uv"); attempt.erase("fiber_uv");
        }
        record["map_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        output << record.dump() << '\n'; output.flush();
        if (!output) throw std::runtime_error("Failed writing registered maps");
        if (++count % 100 == 0) std::cout << "mapped=" << count << std::endl;
    }
    return 0;
}
