#pragma once

// Read-only replay of the same proposal/fitting path used by Fiber review.
// Rejection reasons were not persisted; this reports reproducible failures,
// not a claim about which UI action caused a historical rejection.
inline int auditFiberReviewFits(int argc, char** argv) {
    using namespace vc::constraints;
    using Json = nlohmann::json;
    const bool exportGeometry = std::string(argv[1]) == "--review-fit-dataset";
    const bool overlaps = std::string(argv[1]) == "--review-overlap-audit";
    if (std::filesystem::weakly_canonical(argv[2]) == std::filesystem::weakly_canonical(argv[3]))
        throw std::runtime_error("Audit output must not overwrite the review workspace");
    const auto review = loadReview(argv[2]);
    const size_t maxRejected = argc > 4 ? std::stoull(argv[4]) : 0;
    const size_t maxControls = argc > 5 ? std::stoull(argv[5]) : 12;
    std::ofstream output(argv[3]);
    if (!output) throw std::runtime_error("Cannot open audit output");
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.workingToBaseScale = review.settings.normalScale;
    const auto dataset = vc::lasagna::LasagnaDataset::openLocation(review.settings.normalManifest, options);
    vc::lasagna::LasagnaNormalSampler normals(dataset);
    Catalog catalog;
    std::set<std::string> patches;
    for (const auto& [id, fiber] : review.fibers)
        for (const auto& candidate : fiber.proposals) patches.insert(candidate.id);
    for (const auto& id : patches)
        catalog.patches.push_back({id, review.settings.patchFolder / id.substr(6), {}, {}});
    size_t rejected = 0, controls = 0;
    for (const auto& [id, fiber] : review.fibers) {
        if (exportGeometry ? fiber.proposals.empty() : fiber.decisions.empty()) continue;
        // Match selectReviewFiber on reopening: shared review settings override
        // the potentially stale base settings (accepted alignments use these).
        auto baseGraph = fiber.base;
        baseGraph.settings = review.settings;
        const auto base = renderPlacement(baseGraph);
        if (!base.valid) throw std::runtime_error("Invalid saved fiber: " + base.message);
        const auto prepared = prepareGraph(base);
        size_t localControls = 0;
        // Dataset export visits every candidate identically, without consulting
        // decisions or saved accepted alignments. Audit behavior stays unchanged.
        std::map<std::string, ReviewDecision> requests;
        if (exportGeometry) {
            for (const auto& candidate : fiber.proposals) requests.try_emplace(candidate.id);
        } else requests = fiber.decisions;
        for (const auto& [patch, decision] : requests) {
            if (!exportGeometry && decision.status == ReviewStatus::Skipped) continue;
            const bool accepted = decision.status == ReviewStatus::Accepted;
            if (!exportGeometry && accepted && localControls >= 1) continue;
            if (!exportGeometry && (accepted ? controls >= maxControls : maxRejected && rejected >= maxRejected)) continue;
            if (accepted) { ++controls; ++localControls; } else ++rejected;
            Json row = {{"review_path", argv[2]}, {"review_revision", review.sessionRevision}, {"fiber", id}, {"patch", patch}, {"accepted", accepted},
                        {"radius", review.settings.radius}, {"attempts", Json::array()}};
            if (exportGeometry) row.erase("accepted");
            bool anyValid = false, exception = false;
            size_t available = 0;
            for (const auto& candidate : fiber.proposals) if (candidate.id == patch) ++available;
            row["available_placements"] = available;
            for (size_t index = 0; index < fiber.proposals.size(); ++index) {
                const auto& candidate = fiber.proposals[index];
                if (candidate.id != patch) continue;
                const auto start = std::chrono::steady_clock::now();
                Json attempt = {{"proposal_index", index}, {"contacts", candidate.contacts.size()},
                                {"progress", Json::array()}};
                try {
                    auto edit = proposeEdit(catalog, prepared, candidate, false, normals, nullptr, ContactPolicy::FiberReview);
                    auto overlapRms = [&](const Graph& graph) {
                        double squared=0;
                        for(const auto& target:edit.fixedTargets) {
                            const auto error=placedLocation(graph,target.point)-target.uv;squared+=error.dot(error);
                        }
                        return edit.fixedTargets.empty()?0:std::sqrt(squared/edit.fixedTargets.size());
                    };
                    if(overlaps) {
                        const auto alone=fitEdit(edit,nullptr,{},FitPolicy::FiberReview);
                        addReviewOverlapTargets(edit,fiber);
                        attempt["overlap_targets"]=edit.fixedTargets.size();
                        attempt["fiber_only_valid"]=alone.valid;
                        if(alone.valid)attempt["overlap_rms_before"]=overlapRms(alone.graph);
                    }
                    const auto fit = fitEdit(edit, nullptr, [&](const std::string& text) {
                        attempt["progress"].push_back(text);
                    }, FitPolicy::FiberReview);
                    if (fit.graph.nodes.at(id).uv != fiber.base.nodes.at(id).uv)
                        throw std::runtime_error("Audit fit moved its fixed fiber");
                    attempt["valid"] = fit.valid;
                    attempt["message"] = fit.message;
                    attempt["rms"] = fit.correspondenceRms;
                    attempt["stretch_p95"] = fit.stretchP95;
                    if(overlaps&&fit.valid)attempt["overlap_rms_after"]=overlapRms(fit.graph);
                    if (fit.valid) {
                        const auto layer = renderSourceLayer(fit.graph, false);
                        attempt["layer_valid"] = layer.valid;
                        attempt["layer_message"] = layer.message;
                        if (exportGeometry) {
                            auto uvJson = [](const auto& values) {
                                Json result = Json::array();
                                for (const auto& uv : values) result.push_back({uv[0], uv[1]});
                                return result;
                            };
                            attempt["patch_uv"] = uvJson(fit.graph.nodes.at(patch).uv);
                            attempt["fiber_uv"] = uvJson(fit.graph.nodes.at(id).uv);
                        }
                    }
                    anyValid |= fit.valid;
                } catch (const std::exception& error) {
                    attempt["error"] = error.what();
                    exception = true;
                }
                attempt["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                row["attempts"].push_back(std::move(attempt));
                // Once a placement works, this pair is not an all-placement failure.
                if (anyValid || exception) break;
            }
            row["status"] = exception ? "unavailable" : anyValid ? "fits" : available ? "all_placements_fail" : "no_saved_proposal";
            row["first_placement_failed"] = !row["attempts"].empty() && row["attempts"][0].contains("valid") && !row["attempts"][0]["valid"].get<bool>();
            output << row.dump() << '\n'; output.flush();
            if (!output) throw std::runtime_error("Failed to write audit result");
            if (exportGeometry) std::cout << "candidates=" << rejected + controls;
            else std::cout << "rejected=" << rejected << " controls=" << controls;
            std::cout << " patch=" << patch
                      << " status=" << row["status"] << " attempts=" << row["attempts"].size() << std::endl;
        }
        if (maxRejected && rejected >= maxRejected && controls >= maxControls) break;
    }
    return 0;
}
