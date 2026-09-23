#pragma once
#include "vc/constraints/AutoLink.hpp"

// Read-only replay: run the geometric auto-linker on every fiber of a saved
// review that has a strip and proposals, writing one JSON row per fiber.
inline int autoLinkReview(int argc, char** argv) {
    using namespace vc::constraints;
    using Json = nlohmann::json;
    if (std::filesystem::weakly_canonical(argv[2]) == std::filesystem::weakly_canonical(argv[3]))
        throw std::runtime_error("Auto-link output must not overwrite the review");
    const auto review = loadReview(argv[2]);
    AutoLinkSettings settings;
    if (argc > 4) settings.anchorContact = std::stod(argv[4]);
    if (const char* v = std::getenv("AUTOLINK_GATE_RATIO")) settings.stripGateRatio = std::stod(v);
    if (const char* v = std::getenv("AUTOLINK_GATE_MARGIN")) settings.stripGateMargin = std::stod(v);
    // Strips are rebuilt with the current builder from each fiber's saved dense
    // line, so saved (older-generation) strips never enter the linker.
    vc::lasagna::LasagnaDatasetOpenOptions options; options.workingToBaseScale = review.settings.normalScale;
    const auto dataset = std::make_shared<vc::lasagna::LasagnaDataset>(vc::lasagna::LasagnaDataset::openLocation(review.settings.normalManifest, options));
    vc::lasagna::LasagnaNormalSampler sampler(*dataset);
    std::ofstream output(argv[3]);
    if (!output) throw std::runtime_error("Cannot open auto-link output");
    std::map<std::string, std::shared_ptr<const Node>> cache;
    std::mutex mutex;
    PatchLoader load = [&](const std::string& id) -> std::shared_ptr<const Node> {
        {std::lock_guard lock(mutex); if (auto it = cache.find(id); it != cache.end()) return it->second;}
        std::shared_ptr<const Node> node;
        try { node = loadPatchNode({id, review.settings.patchFolder / id.substr(6), {}, {}}, review.settings); }
        catch (const std::exception&) { node = nullptr; }
        std::lock_guard lock(mutex); cache[id] = node; return node;
    };
    size_t fibers = 0;
    for (const auto& [id, fiber] : review.fibers) {
        if (fiber.proposals.empty() || fiber.base.nodes.empty()) continue;
        const auto start = std::chrono::steady_clock::now();
        const auto& saved = *fiber.base.nodes.at(id).source;
        Fiber source; source.id = id; source.path = saved.path; source.direction = saved.direction; source.line = saved.line; source.controls = saved.controls;
        const auto tick = std::chrono::steady_clock::now();
        const auto strip = makeFiberNode(source, review.settings, sampler);
        const double stripSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tick).count();
        const auto result = autoLinkFiber(*strip, fiber.proposals, load, settings);
        Json row = {{"fiber", id}, {"anchors", result.anchors}, {"linked", result.linked}, {"candidates", Json::array()}, {"strip_seconds", stripSeconds}};
        for (const auto& c : result.candidates) {
            const auto decision = fiber.decisions.find(c.id);
            row["candidates"].push_back({{"patch", c.id}, {"contact", c.contact}, {"strip_same", c.stripSame}, {"strip_adjacent", c.stripAdjacent},
                {"anchor", c.anchor}, {"linked", c.linked}, {"round", c.round}, {"linked_same", c.linkedSame}, {"linked_adjacent", c.linkedAdjacent},
                {"status", decision == fiber.decisions.end() ? "undecided" : decision->second.status == ReviewStatus::Accepted ? "accepted" : decision->second.status == ReviewStatus::Skipped ? "skipped" : "rejected"}});
        }
        row["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        output << row.dump() << '\n'; output.flush();
        std::cout << "fiber=" << id << " candidates=" << result.candidates.size() << " anchors=" << result.anchors << " linked=" << result.linked
                  << " seconds=" << row["seconds"] << std::endl;
        ++fibers;
        cache.clear();
    }
    std::cout << "fibers=" << fibers << std::endl;
    return 0;
}
