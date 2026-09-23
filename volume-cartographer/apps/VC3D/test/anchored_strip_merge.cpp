// Extend an existing fiber merge with graph patches using the shared review fit.
// The caller must verify overlap with measured source patches before invoking this.
#include "vc/constraints/ConstraintGraph.hpp"
#include "vc/constraints/FiberReview.hpp"
#include "vc/lasagna/LineModel.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "../../../core/src/constraints/GraphInternal.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>

namespace cg = vc::constraints;
namespace fs = std::filesystem;

namespace {

struct UnusedNormals final : vc::lasagna::NormalSampler {
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override {
        throw std::logic_error("Patch-only proposal unexpectedly requested normal sampling");
    }
};

cg::Graph acceptedGraph(const cg::ReviewFiber& fiber, double outputSpacing) {
    cg::Graph graph = fiber.base;
    graph.settings.spacing = outputSpacing;
    const std::string fid = graph.seed;
    for (const auto& [id, decision] : fiber.decisions) {
        if (!cg::fitted(decision)) continue;
        const auto& alignment = decision.alignment;
        graph.nodes.emplace(id, alignment.nodes.at(id));
        for (const auto& link : alignment.links.read()) {
            if (link.a.node == fid || link.b.node == fid) graph.links.push_back(link);
        }
    }
    for (const auto& [id, placement] : graph.nodes) {
        if (placement.source->fiber) continue;
        const auto links = cg::detail::patchOverlapContacts(
            *placement.source, graph.nodes, 2 * graph.settings.radius, nullptr, true);
        graph.links.insert(graph.links.end(), links.begin(), links.end());
    }
    return graph;
}

size_t validQuads(const cg::FitResult& part) {
    const auto& grid = *part.surface->rawPointsPtr();
    size_t count = 0;
    for (int row = 0; row + 1 < grid.rows; ++row)
        for (int col = 0; col + 1 < grid.cols; ++col)
            count += grid(row, col)[0] != -1 && grid(row + 1, col)[0] != -1
                && grid(row, col + 1)[0] != -1 && grid(row + 1, col + 1)[0] != -1;
    return count;
}

std::pair<size_t, size_t> retainedComponent(const std::vector<cg::FitResult>& parts) {
    size_t best = 0, bestQuads = 0, droppedQuads = 0;
    for (size_t index = 0; index < parts.size(); ++index) {
        const size_t quads = validQuads(parts[index]);
        droppedQuads += quads;
        if (quads > bestQuads) { best = index; bestQuads = quads; }
    }
    if (!bestQuads) throw std::runtime_error("Merged surface has no complete quads");
    droppedQuads -= bestQuads;
    if (droppedQuads > 50) throw std::runtime_error("Merge detached more than 50 quads");
    return {best, droppedQuads};
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5) {
            std::cerr << "Usage: anchored_strip_merge REVIEW.json OUTPUT.tifxyz OUTPUT_SPACING CANDIDATE.tifxyz...\n";
            return 2;
        }
        const fs::path reviewPath = argv[1], outputPath = argv[2];
        if (fs::exists(outputPath)) throw std::runtime_error("Output already exists");
        const double spacing = std::stod(argv[3]);
        if (!std::isfinite(spacing) || spacing <= 0) throw std::runtime_error("Invalid output spacing");
        const cg::FiberReview review = cg::loadReview(reviewPath);
        const cg::ReviewFiber& fiber = review.fibers.at(review.selected);
        cg::Graph graph = acceptedGraph(fiber, spacing);
        std::map<std::string, cg::Placement> references;
        for (const auto& [id, placement] : graph.nodes)
            if (fiber.decisions.contains(id) && cg::fitted(fiber.decisions.at(id)))
                references.emplace(id, placement);
        std::map<std::string, std::shared_ptr<const cg::PreparedGraph>> preparedReferences;
        std::map<std::string, int> candidateRanks;
        for (int arg = 4; arg < argc; ++arg)
            candidateRanks.emplace("patch:" + fs::path(argv[arg]).filename().string(), arg - 4);
        nlohmann::json report = {{"accepted", nlohmann::json::array()},
                                 {"rejected", nlohmann::json::array()}};
        for (int arg = 4; arg < argc; ++arg) {
            const fs::path candidatePath = argv[arg];
            const auto started = std::chrono::steady_clock::now();
            try {
                cg::PatchEntry entry;
                entry.id = "patch:" + candidatePath.filename().string();
                entry.path = candidatePath;
                if (graph.nodes.contains(entry.id)) throw std::runtime_error("Candidate is already in strip");
                const auto node = cg::loadPatchNode(entry, graph.settings);
                cg::Candidate candidate;
                candidate.id = entry.id;
                candidate.contacts = cg::detail::patchOverlapContacts(
                    *node, references, 2 * graph.settings.radius, nullptr, true);
                if (candidate.contacts.size() < 32)
                    throw std::runtime_error("Too few measured patch contacts");
                std::map<std::string, size_t> contactCounts;
                for (const auto& link : candidate.contacts) ++contactCounts[link.a.node];
                const auto strongest = std::max_element(contactCounts.begin(), contactCounts.end(),
                    [](const auto& a, const auto& b) {
                        return a.second < b.second || (a.second == b.second && a.first > b.first);
                    });
                if (strongest == contactCounts.end() || strongest->second < 32)
                    throw std::runtime_error("No measured source has 32 contacts");
                auto prepared = preparedReferences.find(strongest->first);
                if (prepared == preparedReferences.end()) {
                    cg::Graph reference = fiber.decisions.at(strongest->first).alignment;
                    reference.settings.spacing = spacing;
                    const auto base = cg::renderMergedSurface(reference);
                    if (!base.valid) throw std::runtime_error("Reference render: " + base.message);
                    prepared = preparedReferences.emplace(
                        strongest->first, cg::prepareGraph(base)).first;
                }
                candidate.contacts.erase(std::remove_if(candidate.contacts.begin(), candidate.contacts.end(),
                    [&](const cg::Link& link) { return link.a.node != strongest->first; }), candidate.contacts.end());
                cg::Catalog catalog;
                catalog.patches.push_back(entry);
                UnusedNormals normals;
                auto edit = cg::proposeEdit(catalog, prepared->second, candidate, false, normals,
                                            nullptr, cg::ContactPolicy::FiberReview);
                cg::addReviewOverlapTargets(edit, fiber);
                if (edit.fixedTargets.size() < 32)
                    throw std::runtime_error("Too few review overlap targets");
                const auto fitted = cg::fitEdit(edit, nullptr, {}, cg::FitPolicy::FiberReview);
                if (!fitted.valid) throw std::runtime_error(fitted.message);
                cg::Graph proposed = graph;
                proposed.nodes.emplace(entry.id, fitted.graph.nodes.at(entry.id));
                for (const auto& link : edit.addedLinks)
                    if (link.a.node == proposed.seed || link.b.node == proposed.seed)
                        proposed.links.push_back(link);
                const auto overlaps = cg::detail::patchOverlapContacts(
                    *node, proposed.nodes, 2 * proposed.settings.radius, nullptr, true);
                proposed.links.insert(proposed.links.end(), overlaps.begin(), overlaps.end());
                report["accepted"].push_back({{"patch", candidatePath.filename().string()},
                                              {"measured_contacts", candidate.contacts.size()},
                                              {"review_targets", edit.fixedTargets.size()},
                                              {"fit_rms", fitted.correspondenceRms}});
                graph = std::move(proposed);
            } catch (const std::exception& error) {
                report["rejected"].push_back({{"patch", candidatePath.filename().string()},
                                              {"reason", error.what()}});
            }
            if (std::getenv("VC3D_ANCHORED_PROGRESS")) {
                const auto seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                std::clog << "Candidate " << arg - 3 << "/" << argc - 4 << " "
                          << candidatePath.filename().string() << " " << seconds << " s\n";
            }
        }
        if (!report["accepted"].empty()) {
            cg::FitResult merged;
            while (true) {
                if (std::getenv("VC3D_ANCHORED_PROGRESS")) std::clog << "Rendering combined strip\n";
                merged = cg::renderMergedSurface(graph);
                if (merged.valid) break;
                std::map<std::string, size_t> conflicts;
                for (const auto& collision : merged.projectionCollisions) {
                    for (const auto& id : {collision.a.node, collision.b.node})
                        if (candidateRanks.contains(id) && graph.nodes.contains(id)) ++conflicts[id];
                }
                if (conflicts.empty()) throw std::runtime_error("Merged render: " + merged.message);
                const auto victim = std::max_element(conflicts.begin(), conflicts.end(),
                    [&](const auto& a, const auto& b) {
                        return a.second < b.second || (a.second == b.second
                            && candidateRanks.at(a.first) < candidateRanks.at(b.first));
                    })->first;
                graph.nodes.erase(victim);
                graph.links.erase(std::remove_if(graph.links.begin(), graph.links.end(),
                    [&](const cg::Link& link) {
                        return link.a.node == victim || link.b.node == victim;
                    }), graph.links.end());
                const std::string patch = victim.substr(std::string("patch:").size());
                auto& accepted = report["accepted"];
                accepted.erase(std::remove_if(accepted.begin(), accepted.end(),
                    [&](const nlohmann::json& row) { return row.at("patch") == patch; }), accepted.end());
                report["rejected"].push_back({{"patch", patch},
                                              {"reason", "Projection collision in combined strip"},
                                              {"collision_samples", conflicts.at(victim)}});
                if (accepted.empty()) break;
            }
            if (report["accepted"].empty()) {
                report["exported"] = nullptr;
                std::cout << report.dump(2) << '\n';
                return 0;
            }
            const auto parts = cg::splitSurfaceComponents(merged, 2);
            const auto [best, dropped] = retainedComponent(parts);
            cg::exportSurface(parts[best], outputPath);
            report["exported"] = outputPath.string();
            report["dropped_quads"] = dropped;
        } else {
            report["exported"] = nullptr;
        }
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "anchored_strip_merge: " << error.what() << '\n';
        return 1;
    }
}
