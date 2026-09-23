// Experimental batch consumer of the production FiberReview fitter.
// Build using prepare_fiber_review_benchmark.py; see prototype_fiber_merge.md.
#include "vc/constraints/ConstraintGraph.hpp"
#include "vc/constraints/FiberReview.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "../../../core/src/constraints/GraphInternal.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>

namespace cg = vc::constraints;
namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {
struct Run { size_t first, last; std::set<std::string> patches; };

// Each dense sample represents its Voronoi interval along the fiber. Only
// adjacent occupied samples join; an uncovered sample always separates runs.
// Preserve disjoint runs of the SAME patch (do not replace them by min/max).
std::vector<Run> coverageRuns(const std::vector<std::set<std::string>>& samples) {
    std::vector<Run> runs;
    for (size_t i = 0; i < samples.size(); ++i) {
        if (samples[i].empty()) continue;
        if (runs.empty() || runs.back().last + 1 != i) runs.push_back({i, i, {}});
        runs.back().last = i;
        runs.back().patches.insert(samples[i].begin(), samples[i].end());
    }
    return runs;
}
std::vector<std::set<std::string>> overlapGroups(const std::set<std::string>& patches,
                                              const std::vector<cg::Link>& contacts) {
    std::map<std::string, std::set<std::string>> neighbors;
    for (const auto& link : contacts) {
        neighbors[link.a.node].insert(link.b.node);
        neighbors[link.b.node].insert(link.a.node);
    }
    auto remaining = patches;
    std::vector<std::set<std::string>> groups;
    while (!remaining.empty()) {
        std::vector<std::string> queue{*remaining.begin()}; remaining.erase(queue[0]);
        std::set<std::string> group{queue[0]};
        for (size_t i = 0; i < queue.size(); ++i) for (const auto& next : neighbors[queue[i]])
            if (remaining.erase(next)) { group.insert(next); queue.push_back(next); }
        groups.push_back(std::move(group));
    }
    return groups;
}
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void selfTest() {
    const auto runs = coverageRuns({{}, {"a"}, {"a", "b"}, {"b"}, {}, {"a"}, {}});
    require(runs.size() == 2, "uncovered samples must split coverage");
    require(runs[0].first == 1 && runs[0].last == 3 && runs[0].patches.size() == 2,
            "overlapping and adjacent sample coverage must stay together");
    require(runs[1].patches == std::set<std::string>{"a"}, "preserve repeated patch runs");
    require(coverageRuns({{}, {}}).empty(), "empty coverage");
    require(coverageRuns({{"a"}, {"b"}}).size() == 1, "touching support intervals");
    cg::Link ab{{"a"}, {"b"}}, bc{{"b"}, {"c"}};
    const auto groups = overlapGroups({"a", "b", "c", "d"}, {ab, bc});
    require(groups.size() == 2 && groups[0].size() == 3 && groups[1].contains("d"), "physical overlap chains stay together");
    require(overlapGroups({"a", "b"}, {}).size() == 2, "same fiber alone is not physical overlap");
    std::cout << "Coverage regression checks passed\n";
}
Json read(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot read " + path.string());
    return Json::parse(in);
}
void write(const fs::path& path, const Json& value) {
    std::ofstream out(path); out << value.dump(2) << '\n'; out.close();
    if (!out) throw std::runtime_error("Cannot write " + path.string());
}
double overlapRms(const cg::Graph& graph, const cg::GraphEdit& edit) {
    double sum = 0;
    for (const auto& target : edit.fixedTargets) {
        const auto delta = cg::placedLocation(graph, target.point) - target.uv;
        sum += delta.dot(delta);
    }
    return edit.fixedTargets.empty() ? 0 : std::sqrt(sum / edit.fixedTargets.size());
}
Json exportResult(const cg::FitResult& result, const fs::path& root, const fs::path& relative) {
    Json info = {{"valid", result.valid}, {"message", result.message}};
    if (!result.valid) return info;
    cg::exportSurface(result, root / relative);
    auto meta = read(root / relative / "meta.json");
    meta["constraint_graph"]["includes_fiber_ribbons"] = std::any_of(result.graph.nodes.begin(),result.graph.nodes.end(),[](const auto& p){return p.second.source->fiber;});
    meta["prototype_fiber_merge"] = true;
    write(root / relative / "meta.json", meta);
    info["path"] = relative.string();
    info["origin"] = {result.origin[0], result.origin[1]};
    info["spacing"] = result.graph.settings.spacing;
    info["stretch_p95"] = result.stretchP95;
    return info;
}
size_t fiberSamples(const cg::FitResult& result) {
    size_t count=0;
    const auto& grid=*result.surface->rawPointsPtr();
    for(const auto& [id,p]:result.graph.nodes)if(p.source->fiber)for(size_t i=0;i<p.source->line.size();++i) {
        const auto at=(cg::placedLocation(result.graph,cg::fiberLineLocation(*p.source,i))-result.origin)/result.graph.settings.spacing;
        const int x=int(std::floor(at[0])),y=int(std::floor(at[1]));
        if(x>=0&&y>=0&&x+1<grid.cols&&y+1<grid.rows&&grid(y,x)[0]!=-1&&grid(y+1,x)[0]!=-1&&grid(y,x+1)[0]!=-1&&grid(y+1,x+1)[0]!=-1)++count;
    }
    return count;
}
Json collisionDiagnostics(const cg::FitResult& result) {
    Json pairs=Json::object();
    for(const auto& hit:result.projectionCollisions) {
        const auto delta=hit.xyzB-hit.xyzA;
        const double distance=cv::norm(delta);
        const double normalDistance=std::max(std::abs(delta.dot(hit.normalA)),std::abs(delta.dot(hit.normalB)));
        const double normalCosine=std::abs(hit.normalA.dot(hit.normalB));
        const double aa=hit.tangentU.dot(hit.tangentU),bb=hit.tangentU.dot(hit.tangentV),cc=hit.tangentV.dot(hit.tangentV);
        const double determinant=aa*cc-bb*bb;
        cv::Vec2d offset{0,0};double tangentResidual=distance;
        if(determinant>1e-12) {
            const double u=hit.tangentU.dot(delta),v=hit.tangentV.dot(delta);
            offset={(cc*u-bb*v)/determinant,(aa*v-bb*u)/determinant};
            tangentResidual=cv::norm(delta-hit.tangentU*offset[0]-hit.tangentV*offset[1]);
        }
        auto key=hit.a.node+" | "+hit.b.node;
        if(!pairs.contains(key))pairs[key]={{"count",0},{"max_distance",0.},{"max_normal_distance",0.},
            {"max_tangent_residual",0.},{"min_normal_cosine",1.},{"first_uv",{hit.uv[0],hit.uv[1]}},
            {"first_offset",{offset[0],offset[1]}}};
        auto& row=pairs[key];row["count"]=row["count"].get<size_t>()+1;
        row["max_distance"]=std::max(row["max_distance"].get<double>(),distance);
        row["max_normal_distance"]=std::max(row["max_normal_distance"].get<double>(),normalDistance);
        row["max_tangent_residual"]=std::max(row["max_tangent_residual"].get<double>(),tangentResidual);
        row["min_normal_cosine"]=std::min(row["min_normal_cosine"].get<double>(),normalCosine);
    }
    return {{"samples",result.projectionCollisions.size()},{"pairs",pairs}};
}
Json exportMergedResult(const cg::FitResult& result,const fs::path& root,const fs::path& relative,bool continuous=false) {
    if(!result.valid) {
        Json info={{"valid",false},{"message",result.message}};
        if(!result.projectionCollisions.empty())info["collision_diagnostics"]=collisionDiagnostics(result);
        return info;
    }
    const auto all=cg::splitSurfaceComponents(result,1);
    auto parts=cg::splitSurfaceComponents(result,2);
    Json info={{"valid",!parts.empty()},{"message",parts.empty()?"Only isolated single quads remain":result.message},
        {"components",Json::array()},{"dropped_single_quad_components",all.size()-parts.size()}};
    if(continuous&&!parts.empty()) {
        const auto selected=std::max_element(parts.begin(),parts.end(),[](const auto& a,const auto& b){return fiberSamples(a)<fiberSamples(b);});
        const auto count=fiberSamples(*selected);
        if(!count)return {{"valid",false},{"message","No multi-quad component contains the fiber"}};
        info["fiber_samples_on_component"]=count;
        info["dropped_other_components"]=parts.size()-1;
        auto component=*selected;parts.clear();parts.push_back(std::move(component));
    }
    for(size_t i=0;i<parts.size();++i) {
        const auto path=i?fs::path(relative.string()+"_part_"+std::to_string(i)):relative;
        auto item=exportResult(parts[i],root,path);
        if(!i)for(const auto& [key,value]:item.items())info[key]=value;
        info["components"].push_back(std::move(item));
    }
    info["component_count"]=parts.size();
    return info;
}
}

int main(int argc, char** argv) {
    try {
        double reviewRegularization=1;
        while(argc>=3) {
            const std::string flag=argv[argc-2];
            if(flag=="--shape-weight")reviewRegularization=std::stod(argv[argc-1]);
            else break;
            argc-=2;
        }
        require(std::isfinite(reviewRegularization)&&reviewRegularization>=1,
                "Shape weight must be finite and at least 1");
        const bool continuousOnly=argc>1&&std::string(argv[argc-1])=="--continuous-only";
        const bool fiberOnly=argc>1&&std::string(argv[argc-1])=="--fiber-only";
        const bool withFiber=continuousOnly||fiberOnly||(argc>1&&std::string(argv[argc-1])=="--with-fiber");
        if(withFiber)--argc;
        if (argc == 2 && std::string(argv[1]) == "--self-test") { selfTest(); return 0; }
        if(argc==5&&std::string(argv[1])=="--fuse") {
            const fs::path source=argv[2],output=argv[4];
            require(!fs::exists(output),"Output must be a new directory");
            const auto review=cg::loadReview(source/"review.json");
            const auto report=read(source/"report.json");
            const auto& fiber=review.fibers.at(review.selected);
            cg::Graph graph;graph.settings=review.settings;graph.settings.spacing=report.at("output_spacing");
            std::set<std::string> members;
            if(std::string(argv[3])=="all")for(const auto& id:report.at("assigned"))members.insert(id.get<std::string>());
            else for(const auto& group:report.at("groups"))if(group.at("id")==std::stoul(argv[3]))
                for(const auto& id:group.at("patches"))members.insert(id.get<std::string>());
            require(!members.empty(),"No matching patch group");
            for(const auto& key:members) {
                graph.nodes.emplace(key,fiber.decisions.at(key).alignment.nodes.at(key));
            }
            graph.seed=graph.nodes.begin()->first;
            for(const auto& [id,p]:graph.nodes) {
                const auto links=cg::detail::patchOverlapContacts(*p.source,graph.nodes,2*graph.settings.radius,nullptr,true);
                graph.links.insert(graph.links.end(),links.begin(),links.end());
            }
            if(withFiber) {
                graph.nodes.insert(fiber.base.nodes.begin(),fiber.base.nodes.end());graph.seed=fiber.base.seed;
                for(const auto& key:members)for(const auto& link:fiber.decisions.at(key).alignment.links.read())
                    if(graph.nodes.contains(link.a.node)&&graph.nodes.contains(link.b.node))graph.links.push_back(link);
            }
            fs::create_directories(output);
            const auto result=cg::renderMergedSurface(std::move(graph));
            const auto summary=exportMergedResult(result,output,"merged",withFiber);write(output/"report.json",summary);
            std::cout<<summary.dump(2)<<std::endl;return summary.at("valid").get<bool>()?0:1;
        }
        if (argc < 4 || argc > 6) {
            std::cerr << "Usage: prototype_fiber_merge LINKS.json FIBER_BASENAME NEW_OUTPUT_DIR [OUTPUT_SPACING=10] [REFIT_SWEEPS=1] [--with-fiber|--continuous-only|--fiber-only] [--shape-weight N]\n"
                << "       prototype_fiber_merge --fuse EXISTING_OUTPUT GROUP_INDEX|all NEW_OUTPUT_DIR [--with-fiber]\n";
            return 2;
        }
        const auto input = read(argv[1]);
        require(input.at("type") == "vc_fiber_patch_links" && input.at("version") == 1, "Expected vc_fiber_patch_links v1");
        const std::string basename = argv[2], fid = "fiber:" + basename;
        const auto& requested = input.at("fibers").at(basename);
        const double outputSpacing = argc > 4 ? std::stod(argv[4]) : 10;
        const int sweeps = argc > 5 ? std::stoi(argv[5]) : 1;
        require(std::isfinite(outputSpacing) && outputSpacing > 0 && sweeps >= 0, "Invalid spacing/sweeps");
        const fs::path output = fs::absolute(argv[3]);
        require(!fs::exists(output), "Output must be a new directory");
        const auto settings = cg::settingsFromJson(input.at("settings"));
        // Load production catalog/scales once. Limit discovery to explicit assignments.
        auto catalog = cg::loadCatalog(settings, nullptr, [](const auto& m) { std::cout << m << std::endl; });
        std::set<std::string> assigned;
        for (const auto& row : requested.at("linked")) assigned.insert("patch:" + row.at("patch").get<std::string>());
        if(!fiberOnly)require(!assigned.empty(), "Fiber has no assigned patches");
        std::set<std::string> excluded;
        size_t present = 0;
        for (const auto& p : catalog.patches) {
            if (assigned.contains(p.id)) ++present; else excluded.insert(p.id);
        }
        require(present == assigned.size(), "Some assigned patches are missing from the catalog");
        vc::lasagna::LasagnaDatasetOpenOptions open; open.workingToBaseScale = settings.normalScale;
        const auto dataset = vc::lasagna::LasagnaDataset::openLocation(settings.normalManifest, open);
        vc::lasagna::LasagnaNormalSampler normals(dataset);
        const auto fiber = cg::makeFiberNode(catalog.fibers.at(fid), settings, normals);
        cg::ReviewFiber review; review.path = catalog.fibers.at(fid).path; review.direction = fiber->direction;
        review.base.settings = settings; review.base.seed = fid;
        review.base.nodes.emplace(fid, cg::Placement{fiber, fiber->initialUv});
        const auto base = cg::renderPlacement(review.base);
        require(base.valid, base.message.c_str());
        if(fiberOnly) {
            require(requested.at("direction")=="H","Fiber-only fallback is restricted to horizontal fibers");
            fs::create_directories(output);
            auto graph=review.base;graph.settings.spacing=outputSpacing;
            Json report={{"type","prototype_fiber_merge"},{"version",1},{"fiber",basename},
                {"fiber_only",true},{"source_fiber",review.path.string()},
                {"output_spacing",outputSpacing},{"complete",true}};
            auto info=exportMergedResult(cg::renderMergedSurface(graph),output,fs::path("continuous")/"refined",true);
            if(info.value("valid",false)) {
                auto meta=read(output/info.at("path").get<std::string>()/"meta.json");
                meta["constraint_graph"]["fiber_only"] = true;
                write(output/info.at("path").get<std::string>()/"meta.json",meta);
            }
            report["continuous"]["refined"]=std::move(info);
            write(output/"report.json",report);
            return report["continuous"]["refined"].at("valid").get<bool>()?0:1;
        }
        const auto prepared = cg::prepareGraph(base);
        review.proposals = cg::discover(catalog, review.base, fid, nullptr, {}, {}, 1, excluded, cg::ContactPolicy::FiberReview);
        cg::orderReviewProposals(review);
        review.discovered = true;
        // Keep every assignment. Missing contact eligibility is reported, never
        // treated as permission to discard a patch or invent a new component.
        std::set<std::string> discovered;
        for (const auto& p : review.proposals) discovered.insert(p.id);
        Json report = {{"type", "prototype_fiber_merge"}, {"version", 1},
            {"links", fs::absolute(argv[1]).string()}, {"fiber", basename}, {"settings", input.at("settings")},
            {"output_spacing", outputSpacing}, {"refit_sweeps", sweeps}, {"review_shape_weight",reviewRegularization}, {"assigned", assigned}, {"complete",false}, {"continuous_only",continuousOnly},
            {"fits", Json::array()}, {"groups", Json::array()}, {"unlocated", Json::array()}};
        for (const auto& id : assigned) if (!discovered.contains(id)) report["unlocated"].push_back(id);
        std::vector<std::set<std::string>> coverage(fiber->line.size());
        std::map<std::string, cg::GraphEdit> proposals;
        // Measure coverage before fitting, using all verified contacts at the
        // same 2*radius as review registration. No rigid-fit eligibility gate.
        for (const auto& candidate : review.proposals) {
            auto edit = cg::proposeEdit(catalog, prepared, candidate, false, normals, nullptr, cg::ContactPolicy::FiberReview);
            for (const auto& link : edit.addedLinks) {
                const auto& loc = link.a.node == fid ? link.a : link.b;
                if (loc.node == fid && loc.lineIndex >= 0) coverage.at(size_t(loc.lineIndex)).insert(candidate.id);
            }
            proposals.emplace(candidate.id, std::move(edit));
        }
        const auto contactRuns = coverageRuns(coverage);
        require(!contactRuns.empty(), "No coverage on the dense fiber trace");
        cg::Graph sources; sources.settings = settings;
        for (const auto& [id, edit] : proposals) sources.nodes.emplace(id, edit.addedNodes.at(id));
        std::vector<cg::Link> overlapContacts;
        for (const auto& [id, placement] : sources.nodes) {
            const auto links = cg::detail::patchOverlapContacts(*placement.source, sources.nodes, 2*settings.radius, nullptr, true);
            // Retain either search direction: boundary grids can be asymmetric.
            overlapContacts.insert(overlapContacts.end(), links.begin(), links.end());
        }
        std::vector<Run> runs;
        for (const auto& members : overlapGroups(discovered, overlapContacts)) {
            Run run{coverage.size(), 0, members};
            for (size_t i = 0; i < coverage.size(); ++i) for (const auto& id : coverage[i]) if (members.contains(id)) {
                run.first = std::min(run.first, i); run.last = std::max(run.last, i);
            }
            runs.push_back(std::move(run));
        }
        std::sort(runs.begin(), runs.end(), [](const auto& a, const auto& b) { return std::tie(a.first,a.patches)<std::tie(b.first,b.patches); });
        report["grouping"] = "connected original-XYZ patch overlaps on this assigned fiber; radius=2*review radius, abs(normal dot)>=0.7; no fit-quality or area gates";
        report["contact_runs"] = Json::array();
        for (const auto& run : contactRuns) report["contact_runs"].push_back({{"first", run.first}, {"last", run.last}, {"patches", run.patches}});
        report["overlap_contacts"] = overlapContacts.size();
        fs::create_directories(output);
        std::map<std::string, cg::Graph> initial;
        for (int sweep = 0; sweep <= sweeps; ++sweep) {
            for (const auto& candidate : review.proposals) {
                auto edit = proposals.at(candidate.id);
                const auto prior = review.decisions.find(candidate.id);
                if (prior != review.decisions.end())
                    edit = cg::graphEdit(prepared, prior->second.alignment);
                cg::addReviewOverlapTargets(edit, review);
                Json row = {{"patch", candidate.id}, {"sweep", sweep},
                    {"fiber_contacts", edit.addedLinks.size()}, {"overlap_targets", edit.fixedTargets.size()}};
                if (prior != review.decisions.end()) row["overlap_rms_before"] = overlapRms(prior->second.alignment, edit);
                const auto fitted = cg::fitEdit(edit, nullptr, {}, cg::FitPolicy::FiberReview, reviewRegularization);
                row["valid"] = fitted.valid; row["message"] = fitted.message;
                require(fitted.graph.nodes.at(fid).uv == review.base.nodes.at(fid).uv, "Fit moved the fixed fiber");
                if (fitted.valid) {
                    row["overlap_rms_after"] = overlapRms(fitted.graph, edit);
                    row["attachment_rms"] = fitted.correspondenceRms;
                    cg::ReviewDecision d; d.status = cg::ReviewStatus::Accepted; d.alignment = fitted.graph;
                    d.order = review.decisions.size() + 1; review.decisions[candidate.id] = std::move(d);
                    if (!sweep&&!continuousOnly) initial[candidate.id] = fitted.graph;
                }
                row["retained_previous_fit"] = !fitted.valid && prior != review.decisions.end();
                report["fits"].push_back(row);
                std::cout << "sweep=" << sweep << " patch=" << candidate.id << " valid=" << fitted.valid << std::endl;
                write(output / "report.json", report);
            }
        }
        if(!continuousOnly) {
        report["layers"] = Json::object();
        report["initial_layers"] = Json::object();
        for (const auto& [id, decision] : review.decisions) {
            auto graph = decision.alignment; graph.settings.spacing = outputSpacing;
            report["layers"][id] = exportMergedResult(cg::renderSourceLayer(graph, false), output, fs::path("layers") / id.substr(6));
            if (initial.contains(id)) {
                graph = initial.at(id); graph.settings.spacing = outputSpacing;
                report["initial_layers"][id] = exportMergedResult(cg::renderSourceLayer(graph, false), output, fs::path("initial_layers") / id.substr(6));
            }
        }
        const int axis = fiber->direction == 'V' ? 1 : 0;
        const auto uvAt = [&](size_t i) { return cg::placedLocation(review.base, cg::fiberLineLocation(*fiber, i))[axis]; };
        report["fiber_axis"] = axis;
        report["fiber_uv"] = Json::array();
        for (size_t i = 0; i < fiber->line.size(); ++i) {
            const auto uv = cg::placedLocation(review.base, cg::fiberLineLocation(*fiber, i));
            report["fiber_uv"].push_back({uv[0], uv[1]});
        }
        for (size_t k = 0; k < runs.size(); ++k) {
            const auto& run = runs[k];
            Json group = {{"id", k}, {"patches", run.patches}, {"first_sample", run.first}, {"last_sample", run.last},
                {"arclength", {fiber->lineArclength[run.first], fiber->lineArclength[run.last]}},
                {"coverage_uv", {uvAt(run.first), uvAt(run.last)}}, {"missing_fits", Json::array()}};
            for (const auto& id : run.patches) if (!review.decisions.contains(id)) group["missing_fits"].push_back(id);
            // Unknown assignments may bridge known coverage. Do not silently
            // publish apparently complete groups while those remain unresolved.
            if (!group["missing_fits"].empty() || !report["unlocated"].empty()) {
                group["status"] = "blocked_by_missing_assignment_or_fit";
            } else {
                for (const std::string phase : {"initial", "refined"}) {
                    if (phase == "initial" && std::any_of(run.patches.begin(), run.patches.end(), [&](const auto& id) { return !initial.contains(id); })) {
                        group[phase] = {{"valid", false}, {"message", "Some patches had no valid first-pass fit"}};
                        continue;
                    }
                    cg::Graph graph; graph.settings = settings; graph.settings.spacing = outputSpacing;
                    for (const auto& id : run.patches) {
                        const auto& source = phase == "initial" ? initial.at(id) : review.decisions.at(id).alignment;
                        graph.nodes.emplace(id, source.nodes.at(id));
                    }
                    graph.seed = *run.patches.begin();
                    // Reuse review's real-XYZ matcher for explicit pair links.
                    // This authorizes the shared raster only for actual overlaps.
                    for (const auto& link : overlapContacts)
                        if (run.patches.contains(link.a.node) && run.patches.contains(link.b.node)) graph.links.push_back(link);
                    // Preserve fitted UV. The shared renderer regularizes XYZ;
                    // it never reruns the full graph chart fit or adds ribbons.
                    auto result = cg::renderMergedSurface(graph);
                    group[phase] = exportMergedResult(result, output, fs::path("groups") / ("group_" + std::to_string(k) + "_" + phase));
                }
                group["status"] = group["refined"]["valid"].get<bool>() ? "exported_prototype" : "fusion_failed";
            }
            report["groups"].push_back(group);
            std::cout << "group=" << k << " patches=" << run.patches.size() << " status=" << group["status"] << std::endl;
            write(output / "report.json", report);
        }
        }
        if(withFiber&&report["unlocated"].empty()&&review.decisions.size()==assigned.size()) {
            for(const std::string phase:continuousOnly?std::vector<std::string>{"refined"}:std::vector<std::string>{"initial","refined"}) {
                if(phase=="initial"&&initial.size()!=assigned.size()) {
                    report["continuous"][phase]={{"valid",false},{"message","Some patches had no valid first-pass fit"}};
                    continue;
                }
                auto graph=review.base;graph.settings.spacing=outputSpacing;
                for(const auto& id:assigned) {
                    const auto& alignment=phase=="initial"?initial.at(id):review.decisions.at(id).alignment;
                    graph.nodes.emplace(id,alignment.nodes.at(id));
                    for(const auto& link:alignment.links.read())if(link.a.node==fid||link.b.node==fid)graph.links.push_back(link);
                }
                graph.links.insert(graph.links.end(),overlapContacts.begin(),overlapContacts.end());
                report["continuous"][phase]=exportMergedResult(cg::renderMergedSurface(graph),output,fs::path("continuous")/phase,true);
            }
            write(output/"report.json",report);
        }
        if(continuousOnly&&!report.contains("continuous"))report["continuous"]["refined"]={{"valid",false},{"message","Some assigned patches could not be located or fitted"}};
        cg::FiberReview workspace; workspace.settings = settings; workspace.selected = fid; workspace.fibers.emplace(fid, review);
        workspace.nextOrder = review.decisions.size()+1; cg::saveReview(workspace, output / "review.json");
        report["complete"]=true;write(output/"report.json",report);
        std::cout << "Review artifacts: " << output << std::endl;
        return continuousOnly&&!report["continuous"]["refined"]["valid"].get<bool>()?1:0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
