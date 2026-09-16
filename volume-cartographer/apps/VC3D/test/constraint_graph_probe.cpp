#include "vc/constraints/ConstraintGraph.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "../../../core/src/constraints/GraphFiberJunctions.hpp"
#include <iostream>
#include <chrono>
#include <set>
#include <fstream>
#include <cstdlib>
namespace {
vc::constraints::Catalog benchmarkCatalog(vc::constraints::Settings settings) {
    // Use a local copy of a real catalog without changing the saved graph or
    // invalidating its derived-surface checksum.
    if(const auto* path=std::getenv("VC3D_BENCHMARK_PATCH_FOLDER"))settings.patchFolder=path;
    return vc::constraints::loadCatalog(settings);
}
}
int main(int argc,char** argv){
    if(argc>=3&&(std::string(argv[1])=="--render-benchmark"||std::string(argv[1])=="--fit-benchmark")) {
        try {
            using namespace vc::constraints;
            const auto accepted=loadSession(argv[2]);
            if(!accepted.valid)throw std::runtime_error(accepted.message);
            for(int i=0;i<(argc>3?std::stoi(argv[3]):5);++i) {
                const auto start=std::chrono::steady_clock::now();
                auto result=std::string(argv[1])=="--fit-benchmark"?fit(accepted.graph):renderPlacement(accepted.graph);
                const auto end=std::chrono::steady_clock::now();
                if(!result.valid)throw std::runtime_error(result.message);
                std::cout<<"iteration="<<i<<" elapsed_ms="<<std::chrono::duration<double,std::milli>(end-start).count()
                    <<" rms="<<result.correspondenceRms<<" stretch="<<result.stretchP95<<std::endl;
                if(i==0&&argc>4)saveSession(result,argv[4]);
            }
            return 0;
        }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
    }
    if(argc==3&&std::string(argv[1])=="--junctions") {
        try {
            using namespace vc::constraints;
            const auto result=loadSession(argv[2]);
            if(!result.valid)throw std::runtime_error(result.message);
            const detail::FiberJunctions junctions(result.graph);
            size_t count=0,sampled=0;double beforeSquared=0,afterSquared=0,surfaceSquared=0,surfaceMax=0;
            for(const auto& link:result.graph.links) {
                const auto [h,v]=detail::horizontalVerticalLink(result.graph,link);if(!h)continue;
                const auto& horizontal=*result.graph.nodes.at(h->node).source;
                const auto& vertical=*result.graph.nodes.at(v->node).source;
                const auto hw=detail::weights(horizontal,*h),vw=detail::weights(vertical,*v);
                const auto target=detail::interpolate(horizontal.xyz,hw),source=detail::interpolate(vertical.xyz,vw);
                double column=0;for(int i=0;i<3;++i)column+=(vw.indices[i]%vertical.cols)*vw.values[i];
                const auto before=source-target,after=before+junctions.offset(vertical,column);
                ++count;beforeSquared+=before.dot(before);afterSquared+=after.dot(after);
                const auto sample=result.surface->sampleAtSurface(placedLocation(result.graph,*h));
                if(sample.valid()) {
                    const double error=cv::norm(cv::Vec3d(sample.volume)-target);
                    ++sampled;surfaceSquared+=error*error;surfaceMax=std::max(surfaceMax,error);
                }
            }
            std::cout<<"hv_links="<<count<<" source_xyz_rms="<<std::sqrt(beforeSquared/std::max(size_t(1),count))
                <<" corrected_xyz_rms="<<std::sqrt(afterSquared/std::max(size_t(1),count))
                <<" sampled_links="<<sampled<<" surface_to_horizontal_rms="<<std::sqrt(surfaceSquared/std::max(size_t(1),sampled))
                <<" surface_to_horizontal_max="<<surfaceMax<<std::endl;
            return 0;
        }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
    }
    if(argc>=3&&std::string(argv[1])=="--discovery-benchmark") {
        try {
            using namespace vc::constraints;
            const auto result=loadSession(argv[2]);const auto catalog=benchmarkCatalog(result.graph.settings);
            std::vector<std::string> selections{result.graph.seed};
            for(const auto& [id,node]:result.graph.nodes)if(!node.source->fiber){selections.push_back(id);break;}
            if(argc>4)selections={std::string(argv[4])=="all"?std::string{}:argv[4]};
            std::cout<<"catalog="<<catalog.patches.size()<<" patches, "<<catalog.fibers.size()<<" fibers"<<std::endl;
            const int iterations=argc>3?std::stoi(argv[3]):3;
            const auto strides=argc>6?std::vector<int>{std::stoi(argv[6])}:std::vector<int>{1,4,8};
            for(const auto& selection:selections) {
                std::set<std::string> baseline;
                const bool forward=selection.empty()||result.graph.nodes.at(selection).source->fiber;
                for(int iteration=0;iteration<iterations;++iteration)for(int stride:strides) {
                    size_t tested=0;
                    auto count=[&](size_t n){if(n)tested+=1+(n-1)/stride+((n-1)%stride!=0);};
                    if(selection.empty()){for(const auto& [id,p]:result.graph.nodes)if(p.source->fiber)count(p.source->line.size());}
                    else if(forward)count(result.graph.nodes.at(selection).source->line.size());
                    else for(const auto& [id,f]:catalog.fibers)if(!result.graph.nodes.contains(id))count(f.line.size());
                    const auto start=std::chrono::steady_clock::now();
                    auto phase=start;std::string phaseName;
                    auto stamp=[&](const std::string& name){
                        if(name.starts_with("Searched "))return;
                        const auto now=std::chrono::steady_clock::now();
                        if(!phaseName.empty())std::cout<<"phase="<<phaseName<<" stride="<<stride<<" ms="<<std::chrono::duration<double,std::milli>(now-phase).count()<<'\n';
                        phase=now;phaseName=name;
                    };
                    double first=-1;
                    auto candidates=discover(catalog,result.graph,selection,nullptr,stamp,[&](const auto&){
                        if(first<0)first=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                    },stride);
                    stamp("Complete");
                    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                    size_t contacts=0;std::set<std::string> ids;
                    for(const auto& c:candidates){contacts+=c.contacts.size();ids.insert(c.id);}
                    if(stride==1)baseline=ids;
                    size_t missed=0;for(const auto& id:baseline)missed+=!ids.contains(id);
                    std::cout<<"direction="<<(forward?"fiber_to_patch":"patch_to_fiber")<<" iteration="<<iteration<<" stride="<<stride<<" ms="<<ms<<" first_ms="<<first<<" tested_points="<<tested<<" proposals="<<candidates.size()<<" contacts="<<contacts<<" missed_ids="<<missed<<" selected="<<selection<<std::endl;
                    if(argc>5&&iteration==0) {
                        nlohmann::json records=nlohmann::json::array();
                        for(const auto& c:candidates){
                            nlohmann::json links=nlohmann::json::array();
                            for(const auto& l:c.contacts)links.push_back({l.a.node,l.a.lineIndex,l.a.grid[0],l.a.grid[1],l.b.node,l.b.lineIndex,l.b.grid[0],l.b.grid[1]});
                            records.push_back({{"id",c.id},{"fiber",c.fiber},{"distance",c.distance},{"contacts",links}});
                        }
                        std::ofstream out(std::string(argv[5])+"."+std::to_string(stride)+".json");out<<records;
                    }
                }
            }
            return 0;
        }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
    }
    if(argc==4&&std::string(argv[1])=="--compare-attachments") {
        try {
            using namespace vc::constraints;
            std::ifstream beforeFile(argv[2]),afterFile(argv[3]);
            const auto before=graphFromJson(nlohmann::json::parse(beforeFile));
            const auto after=graphFromJson(nlohmann::json::parse(afterFile));
            struct Errors {size_t count=0;double beforeSquared=0,afterSquared=0,beforeMax=0,afterMax=0;};
            std::map<std::pair<std::string,std::string>,Errors> errors;
            for(const auto& link:after.links){
                if(link.keypoint||!after.nodes.at(link.a.node).source->fiber||after.nodes.at(link.b.node).source->fiber)continue;
                auto& e=errors[{link.a.node,link.b.node}];++e.count;
                const auto a=placedLocation(before,link.a)-placedLocation(before,link.b);
                const auto b=placedLocation(after,link.a)-placedLocation(after,link.b);
                e.beforeSquared+=a.dot(a);e.afterSquared+=b.dot(b);
                e.beforeMax=std::max(e.beforeMax,cv::norm(a));e.afterMax=std::max(e.afterMax,cv::norm(b));
            }
            for(const auto& [pair,e]:errors)std::cout<<pair.first<<" -> "<<pair.second<<" contacts="<<e.count
                <<" before_rms="<<std::sqrt(e.beforeSquared/e.count)<<" after_rms="<<std::sqrt(e.afterSquared/e.count)
                <<" before_max="<<e.beforeMax<<" after_max="<<e.afterMax<<std::endl;
            return 0;
        }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
    }
    if(argc>=3&&(std::string(argv[1])=="--proposal-benchmark"||std::string(argv[1])=="--local-proposal-benchmark"||std::string(argv[1])=="--local-proposal-scan"||std::string(argv[1])=="--clicked-proposal-benchmark"||std::string(argv[1])=="--proposal-contacts")) {
        try {
            using namespace vc::constraints;
            const auto result=loadSession(argv[2]);
            const auto catalog=benchmarkCatalog(result.graph.settings);
            vc::lasagna::LasagnaDatasetOpenOptions options;options.workingToBaseScale=result.graph.settings.normalScale;
            const auto dataset=vc::lasagna::LasagnaDataset::openLocation(result.graph.settings.normalManifest,options);
            vc::lasagna::LasagnaNormalSampler normals(dataset);
            auto selected=result.graph.seed;
            if(argc>6) {
                selected=argv[6];
                if(selected=="patch")for(const auto& [id,p]:result.graph.nodes)if(!p.source->fiber){selected=id;break;}
            }
            auto candidates=discover(catalog,result.graph,selected,nullptr,{},{},1);
            if(candidates.empty())throw std::runtime_error("No seed proposals");
            if(std::string(argv[1])=="--proposal-contacts"){
                for(size_t i=0;i<std::min(candidates.size(),argc>3?std::stoul(argv[3]):size_t(30));++i){
                    auto draft=propose(catalog,result.graph,candidates[i],true,normals);
                    std::map<std::string,std::pair<size_t,double>> errors;
                    for(const auto& link:draft.links)if(link.b.node==candidates[i].id&&draft.nodes.at(link.a.node).source->fiber){
                        auto& [count,squared]=errors[link.a.node];++count;
                        const auto error=placedLocation(draft,link.a)-placedLocation(draft,link.b);squared+=error.dot(error);
                    }
                    std::cout<<"candidate_index="<<i<<" id="<<candidates[i].id<<" fibers="<<errors.size();
                    for(const auto& [id,values]:errors)std::cout<<" "<<id<<":"<<values.first<<" rms="<<std::sqrt(values.second/values.first);
                    std::cout<<std::endl;
                }
                return 0;
            }
            const bool scan=std::string(argv[1])=="--local-proposal-scan";
            for(int i=0;i<(argc>3?std::stoi(argv[3]):3);++i) {
                const auto candidateIndex=(argc>5?std::stoul(argv[5]):0)+(scan?i:0);
                if(candidateIndex>=candidates.size())break;
                const auto& candidate=candidates[candidateIndex];
                std::cout<<"candidate="<<candidate.id<<" candidate_index="<<candidateIndex<<" nodes="<<result.graph.nodes.size()<<" contacts="<<candidate.contacts.size()<<std::endl;
                std::atomic_bool stop{false};
                auto start=std::chrono::steady_clock::now(),phase=start;
                auto stamp=[&](const std::string& name){auto now=std::chrono::steady_clock::now();std::cout<<name<<" phase_ms="<<std::chrono::duration<double,std::milli>(now-phase).count()<<" total_ms="<<std::chrono::duration<double,std::milli>(now-start).count()<<std::endl;phase=now;if(scan&&name=="Fitting full graph")stop=true;};
                auto draft=propose(catalog,result.graph,candidate,true,normals);
                // Reproduce the old clicked-cluster-only fit for paired
                // real-data comparisons, without changing production code.
                if(std::string(argv[1])=="--clicked-proposal-benchmark"){
                    std::erase_if(draft.links,[&](const auto& link){
                        if(link.b.node!=candidate.id||!draft.nodes.at(link.a.node).source->fiber)return false;
                        return std::none_of(candidate.contacts.begin(),candidate.contacts.end(),[&](const auto& selected){
                            return link.a.node==selected.a.node&&link.a.lineIndex==selected.a.lineIndex&&link.b.grid==selected.b.grid;
                        });
                    });
                }
                stamp("propose");
                auto source=renderProposalSources(draft,result.graph,candidate.id);stamp("source_render");
                auto placement=renderPlacement(draft);stamp("initial_placement");
                std::cout<<"placement_valid="<<placement.valid<<" message="<<placement.message<<std::endl;
                FitResult fitResult;
                try {
                    fitResult=std::string(argv[1])!="--proposal-benchmark"?fitIncremental(std::move(draft),result,&stop,stamp):fit(std::move(draft),nullptr,stamp);
                }catch(const std::exception&){if(!stop)throw;std::cout<<"candidate_index="<<candidateIndex<<" requires_global=1"<<std::endl;continue;}
                stamp("fit_complete");
                std::cout<<"iteration="<<i<<" valid="<<fitResult.valid<<" rms="<<fitResult.correspondenceRms<<" stretch="<<fitResult.stretchP95<<" message="<<fitResult.message<<std::endl;
                if(fitResult.valid){
                    auto fibers=renderSourceLayer(fitResult.graph,true),patches=renderSourceLayer(fitResult.graph,false);
                    std::cout<<"source_layers_valid="<<fibers.valid<<","<<patches.valid<<std::endl;
                    if(!fibers.valid||!patches.valid)throw std::runtime_error("Cannot render independent source layers");
                }
                if(argc>4) {std::ofstream out(std::string(argv[4])+"."+std::to_string(i)+".json");out<<graphToJson(fitResult.graph);}
                if(argc>4&&i==0&&fitResult.valid&&std::string(argv[1])=="--local-proposal-benchmark") {
                    const auto path=std::string(argv[4])+".session.json";saveSession(fitResult,path);auto restored=loadSession(path);
                    if(!restored.valid||restored.origin!=fitResult.origin||restored.surface->rawPointsPtr()->size()!=fitResult.surface->rawPointsPtr()->size()||
                       cv::norm(*restored.surface->rawPointsPtr(),*fitResult.surface->rawPointsPtr(),cv::NORM_INF)!=0)throw std::runtime_error("Incremental surface did not round trip exactly");
                    std::cout<<"surface_roundtrip=exact"<<std::endl;
                }
            }
            return 0;
        }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
    }
    if(argc==4&&std::string(argv[1])=="--audit") {
        try {
            vc::constraints::Settings settings;settings.fiberFolder=argv[2];settings.normalManifest=argv[3];
            const auto catalog=vc::constraints::loadCatalog(settings);
            const auto& audit=catalog.cycleAudit;
            std::cout<<"fibers="<<catalog.fibers.size()<<" links="<<catalog.links.size()<<" components="<<catalog.components.size()<<" input_warnings="<<catalog.warnings.size()<<"\n"<<audit.summary<<"\n";
            for(const auto& detail:audit.details)std::cout<<detail<<"\n";
            for(const auto& warning:catalog.warnings)std::cout<<warning<<"\n";
            return !audit.available||!audit.issues.empty()||!catalog.warnings.empty()?1:0;
        }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
    }
    if(argc==4&&std::string(argv[1])=="--refit") {
        try {
            std::ifstream input(argv[2]);auto graph=vc::constraints::graphFromJson(nlohmann::json::parse(input));
            auto result=vc::constraints::fit(std::move(graph),nullptr,[](const auto& s){std::cout<<s<<std::endl;});
            std::cout<<"valid="<<result.valid<<" rms="<<result.correspondenceRms<<" stretch="<<result.stretchP95<<" message="<<result.message<<std::endl;
            if(!result.valid)return 1;
            vc::constraints::saveSession(result,argv[3]);return 0;
        }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
    }
    if(argc>=3&&std::string(argv[1])=="--grow") {
        try {
            auto result=vc::constraints::loadSession(argv[2]);
            if(!result.valid)throw std::runtime_error(result.message);
            auto progress=[](const std::string& s){std::cout<<s<<std::endl;};
            const auto settings=result.graph.settings;
            auto catalog=vc::constraints::loadCatalog(settings,nullptr,progress);
            vc::lasagna::LasagnaDatasetOpenOptions options;options.workingToBaseScale=settings.normalScale;
            auto dataset=vc::lasagna::LasagnaDataset::openLocation(settings.normalManifest,options);
            vc::lasagna::LasagnaNormalSampler normals(dataset);
            std::string patch;
            for(const auto& [id,p]:result.graph.nodes)if(!p.source->fiber){patch=id;break;}
            if(patch.empty())throw std::runtime_error("Growth validation needs a committed patch");
            auto candidates=vc::constraints::discover(catalog,result.graph,patch,nullptr,progress);
            if(candidates.empty()) {
                const auto alternatives=vc::constraints::discover(catalog,result.graph,result.graph.seed,nullptr,progress);
                for(size_t i=0;i<std::min(size_t(30),alternatives.size());++i) {
                    auto graph=vc::constraints::propose(catalog,result.graph,alternatives[i],true,normals);
                    auto nearby=vc::constraints::discover(catalog,graph,alternatives[i].id,nullptr,progress);
                    if(nearby.empty())continue;
                    auto fitted=vc::constraints::fit(graph,nullptr,progress);
                    if(!fitted.valid)continue;
                    result=std::move(fitted);candidates=std::move(nearby);
                    std::cout<<"bridge patch="<<alternatives[i].id<<std::endl;break;
                }
            }
            bool grew=false;
            std::cout<<"fiber growth proposals="<<candidates.size()<<std::endl;
            for(size_t i=0;i<std::min(size_t(5),candidates.size());++i) {
                auto graph=vc::constraints::propose(catalog,result.graph,candidates[i],true,normals);
                auto fit=vc::constraints::fit(graph,nullptr,progress);
                std::cout<<"fiber growth="<<candidates[i].id<<" added="<<graph.nodes.size()-result.graph.nodes.size()<<" valid="<<fit.valid<<" rms="<<fit.correspondenceRms<<" message="<<fit.message<<std::endl;
                if(fit.valid){result=std::move(fit);grew=true;break;}
            }
            if(!grew)throw std::runtime_error("No valid fiber growth among tested proposals");
            vc::constraints::saveSession(result,std::string(argv[2])+".fiber-grown.json");
            candidates=vc::constraints::discover(catalog,result.graph,result.graph.seed,nullptr,progress);
            bool overlapped=false;
            size_t overlapAttempts=0;
            for(size_t i=0;i<candidates.size()&&overlapAttempts<5;++i) {
                auto graph=vc::constraints::propose(catalog,result.graph,candidates[i],true,normals);
                bool overlap=false;
                for(const auto& l:graph.links)overlap|=!graph.nodes.at(l.a.node).source->fiber&&!graph.nodes.at(l.b.node).source->fiber;
                if(!overlap)continue;
                ++overlapAttempts;
                auto fit=vc::constraints::fit(graph,nullptr,progress);
                std::cout<<"overlapping patch="<<candidates[i].id<<" valid="<<fit.valid<<" rms="<<fit.correspondenceRms<<" message="<<fit.message<<std::endl;
                if(fit.valid){result=std::move(fit);overlapped=true;break;}
            }
            if(!overlapped)throw std::runtime_error("No valid overlapping patch among tested proposals");
            const auto path=std::string(argv[2])+".grown.json";
            vc::constraints::saveSession(result,path);
            auto reopened=vc::constraints::loadSession(path);
            if(!reopened.valid||vc::constraints::graphToJson(reopened.graph)!=vc::constraints::graphToJson(result.graph))throw std::runtime_error("Growth session did not round trip");
            std::cout<<"growth and overlap roundtrip=passed nodes="<<result.graph.nodes.size()<<std::endl;
            return 0;
        }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
    }
    if(argc<5){std::cerr<<"usage: constraint_graph_probe FIBERS PATCHES NORMAL_MANIFEST OUTPUT_SESSION [COMPONENT]\n";return 2;}
    try{
        vc::constraints::Settings settings;settings.fiberFolder=argv[1];settings.patchFolder=argv[2];settings.normalManifest=argv[3];
        auto progress=[](const std::string& s){std::cout<<s<<std::endl;};
        auto began=std::chrono::steady_clock::now();auto catalog=vc::constraints::loadCatalog(settings,nullptr,progress);
        std::cout<<"catalog fibers="<<catalog.fibers.size()<<" patches="<<catalog.patches.size()<<" components="<<catalog.components.size()<<" warnings="<<catalog.warnings.size()<<std::endl;
        for(size_t i=0;i<std::min(size_t(8),catalog.warnings.size());++i)std::cout<<catalog.warnings[i]<<std::endl;
        vc::lasagna::LasagnaDatasetOpenOptions options;options.workingToBaseScale=settings.normalScale;
        auto dataset=vc::lasagna::LasagnaDataset::openLocation(settings.normalManifest,options);vc::lasagna::LasagnaNormalSampler normals(dataset);
        size_t component=argc>5?std::stoul(argv[5]):0;
        std::cout<<"seed component="<<component<<" fibers="<<catalog.components.at(component).size()<<std::endl;
        auto graph=vc::constraints::seedGraph(catalog,component,settings,normals);
        size_t vertices=0;for(const auto& [id,p]:graph.nodes)vertices+=p.uv.size();std::cout<<"source vertices="<<vertices<<std::endl;
        auto result=vc::constraints::fit(graph,nullptr,progress);
        std::cout<<"seed valid="<<result.valid<<" error="<<result.message<<" stretch="<<result.stretchP95<<" rms="<<result.correspondenceRms<<std::endl;
        if(!result.valid)return 1;
        vc::constraints::saveSession(result,argv[4]);
        auto candidates=vc::constraints::discover(catalog,result.graph,result.graph.seed,nullptr,progress);
        std::cout<<"seed-fiber proposals="<<candidates.size()<<std::endl;
        bool accepted=false;
        for(size_t i=0;i<std::min(size_t(20),candidates.size());++i){
            auto proposal=vc::constraints::propose(catalog,result.graph,candidates[i],true,normals);
            auto fitted=vc::constraints::fit(proposal,nullptr,progress);
            std::cout<<"proposal="<<candidates[i].id<<" valid="<<fitted.valid<<" error="<<fitted.message<<" rms="<<fitted.correspondenceRms<<std::endl;
            if(fitted.valid){
                auto path=std::string(argv[4])+".proposal.json";
                vc::constraints::saveSession(fitted,path);
                auto reopened=vc::constraints::loadSession(path);
                if(!reopened.valid||vc::constraints::graphToJson(reopened.graph)!=vc::constraints::graphToJson(fitted.graph))throw std::runtime_error("Proposal session did not round trip");
                std::cout<<"proposal roundtrip=passed"<<std::endl;
                accepted=true;break;
            }
        }
        std::cout<<"elapsed="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()<<std::endl;
        return accepted?0:1;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
