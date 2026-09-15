#include "vc/constraints/ConstraintGraph.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <iostream>
#include <chrono>
#include <set>
#include <fstream>
int main(int argc,char** argv){
    if(argc>=3&&(std::string(argv[1])=="--proposal-benchmark"||std::string(argv[1])=="--local-proposal-benchmark")) {
        try {
            using namespace vc::constraints;
            const auto result=loadSession(argv[2]);
            const auto catalog=loadCatalog(result.graph.settings);
            vc::lasagna::LasagnaDatasetOpenOptions options;options.workingToBaseScale=result.graph.settings.normalScale;
            const auto dataset=vc::lasagna::LasagnaDataset::openLocation(result.graph.settings.normalManifest,options);
            vc::lasagna::LasagnaNormalSampler normals(dataset);
            auto candidates=discover(catalog,result.graph,result.graph.seed);
            if(candidates.empty())throw std::runtime_error("No seed proposals");
            const auto& candidate=candidates.at(argc>5?std::stoul(argv[5]):0);
            std::cout<<"candidate="<<candidate.id<<" nodes="<<result.graph.nodes.size()<<" contacts="<<candidate.contacts.size()<<std::endl;
            for(int i=0;i<(argc>3?std::stoi(argv[3]):3);++i) {
                auto start=std::chrono::steady_clock::now(),phase=start;
                auto stamp=[&](const std::string& name){auto now=std::chrono::steady_clock::now();std::cout<<name<<" phase_ms="<<std::chrono::duration<double,std::milli>(now-phase).count()<<" total_ms="<<std::chrono::duration<double,std::milli>(now-start).count()<<std::endl;phase=now;};
                auto draft=propose(catalog,result.graph,candidate,true,normals);stamp("propose");
                auto source=renderProposalSources(draft,result.graph,candidate.id);stamp("source_render");
                auto placement=renderPlacement(draft);stamp("initial_placement");
                std::cout<<"placement_valid="<<placement.valid<<" message="<<placement.message<<std::endl;
                auto fitResult=std::string(argv[1])=="--local-proposal-benchmark"?fitIncremental(std::move(draft),result,nullptr,stamp):fit(std::move(draft),nullptr,stamp);stamp("fit_complete");
                std::cout<<"iteration="<<i<<" valid="<<fitResult.valid<<" rms="<<fitResult.correspondenceRms<<" stretch="<<fitResult.stretchP95<<" message="<<fitResult.message<<std::endl;
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
