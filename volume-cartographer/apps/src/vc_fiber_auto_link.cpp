// Geometric fiber/patch auto-linking without the VC3D UI. Writes a
// vc_fiber_patch_links JSON that spiral-fitting/auto_link_fibers.py turns into
// explicit_link tags. See apps/VC3D/ConstraintGraph.md, "Automatic linking".
#include "vc/constraints/ConstraintGraph.hpp"
#include "vc/constraints/AutoLink.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <boost/program_options.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <atomic>
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace po=boost::program_options;
using namespace vc::constraints;
using Json=nlohmann::json;

int main(int argc,char** argv) {
    po::options_description options("vc_fiber_auto_link options");
    std::string fibers,patches,normals,output,settingsPath,only,stripCache,stripKey;
    int jobs=0;std::string shard;
    Settings settings;settings.radius=1.5;
    AutoLinkSettings autoLink;FastStripOptions fastStrip;
    options.add_options()
        ("help,h","print this help")
        ("fibers",po::value(&fibers)->required(),"folder of vc3d_fiber JSON files")
        ("patches",po::value(&patches)->required(),"folder of tifxyz patches")
        ("normals",po::value(&normals)->required(),"Lasagna normals manifest (.lasagna.json)")
        ("output,o",po::value(&output)->required(),"output vc_fiber_patch_links JSON")
        ("fiber-scale",po::value(&settings.fiberScale)->default_value(settings.fiberScale),"fiber coordinate scale")
        ("patch-scale",po::value(&settings.patchScale)->default_value(settings.patchScale),"patch coordinate scale")
        ("normal-scale",po::value(&settings.normalScale)->default_value(settings.normalScale),"normal manifest to working scale")
        ("spacing",po::value(&settings.spacing)->default_value(settings.spacing),"strip sample spacing in graph voxels")
        ("radius",po::value(&settings.radius)->default_value(settings.radius),"discovery proximity in graph voxels")
        ("anchor-contact",po::value(&autoLink.anchorContact)->default_value(autoLink.anchorContact),"minimum merged fiber contact of an anchor patch, in graph voxels")
        ("grow-same",po::value(&autoLink.growSame)->default_value(autoLink.growSame),"minimum same-sheet vertex fraction for growth")
        ("anchor-strip-ratio",po::value(&autoLink.stripGateRatio)->default_value(autoLink.stripGateRatio),"anchor gate: strip same-sheet fraction must exceed this multiple of the adjacent-sheet fraction")
        ("anchor-strip-margin",po::value(&autoLink.stripGateMargin)->default_value(autoLink.stripGateMargin),"anchor gate: additive margin on top of the ratio")
        ("autolink-settings",po::value(&settingsPath),"JSON file overriding all auto-link parameters")
        ("only",po::value(&only),"process a single fiber (file name)")
        ("strip-step",po::value(&fastStrip.step)->default_value(fastStrip.step),"fast strip: marching step across the sheet (graph voxels)")
        ("strip-follow-angle",po::value(&fastStrip.followAngle)->default_value(fastStrip.followAngle),"fast strip: max degrees a sampled normal may turn per step and still be followed")
        ("strip-normal-sigma",po::value(&fastStrip.normalSigma)->default_value(fastStrip.normalSigma),"fast strip: smoothing of per-column normals along the fiber (columns)")
        ("strip-offset-sigma",po::value(&fastStrip.offsetSigma)->default_value(fastStrip.offsetSigma),"fast strip: smoothing of row offsets along the fiber (columns)")
        ("strip-snap",po::value<int>()->default_value(1),"fast strip: re-centre marched points on the Lasagna pred_dt sheet score (1) or not (0)")
        ("strip-snap-range",po::value(&fastStrip.snapRange)->default_value(fastStrip.snapRange),"fast strip: search range along the normal for the sheet peak (graph voxels)")
        ("strip-cache",po::value(&stripCache),"folder caching fitted fiber strips across runs (keyed by fiber content, settings and strip version)")
        ("jobs,j",po::value(&jobs)->default_value(0),"worker processes over fiber shards (0 = hardware threads / 2, 1 = in this process). The strip fit contends on shared state when run on threads, so workers are processes.")
        ("shard",po::value(&shard),"internal: process fibers K/N of the sorted catalog (K from 0)");
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc,argv,options),vm);
        if(vm.count("help")){std::cout<<options<<std::endl;return 0;}
        po::notify(vm);
        settings.fiberFolder=fibers;settings.patchFolder=patches;settings.normalManifest=normals;settings.validate();
        if(!settingsPath.empty()){std::ifstream in(settingsPath);if(!in)throw std::runtime_error("Cannot read "+settingsPath);autoLink=autoLinkSettingsFromJson(Json::parse(in));}
        if(std::filesystem::exists(output)&&std::filesystem::equivalent(output,fibers))throw std::runtime_error("Output must not be the fiber folder");
        {std::ostringstream key;key<<"fast:"<<fastStrip.step<<":"<<fastStrip.followAngle<<":"<<fastStrip.normalSigma<<":"<<fastStrip.offsetSigma<<":"<<fastStrip.outlierAngle<<":snap"<<vm["strip-snap"].as<int>()<<":"<<fastStrip.snapRange;stripKey=key.str();}
    } catch(const std::exception& e){std::cerr<<e.what()<<"\n"<<options<<std::endl;return 1;}
    try {
        const auto start=std::chrono::steady_clock::now();
        vc::lasagna::LasagnaDatasetOpenOptions open;open.workingToBaseScale=settings.normalScale;
        const auto dataset=vc::lasagna::LasagnaDataset::openLocation(settings.normalManifest,open);
        vc::lasagna::LasagnaNormalSampler sampler(dataset);
        if(!vm["strip-snap"].as<int>())fastStrip.sheetScore=[](const cv::Vec3d&){return std::optional<double>{};};
        else if(!sampler.hasPredDtChannel())std::cerr<<"warning: normals manifest has no pred_dt channel; strips will not snap to the sheet"<<std::endl;
        std::cout<<"Loading catalog"<<std::endl;
        const auto catalog=loadCatalog(settings,nullptr,[](const std::string& m){std::cout<<m<<std::endl;});
        for(const auto& warning:catalog.warnings)std::cerr<<"warning: "<<warning<<std::endl;
        std::map<std::string,PatchEntry> entries;for(const auto& p:catalog.patches)entries.emplace(p.id,p);
        PatchLoader load=[&](const std::string& id)->std::shared_ptr<const Node> {
            const auto found=entries.find(id);if(found==entries.end())return nullptr;
            try{return loadPatchNode(found->second,settings);}catch(const std::exception& e){std::cerr<<"warning: "<<id<<": "<<e.what()<<std::endl;return nullptr;}
        };
        Json out={{"type","vc_fiber_patch_links"},{"version",1},{"settings",settingsToJson(settings)},{"autolink",autoLinkSettingsToJson(autoLink)},{"fibers",Json::object()}};
        std::vector<std::pair<std::string,const Fiber*>> work;
        size_t shardIndex=0,shardCount=1;
        if(!shard.empty()){const auto slash=shard.find('/');if(slash==std::string::npos)throw std::runtime_error("--shard needs K/N");shardIndex=std::stoul(shard.substr(0,slash));shardCount=std::stoul(shard.substr(slash+1));if(!shardCount||shardIndex>=shardCount)throw std::runtime_error("Invalid --shard");}
        {
            // Longest fibers first, dealt round-robin to shards: the strip fit
            // cost grows with fiber length, so this balances the workers' tails.
            std::vector<std::pair<std::string,const Fiber*>> all;
            for(const auto& [id,fiber]:catalog.fibers)if(only.empty()||fiber.path.filename().string()==only)all.emplace_back(id,&fiber);
            std::stable_sort(all.begin(),all.end(),[](const auto& a,const auto& b){return a.second->line.size()>b.second->line.size();});
            for(size_t index=0;index<all.size();++index)if(index%shardCount==shardIndex)work.push_back(all[index]);
        }
        size_t done=0,linkedTotal=0,candidateTotal=0;
        // Fibers are independent, but the strip fit slows an order of magnitude
        // when several run on threads of one process, so each worker is a child
        // process over a shard of the sorted catalog; results merge by name.
        const size_t width=jobs>0?size_t(jobs):std::max<size_t>(1,std::thread::hardware_concurrency()/2);
#if !defined(_WIN32)
        if(shard.empty()&&width>1&&work.size()>1) {
            const size_t children=std::min(width,work.size());
            std::vector<pid_t> pids;std::vector<std::string> parts;
            for(size_t k=0;k<children;++k) {
                parts.push_back(output+".part"+std::to_string(k));
                std::vector<std::string> args;for(int i=0;i<argc;++i)args.push_back(argv[i]);
                // Children keep every option except the output, job count and shard.
                for(size_t i=1;i<args.size();){const auto& a=args[i];const bool valued=a=="--output"||a=="-o"||a=="--jobs"||a=="-j"||a=="--shard";
                    if(valued){args.erase(args.begin()+i,args.begin()+std::min(i+2,args.size()));}
                    else if(a.rfind("--output=",0)==0||a.rfind("--jobs=",0)==0||a.rfind("--shard=",0)==0||a.rfind("-o",0)==0&&a.size()>2||a.rfind("-j",0)==0&&a.size()>2)args.erase(args.begin()+i);
                    else ++i;}
                args.insert(args.end(),{"--jobs","1","--shard",std::to_string(k)+"/"+std::to_string(children),"--output",parts.back()});
                std::vector<char*> argvChild;for(auto& a:args)argvChild.push_back(a.data());argvChild.push_back(nullptr);
                const pid_t pid=fork();
                if(pid<0)throw std::runtime_error("fork failed");
                if(pid==0){execv("/proc/self/exe",argvChild.data());execvp(argv[0],argvChild.data());_exit(127);}
                pids.push_back(pid);
            }
            std::cout<<"Started "<<children<<" worker processes over "<<work.size()<<" fibers"<<std::endl;
            bool failed=false;
            for(pid_t pid:pids){int status=0;waitpid(pid,&status,0);if(!WIFEXITED(status)||WEXITSTATUS(status)!=0)failed=true;}
            if(failed)throw std::runtime_error("A worker process failed; partial outputs kept as "+output+".part*");
            for(const auto& part:parts) {
                std::ifstream in(part);if(!in)throw std::runtime_error("Missing worker output "+part);
                auto partial=Json::parse(in);
                for(auto& [name,row]:partial.at("fibers").items()){
                    ++done;candidateTotal+=row["linked"].size()+row["unlinked"].size();linkedTotal+=row["linked"].size();
                    out["fibers"][name]=std::move(row);
                }
                std::filesystem::remove(part);
            }
        } else
#endif
        for(const auto& [id,fiberPtr]:work) {
            const auto& fiber=*fiberPtr;const auto name=fiber.path.filename().string();
            const auto tick=std::chrono::steady_clock::now();
            Json row={{"path",fiber.path.string()},{"direction",std::string(1,fiber.direction)},{"linked",Json::array()},{"unlinked",Json::array()}};
            try {
                auto phase=std::chrono::steady_clock::now();
                auto lap=[&]{const auto now=std::chrono::steady_clock::now();const double s=std::chrono::duration<double>(now-phase).count();phase=now;return s;};
                std::shared_ptr<const Node> node;
                if(!stripCache.empty())node=loadCachedFiberNode(stripCache,fiber,settings,stripKey);
                row["strip_cached"]=node!=nullptr;
                if(!node){node=makeFastFiberNode(fiber,settings,sampler,nullptr,fastStrip);if(!stripCache.empty())saveCachedFiberNode(stripCache,fiber,settings,*node,stripKey);}
                row["strip_seconds"]=lap();
                Graph graph;graph.settings=settings;graph.seed=id;graph.nodes.emplace(id,Placement{node,node->initialUv});
                const auto proposals=discover(catalog,graph,id,nullptr,{},{},1,{},ContactPolicy::FiberReview);row["discover_seconds"]=lap();
                const auto result=autoLinkFiber(*node,proposals,load,autoLink);row["autolink_seconds"]=lap();
                for(const auto& c:result.candidates) {
                    Json entry={{"patch",c.id.substr(c.id.rfind(':')==std::string::npos?0:6)},{"contact",c.contact},{"anchor",c.anchor},{"round",c.round},
                                {"strip_same",c.stripSame},{"strip_adjacent",c.stripAdjacent},{"linked_same",c.linkedSame},{"linked_adjacent",c.linkedAdjacent}};
                    row[c.linked?"linked":"unlinked"].push_back(std::move(entry));
                }
                row["anchors"]=result.anchors;linkedTotal+=result.linked;candidateTotal+=result.candidates.size();
            } catch(const std::exception& e){row["error"]=e.what();std::cerr<<"warning: "<<name<<": "<<e.what()<<std::endl;}
            row["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();
            std::cout<<"fiber="<<name<<" ("<<++done<<"/"<<work.size()<<(shard.empty()?"":" of shard "+shard)<<") candidates="<<row["linked"].size()+row["unlinked"].size()<<" linked="<<row["linked"].size()<<" seconds="<<row["seconds"]
                     <<" strip="<<row.value("strip_seconds",0.)<<" discover="<<row.value("discover_seconds",0.)<<" autolink="<<row.value("autolink_seconds",0.)<<std::endl;
            out["fibers"][name]=std::move(row);
        }
        out["summary"]={{"fibers",done},{"candidates",candidateTotal},{"linked",linkedTotal},{"jobs",width},
                        {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
        auto temporary=output+".tmp";
        {std::ofstream o(temporary,std::ios::trunc);o<<out.dump(1);if(!o)throw std::runtime_error("Failed to write "+output);}
        std::filesystem::rename(temporary,output);
        std::cout<<"fibers="<<done<<" candidates="<<candidateTotal<<" linked="<<linkedTotal<<" jobs="<<width<<" output="<<output<<std::endl;
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
