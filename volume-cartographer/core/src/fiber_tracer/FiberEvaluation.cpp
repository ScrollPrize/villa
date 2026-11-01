#include "vc/fiber_tracer/FiberEvaluation.hpp"
#include "vc/fiber_tracer/FiberReverseConsistency.hpp"
#include "vc/fiber_tracer/detail/FiberPackedStorage.hpp"
#include "vc/fiber_tracer/detail/PolylineSegmentIndex.hpp"
#include "vc/fiber_tracer/detail/FiberRouteOwnership.hpp"
#include "vc/fiber_tracer/detail/FiberContinuationGraph.hpp"
#include <numeric>
#include <atomic>
#include <chrono>
#include <cmath>
#include <list>
#include <random>
#include <set>

namespace vc::fiber_tracer {

ReplayConsistencyPolicy fiberChunkReplayConsistency(const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler& normals, const FiberTraceConfig& trace,
    const FiberChunkReplayConfig& config, double interval)
{
    if (!std::isfinite(interval) || interval < 0)
        throw std::invalid_argument("reverse interval must be finite and nonnegative");
    ReplayConsistencyPolicy policy;
    policy.intervalBase = interval;
    if (interval > 0) policy.verify = [&predictions, &normals, trace, config](const auto& points) {
        const auto path = makePolylineArcGeometry(points);
        return verifyFiberReverseSpan(predictions, normals, trace, path, path.length(),
            config.errorThresholdBase, config.errorThresholdBase * config.tangentialThresholdFactor) <= 1e-12;
    };
    return policy;
}

void FiberEvaluationControl::cancel(const std::string& reason)
{
    std::lock_guard lock(mutex_);
    if (status_.finished || status_.cancelled) return;
    status_.cancelled = true;
    status_.error = reason.empty() ? "fiber evaluation cancelled" : reason;
    if (queue_) queue_->cancel(status_.error);
}

FiberEvaluationStatus FiberEvaluationControl::status() const
{
    std::lock_guard lock(mutex_);
    return status_;
}

class FiberEvaluationRun {
public:
    FiberEvaluationRun(FiberEvaluationControl& control, size_t total, bool volume=false) : control_(control) {
        std::lock_guard lock(control_.mutex_);
        if (control_.status_.started) throw std::invalid_argument("evaluation control already used");
        control_.status_.started = true;
        control_.status_.total = total;
        control_.status_.chunkVolume = volume;
    }
    ~FiberEvaluationRun() {
        std::lock_guard lock(control_.mutex_);
        control_.queue_ = nullptr;
        control_.status_.finished = true;
    }
    void attach(FiberCaseQueue* queue) {
        std::lock_guard lock(control_.mutex_);
        control_.queue_ = queue;
        if (queue && control_.status_.cancelled) queue->cancel(control_.status_.error);
    }
    void check() const {
        const auto s = control_.status();
        if (s.cancelled) throw std::runtime_error(s.error);
    }
    void completed(const std::string& outcome={}) {
        std::lock_guard lock(control_.mutex_);
        ++control_.status_.completed;
        control_.status_.built += outcome=="built";
        control_.status_.reused += outcome=="reused";
        control_.status_.annotated += outcome=="annotated";
    }
    void finish() {
        std::lock_guard lock(control_.mutex_);
        if (control_.status_.cancelled) throw std::runtime_error(control_.status_.error);
        control_.status_.finished = true;
    }
    void total(size_t count) {std::lock_guard lock(control_.mutex_);control_.status_.total=count;}
    void failed(const std::string& message) {
        std::lock_guard lock(control_.mutex_);
        if (control_.status_.error.empty()) control_.status_.error = message;
    }
private:
    FiberEvaluationControl& control_;
};

void prepareFiberChunkVolume(const FiberPredictionSource& predictions,const FiberTraceConfig& trace,
    double scale,const vc::lasagna::NormalSampler& normals,const FiberChunkTraceConfig& chunks,
    const std::filesystem::path& output,const std::string& identity,
    const std::vector<std::array<int,3>>& keys,FiberEvaluationControl& control,size_t workers,size_t contexts)
{
    FiberEvaluationRun run(control,keys.size(),true);
    try {
        run.check();
        if(!workers||!contexts||trace.parallelThreads!=1||trace.profile)
            throw std::invalid_argument("volume requires positive workers/contexts, parallel_threads=1 and no shared profile");
        if(!predictions.supportsConcurrentSampling()||!normals.supportsConcurrentSampling())
            throw std::invalid_argument("volume requires concurrent sampling sources");
        if(std::set<std::array<int,3>>(keys.begin(),keys.end()).size()!=keys.size())
            throw std::invalid_argument("duplicate volume chunk keys");
        std::atomic<size_t> next{0};
        auto queue=std::make_unique<FiberCaseQueue>(workers,keys.size(),[&] {
            run.check();
            const auto key=keys[next.fetch_add(1)];
            std::string outcome;
            try {
                // No cache object/future retains completed geometry.
                auto chunk=prepareFiberChunkPaths(key,predictions,trace,normals,chunks,scale,output,identity,outcome,
                    FiberCaseQueue::parallelFor,[](const auto& ready){FiberCaseQueue::await(ready);});
            } catch(const std::exception& e) {
                throw std::runtime_error("chunk "+std::to_string(key[0])+"_"+std::to_string(key[1])+"_"+
                    std::to_string(key[2])+": "+e.what());
            }
            run.completed(outcome);
        },contexts);
        run.attach(queue.get());
        try {queue->join();}catch(...){run.attach(nullptr);throw;}
        run.attach(nullptr);run.finish();
    }catch(const std::exception& e){run.failed(e.what());throw;}
    catch(...){run.failed("unknown chunk volume error");throw;}
}

namespace {
using RouteKey=std::array<int,3>;
using RouteJson=nlohmann::json;
struct OutsideRouteInventory {};
std::string routeStem(const RouteKey& k) {
    return std::to_string(k[0])+"_"+std::to_string(k[1])+"_"+std::to_string(k[2]);
}
RouteJson routeXYZ(const cv::Vec3d& p) {return {p[0],p[1],p[2]};}

// Ready LRU bytes exclude active reader leases and in-flight loads, like the
// volume cache. No lock is held during I/O, annotation validation or sampling.
class RouteChunkReader {
    struct Ready {
        std::shared_ptr<const FiberChunkPaths> chunk;
        size_t bytes;
        std::list<RouteKey>::iterator age;
    };
    std::mutex mutex;
    std::map<RouteKey,Ready> ready;
    std::map<RouteKey,std::shared_future<std::shared_ptr<const FiberChunkPaths>>> loading;
    std::list<RouteKey> ages;
    size_t bytes=0,limit;
    FiberChunkProvider load;
public:
    RouteChunkReader(size_t budget,FiberChunkProvider provider):limit(budget),load(std::move(provider)) {}
    std::shared_ptr<const FiberChunkPaths> get(const RouteKey& key) {
        std::shared_future<std::shared_ptr<const FiberChunkPaths>> future;
        std::shared_ptr<std::promise<std::shared_ptr<const FiberChunkPaths>>> promise;
        {
            std::lock_guard lock(mutex);
            if(auto i=ready.find(key);i!=ready.end()) {
                ages.splice(ages.begin(),ages,i->second.age);return i->second.chunk;
            }
            if(auto i=loading.find(key);i!=loading.end())future=i->second;
            else {promise=std::make_shared<std::promise<std::shared_ptr<const FiberChunkPaths>>>();
                future=promise->get_future().share();loading.emplace(key,future);}
        }
        if(!promise)return future.get(); // loader is synchronous, never waits on a queued job
        try {
            auto chunk=load(key);
            // Only the small validated filter specification remains as JSON;
            // include generous container/node overhead, not just serialized bytes.
            size_t size=sizeof(FiberChunkPaths)+4096+16*chunk->conflictAnnotation.dump().size();
            size+=chunk->lines.capacity()*sizeof(std::vector<cv::Vec3d>);
            for(const auto& line:chunk->lines)size+=line.capacity()*sizeof(cv::Vec3d);
            for(const auto& ends:chunk->endReasons)size+=sizeof(ends)+ends[0].capacity()+ends[1].capacity();
            size+=chunk->conflictFlags.capacity()/8;
            {
                std::lock_guard lock(mutex);
                if(size<=limit) {
                    while(bytes+size>limit&&!ages.empty()) {
                        auto i=ready.find(ages.back());bytes-=i->second.bytes;ready.erase(i);ages.pop_back();
                    }
                    ages.push_front(key);ready.emplace(key,Ready{chunk,size,ages.begin()});bytes+=size;
                }
                promise->set_value(chunk);loading.erase(key);
            }
            return chunk;
        }catch(...) {
            promise->set_exception(std::current_exception());
            std::lock_guard lock(mutex);loading.erase(key);throw;
        }
    }
};

std::vector<std::pair<size_t,size_t>> routeSeeds(const FiberChunkPaths& chunk,const FiberChunkTraceConfig& c) {
    validateFiberChunkConflictAnnotation(chunk,c);
    std::vector<std::pair<size_t,size_t>> result;
    const cv::Vec3d center((chunk.key[0]+.5)*c.coreBase,(chunk.key[1]+.5)*c.coreBase,(chunk.key[2]+.5)*c.coreBase);
    for(size_t id=0;id<chunk.lines.size();++id) {
        if(c.conflictNormalBase>0&&chunk.conflictFlags[id])continue;
        const auto& line=chunk.lines[id];if(line.size()<2)continue;
        size_t selected=line.size();double best=std::numeric_limits<double>::infinity();
        for(size_t v=0;v<line.size();++v) {
            bool owned=true;
            for(int a=0;a<3;++a)owned&=line[v][a]>=chunk.key[a]*c.coreBase&&line[v][a]<(chunk.key[a]+1)*c.coreBase;
            const auto tangent=v+1<line.size()?line[v+1]-line[v]:line[v]-line[v-1];
            if(!owned||cv::norm(tangent)<1e-9)continue;
            const double d=cv::norm(line[v]-center);
            if(d<best){best=d;selected=v;}
        }
        if(selected<line.size())result.emplace_back(id,selected);
    }
    return result;
}
} // namespace

void exportFiberChunkRoutes(const FiberPredictionSource& source,const FiberTraceConfig& trace,double scale,
    const vc::lasagna::NormalSampler& normals,const FiberChunkTraceConfig& chunks,
    const std::filesystem::path& directory,const std::filesystem::path& output,const std::string& identity,
    const std::vector<RouteKey>& keys,const std::vector<RouteKey>& seedKeys,const FiberChunkRouteVolumeConfig& config,
    FiberEvaluationControl& control,size_t workers,size_t contexts)
{
    FiberEvaluationRun run(control,seedKeys.size(),true);
    try {
        run.check();validateFiberTraceConfig(trace);
        if(!workers||!contexts||!config.cacheBytes||!config.pagePoints||config.pagePoints>1000000||
           !std::isfinite(config.maximumDistanceBase)||config.maximumDistanceBase<=0||
           config.maximumSteps<=0||config.maximumSteps>1000000||!std::isfinite(scale)||scale<=0||
           std::abs(trace.stepVoxels*scale-16)>1e-6||trace.parallelThreads!=1||trace.profile)
            throw std::invalid_argument("invalid route export budgets or tracing configuration (requires 16 base step)");
        if(!source.supportsConcurrentSampling()||!normals.supportsConcurrentSampling())
            throw std::invalid_argument("route export requires concurrent sources");
        if(directory.empty()||output.empty()||identity.empty()||
           std::filesystem::weakly_canonical(directory)==std::filesystem::weakly_canonical(output))
            throw std::invalid_argument("route export requires distinct source/output directories and identity");
        const std::set<RouteKey> inventory(keys.begin(),keys.end());
        if(inventory.size()!=keys.size()||std::set<RouteKey>(seedKeys.begin(),seedKeys.end()).size()!=seedKeys.size())
            throw std::invalid_argument("duplicate route inventory/seed keys");
        for(const auto& k:inventory)for(auto v:k)if(v<0)throw std::invalid_argument("negative route inventory key");
        for(const auto& k:seedKeys)if(!inventory.contains(k))throw std::invalid_argument("seed owner outside route inventory");
        std::filesystem::create_directories(output);
        auto readConfig=chunks;readConfig.cacheOnly=true;
        const auto construction=fiberChunkCacheSpecification(readConfig,trace,scale,identity);
        // Performance-only settings must not invalidate deterministic output.
        auto matching=fiberChunkRouteMatchingSpecification(chunks);
        RouteJson binding={{"format","vc-fiber-routes-v4-reduced"},{"construction",construction},{"route",matching},
            {"path_reduction",config.reduceDuplicatePaths?"terminal-complete-link-v2":"disabled"},
            {"maximum_distance_base",config.maximumDistanceBase},{"maximum_steps",config.maximumSteps},
            {"page_points",config.pagePoints},{"seed_keys",seedKeys},{"inventory",RouteJson::array()}};
        if(!config.rejectAmbiguousPaths)binding["reject_ambiguous_paths"]=false;
        std::vector<RouteKey> ordered(inventory.begin(),inventory.end());
        struct File {std::filesystem::path path;uintmax_t size;std::filesystem::file_time_type time;};
        std::vector<File> files(ordered.size());std::atomic<size_t> next{0};
        // Metadata inventory is parallel, not a serial walk through every payload.
        FiberCaseQueue preflight(workers,ordered.size(),[&] {
            run.check();const auto i=next.fetch_add(1);const auto stem=routeStem(ordered[i]);
            auto path=directory/(stem+".msgpack.zst");
            if(!std::filesystem::exists(path))path=directory/(stem+".json");
            if(!std::filesystem::is_regular_file(path))throw std::runtime_error("expected route source chunk missing: "+path.string());
            files[i]={path,std::filesystem::file_size(path),std::filesystem::last_write_time(path)};
        },1);
        preflight.join();run.check();
        std::map<RouteKey,size_t> fileIndex;
        for(size_t i=0;i<ordered.size();++i) {
            fileIndex.emplace(ordered[i],i);
            binding["inventory"].push_back({{"key",ordered[i]},{"file",files[i].path.filename().string()},
                {"bytes",files[i].size},{"mtime",files[i].time.time_since_epoch().count()}});
        }
        const auto manifest=output/"routes.manifest.msgpack.zst";
        std::string generation;
        if(std::filesystem::exists(manifest)) {
            auto previous=detail::readFiberPacked(manifest);generation=previous.at("generation").get<std::string>();
            previous.erase("generation");
            if(previous!=binding)throw std::runtime_error("route export source/inventory/configuration mismatch");
        } else {
            std::random_device random;generation=std::to_string(random())+"-"+std::to_string(random());
            auto document=binding;document["generation"]=generation;detail::writeFiberPacked(manifest,document);
        }
        binding=RouteJson();
        // A key's load is coalesced. Publish decisions before its immutable view;
        // eviction/reload reapplies them without recomputing equivalence.
        std::vector<std::optional<FiberChunkPathReduction>> reductions(ordered.size());
        RouteChunkReader cache(config.cacheBytes,[&](const RouteKey& key)->std::shared_ptr<const FiberChunkPaths> {
            run.check();const auto i=fileIndex.find(key);
            if(i==fileIndex.end()) {
                auto empty=std::make_shared<FiberChunkPaths>();empty->key=key;
                for(int a=0;a<3;++a){empty->lo[a]=key[a]*chunks.coreBase-chunks.haloBase;empty->hi[a]=(key[a]+1)*chunks.coreBase+chunks.haloBase;}
                return empty;
            }
            const auto& file=files[i->second];
            if(std::filesystem::file_size(file.path)!=file.size||std::filesystem::last_write_time(file.path)!=file.time)
                throw std::runtime_error("route source changed during export: "+file.path.string());
            std::string outcome;
            auto loaded=prepareFiberChunkPaths(key,source,trace,normals,readConfig,scale,directory,identity,outcome,{},{},true);
            // Reports/timings can contain large JSON trees. They were validated
            // above but are not needed by route matching; don't retain them.
            loaded->timings=RouteJson::object();
            if(!loaded->conflictAnnotation.is_null())
                loaded->conflictAnnotation={{"specification",loaded->conflictAnnotation.at("specification")}};
            if(config.reduceDuplicatePaths){
                auto& reduction=reductions[i->second];
                if(!reduction)reduction=reduceFiberChunkPaths(*loaded,chunks,
                    [&](const auto& points){return sampleFiberChunkWidths(source,points,scale);},trace.widthRelativeTolerance,
                    config.rejectAmbiguousPaths);
                reduction->apply(*loaded);
            }
            return loaded;
        });
        const auto graphStart=std::chrono::steady_clock::now();
        struct Seed {size_t line,vertex;std::vector<std::pair<double,double>> targets;};
        struct End {size_t line;cv::Vec3d point;RouteKey target;bool reverse;};
        struct Meta {size_t offset=0;std::vector<double> lengths;std::vector<uint8_t> eligible;std::vector<Seed> seeds;std::vector<End> ends;};
        std::vector<Meta> metadata(ordered.size());
        const std::set<RouteKey> requested(seedKeys.begin(),seedKeys.end());
        next=0;
        FiberCaseQueue scan(workers,ordered.size(),[&]{
            run.check();const auto i=next.fetch_add(1);const auto chunk=cache.get(ordered[i]);auto& m=metadata[i];
            m.lengths.resize(chunk->lines.size());m.eligible.resize(chunk->lines.size());
            for(size_t line=0;line<chunk->lines.size();++line){
                const auto& points=chunk->lines[line];
                if(points.size()<2||(chunks.conflictNormalBase>0&&chunk->conflictFlags[line]))continue;
                const auto geometry=makePolylineArcGeometry(points);m.lengths[line]=geometry.length();
                if(!(geometry.length()>0))continue;m.eligible[line]=1;
                // All eligible paths, including halo-only paths, participate.
                // Endpoint lookup is a conservative superset: no history,
                // terminal-reason, score or ownership pruning.
                for(bool reverse:{false,true}){
                    const auto point=reverse?points.front():points.back();
                    size_t p=reverse?1:points.size()-2;
                    while(cv::norm(points[p]-point)<1e-9 && (reverse?p+1<points.size():p>0))
                        p=reverse?p+1:p-1;
                    const auto previous=points[p];
                    const auto next=nextFiberChunk(*chunk,point,previous,chunk->endReasons[line][reverse?0:1],chunks);
                    if(next.stop.empty())m.ends.push_back({line,point,next.key,reverse});
                }
            }
            std::map<size_t,size_t> preferred;
            for(auto [line,vertex]:routeSeeds(*chunk,chunks))preferred[line]=vertex;
            if(requested.contains(ordered[i]))for(size_t line=0;line<m.eligible.size();++line){
                if(!m.eligible[line])continue;
                std::vector<std::pair<double,double>> targets;
                const cv::Vec3d lo(ordered[i][0]*chunks.coreBase,ordered[i][1]*chunks.coreBase,ordered[i][2]*chunks.coreBase);
                (void)clipPolylineToHalfOpenBox(chunk->lines[line],lo,lo+cv::Vec3d::all(chunks.coreBase),&targets);
                if(!targets.empty())m.seeds.push_back({line,preferred.contains(line)?preferred.at(line):0,std::move(targets)});
            }
        },1);scan.join();run.check();
        RouteJson reductionData=RouteJson::array();size_t originalEligible=0,fusedPaths=0,ambiguousPaths=0,terminalPairs=0;
        size_t retainedAmbiguousPaths=0;
        for(size_t i=0;i<reductions.size();++i)if(reductions[i]){
            auto entry=reductions[i]->json();entry["key"]=ordered[i];reductionData.push_back(std::move(entry));
            originalEligible+=reductions[i]->eligible;fusedPaths+=reductions[i]->fusedPaths;
            ambiguousPaths+=reductions[i]->ambiguousPaths;terminalPairs+=reductions[i]->terminalPairs;
            retainedAmbiguousPaths+=reductions[i]->retainedAmbiguousPaths;
        }
        RouteJson reductionReport={{"generation",generation},{"enabled",config.reduceDuplicatePaths},
            {"original_eligible",originalEligible},{"fused_paths",fusedPaths},{"ambiguous_paths",ambiguousPaths},
            {"terminal_pairs",terminalPairs},{"chunks",std::move(reductionData)}};
        if(!config.rejectAmbiguousPaths){reductionReport["reject_ambiguous_paths"]=false;
            reductionReport["retained_ambiguous_paths"]=retainedAmbiguousPaths;}
        const auto reductionFile=output/"path_reduction.msgpack.zst";
        if(std::filesystem::exists(reductionFile)){
            if(detail::readFiberPacked(reductionFile)!=reductionReport)throw std::runtime_error("path reduction changed");
        }else detail::writeFiberPacked(reductionFile,reductionReport);
        size_t nodes=0;
        for(auto& m:metadata){m.offset=nodes;nodes+=m.lengths.size();}
        struct Incoming {size_t node;cv::Vec3d point;detail::FiberPathEnd end;};
        std::vector<std::vector<Incoming>> incoming(ordered.size());
        for(size_t i=0;i<metadata.size();++i)for(const auto& e:metadata[i].ends){
            if(auto target=fileIndex.find(e.target);target!=fileIndex.end())
                incoming[target->second].push_back({metadata[i].offset+e.line,e.point,{ordered[i],e.line,e.reverse}});
        }
        std::vector<std::vector<std::pair<size_t,size_t>>> edges(ordered.size());next=0;
        std::vector<std::vector<detail::FiberContinuationGraph::Edge>> candidates(ordered.size());
        FiberCaseQueue connect(workers,ordered.size(),[&]{
            run.check();const auto i=next.fetch_add(1);if(incoming[i].empty())return;
            const auto chunk=cache.get(ordered[i]);
            const detail::ChunkSegmentIndex lookup(chunk->lines,[](const auto& p)->const auto&{return p;});
            for(const auto& request:incoming[i]){
                std::set<size_t> hits;lookup.nearbyPaths(request.point,chunks.joinDistanceBase,[&](size_t line){
                    if(metadata[i].eligible[line])hits.insert(line);
                });
                for(auto line:hits)edges[i].emplace_back(request.node,metadata[i].offset+line);
                if(hits.empty())continue;
                const auto from=cache.get(request.end.key);
                auto history=from->lines.at(request.end.line);
                if(request.end.reverse)std::reverse(history.begin(),history.end());
                history.erase(std::unique(history.begin(),history.end(),[](const auto& a,const auto& b){return cv::norm(a-b)<1e-9;}),history.end());
                if(history.size()<2)continue;
                const auto direction=(history.back()-history[history.size()-2])/cv::norm(history.back()-history[history.size()-2]);
                const auto widths=[&](const auto& points){return sampleFiberChunkWidths(source,points,scale);};
                const std::vector<size_t> selected(hits.begin(),hits.end());
                for(const auto& match:findFiberChunkMatches(*chunk,history,direction,chunks,false,nullptr,widths,trace.widthRelativeTolerance,&selected)){
                    if(cv::norm(match.point-history.back())>1e-9 &&
                       !checkFiberChunkSplice(source,trace,scale,history.back(),match.point,direction).empty())continue;
                    // Exit at source -> entry at target. The same geometric link
                    // discovered from the other side is one undirected edge.
                    candidates[i].push_back({request.end,{ordered[i],match.line,!match.reverse}});
                }
            }
        },1);connect.join();run.check();
        detail::FiberContinuationGraph continuations;
        for(auto& list:candidates){for(const auto& edge:list)continuations.add(edge.first,edge.second);list.clear();list.shrink_to_fit();}
        RouteJson connectionData=RouteJson::array();size_t acceptedConnections=0,ambiguousEnds=0;
        for(const auto& [end,degree]:continuations.degrees())ambiguousEnds+=degree>1;
        for(const auto& [a,b]:continuations.edges()){
            const auto reason=continuations.reason(a,b);acceptedConnections+=reason.empty();
            connectionData.push_back({{"from",{a.key,a.line,a.reverse}},{"to",{b.key,b.line,b.reverse}},
                {"from_degree",continuations.degree(a)},{"to_degree",continuations.degree(b)},{"rejection",reason}});
        }
        RouteJson prefilter={{"generation",generation},{"algorithm","unique-both-ends-v1"},
            {"candidate_connections",continuations.edges().size()},{"accepted_connections",acceptedConnections},
            {"ambiguous_ends",ambiguousEnds},{"connections",std::move(connectionData)}};
        const auto prefilterFile=output/"continuations.msgpack.zst";
        if(std::filesystem::exists(prefilterFile)){
            if(detail::readFiberPacked(prefilterFile)!=prefilter)throw std::runtime_error("continuation prefilter changed");
        }else detail::writeFiberPacked(prefilterFile,prefilter);
        std::vector<size_t> parent(nodes);std::iota(parent.begin(),parent.end(),0);
        auto root=[&](size_t v){while(parent[v]!=v){parent[v]=parent[parent[v]];v=parent[v];}return v;};
        for(auto& list:edges){for(auto [a,b]:list){a=root(a);b=root(b);if(a!=b)parent[std::max(a,b)]=std::min(a,b);}list.clear();list.shrink_to_fit();}
        incoming.clear();incoming.shrink_to_fit();
        for(size_t i=0;i<nodes;++i)parent[i]=root(i);
        struct Work {size_t root;std::vector<std::pair<size_t,size_t>> seeds;}; // metadata index, seed index
        std::map<size_t,std::vector<std::pair<size_t,size_t>>> byComponent;
        std::map<size_t,size_t> componentSizes;
        for(size_t i=0;i<metadata.size();++i){
            for(size_t line=0;line<metadata[i].eligible.size();++line)if(metadata[i].eligible[line])
                ++componentSizes[parent[metadata[i].offset+line]];
            for(size_t seed=0;seed<metadata[i].seeds.size();++seed)
                byComponent[parent[metadata[i].offset+metadata[i].seeds[seed].line]].emplace_back(i,seed);
        }
        std::vector<Work> work;std::vector<size_t> sizes;
        for(auto& [id,seeds]:byComponent){work.push_back({id,std::move(seeds)});sizes.push_back(componentSizes[id]);}
        std::sort(sizes.rbegin(),sizes.rend());
        RouteJson componentReport={
            {"generation",generation},{"components",work.size()},{"eligible_pieces",std::accumulate(sizes.begin(),sizes.end(),size_t(0))},
            {"largest_components",std::vector<size_t>(sizes.begin(),sizes.begin()+std::min(size_t(20),sizes.size()))},
            {"graph_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-graphStart).count()}};
        const auto componentFile=output/"components.msgpack.zst";
        if(std::filesystem::exists(componentFile)){
            auto previous=detail::readFiberPacked(componentFile),expected=componentReport;
            previous.erase("graph_seconds");expected.erase("graph_seconds");
            if(previous!=expected)throw std::runtime_error("ownership component inventory changed");
        }else detail::writeFiberPacked(componentFile,componentReport);
        run.total(work.size());next=0;
        const FiberChunkContinuationGate continuation=[&](const auto& previous,const auto& next){
            return continuations.reasonForJoin(previous,next);
        };
        auto queue=std::make_unique<FiberCaseQueue>(workers,work.size(),[&] {
            run.check();const auto& job=work[next.fetch_add(1)];const auto stem="component_"+std::to_string(job.root);
            const auto indexPath=output/(stem+".routes.msgpack.zst");
            RouteJson seeds=RouteJson::array();
            for(auto [i,s]:job.seeds){const auto& seed=metadata[i].seeds[s];
                seeds.push_back({ordered[i],seed.line,seed.vertex,seed.targets});}
            std::map<size_t,detail::FiberRouteOwnership> owned;
            auto nodeFor=[&](const RouteKey& key,size_t line){
                const auto i=fileIndex.at(key);
                if(line>=metadata[i].eligible.size()||!metadata[i].eligible[line])
                    throw std::runtime_error("route selected an ineligible cached piece");
                const auto id=metadata[i].offset+line;
                if(parent[id]!=job.root)throw std::runtime_error("route crossed ownership component");
                return std::pair{id,metadata[i].lengths[line]};
            };
            auto ownership=[&](size_t id,double length)->detail::FiberRouteOwnership&{
                return owned.try_emplace(id,length).first->second;
            };
            auto coverageComplete=[&]{
                for(auto [i,s]:job.seeds){const auto& seed=metadata[i].seeds[s];const auto id=metadata[i].offset+seed.line;
                    const auto& arcs=ownership(id,metadata[i].lengths[seed.line]);
                    for(auto [lo,hi]:seed.targets)if(arcs.uncovered(lo,hi))
                        throw std::runtime_error("unexported core interval in completed component");}
            };
            if(std::filesystem::exists(indexPath)){
                const auto index=detail::readFiberPacked(indexPath);
                if(index.at("generation")!=generation||index.at("component")!=job.root||index.at("seeds")!=seeds)
                    throw std::runtime_error("incompatible owned route component");
                size_t totalPoints=0,totalFibers=0;
                for(size_t page=0;page<index.at("pages").size();++page){
                    run.check();const auto data=detail::readFiberPacked(output/(stem+".routes."+std::to_string(page)+".msgpack.zst"));
                    if(data.at("generation")!=generation||data.at("component")!=job.root||data.at("page")!=page)
                        throw std::runtime_error("incompatible owned route page");
                    size_t pagePoints=0;
                    for(const auto& fiber:data.at("fibers")){
                        const auto key=fiber.at("source_key").get<RouteKey>();const auto line=fiber.at("source_line").get<size_t>();
                        const auto [id,length]=nodeFor(key,line);const double arc=fiber.at("source_arc");
                        if(!std::isfinite(arc)||arc<0||arc>length)throw std::runtime_error("invalid owned seed arc");
                        bool selected=false;
                        for(auto [i,s]:job.seeds)if(ordered[i]==key&&metadata[i].seeds[s].line==line)
                            for(auto [lo,hi]:metadata[i].seeds[s].targets)selected|=arc>=lo&&arc<=hi;
                        if(!selected)throw std::runtime_error("owned route seed outside selected core intervals");
                        const auto sourceChunk=cache.get(key);const auto seed=samplePolylineArc(makePolylineArcGeometry(sourceChunk->lines[line]),arc).point;
                        const auto& points=fiber.at("points_base");const auto seedIndex=fiber.at("seed_index").get<size_t>();
                        if(points.size()<2||seedIndex>=points.size()||fiber.at("seed_base")!=routeXYZ(seed)||points[seedIndex]!=routeXYZ(seed))
                            throw std::runtime_error("owned route seed mismatch");
                        double measured=0;cv::Vec3d previous;bool first=true;
                        for(const auto& p:points){
                            if(p.size()!=3)throw std::runtime_error("invalid route point");
                            for(const auto& v:p)if(!v.is_number()||!std::isfinite(v.get<double>()))throw std::runtime_error("nonfinite route point");
                            cv::Vec3d point(p[0],p[1],p[2]);if(!first)measured+=cv::norm(point-previous);previous=point;first=false;++pagePoints;
                        }
                        if(fiber.at("length_base")!=measured)throw std::runtime_error("owned route length mismatch");
                        if(fiber.at("stop_reasons").size()!=2||fiber.at("joins").size()!=2||fiber.at("owned_arcs").empty())
                            throw std::runtime_error("invalid owned route provenance");
                        for(const auto& reason:fiber.at("stop_reasons"))
                            if(!reason.is_string()||reason.get<std::string>().empty())throw std::runtime_error("invalid owned route stop");
                        if(fiber.at("cyclic_candidate").get<bool>()!=(fiber.at("stop_reasons")[0]=="chunk_route_cycle"||fiber.at("stop_reasons")[1]=="chunk_route_cycle"))
                            throw std::runtime_error("owned route cycle flag mismatch");
                        for(const auto& half:fiber.at("joins"))for(size_t j=1;j<half.size();++j){
                            const auto& a=half[j-1];const auto& b=half[j];
                            if(!continuations.reason({a.at("key").get<RouteKey>(),a.at("line").get<size_t>(),a.at("reverse").get<bool>()},
                                {b.at("key").get<RouteKey>(),b.at("line").get<size_t>(),!b.at("reverse").get<bool>()}).empty())
                                throw std::runtime_error("saved route crosses rejected continuation");
                        }
                        std::map<std::pair<size_t,bool>,std::vector<std::pair<double,double>>> supported;
                        for(const auto& half:fiber.at("joins"))for(const auto& join:half){
                            const auto targetKey=join.at("key").get<RouteKey>();const auto targetLine=join.at("line").get<size_t>();
                            const auto [target,targetLength]=nodeFor(targetKey,targetLine);const bool reverse=join.at("reverse").get<bool>();
                            auto path=cache.get(targetKey)->lines.at(targetLine);if(reverse)std::reverse(path.begin(),path.end());
                            const double at=join.at("arc"),begin=join.at("represented_overlap_begin_arc");
                            const double overlap=join.at("overlap_base"),splice=join.at("splice_base");
                            if(!std::isfinite(at)||!std::isfinite(begin)||!std::isfinite(overlap)||!std::isfinite(splice)||
                               at<0||at>targetLength||begin<0||begin>at||overlap<0||splice<0||splice>chunks.joinDistanceBase+1e-7)
                                throw std::runtime_error("invalid owned route join arc");
                            const auto sample=samplePolylineArc(makePolylineArcGeometry(path),at);
                            const auto point=join.at("point_base").get<std::array<double,3>>();
                            if(!std::isfinite(point[0])||!std::isfinite(point[1])||!std::isfinite(point[2])||
                               cv::norm(sample.point-cv::Vec3d(point[0],point[1],point[2]))>1e-7||join.at("segment").get<size_t>()+1>=path.size())
                                throw std::runtime_error("invalid owned route join point");
                            supported[{target,false}].emplace_back(reverse?0:at,reverse?targetLength-at:targetLength);
                            supported[{target,true}].emplace_back(reverse?targetLength-at:begin,reverse?targetLength-begin:at);
                        }
                        double claimedLength=0;
                        for(const auto& claim:fiber.at("owned_arcs")){
                            const auto [target,targetLength]=nodeFor(claim.at("key").get<RouteKey>(),claim.at("line").get<size_t>());
                            const double lo=claim.at("lo"),hi=claim.at("hi");
                            auto ranges=supported[{target,claim.at("alias").get<bool>()}];std::sort(ranges.begin(),ranges.end());
                            double reached=lo;
                            for(auto [a,b]:ranges)if(a<=reached+1e-7)reached=std::max(reached,b);
                            if(reached<hi-1e-7)throw std::runtime_error("owned arc not supported by accepted joins");
                            ownership(target,targetLength).add(lo,hi);
                            if(!claim.at("alias").get<bool>())claimedLength+=hi-lo;
                        }
                        if(claimedLength>measured+1e-7*std::max(1.,measured))
                            throw std::runtime_error("claimed route exceeds emitted geometry");
                        ++totalFibers;
                    }
                    if(index.at("pages")[page].at("fibers")!=data.at("fibers").size()||index.at("pages")[page].at("points")!=pagePoints)
                        throw std::runtime_error("owned route page count mismatch");
                    totalPoints+=pagePoints;
                }
                if(index.at("points")!=totalPoints||index.at("fibers")!=totalFibers)throw std::runtime_error("owned route count mismatch");
                coverageComplete();run.completed("reused");return;
            }
            RouteJson pages=RouteJson::array(),fibers=RouteJson::array();size_t pagePoints=0,totalPoints=0,totalFibers=0;
            auto flush=[&]{
                if(fibers.empty())return;const auto number=pages.size(),count=fibers.size();
                RouteJson page={{"generation",generation},{"component",job.root},{"page",number},{"fibers",std::move(fibers)}};
                detail::writeFiberPacked(output/(stem+".routes."+std::to_string(number)+".msgpack.zst"),page);
                pages.push_back({{"fibers",count},{"points",pagePoints}});
                totalPoints+=pagePoints;totalFibers+=count;pagePoints=0;fibers=RouteJson::array();
            };
            for(auto [i,s]:job.seeds){
                const auto& item=metadata[i].seeds[s];const auto key=ordered[i];const auto lineId=item.line;
                const auto chunk=cache.get(key);const auto geometry=makePolylineArcGeometry(chunk->lines[lineId]);
                const auto sourceId=metadata[i].offset+lineId;auto& sourceOwned=ownership(sourceId,geometry.length());
                // Continue uncovered core intervals after a partial handoff or budget stop;
                // never mark an unvisited prefix/suffix as consumed.
                for(;;){
                    run.check();std::optional<std::pair<double,double>> gap;
                    for(auto [lo,hi]:item.targets)if((gap=sourceOwned.uncovered(lo,hi)))break;
                    if(!gap)break;
                    const double preferred=geometry.vertexArcs.at(item.vertex);
                    const double seedArc=gap->second-gap->first<=4*sourceOwned.tolerance()?gap->first:
                        (preferred>gap->first+sourceOwned.tolerance()&&preferred<gap->second-sourceOwned.tolerance()?
                        preferred:(gap->first+gap->second)*.5);
                    const auto sample=samplePolylineArc(geometry,seedArc);const auto seed=sample.point;
                    std::array<std::vector<cv::Vec3d>,2> halves;std::array<std::string,2> stops;
                    RouteJson joins=RouteJson::array({RouteJson::array(),RouteJson::array()});
                    std::map<std::pair<RouteKey,size_t>,detail::FiberRouteOwnership> claims;
                    std::map<std::pair<RouteKey,size_t>,detail::FiberRouteOwnership> aliases;
                    auto policy=detail::fiberRouteOwnershipPolicy([&](const FiberChunkRouteJoin& j){
                        const auto [id,length]=nodeFor(j.key,j.match.line);
                        return std::pair{&ownership(id,length),length};
                    },[&](const FiberChunkRouteJoin& j,double lo,double hi,bool alias){
                        const auto [id,length]=nodeFor(j.key,j.match.line);
                        (alias?aliases:claims).try_emplace({j.key,j.match.line},length).first->second.add(lo,hi);
                    });
                    for(int half=0;half<2;++half){
                        FiberChunkProvider provider=[&](const RouteKey& target){
                            run.check();if(!inventory.contains(target))throw OutsideRouteInventory{};return cache.get(target);
                        };
                        FiberChunkInitialRegion region;region.center=seed;
                        const FiberChunkRouteStart start{lineId,sample.segmentIndex,half==0,seedArc};
                        try{stops[half]=traceFiberChunkRoute(provider,chunks,source,trace,scale,region,
                            half==0?-sample.tangent:sample.tangent,config.maximumDistanceBase,config.maximumSteps,
                            [&](const cv::Vec3d& p,double,int step){if(step%256==0)run.check();halves[half].push_back(p);return true;},
                            [&](const FiberChunkRouteJoin& j){joins[half].push_back({{"key",j.key},{"line",j.match.line},
                                {"reverse",j.match.reverse},{"arc",j.match.arc},{"point_base",routeXYZ(j.match.point)},
                                {"segment",j.match.segment},{"overlap_base",j.match.overlap},{"splice_base",j.spliceDistanceBase},
                                {"represented_overlap_begin_arc",j.representedOverlapBeginArc}});},
                            &start,&normals,&policy,continuation);
                        }catch(const OutsideRouteInventory&){stops[half]="chunk_inventory_boundary";}
                    }
                    if(claims.empty()||sourceOwned.uncovered(gap->first,gap->second)==gap)
                        throw std::runtime_error("owned route failed to cover its selected seed interval");
                    RouteJson points=RouteJson::array(),arcs=RouteJson::array();double length=0;cv::Vec3d previous;bool first=true;
                    auto append=[&](const cv::Vec3d& p){if(!first)length+=cv::norm(p-previous);first=false;previous=p;points.push_back(routeXYZ(p));};
                    for(auto p=halves[0].rbegin();p!=halves[0].rend();++p)append(*p);
                    append(seed);for(const auto& p:halves[1])append(p);
                    for(const auto& [target,claim]:claims)for(auto [lo,hi]:claim.intervals())
                        arcs.push_back({{"key",target.first},{"line",target.second},{"lo",lo},{"hi",hi},{"alias",false}});
                    for(const auto& [target,claim]:aliases)for(auto [lo,hi]:claim.intervals())
                        arcs.push_back({{"key",target.first},{"line",target.second},{"lo",lo},{"hi",hi},{"alias",true}});
                    if(pagePoints&&pagePoints+points.size()>config.pagePoints)flush();
                    pagePoints+=points.size();
                    fibers.push_back({{"source_key",key},{"source_line",lineId},{"source_arc",seedArc},
                        {"seed_base",routeXYZ(seed)},{"seed_index",halves[0].size()},{"length_base",length},{"points_base",std::move(points)},
                        {"cyclic_candidate",stops[0]=="chunk_route_cycle"||stops[1]=="chunk_route_cycle"},
                        {"stop_reasons",stops},{"joins",std::move(joins)},{"owned_arcs",std::move(arcs)}});
                    if(pagePoints>=config.pagePoints)flush();
                }
            }
            coverageComplete();flush();
            detail::writeFiberPacked(indexPath,{{"generation",generation},{"component",job.root},{"seeds",seeds},
                {"pages",pages},{"points",totalPoints},{"fibers",totalFibers}});
            run.completed("built");
        },contexts);
        run.attach(queue.get());
        try{queue->join();}catch(...){run.attach(nullptr);throw;}
        run.attach(nullptr);run.finish();
    }catch(const std::exception& e){run.failed(e.what());throw;}
    catch(...){run.failed("unknown route export error");throw;}
}

std::optional<PolylineArcSample> findFiberReplayRestartSeed(
    const FiberPredictionSource& predictions, const FiberTraceConfig& trace, double scale,
    const vc::lasagna::NormalSampler& normalSampler, double normalWorkingToBaseScale,
    const PolylineArcGeometry& reference, double fromArc, double endArcBase,
    double nominalStepBase)
{
    const double grid = scale;
    const double minDot = std::cos(25.0 * std::acos(-1.0) / 180.0);
    for (double arc = fromArc; arc < endArcBase; arc += nominalStepBase) {
        auto ref = samplePolylineArc(reference, arc);
        const auto tangent = ref.tangent / cv::norm(ref.tangent);
        const auto normal = normalSampler.sampleNormal(ref.point / normalWorkingToBaseScale);
        cv::Vec3d u = normal.valid ? normal.normal : cv::Vec3d(0, 0, 1);
        u -= tangent * u.dot(tangent);
        if (cv::norm(u) < 1e-6) {
            u = cv::Vec3d(0, 1, 0) - tangent * tangent[1];
            if (cv::norm(u) < 1e-6) u = cv::Vec3d(1, 0, 0) - tangent * tangent[0];
        }
        u /= cv::norm(u);
        const cv::Vec3d v = tangent.cross(u);
        const auto sample = [&](const cv::Vec3d& point) {
            FiberPredictionSampleOption best;
            for (const auto& option : predictions.sample(point / grid, tangent).options)
                if (option.valid && option.presence > best.presence) best = option;
            return best;
        };
        std::optional<PolylineArcSample> chosen;
        double bestDistance = std::numeric_limits<double>::infinity();
        for (int i = -int(std::ceil(10 / grid)); i <= int(std::ceil(10 / grid)); ++i)
            for (int j = -int(std::ceil(20 / grid)); j <= int(std::ceil(20 / grid)); ++j) {
                const cv::Vec3d point = ref.point + grid * (i * u + j * v);
                const auto distance = measureFiberReplayThreshold(point, ref.point,
                    normalSampler, normalWorkingToBaseScale, 10, 2);
                if (fiberReplayThresholdExceeded(distance, 10)) continue;
                const auto value = sample(point);
                cv::Vec3d dir(value.direction);
                if (!value.valid || value.presence <= 0 || value.presence < trace.replayPresenceCutoff || cv::norm(dir) <= 1.0e-12) continue;
                dir /= cv::norm(dir);
                const double dot = dir.dot(tangent);
                if (std::abs(dot) < minDot) continue;
                if (trace.widthRelativeTolerance > 0) {
                    std::vector<float> widths;
                    predictions.sampleWidths({cv::Vec3f(point / grid)}, widths);
                    if (widths.size() != 1 || !std::isfinite(widths.front()) || widths.front() <= 0)
                        continue;
                }
                bool peak = true, strict = false;
                for (int a = -1; a <= 1; ++a) for (int b = -1; b <= 1; ++b) {
                    if (!a && !b) continue;
                    const double other = sample(point + grid * (a * u + b * v)).presence;
                    if (other > value.presence) peak = false;
                    if (other < value.presence) strict = true;
                }
                if (peak && strict && distance.thresholdErrorRatio < bestDistance) {
                    bestDistance = distance.thresholdErrorRatio;
                    chosen = ref;
                    chosen->point = point;
                    chosen->tangent = dot < 0 ? -dir : dir;
                }
            }
        if (chosen) return chosen;
    }
    return std::nullopt;
}

static FiberReplayTraceResult replayBeamCase(
    const FiberPredictionSource& predictions, const FiberTraceConfig& trace, double scale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalScale,
    const FiberEvaluationCase& c, const FiberChunkReplayConfig& config,
    double reverseInterval, const FiberTraceProgressCallback& progress, bool trainingProtocol,
    bool predictionBasedSeeding = false)
{
    const FiberTraceCoordinateAdapter coordinates(scale);
    validateFiberTraceConfig(trace);
    if(!trainingProtocol && (trace.replayReverseDelaySteps>0 || trace.replayReverseConvergenceMinBase>0))
        throw std::invalid_argument("delayed reverse verification requires training replay protocol");
    const auto reference = makePolylineArcGeometry(c.referenceBase);
    const double end = c.endArcBase.value_or(reference.length());
    const double interval = !trainingProtocol || reverseInterval > 0 ? reverseInterval : trace.replayReverseIntervalBase;
    auto policy = fiberChunkReplayConsistency(predictions, tracingNormals, trace, config, interval);
    policy.seedAllStarts = predictionBasedSeeding;
    policy.toStart = trainingProtocol && trace.replayReverseToStart;
    policy.allEndpoints = trainingProtocol && trace.replayReverseAllEndpoints;
    policy.nextSeed = [&](double arc) -> std::optional<PolylineArcSample> {
        if (arc >= end) return std::nullopt;
        return samplePolylineArc(reference, arc);
    };
    double widthMean = trace.widthReferenceMean;
    bool rejectedPresencePoint = false;
    float committedModulatedPresence = std::numeric_limits<float>::quiet_NaN();
    if (trainingProtocol && trace.replayPresenceCutoff > 0) {
        policy.acceptPoint = [&](const cv::Vec3d& point, const cv::Vec3d& previous) {
            if (trace.replayPresenceModulated) {
                if (!std::isfinite(committedModulatedPresence))
                    throw std::logic_error("missing committed modulated presence");
                rejectedPresencePoint = committedModulatedPresence < trace.replayPresenceCutoff;
                return !rejectedPresencePoint;
            }
            const auto direction = point - previous;
            float presence = 0;
            for (const auto& option : predictions.sample(coordinates.baseToTrace(point), direction).options)
                if (option.valid && std::isfinite(option.presence)) presence = std::max(presence, option.presence);
            rejectedPresencePoint = presence < trace.replayPresenceCutoff;
            return !rejectedPresencePoint;
        };
    }
    if (trainingProtocol && (predictionBasedSeeding || interval > 0 || policy.acceptPoint)) {
        policy.nextSeed = [&](double arc) {
            return findFiberReplayRestartSeed(predictions, trace, scale, evaluationNormals,
                normalScale, reference, arc, end, trace.stepVoxels * scale);
        };
    }
    if (interval > 0) policy.verify = [&](const auto& points) {
        auto reverse = trace;
        reverse.traceToBaseScale = scale;
        if (reverse.widthRelativeTolerance > 0) reverse.widthReferenceMean = widthMean;
        const auto path = makePolylineArcGeometry(points);
        return verifyFiberReverseSpan(predictions, tracingNormals, reverse, path, path.length(),
            trainingProtocol ? trace.replayReverseNormalRadiusBase : config.errorThresholdBase,
            trainingProtocol ? trace.replayReverseTangentialRadiusBase : config.errorThresholdBase * config.tangentialThresholdFactor,
            trainingProtocol ? trace.replayReverseHistorySteps : 0) <= 1e-12;
    };
    if (interval > 0 && trainingProtocol &&
        (trace.replayReverseDelaySteps > 0 || trace.replayReverseConvergenceMinBase > 0)) {
        policy.delaySteps=static_cast<size_t>(trace.replayReverseDelaySteps);
        policy.verifyCheckpoint=[&](const auto& points,size_t checkpoint,size_t checked,bool) {
            const auto begin=std::chrono::steady_clock::now();
            auto reverse=trace;
            reverse.traceToBaseScale=scale;
            if(reverse.widthRelativeTolerance>0) reverse.widthReferenceMean=widthMean;
            const auto path=makePolylineArcGeometry(points);
            // Arc geometry keeps vertices (including zero-length edges), so the
            // checkpoint and verified indices refer to the exact committed path.
            ReverseSpanOptions options;
            options.initialDirections=fiberReverseFutureHistory(path,checkpoint,policy.delaySteps);
            options.verifiedThroughArc=path.vertexArcs.at(checked);
            options.minimumTravel=trace.replayReverseConvergenceMinBase;
            options.convergenceDistance=trace.replayReverseConvergenceDistanceBase;
            options.convergenceAngleDegrees=trace.replayReverseConvergenceAngleDegrees;
            options.convergenceSteps=trace.replayReverseConvergenceSteps;
            const auto check=verifyFiberReverseSpanDetailed(predictions,tracingNormals,reverse,path,
                path.vertexArcs.at(checkpoint),trace.replayReverseNormalRadiusBase,
                trace.replayReverseTangentialRadiusBase,0,options);
            return ReplayReverseCheck{check.passed,check.travelled,check.converged,
                options.initialDirections.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()};
        };
    }
    const ReplaySegmentRunner run = [&](const PolylineArcSample& seed, double remaining,
        int maximumSteps, const ReplayPointObserver& observe, const FiberTraceProgressCallback& report) {
        FiberTraceOneWayRequest request;
        request.startPoint = coordinates.baseToTrace(seed.point);
        request.targetPoint = coordinates.baseToTrace(samplePolylineArc(reference, end).point);
        request.initialDirection = seed.tangent;
        request.budgetSpanVoxels = remaining / scale;
        request.config = trace;
        // Candidate feasibility must use the same cutoff as committed-step
        // acceptance. Clearing it here lets an infeasible speculative branch
        // displace a feasible one before the observer can reject that step.
        request.config.traceToBaseScale = scale;
        widthMean = trace.widthReferenceMean;
        double acceptedWidthMean = widthMean;
        return traceFiberReplaySegment(predictions, request, maximumSteps, tracingNormals, report,
            [&](const auto& point, double loss, int step) {
                rejectedPresencePoint=false;
                const bool proceed=observe(coordinates.traceToBase(point), loss, step);
                if(policy.verifyCheckpoint && rejectedPresencePoint) widthMean=acceptedWidthMean;
                else acceptedWidthMean=widthMean;
                return proceed;
            }, widthMean, &committedModulatedPresence);
    };
    return traceReferenceReplayCore(reference, c.beginArcBase, end, trace.stepVoxels * scale,
        trace.maxStepFactor, config.errorThresholdBase, config.excursionGraceBase,
        config.tangentialThresholdFactor, config.matchRefineSteps, evaluationNormals,
        normalScale, run, progress, {}, policy);
}

FiberReplayTraceResult replayFiberBeamCase(
    const FiberPredictionSource& predictions, const FiberTraceConfig& trace, double scale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalScale,
    const FiberEvaluationCase& c, const FiberChunkReplayConfig& config,
    double reverseInterval, const FiberTraceProgressCallback& progress)
{
    return replayBeamCase(predictions, trace, scale, tracingNormals, evaluationNormals,
        normalScale, c, config, reverseInterval, progress, false);
}

FiberReplayTraceResult replayFiberTrainingCase(
    const FiberPredictionSource& predictions, const FiberTraceConfig& trace, double scale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalScale,
    const FiberEvaluationCase& c, const FiberChunkReplayConfig& config,
    const FiberTraceProgressCallback& progress, bool predictionBasedSeeding)
{
    return replayBeamCase(predictions, trace, scale, tracingNormals, evaluationNormals,
        normalScale, c, config, 0, progress, true, predictionBasedSeeding);
}

static std::vector<FiberReplayTraceResult> evaluateCases(
    FiberChunkTraceCache* cache, const FiberPredictionSource& predictions,
    const FiberTraceConfig& traceInput, double scale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalScale,
    const std::vector<FiberEvaluationCase>& input, const FiberChunkReplayConfig& replayInput,
    FiberEvaluationControl& control, size_t workers, size_t contexts, double reverseInterval)
{
    FiberEvaluationRun run(control, input.size());
    try {
        run.check();
        if (!workers || !contexts) throw std::invalid_argument("evaluation worker/context budgets must be positive");
        if (traceInput.parallelThreads != 1)
            throw std::invalid_argument("evaluation requires parallel_threads=1; the case queue owns parallelism");
        if (!predictions.supportsConcurrentSampling() || !tracingNormals.supportsConcurrentSampling() ||
            !evaluationNormals.supportsConcurrentSampling())
            throw std::invalid_argument("evaluation requires concurrent sampling sources");
        if (traceInput.profile)
            throw std::invalid_argument("batch evaluation cannot share a mutable trace profile");
        const auto trace = traceInput;
        const auto config = replayInput;
        const auto cases = input;
        const auto policy = fiberChunkReplayConsistency(predictions, tracingNormals, trace, config, reverseInterval);
        std::vector<PolylineArcGeometry> references;
        for (const auto& c : cases) {
            auto line = makePolylineArcGeometry(c.referenceBase);
            const auto end = c.endArcBase.value_or(line.length());
            if (!std::isfinite(c.beginArcBase) || !std::isfinite(end) ||
                c.beginArcBase < 0 || end <= c.beginArcBase || end > line.length())
                throw std::invalid_argument("evaluation case has invalid reference arc interval");
            references.push_back(std::move(line));
        }
        std::vector<FiberReplayTraceResult> results(cases.size());
        std::atomic<size_t> next{0};
        // Keep the queue alive until detached from cancel(), including on error.
        auto queue = std::make_unique<FiberCaseQueue>(workers, cases.size(), [&] {
            run.check();
            const size_t i = next.fetch_add(1);
            if (!cache) results[i] = replayFiberBeamCase(predictions, trace, scale, tracingNormals,
                evaluationNormals, normalScale, cases[i], config, reverseInterval,
                [&](const auto&) { run.check(); });
            else results[i] = traceFiberChunkReferenceReplay(*cache, predictions, trace, scale, tracingNormals,
                references[i], cases[i].beginArcBase, cases[i].endArcBase.value_or(references[i].length()),
                evaluationNormals, normalScale, config, FiberCaseQueue::parallelFor,
                [](const auto& ready) { FiberCaseQueue::await(ready); },
                [&](const auto&) { run.check(); }, {}, policy);
            run.check();
            run.completed();
        }, contexts);
        run.attach(queue.get());
        try { queue->join(); }
        catch (...) { run.attach(nullptr); throw; }
        run.attach(nullptr);
        run.finish();
        return results;
    } catch (const std::exception& e) {
        run.failed(e.what());
        throw;
    } catch (...) {
        run.failed("unknown evaluation error");
        throw;
    }
}
std::vector<FiberReplayTraceResult> evaluateFiberChunkCases(
    FiberChunkTraceCache& cache, const FiberPredictionSource& predictions,
    const FiberTraceConfig& trace, double scale, const vc::lasagna::NormalSampler& normals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalScale,
    const std::vector<FiberEvaluationCase>& cases, const FiberChunkReplayConfig& config,
    FiberEvaluationControl& control, size_t workers, size_t contexts, double reverseInterval)
{
    return evaluateCases(&cache, predictions, trace, scale, normals, evaluationNormals,
        normalScale, cases, config, control, workers, contexts, reverseInterval);
}

std::vector<FiberReplayTraceResult> evaluateFiberBeamCases(
    const FiberPredictionSource& predictions, const FiberTraceConfig& trace, double scale,
    const vc::lasagna::NormalSampler& normals, const vc::lasagna::NormalSampler& evaluationNormals,
    double normalScale, const std::vector<FiberEvaluationCase>& cases,
    const FiberChunkReplayConfig& config, FiberEvaluationControl& control,
    size_t workers, size_t contexts, double reverseInterval)
{
    return evaluateCases(nullptr, predictions, trace, scale, normals, evaluationNormals,
        normalScale, cases, config, control, workers, contexts, reverseInterval);
}
} // namespace vc::fiber_tracer
