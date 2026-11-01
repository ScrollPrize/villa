#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include "vc/fiber_tracer/FiberEvaluation.hpp"
#include "vc/fiber_tracer/FiberWidth.hpp"
#include "vc/fiber_tracer/detail/PolylineSegmentIndex.hpp"
#include "vc/fiber_tracer/detail/FiberContinuationGraph.hpp"
#include "vc/fiber_tracer/detail/FiberRouteOwnership.hpp"
#include "vc/fiber_tracer/detail/FiberPackedStorage.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>

namespace vc::fiber_tracer {
namespace {
constexpr double stepBase=16;
cv::Vec3d unit(const cv::Vec3d& p) {
    const double n=cv::norm(p);
    return n>1e-12?p/n:cv::Vec3d(0,0,0);
}
std::array<int,3> keyAt(const cv::Vec3d& p,double core) {
    if(!std::isfinite(core)||core<=0) throw std::invalid_argument("invalid chunk core size");
    std::array<int,3> key;
    for(int a=0;a<3;++a) {
        const double v=std::floor(p[a]/core);
        if(!std::isfinite(v)||v<std::numeric_limits<int>::min()||v>std::numeric_limits<int>::max())
            throw std::invalid_argument("chunk key outside integer range");
        key[a]=static_cast<int>(v);
    }
    return key;
}
std::shared_ptr<const FiberChunkPaths> getChunk(const FiberChunkProvider& get,const std::array<int,3>& key) {
    auto chunk=get(key);
    if(!chunk||chunk->key!=key||chunk->endReasons.size()!=chunk->lines.size())
        throw std::runtime_error("invalid chunk provider result");
    for(int a=0;a<3;++a)
        if(!std::isfinite(chunk->lo[a])||!std::isfinite(chunk->hi[a])||chunk->lo[a]>=chunk->hi[a])
            throw std::runtime_error("invalid chunk provider bounds");
    return chunk;
}
void validateRoute(const FiberTraceConfig& t,double scale) {
    validateFiberTraceConfig(t);
    if(!std::isfinite(scale)||scale<=0||std::abs(t.stepVoxels*scale-stepBase)>1e-6)
        throw std::invalid_argument("chunk route requires 16 base voxel steps");
}
std::vector<float> widthsAt(const FiberPredictionSource& source,const std::vector<cv::Vec3d>& points,double scale) {
    std::vector<cv::Vec3f> scaled;scaled.reserve(points.size());
    for(const auto& p:points) scaled.emplace_back(p/scale);
    std::vector<float> values;source.sampleWidths(scaled,values);return values;
}
// The existing route policy checks the bridge at <=2 base voxel spacing,
// against prediction evidence only. Never consult the reference here.
std::string checkSplice(const FiberPredictionSource& source,const FiberTraceConfig& t,double scale,
    const cv::Vec3d& previous,const cv::Vec3d& point,const cv::Vec3d& direction)
{
    const double count=std::ceil(cv::norm(point-previous)/2);
    if(!std::isfinite(count)||count>std::numeric_limits<int>::max())
        throw std::invalid_argument("splice too long");
    const auto referenceWidth=t.widthRelativeTolerance>0?widthsAt(source,{previous},scale):std::vector<float>{};
    for(int n=1;n<=static_cast<int>(count);++n) {
        const auto q=previous+(point-previous)*(n/count);
        FiberPredictionSampleOption best;
        for(const auto& o:source.sample(q/scale,direction).options)
            if(o.valid&&std::isfinite(o.presence)&&(!best.valid||o.presence>best.presence)) best=o;
        double presence=best.valid?best.presence*std::abs(unit(cv::Vec3d(best.direction)).dot(direction)):0;
        if(t.widthRelativeTolerance>0) {
            const auto w=widthsAt(source,{q},scale);
            if(referenceWidth.size()!=1||!std::isfinite(referenceWidth[0])||referenceWidth[0]<=0||
               w.size()!=1||!std::isfinite(w[0])||w[0]<=0) return "chunk_invalid_splice_width";
            presence*=fiberWidthPresenceScale(w[0],referenceWidth[0],t.widthRelativeTolerance,t.widthCosineWeight);
        }
        if(!best.valid||!std::isfinite(presence)||presence<t.replayPresenceCutoff) return "chunk_unsupported_splice";
    }
    return {};
}
FiberChunkInitialRegion initialRegion(const cv::Vec3d& center,const vc::lasagna::NormalSampler& normal,
    double scale,const FiberChunkReplayConfig& c)
{
    FiberChunkInitialRegion region;region.center=center;
    auto squared=[&](const cv::Vec3d& delta) {
        const double r=measureFiberReplayThreshold(center+delta,center,normal,scale,
            c.errorThresholdBase,c.tangentialThresholdFactor).thresholdErrorRatio;
        return r*r;
    };
    for(int a=0;a<3;++a) {cv::Vec3d axis(0,0,0);axis[a]=1;region.metric(a,a)=squared(axis);}
    for(int a=0;a<3;++a) for(int b=a+1;b<3;++b) {
        cv::Vec3d axes(0,0,0);axes[a]=axes[b]=1;
        region.metric(a,b)=region.metric(b,a)=(squared(axes)-region.metric(a,a)-region.metric(b,b))*.5;
    }
    return region;
}
}

std::array<int,3> fiberChunkKeyAt(const cv::Vec3d& point,double coreBase){return keyAt(point,coreBase);}
std::vector<float> sampleFiberChunkWidths(const FiberPredictionSource& source,const std::vector<cv::Vec3d>& points,double scale){
    return widthsAt(source,points,scale);
}
std::string checkFiberChunkSplice(const FiberPredictionSource& source,const FiberTraceConfig& trace,double scale,
    const cv::Vec3d& previous,const cv::Vec3d& point,const cv::Vec3d& direction){
    return checkSplice(source,trace,scale,previous,point,direction);
}
FiberChunkNext nextFiberChunk(const FiberChunkPaths& chunk,const cv::Vec3d& end,const cv::Vec3d& previous,
    const std::string& reason,const FiberChunkTraceConfig& c) {
    if(reason=="presence_cutoff"||reason=="invalid_width")return {{},reason};
    if(reason.starts_with("fused_member_stop:"))return {{},reason};
    const auto dir=unit(end-previous);bool boundary=false;
    for(int a=0;a<3;++a)boundary|=(dir[a]>0&&chunk.hi[a]-end[a]<32)||(dir[a]<0&&end[a]-chunk.lo[a]<32);
    if(!boundary)return {{},"chunk_local_stop:"+reason};
    const auto next=keyAt(end+dir*stepBase,c.coreBase);
    if(next==chunk.key)return {{},"chunk_no_neighbor_progress"};
    return {next,{}};
}

std::string traceFiberChunkRoute(const FiberChunkProvider& get,const FiberChunkTraceConfig& c,
    const FiberPredictionSource& source,const FiberTraceConfig& t,double scale,
    const FiberChunkInitialRegion& region,const cv::Vec3d& direction,double maximumDistance,int maximumSteps,
    const ReplayPointObserver& observe,const FiberChunkJoinObserver& joins,const FiberChunkRouteStart* start,
    const vc::lasagna::NormalSampler* normals,const FiberChunkRouteOwnership* ownership,
    const FiberChunkContinuationGate& continuation)
{
    validateRoute(t,scale);
    if(!get||!observe||!std::isfinite(maximumDistance)||maximumDistance<=0||maximumSteps<=0)
        throw std::invalid_argument("chunk route requires callbacks and positive finite budgets");
    auto key=keyAt(region.center,c.coreBase);
    std::vector<cv::Vec3d> history{region.center};
    std::set<std::tuple<std::array<int,3>,size_t,bool>> visited;
    bool initial=true;int steps=0;double distance=0;
    std::optional<FiberChunkRouteJoin> previousJoin;
    auto widths=[&](const auto& points){return widthsAt(source,points,scale);};
    // Limit distance as well as observer count even inside a long cached path.
    auto emit=[&](const cv::Vec3d& point)->std::string {
        if(steps>=maximumSteps||distance>=maximumDistance) return "chunk_length_limit";
        const double length=cv::norm(point-history.back());
        if(length<1e-9) return {};
        const double advance=std::min(length,maximumDistance-distance);
        const cv::Vec3d q=advance<length?history.back()+(point-history.back())*(advance/length):point;
        distance+=advance;history.push_back(q);
        if(!observe(q,0,++steps)) return "observer_stop";
        return steps>=maximumSteps||distance>=maximumDistance?"chunk_length_limit":"";
    };
    while(steps<maximumSteps&&distance<maximumDistance) {
        const auto chunk=getChunk(get,key);
        FiberChunkMatch match;
        if(initial&&start) {
            validateFiberChunkConflictAnnotation(*chunk,c);
            if(start->line>=chunk->lines.size()||start->vertex>=chunk->lines[start->line].size())
                throw std::invalid_argument("explicit chunk route seed outside path");
            if(c.conflictNormalBase>0) {
                if(chunk->conflictAnnotation.is_null()||chunk->conflictFlags.size()!=chunk->lines.size())
                    throw std::invalid_argument("explicit chunk route seed requires prepared flags");
                if(chunk->conflictFlags[start->line])return "chunk_seed_conflict";
            }
            const auto& points=chunk->lines[start->line];
            const auto geometry=makePolylineArcGeometry(points);
            const double arc=start->arc>=0?start->arc:geometry.vertexArcs.at(start->vertex);
            if(!std::isfinite(arc)||arc>geometry.length()||!std::isfinite(cv::norm(region.center))||
                cv::norm(samplePolylineArc(geometry,arc).point-region.center)>1e-9)
                throw std::invalid_argument("explicit chunk route seed coordinate mismatch");
            match.valid=points.size()>1;match.line=start->line;match.reverse=start->reverse;
            auto oriented=points;if(start->reverse)std::reverse(oriented.begin(),oriented.end());
            match.arc=start->reverse?geometry.length()-arc:arc;
            match.segment=samplePolylineArc(makePolylineArcGeometry(oriented),match.arc).segmentIndex;
            match.point=region.center;
        } else match=matchFiberChunkPaths(*chunk,history,direction,c,initial,initial?&region:nullptr,
            widths,t.widthRelativeTolerance,normals,scale);
        if(match.ambiguous)return "chunk_ambiguous_continuation";
        if(!match.valid) return initial?"chunk_no_initial_path":"chunk_no_overlap";
        if(!visited.emplace(key,match.line,match.reverse).second) return "chunk_route_cycle";
        auto line=chunk->lines.at(match.line);if(match.reverse) std::reverse(line.begin(),line.end());
        const double splice=initial?0:cv::norm(match.point-history.back());
        if(initial) {
            // Backward context is for overlap matching only: no emitted points
            // and no charged steps/distance before the initial snap.
            history.assign(line.begin(),line.begin()+match.segment+1);
            if(cv::norm(history.back()-match.point)>1e-9) history.push_back(match.point);
        }
        if(splice>1e-9) {
            const auto tangent=history.size()>1?unit(history.back()-history[history.size()-2]):unit(direction);
            const auto reason=checkSplice(source,t,scale,history.back(),match.point,tangent);
            if(!reason.empty()) return reason;
        }
        FiberChunkRouteJoin joining{key,match,initial,splice,match.arc};
        if(previousJoin&&continuation){
            const auto reason=continuation(*previousJoin,joining);
            if(!reason.empty())return reason;
        }
        previousJoin=joining;
        double available=ownership?ownership->available(joining):std::numeric_limits<double>::infinity();
        if(std::isnan(available)||available<0)throw std::logic_error("invalid route ownership allowance");
        if(available==0)return "chunk_piece_already_owned";
        if(ownership&&ownership->alias&&!initial){
            // Initial backward context is not emitted. Only alias overlap
            // supported by geometry actually emitted in this half-route.
            double begin=match.arc;
            for(auto [historyDistance,targetArc]:match.overlapArcs)
                if(historyDistance<=distance)begin=targetArc;
            joining.representedOverlapBeginArc=begin;
            if(begin<match.arc)ownership->alias(joining,begin,match.arc);
        }
        if(joins)joins(joining);
        if(splice>1e-9)if(const auto stop=emit(match.point);!stop.empty())return stop;
        const int before=steps;
        double advanced=0;
        for(size_t j=match.segment+1;j<line.size();++j) {
            const auto delta=line[j]-history.back();const double length=cv::norm(delta);
            if(length<1e-9)continue;
            const double remaining=available-advanced;
            if(remaining<=0)return "chunk_piece_already_owned";
            const auto target=length>remaining?history.back()+delta*(remaining/length):line[j];
            const auto previousPoint=history.back();const auto stop=emit(target);const double used=cv::norm(history.back()-previousPoint);
            if(ownership&&used>0)ownership->consume(joining,match.arc+advanced,match.arc+advanced+used);
            advanced+=used;
            if(!stop.empty())return stop;
            if(advanced>=available)return "chunk_piece_already_owned";
        }
        const auto& reason=chunk->endReasons.at(match.line)[match.reverse?0:1];
        if(reason=="presence_cutoff"||reason=="invalid_width") return reason;
        if(steps==before||history.size()<2) return "chunk_no_progress";
        const auto next=nextFiberChunk(*chunk,history.back(),history[history.size()-2],reason,c);
        if(!next.stop.empty())return next.stop;
        key=next.key;initial=false;
        // Matching examines at most 2*halo of trailing history, including the
        // complete segment crossing that distance. Earlier points have already
        // been emitted and cannot affect any later match/splice/stop decision.
        // Keep that exact suffix rather than revalidating an ever-growing fiber.
        double tail=0;size_t first=history.size()-1;
        const double context=std::max(2*c.haloBase,c.conflictAgreementBase+c.conflictSeparationBase+32);
        while(first>0&&tail<context) {tail+=cv::norm(history[first]-history[first-1]);--first;}
        // The matcher accumulates subdivided lengths; keep a predecessor beyond
        // the threshold so rounding at exactly 2*halo cannot discard context.
        if(first)--first;
        if(first)history.erase(history.begin(),history.begin()+first);
    }
    return "chunk_length_limit";
}

nlohmann::json fiberChunkRouteMatchingSpecification(const FiberChunkTraceConfig& config) {
    auto matching=fiberChunkConfigJson(config);
    for(const auto* name:{"gpu_library","gpu_devices","gpu_batch_halves","gpu_batch_memory_mib",
            "cache_only","worker_cache_bytes","channel_hints","pin_sampling_region","benchmark_export"})matching.erase(name);
    return matching;
}
struct FiberChunkRoutingRules::Impl {
    std::map<std::array<int,3>,FiberChunkPathReduction> reductions;
    detail::FiberContinuationGraph graph;
    std::filesystem::path source;
    nlohmann::json construction,matching;
};
FiberChunkRoutingRules::FiberChunkRoutingRules(const std::filesystem::path& directory,
    const std::filesystem::path& source,const FiberChunkTraceConfig& config,
    const FiberTraceConfig& trace,double scale,const std::string& identity)
{
    auto impl=std::make_shared<Impl>();
    const auto manifest=detail::readFiberPacked(directory/"routes.manifest.msgpack.zst");
    auto read=config;read.cacheOnly=true;
    if(manifest.at("construction")!=fiberChunkCacheSpecification(read,trace,scale,identity)||
       manifest.at("route")!=fiberChunkRouteMatchingSpecification(config))
        throw std::runtime_error("routing rules source/configuration mismatch");
    impl->source=std::filesystem::weakly_canonical(source);
    impl->construction=manifest.at("construction");impl->matching=manifest.at("route");
    const auto reductions=detail::readFiberPacked(directory/"path_reduction.msgpack.zst");
    const auto graph=detail::readFiberPacked(directory/"continuations.msgpack.zst");
    if(reductions.at("generation")!=manifest.at("generation")||graph.at("generation")!=manifest.at("generation")||
       graph.at("algorithm")!="unique-both-ends-v1"||!reductions.at("enabled").get<bool>())
        throw std::runtime_error("incompatible routing rule generation/algorithm");
    for(const auto& entry:reductions.at("chunks")){
        if(entry.at("algorithm")!="terminal-complete-link-v2"||
           entry.value("reject_ambiguous_paths",true)!=manifest.value("reject_ambiguous_paths",true))
            throw std::runtime_error("routing reduction mode mismatch");
        FiberChunkPathReduction r;
        r.representative=entry.at("representative").get<std::vector<int64_t>>();
        r.endReasons=entry.at("end_reasons").get<std::vector<std::array<std::string,2>>>();
        if(!impl->reductions.emplace(entry.at("key").get<std::array<int,3>>(),std::move(r)).second)
            throw std::runtime_error("duplicate routing reduction key");
    }
    const auto& inventory=manifest.at("inventory");std::atomic<size_t> next{0};
    if(inventory.size()!=impl->reductions.size())throw std::runtime_error("incomplete routing reduction inventory");
    FiberCaseQueue verify(16,inventory.size(),[&]{
        const auto& file=inventory.at(next.fetch_add(1));
        const auto name=file.at("file").get<std::string>();
        if(std::filesystem::path(name).filename()!=name)throw std::runtime_error("invalid rule source filename");
        const auto path=source/name;
        if(!impl->reductions.contains(file.at("key").get<std::array<int,3>>())||
           std::filesystem::file_size(path)!=file.at("bytes").get<uintmax_t>()||
           std::filesystem::last_write_time(path).time_since_epoch().count()!=file.at("mtime").get<int64_t>())
            throw std::runtime_error("routing rules source inventory changed: "+path.string());
    },1);verify.join();
    auto endpoint=[](const auto& e){return detail::FiberPathEnd{e.at(0).template get<std::array<int,3>>(),
        e.at(1).template get<size_t>(),e.at(2).template get<bool>()};};
    for(const auto& e:graph.at("connections"))impl->graph.add(endpoint(e.at("from")),endpoint(e.at("to")));
    for(const auto& e:graph.at("connections")){
        const auto a=endpoint(e.at("from")),b=endpoint(e.at("to"));
        if(impl->graph.degree(a)!=e.at("from_degree").get<size_t>()||impl->graph.degree(b)!=e.at("to_degree").get<size_t>()||
           impl->graph.reason(a,b)!=e.at("rejection").get<std::string>())throw std::runtime_error("invalid continuation graph degrees");
    }
    impl_=std::move(impl);
}
std::shared_ptr<const FiberChunkPaths> FiberChunkRoutingRules::apply(std::shared_ptr<const FiberChunkPaths> raw) const {
    const auto it=impl_->reductions.find(raw->key);
    if(it==impl_->reductions.end())throw std::runtime_error("chunk outside complete routing rule inventory");
    auto result=std::make_shared<FiberChunkPaths>(*raw);it->second.apply(*result);return result;
}
std::string FiberChunkRoutingRules::continuation(const FiberChunkRouteJoin& previous,const FiberChunkRouteJoin& next) const {
    return impl_->graph.reasonForJoin(previous,next);
}
void FiberChunkRoutingRules::validate(const FiberChunkTraceCache& cache,const FiberTraceConfig& trace,double scale) const {
    if(std::filesystem::weakly_canonical(cache.directory())!=impl_->source||
       fiberChunkCacheSpecification(cache.config(),trace,scale,cache.sourceIdentity())!=impl_->construction||
       fiberChunkRouteMatchingSpecification(cache.config())!=impl_->matching)
        throw std::invalid_argument("routing rules do not belong to this replay cache/configuration");
}

struct FiberStoredRouteSet::Impl {
    std::vector<PolylineArcGeometry> paths;
    std::vector<std::array<std::string,2>> reasons;
    detail::ChunkSegmentIndex index;
    Impl(std::vector<std::vector<cv::Vec3d>> lines,std::vector<std::array<std::string,2>> stops):reasons(std::move(stops)){
        if(lines.size()!=reasons.size())throw std::invalid_argument("stored route endpoint count mismatch");
        for(const auto& line:lines){
            if(line.size()<2)throw std::invalid_argument("stored route needs two points");
            for(const auto& p:line)for(int a=0;a<3;++a)if(!std::isfinite(p[a]))throw std::invalid_argument("nonfinite stored route");
            paths.push_back(makePolylineArcGeometry(line));
            if(!(paths.back().length()>0))throw std::invalid_argument("zero-length stored route");
        }
        index=detail::ChunkSegmentIndex(paths,[](const auto& p)->const auto&{return p.points;});
    }
};
FiberStoredRouteSet::FiberStoredRouteSet(std::vector<std::vector<cv::Vec3d>> lines,
    std::vector<std::array<std::string,2>> reasons):impl_(std::make_shared<Impl>(std::move(lines),std::move(reasons))){}

FiberReplayTraceResult FiberStoredRouteSet::replay(const FiberPredictionSource& source,const FiberTraceConfig& trace,
    double scale,const PolylineArcGeometry& reference,const vc::lasagna::NormalSampler& normals,double normalScale,
    const FiberChunkTraceConfig& input,const FiberChunkReplayConfig& config) const
{
    validateRoute(trace,scale);
    auto matching=input;matching.conflictNormalBase=matching.conflictTangentBase=0; // input already filtered
    auto initial=config;
    if(input.initialNormalRadiusBase>0){initial.errorThresholdBase=input.initialNormalRadiusBase;
        initial.tangentialThresholdFactor=input.initialTangentialRadiusBase/input.initialNormalRadiusBase;}
    if(!(initial.errorThresholdBase>0&&initial.tangentialThresholdFactor>0))throw std::invalid_argument("invalid stored-route seed ellipse");
    const auto select=[&](double arc){
        const auto ref=samplePolylineArc(reference,arc);
        const auto region=initialRegion(ref.point,normals,normalScale,initial);
        std::set<size_t> found;
        const double radius=initial.errorThresholdBase*std::max(1.,initial.tangentialThresholdFactor);
        impl_->index.nearbyPaths(ref.point,radius,[&](size_t id){found.insert(id);});
        FiberChunkPaths candidates;std::vector<size_t> ids(found.begin(),found.end());
        for(auto id:ids)candidates.lines.push_back(impl_->paths[id].points);
        auto match=matchFiberChunkPaths(candidates,{ref.point},ref.tangent,matching,true,&region);
        if(match.valid)match.line=ids.at(match.line);
        return match;
    };
    ReplayConsistencyPolicy policy;policy.seedAllStarts=true;
    policy.nextSeed=[&](double from)->std::optional<PolylineArcSample>{
        while(from<reference.length()){
            auto seed=input.presencePeakRestarts?
                findFiberReplayRestartSeed(source,trace,scale,normals,normalScale,reference,from,reference.length(),stepBase):
                std::optional(samplePolylineArc(reference,from));
            if(!seed)return std::nullopt;
            const auto match=select(seed->arc);
            if(match.valid){seed->point=match.point;return seed;}
            if(input.presencePeakRestarts)return seed;
            from=seed->arc+stepBase;
        }
        return std::nullopt;
    };
    const ReplaySegmentRunner run=[&](const PolylineArcSample& seed,double remaining,int maximumSteps,
        const ReplayPointObserver& observe,const FiberTraceProgressCallback&){
        const auto match=select(seed.arc);
        if(!match.valid)return std::string("chunk_no_initial_path");
        auto line=impl_->paths[match.line].points;
        if(match.reverse)std::reverse(line.begin(),line.end());
        const double budget=trace.maxStepFactor*remaining;
        double advanced=0;int steps=0;auto previous=match.point;
        // Preserve the actual committed vertices, including boundary splices;
        // resampling could cut a corner and change the reference-match metric.
        for(size_t j=match.segment+1;j<line.size();++j){
            const double distance=cv::norm(line[j]-previous);
            if(distance<1e-9)continue;
            if(steps>=maximumSteps||advanced>=budget)return std::string("chunk_length_limit");
            const double used=std::min(distance,budget-advanced);
            const auto point=used<distance?previous+(line[j]-previous)*(used/distance):line[j];
            advanced+=used;previous=point;
            if(!observe(point,0,++steps))return std::string("observer_stop");
            if(steps>=maximumSteps||advanced>=budget)return std::string("chunk_length_limit");
        }
        auto reason=impl_->reasons[match.line][match.reverse?0:1];
        // Keep the original terminal semantics through fusion's provenance wrapper.
        const std::string prefix="fused_member_stop:";
        while(reason.starts_with(prefix))reason.erase(0,prefix.size());
        return reason.empty()?std::string("stored_route_end"):reason;
    };
    return traceReferenceReplayCore(reference,0,reference.length(),stepBase,trace.maxStepFactor,
        config.errorThresholdBase,config.excursionGraceBase,config.tangentialThresholdFactor,
        config.matchRefineSteps,normals,normalScale,run,{},{},policy);
}

FiberReplayTraceResult traceFiberChunkReferenceReplay(FiberChunkTraceCache& cache,const FiberPredictionSource& source,
    const FiberTraceConfig& t,double scale,const vc::lasagna::NormalSampler& tracingNormals,
    const PolylineArcGeometry& reference,double start,double end,const vc::lasagna::NormalSampler& normals,
    double normalScale,const FiberChunkReplayConfig& c,const FiberChunkParallelFor& parallel,
    const FiberChunkWait& wait,const FiberTraceProgressCallback& progress,const FiberReplayFailureCallback& failure,
    const ReplayConsistencyPolicy& consistency,const FiberChunkJoinObserver& joins,const FiberChunkRoutingRules* rules)
{
    validateRoute(t,scale);
    if(rules)rules->validate(cache,t,scale);
    if(!std::isfinite(c.errorThresholdBase)||c.errorThresholdBase<=0)
        throw std::invalid_argument("chunk replay requires a positive initial acceptance radius");
    const auto& config=cache.config();
    auto initial=c;
    if(config.initialNormalRadiusBase>0) {
        initial.errorThresholdBase=config.initialNormalRadiusBase;
        initial.tangentialThresholdFactor=config.initialTangentialRadiusBase/config.initialNormalRadiusBase;
    }
    const FiberChunkProvider get=[&](const auto& key){
        auto raw=cache.get(key,source,t,tracingNormals,scale,parallel,wait);
        return rules?rules->apply(raw):raw;
    };
    auto policy=consistency;policy.seedAllStarts=true;
    policy.nextSeed=[&](double from)->std::optional<PolylineArcSample> {
        while(from<end) {
            auto seed=config.presencePeakRestarts?
                findFiberReplayRestartSeed(source,t,scale,normals,normalScale,reference,from,end,stepBase):
                (consistency.nextSeed?consistency.nextSeed(from):std::optional(samplePolylineArc(reference,from)));
            if(!seed) return std::nullopt;
            if(!std::isfinite(seed->arc)||seed->arc<from||seed->arc>=end)
                throw std::invalid_argument("chunk seed search did not advance within reference bounds");
            const auto ref=samplePolylineArc(reference,seed->arc);
            const auto region=initialRegion(ref.point,normals,normalScale,initial);
            const auto chunk=getChunk(get,keyAt(ref.point,config.coreBase));
            const auto match=matchFiberChunkPaths(*chunk,{ref.point},ref.tangent,config,true,&region);
            if(match.valid) {
                seed->point=match.point;
                if(!config.presencePeakRestarts)seed->tangent=ref.tangent;
                return seed;
            }
            if(config.presencePeakRestarts)return seed; // report no initial path, as historical replay did
            const double next=seed->arc+stepBase;
            if(next<=from) throw std::invalid_argument("chunk seed search cannot advance");
            from=next;
        }
        return std::nullopt;
    };
    const ReplaySegmentRunner run=[&](const PolylineArcSample& seed,double remaining,int maximumSteps,
        const ReplayPointObserver& observe,const FiberTraceProgressCallback& report) {
        const auto ref=samplePolylineArc(reference,seed.arc);
        const auto region=initialRegion(ref.point,normals,normalScale,initial);
        const ReplayPointObserver forwarding=[&](const cv::Vec3d& p,double loss,int step) {
            const bool keep=observe(p,loss,step);
            if(report) {
                FiberTraceProgress event;event.step=step;event.maxSteps=maximumSteps;event.phase="chunk_route";
                report(event);
            }
            return keep;
        };
        std::map<std::pair<std::array<int,3>,size_t>,detail::FiberRouteOwnership> owned;
        std::map<std::pair<std::array<int,3>,size_t>,double> lengths;
        auto ownership=detail::fiberRouteOwnershipPolicy([&](const FiberChunkRouteJoin& j){
            const auto key=std::pair{j.key,j.match.line};
            auto [it,inserted]=lengths.try_emplace(key,0);
            if(inserted)it->second=makePolylineArcGeometry(get(j.key)->lines.at(j.match.line)).length();
            const auto length=it->second;
            auto& arcs=owned.try_emplace({j.key,j.match.line},length).first->second;
            return std::pair{&arcs,length};
        });
        FiberChunkContinuationGate gate;
        if(rules)gate=[&](const auto& a,const auto& b){return rules->continuation(a,b);};
        return traceFiberChunkRoute(get,config,source,t,scale,region,ref.tangent,
            t.maxStepFactor*remaining,maximumSteps,forwarding,joins,nullptr,&tracingNormals,rules?&ownership:nullptr,gate);
    };
    return traceReferenceReplayCore(reference,start,end,stepBase,t.maxStepFactor,c.errorThresholdBase,
        c.excursionGraceBase,c.tangentialThresholdFactor,c.matchRefineSteps,normals,normalScale,
        run,progress,failure,policy);
}
} // namespace vc::fiber_tracer
