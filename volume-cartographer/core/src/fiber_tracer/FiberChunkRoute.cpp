#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include "vc/fiber_tracer/FiberWidth.hpp"
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

std::string traceFiberChunkRoute(const FiberChunkProvider& get,const FiberChunkTraceConfig& c,
    const FiberPredictionSource& source,const FiberTraceConfig& t,double scale,
    const FiberChunkInitialRegion& region,const cv::Vec3d& direction,double maximumDistance,int maximumSteps,
    const ReplayPointObserver& observe,const FiberChunkJoinObserver& joins)
{
    validateRoute(t,scale);
    if(!get||!observe||!std::isfinite(maximumDistance)||maximumDistance<=0||maximumSteps<=0)
        throw std::invalid_argument("chunk route requires callbacks and positive finite budgets");
    auto key=keyAt(region.center,c.coreBase);
    std::vector<cv::Vec3d> history{region.center};
    std::set<std::tuple<std::array<int,3>,size_t,bool>> visited;
    bool initial=true;int steps=0;double distance=0;
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
        const auto match=matchFiberChunkPaths(*chunk,history,direction,c,initial,initial?&region:nullptr,
            widths,t.widthRelativeTolerance);
        if(!match.valid) return initial?"chunk_no_initial_path":"chunk_no_overlap";
        if(!visited.emplace(key,match.line,match.reverse).second) return "chunk_route_cycle";
        auto line=chunk->lines.at(match.line);if(match.reverse) std::reverse(line.begin(),line.end());
        const double splice=initial?0:cv::norm(match.point-history.back());
        if(joins) joins({key,match,initial,splice});
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
            if(const auto stop=emit(match.point);!stop.empty()) return stop;
        }
        const int before=steps;
        for(size_t j=match.segment+1;j<line.size();++j)
            if(const auto stop=emit(line[j]);!stop.empty()) return stop;
        const auto& reason=chunk->endReasons.at(match.line)[match.reverse?0:1];
        if(reason=="presence_cutoff"||reason=="invalid_width") return reason;
        if(steps==before||history.size()<2) return "chunk_no_progress";
        const auto dir=unit(history.back()-history[history.size()-2]);
        bool boundary=false;
        for(int a=0;a<3;++a) boundary|=(dir[a]>0&&chunk->hi[a]-history.back()[a]<32)||
            (dir[a]<0&&history.back()[a]-chunk->lo[a]<32);
        if(!boundary) return "chunk_local_stop:"+reason;
        const auto next=keyAt(history.back()+dir*stepBase,c.coreBase);
        if(next==key) return "chunk_no_neighbor_progress";
        key=next;initial=false;
    }
    return "chunk_length_limit";
}

FiberReplayTraceResult traceFiberChunkReferenceReplay(FiberChunkTraceCache& cache,const FiberPredictionSource& source,
    const FiberTraceConfig& t,double scale,const vc::lasagna::NormalSampler& tracingNormals,
    const PolylineArcGeometry& reference,double start,double end,const vc::lasagna::NormalSampler& normals,
    double normalScale,const FiberChunkReplayConfig& c,const FiberChunkParallelFor& parallel,
    const FiberChunkWait& wait,const FiberTraceProgressCallback& progress,const FiberReplayFailureCallback& failure,
    const ReplayConsistencyPolicy& consistency,const FiberChunkJoinObserver& joins)
{
    validateRoute(t,scale);
    if(!std::isfinite(c.errorThresholdBase)||c.errorThresholdBase<=0)
        throw std::invalid_argument("chunk replay requires a positive initial acceptance radius");
    const auto& config=cache.config();
    const FiberChunkProvider get=[&](const auto& key){return cache.get(key,source,t,tracingNormals,scale,parallel,wait);};
    auto policy=consistency;policy.seedAllStarts=true;
    policy.nextSeed=[&](double from)->std::optional<PolylineArcSample> {
        while(from<end) {
            auto seed=consistency.nextSeed?consistency.nextSeed(from):std::optional(samplePolylineArc(reference,from));
            if(!seed) return std::nullopt;
            if(!std::isfinite(seed->arc)||seed->arc<from||seed->arc>=end)
                throw std::invalid_argument("chunk seed search did not advance within reference bounds");
            const auto ref=samplePolylineArc(reference,seed->arc);
            const auto region=initialRegion(ref.point,normals,normalScale,c);
            const auto chunk=getChunk(get,keyAt(ref.point,config.coreBase));
            const auto match=matchFiberChunkPaths(*chunk,{ref.point},ref.tangent,config,true,&region);
            if(match.valid) {seed->point=match.point;seed->tangent=ref.tangent;return seed;}
            const double next=seed->arc+stepBase;
            if(next<=from) throw std::invalid_argument("chunk seed search cannot advance");
            from=next;
        }
        return std::nullopt;
    };
    const ReplaySegmentRunner run=[&](const PolylineArcSample& seed,double remaining,int maximumSteps,
        const ReplayPointObserver& observe,const FiberTraceProgressCallback& report) {
        const auto ref=samplePolylineArc(reference,seed.arc);
        const auto region=initialRegion(ref.point,normals,normalScale,c);
        const ReplayPointObserver forwarding=[&](const cv::Vec3d& p,double loss,int step) {
            const bool keep=observe(p,loss,step);
            if(report) {
                FiberTraceProgress event;event.step=step;event.maxSteps=maximumSteps;event.phase="chunk_route";
                report(event);
            }
            return keep;
        };
        return traceFiberChunkRoute(get,config,source,t,scale,region,ref.tangent,
            t.maxStepFactor*remaining,maximumSteps,forwarding,joins);
    };
    return traceReferenceReplayCore(reference,start,end,stepBase,t.maxStepFactor,c.errorThresholdBase,
        c.excursionGraceBase,c.tangentialThresholdFactor,c.matchRefineSteps,normals,normalScale,
        run,progress,failure,policy);
}
} // namespace vc::fiber_tracer
