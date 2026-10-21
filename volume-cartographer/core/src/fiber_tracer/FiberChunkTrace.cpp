#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include "vc/fiber_tracer/FiberWidth.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <opencv2/core.hpp>

namespace vc::fiber_tracer {
namespace {
using Json=nlohmann::json;
constexpr double pi=3.14159265358979323846;
cv::Vec3d unit(const cv::Vec3d& p) {double n=cv::norm(p);return n>1e-12?p/n:cv::Vec3d(0,0,0);}
bool finite(const cv::Vec3d& p) {return std::isfinite(p[0])&&std::isfinite(p[1])&&std::isfinite(p[2]);}
bool inside(const cv::Vec3d& p,const cv::Vec3d& lo,const cv::Vec3d& hi) {
    for(int a=0;a<3;++a) if(p[a]<lo[a] || p[a]>=hi[a]) return false;
    return true;
}
FiberPredictionSampleOption best(const FiberPredictionSample& s) {
    FiberPredictionSampleOption b;
    for(const auto& o:s.options) if(o.valid && std::isfinite(o.presence) && (!b.valid || o.presence>b.presence)) b=o;
    return b;
}
Json xyz(const cv::Vec3d& p) {return Json::array({p[0],p[1],p[2]});}
cv::Vec3d vec(const Json& p) {
    if(!p.is_array() || p.size()!=3) throw std::runtime_error("invalid chunk point");
    cv::Vec3d v{p[0].get<double>(),p[1].get<double>(),p[2].get<double>()};
    if(!finite(v)) throw std::runtime_error("nonfinite chunk point");
    return v;
}
int validate(const FiberChunkTraceConfig& c,double scale) {
    for(double v:{c.coreBase,c.haloBase,c.gridBase,c.nmsBase,c.tubeStepBase,c.rimNormalBase,c.rimTangentBase,
                  c.joinDistanceBase,c.joinMinimumBase,scale})
        if(!std::isfinite(v)||v<=0) throw std::invalid_argument("chunk geometry must be positive and finite");
    for(double v:{c.seedPresence,c.tubePresence,c.rimDrop})
        if(!std::isfinite(v)||v<0||v>1) throw std::invalid_argument("invalid chunk presence threshold");
    for(double v:{c.tubeTurnDegrees,c.tubeTotalDegrees,c.joinAngleDegrees})
        if(!std::isfinite(v)||v<0||v>180) throw std::invalid_argument("invalid chunk angle");
    if(!std::isfinite(c.tubeHalfBase)||c.tubeHalfBase<0||c.nmsBase<c.gridBase)
        throw std::invalid_argument("invalid chunk tube or NMS size");
    const double side=std::round((c.coreBase+2*c.haloBase)/c.gridBase);
    const double radius=std::round(c.nmsBase/c.gridBase);
    if(!std::isfinite(side)||side<3||side>=std::numeric_limits<int>::max()||
       radius*2>=side||side*side*side>static_cast<double>(c.maxGridSamples)||
       side*side*side>static_cast<double>(std::numeric_limits<size_t>::max()))
        throw std::invalid_argument("chunk seed grid exceeds budget or has no NMS interior");
    return static_cast<int>(side);
}

bool tubeSeed(const FiberPredictionSource& source,const vc::lasagna::NormalSampler& normals,
    const cv::Vec3d& seed,const cv::Vec3d& direction,const FiberChunkPaths& chunk,
    const FiberChunkTraceConfig& c,double scale)
{
    for(int sign:{-1,1}) {
        auto p=seed; auto d=direction*sign;
        for(double along=0;along<=c.tubeHalfBase;along+=c.tubeStepBase) {
            if(!inside(p,chunk.lo,chunk.hi)) return false;
            auto center=best(source.sample(p/scale,d));
            if(!center.valid || center.presence<c.tubePresence) return false;
            auto nd=unit(cv::Vec3d(center.direction)); if(nd.dot(d)<0) nd=-nd;
            if(nd.dot(d)<std::cos(c.tubeTurnDegrees*pi/180) ||
               nd.dot(direction*sign)<std::cos(c.tubeTotalDegrees*pi/180)) return false;
            auto n=normals.sampleNormal(p/scale);
            auto u=n.valid?unit(n.normal-nd*n.normal.dot(nd)):cv::Vec3d(0,0,0);
            if(cv::norm(u)<.5) return false;
            auto v=unit(nd.cross(u));
            std::vector<cv::Vec3d> ring,dirs;
            for(int a=0;a<8;++a) {
                auto q=p+u*(c.rimNormalBase*std::cos(a*pi/4))+v*(c.rimTangentBase*std::sin(a*pi/4));
                if(!inside(q,chunk.lo,chunk.hi)) return false;
                ring.push_back(q/scale); dirs.push_back(nd);
            }
            std::vector<FiberPredictionSample> samples;
            source.sampleBatch(ring,dirs,1,samples);
            int separated=0;
            for(const auto& s:samples) {
                auto o=best(s);
                if(!o.valid || o.presence<=center.presence-c.rimDrop) ++separated;
            }
            if(separated<6) return false;
            if(along+c.tubeStepBase>c.tubeHalfBase) break;
            // Midpoint integration follows gentle curvature without optimizing toward GT.
            auto mid=p+nd*(c.tubeStepBase*.5);
            if(!inside(mid,chunk.lo,chunk.hi)) return false;
            auto mo=best(source.sample(mid/scale,nd));
            if(!mo.valid) return false;
            auto md=unit(cv::Vec3d(mo.direction)); if(md.dot(nd)<0) md=-md;
            if(md.dot(nd)<std::cos(c.tubeTurnDegrees*pi/180)) return false;
            p+=md*c.tubeStepBase; d=nd;
        }
    }
    return true;
}

struct Projection { double arc=0,distance=1e30; cv::Vec3d point,tangent; size_t segment=0; };
Projection project(const std::vector<cv::Vec3d>& line,const cv::Vec3d& point,const cv::Vec3d& direction,double minDot,
    double lowerArc=0,double upperArc=std::numeric_limits<double>::infinity(),
    const FiberChunkInitialRegion* region=nullptr,bool anisotropic=false) {
    Projection best; double arc=0;
    double bestCost=std::numeric_limits<double>::infinity();
    for(size_t j=1;j<line.size();++j) {
        auto v=line[j]-line[j-1]; double len=cv::norm(v); if(len<1e-9) continue;
        if(v.dot(direction)/len>=minDot && arc<=upperArc && arc+len>=lowerArc) {
            double lo=std::max(0.,(lowerArc-arc)/len),hi=std::min(1.,(upperArc-arc)/len);
            if(region) {
                const auto delta=line[j-1]-region->center;
                const double a=v.dot(region->metric*v), b=delta.dot(region->metric*v);
                const double c=delta.dot(region->metric*delta)-1;
                const double disc=b*b-a*c;
                if(disc<0) { arc+=len; continue; }
                const double root=std::sqrt(disc);
                lo=std::max(lo,(-b-root)/a); hi=std::min(hi,(-b+root)/a);
                if(lo>hi) { arc+=len; continue; }
            }
            double t=std::clamp(anisotropic?
                (point-line[j-1]).dot(region->metric*v)/v.dot(region->metric*v):
                (point-line[j-1]).dot(v)/(len*len),lo,hi);
            auto p=line[j-1]+v*t; double d=cv::norm(p-point);
            const auto delta=p-point;
            const double cost=anisotropic?delta.dot(region->metric*delta):d;
            if(cost<bestCost) {bestCost=cost;best={arc+t*len,d,p,v/len,j-1};}
        }
        arc+=len;
    }
    return best;
}

} // namespace

FiberTraceOneWayResult traceFiberChunkHalf(const FiberPredictionSource& source,
    const cv::Vec3d& seed,const cv::Vec3d& direction,double distance,
    const FiberTraceConfig& config,const vc::lasagna::NormalSampler* normals,
    const std::array<cv::Vec3d,2>& bounds)
{
    if(!inside(seed,bounds[0],bounds[1])) throw std::invalid_argument("chunk seed outside bounds");
    auto cfg=config;cfg.committedBounds=bounds;cfg.preserveInitialDirection=true;
    bool crossed=false;
    auto result=traceFiberExtrapolation(source,seed,direction,distance,cfg,normals,{},false,
        [&](const cv::Vec3d& p,double,int) {crossed=!inside(p,bounds[0],bounds[1]);return !crossed;});
    if(crossed) result.reason="chunk_boundary";
    return result;
}

std::shared_ptr<FiberChunkPaths> buildFiberChunkPaths(const std::array<int,3>& key,
    const FiberPredictionSource& source,const FiberTraceConfig& trace,
    const vc::lasagna::NormalSampler& normals,const FiberChunkTraceConfig& c,double scale,
    const FiberChunkParallelFor& parallelFor)
{
    const int side=validate(c,scale);
    validateFiberTraceConfig(trace);
    auto out=std::make_shared<FiberChunkPaths>();out->key=key;
    for(int a=0;a<3;++a) {
        out->lo[a]=double(key[a])*c.coreBase-c.haloBase;
        out->hi[a]=(double(key[a])+1)*c.coreBase+c.haloBase;
    }
    if(!finite(out->lo)||!finite(out->hi)) throw std::invalid_argument("chunk bounds overflow");
    auto cfg=trace;cfg.profile=nullptr;cfg.parallelThreads=1;cfg.traceToBaseScale=scale;
    cfg.pinnedRegion=nullptr;cfg.committedBounds.reset();
    std::shared_ptr<vc::render::PinnedChunkRegion> pinned;
    if(c.pinSamplingRegion) {
        const auto* field=dynamic_cast<const FiberPredictionField*>(&source);
        const auto* normalField=dynamic_cast<const vc::lasagna::LasagnaNormalSampler*>(&normals);
        if(field&&normalField) {
            const double margin=trace.stepVoxels*std::max(1,trace.beamLookaheadSteps)*scale;
            const auto pad=cv::Vec3d::all(margin);
            pinned=field->pinRegion((out->lo-pad)/scale,(out->hi+pad)/scale);
            normalField->pinRegion(*pinned,(out->lo-pad)/scale,(out->hi+pad)/scale);
            pinned->seal();cfg.pinnedRegion=pinned.get();
        }
    }
    auto idx=[&](int x,int y,int z){return (size_t(z)*side+y)*side+x;};
    auto point=[&](int x,int y,int z){return out->lo+cv::Vec3d(x,y,z)*c.gridBase;};
    std::vector<float> presence(size_t(side)*side*side);
    for(int z=0;z<side;++z) {
        std::vector<cv::Vec3d> points,dirs(size_t(side)*side,cv::Vec3d(1,0,0));
        points.reserve(dirs.size());
        for(int y=0;y<side;++y) for(int x=0;x<side;++x) points.push_back(point(x,y,z)/scale);
        std::vector<FiberPredictionSample> samples;
        source.sampleBatch(points,dirs,1,samples);
        if(samples.size()!=points.size()) throw std::runtime_error("chunk prediction batch mismatch");
        for(size_t i=0;i<samples.size();++i) {auto p=best(samples[i]);presence[size_t(z)*side*side+i]=p.valid?p.presence:0;}
    }
    const int r=std::lround(c.nmsBase/c.gridBase);
    std::vector<std::array<int,3>> candidates;
    for(int z=r;z<side-r;++z) for(int y=r;y<side-r;++y) for(int x=r;x<side-r;++x) {
        const float p=presence[idx(x,y,z)];if(p<c.seedPresence) continue;
        bool peak=true;
        for(int dz=-r;dz<=r&&peak;++dz) for(int dy=-r;dy<=r&&peak;++dy) for(int dx=-r;dx<=r;++dx)
            if(presence[idx(x+dx,y+dy,z+dz)]>p) {peak=false;break;}
        if(peak) candidates.push_back({x,y,z});
    }
    std::stable_sort(candidates.begin(),candidates.end(),[&](auto a,auto b){return presence[idx(a[0],a[1],a[2])]>presence[idx(b[0],b[1],b[2])];});
    std::vector<bool> suppressed(presence.size());
    std::vector<std::array<int,3>> selected;
    for(const auto& v:candidates) {
        if(suppressed[idx(v[0],v[1],v[2])]) continue;
        ++out->maxima;
        for(int dz=-r;dz<=r;++dz) for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx)
            suppressed[idx(v[0]+dx,v[1]+dy,v[2]+dz)]=true;
        selected.push_back(v);
    }
    struct SeedResult {bool accepted=false;std::vector<cv::Vec3d> line;std::array<std::string,2> reasons;};
    std::vector<SeedResult> results(selected.size());
    const auto process=[&](size_t i) {
        const auto& v=selected[i];auto& result=results[i];
        const auto p=point(v[0],v[1],v[2]);const auto o=best(source.sample(p/scale,{1,0,0}));
        const auto d=unit(cv::Vec3d(o.direction));
        if(!o.valid||cv::norm(d)<.5||!tubeSeed(source,normals,p,d,*out,c,scale)) return;
        if(cfg.widthRelativeTolerance>0) {
            std::vector<float> w;source.sampleWidths({cv::Vec3f(p/scale)},w);
            if(w.size()!=1||!std::isfinite(w[0])||w[0]<=0) return;
        }
        result.accepted=true;
        const std::array<cv::Vec3d,2> bounds{out->lo/scale,out->hi/scale};
        auto fw=traceFiberChunkHalf(source,p/scale,d,4*(c.coreBase+2*c.haloBase)/scale,cfg,&normals,bounds);
        auto bw=traceFiberChunkHalf(source,p/scale,-d,4*(c.coreBase+2*c.haloBase)/scale,cfg,&normals,bounds);
        for(auto it=bw.points.rbegin();it!=bw.points.rend();++it) result.line.push_back(*it*scale);
        for(size_t j=1;j<fw.points.size();++j) result.line.push_back(fw.points[j]*scale);
        result.reasons={bw.reason,fw.reason};
    };
    if(parallelFor&&source.supportsConcurrentSampling()&&normals.supportsConcurrentSampling()) parallelFor(selected.size(),process);
    else for(size_t i=0;i<selected.size();++i) process(i);
    for(auto& result:results) {
        out->seeds+=result.accepted;
        if(result.line.size()>=2) {out->lines.push_back(std::move(result.line));out->endReasons.push_back(result.reasons);}
    }
    return out;
}

namespace {
Json specification(const FiberChunkTraceConfig& c,const FiberTraceConfig& t,double scale,const std::string& identity) {
    return {{"version",1},{"source_identity",identity},{"trace_to_base",scale},
        {"chunk",{{"core",c.coreBase},{"halo",c.haloBase},{"grid",c.gridBase},{"seed_presence",c.seedPresence},
          {"nms",c.nmsBase},{"tube_half",c.tubeHalfBase},{"tube_step",c.tubeStepBase},{"tube_presence",c.tubePresence},
          {"tube_turn",c.tubeTurnDegrees},{"tube_total",c.tubeTotalDegrees},{"rim_drop",c.rimDrop},
          {"rim_normal",c.rimNormalBase},{"rim_tangent",c.rimTangentBase}}},
        {"trace",{{"step",t.stepVoxels},{"cone",t.coneAngleDegrees},{"cone_step",t.coneAngleStepDegrees},
          {"cone_grid",t.coneGridSize},{"beam",t.beamWidth},{"prune",t.beamPruneDistanceVoxels},
          {"lookahead",t.beamLookaheadSteps},{"lazy",t.lazyLookahead},{"parent_cap",t.lookaheadParentCap},
          {"retry_cap",t.lookaheadRetryParentCap},{"smoothness",t.smoothnessWeight},
          {"normal",t.smoothnessNormalWeight},{"tangent",t.smoothnessTangentWeight},
          {"free_angle",t.smoothnessFreeAngleDegrees},{"cumulative_steps",t.cumulativeSmoothnessSteps},
          {"cumulative_tangent",t.cumulativeSmoothnessTangentWeight},{"initial_free_angle",t.initialFreeAngleDegrees},
          {"width_tolerance",t.widthRelativeTolerance},{"width_cosine",t.widthCosineWeight},
          {"width_reference",t.widthReferenceMean},{"initial_width",t.initialWidthMean},
          {"embedding_weight",t.embeddingWeight},{"embedding_spacing",t.embeddingHistorySpacingBaseVoxels},
          {"embedding_window",t.embeddingHistoryWindowBaseVoxels},{"embedding_count",t.embeddingHistoryCount},
          {"embedding_falloff",t.embeddingHistoryFalloff},{"embedding_mode",t.embeddingReferenceMode},
          {"presence_cutoff",t.replayPresenceCutoff},{"presence_modulated",t.replayPresenceModulated}}}};
}
std::shared_ptr<FiberChunkPaths> parseChunk(const Json& j) {
    auto chunk=std::make_shared<FiberChunkPaths>();
    chunk->key=j.at("key").get<std::array<int,3>>();chunk->lo=vec(j.at("lo"));chunk->hi=vec(j.at("hi"));
    for(int a=0;a<3;++a) if(chunk->lo[a]>=chunk->hi[a]) throw std::runtime_error("invalid cached chunk bounds");
    chunk->maxima=j.at("maxima").get<size_t>();chunk->seeds=j.at("seeds").get<size_t>();
    chunk->endReasons=j.at("end_reasons").get<std::vector<std::array<std::string,2>>>();
    for(const auto& l:j.at("lines")) {
        std::vector<cv::Vec3d> line;for(const auto& p:l) line.push_back(vec(p));
        if(line.size()<2) throw std::runtime_error("cached chunk path has fewer than two points");
        chunk->lines.push_back(std::move(line));
    }
    if(chunk->endReasons.size()!=chunk->lines.size()||chunk->seeds>chunk->maxima||chunk->lines.size()>chunk->seeds)
        throw std::runtime_error("inconsistent cached chunk counts");
    return chunk;
}
Json readChunk(const std::filesystem::path& path) {
    std::ifstream input(path);
    if(!input) throw std::runtime_error("cannot read chunk paths: "+path.string());
    return Json::parse(input);
}
}

std::shared_ptr<FiberChunkPaths> loadFiberChunkPaths(const std::filesystem::path& path) {return parseChunk(readChunk(path));}

FiberChunkTraceCache::FiberChunkTraceCache(FiberChunkTraceConfig c,std::filesystem::path output,std::string identity)
    :config_(c),output_(std::move(output)),identity_(std::move(identity))
{
    validate(config_,1);
    if(identity_.empty()||output_.empty()) throw std::invalid_argument("chunk cache requires source identity and output directory");
}

std::shared_ptr<const FiberChunkPaths> FiberChunkTraceCache::get(const std::array<int,3>& key,
    const FiberPredictionSource& source,const FiberTraceConfig& trace,const vc::lasagna::NormalSampler& normals,
    double scale,const FiberChunkParallelFor& parallelFor,const FiberChunkWait& wait)
{
    validate(config_,scale);validateFiberTraceConfig(trace);
    const auto spec=specification(config_,trace,scale,identity_);
    std::shared_ptr<std::promise<std::shared_ptr<const FiberChunkPaths>>> promise;
    std::shared_future<std::shared_ptr<const FiberChunkPaths>> future;
    {
        std::lock_guard lock(mutex_);
        if(specification_.is_null()) specification_=spec;
        else if(specification_!=spec) throw std::invalid_argument("chunk cache configuration changed");
        auto it=chunks_.find(key);
        if(it!=chunks_.end()) future=it->second;
        else {
            promise=std::make_shared<std::promise<std::shared_ptr<const FiberChunkPaths>>>();
            future=promise->get_future().share();chunks_[key]=future;
        }
    }
    if(!promise) {
        auto ready=[&]{return future.wait_for(std::chrono::seconds(0))==std::future_status::ready;};
        if(!ready()&&wait) {
            wait(ready);
            if(!ready()) throw std::logic_error("chunk wait returned before ready");
        }
        return future.get();
    }
    try {
        const auto path=output_/(std::to_string(key[0])+"_"+std::to_string(key[1])+"_"+std::to_string(key[2])+".json");
        std::shared_ptr<FiberChunkPaths> chunk;
        if(std::filesystem::exists(path)) {
            const auto j=readChunk(path);
            if(j.at("specification")!=spec) throw std::runtime_error("cached chunk prediction identity or configuration mismatch");
            chunk=parseChunk(j);
            if(chunk->key!=key) throw std::runtime_error("cached chunk key mismatch");
        } else {
            chunk=buildFiberChunkPaths(key,source,trace,normals,config_,scale,parallelFor);
            Json lines=Json::array();
            for(const auto& line:chunk->lines) {Json points=Json::array();for(const auto& p:line) points.push_back(xyz(p));lines.push_back(points);}
            std::filesystem::create_directories(output_);
            vc::core::util::atomicWriteString(path,Json{{"key",key},{"lo",xyz(chunk->lo)},{"hi",xyz(chunk->hi)},
                {"maxima",chunk->maxima},{"seeds",chunk->seeds},{"lines",lines},{"end_reasons",chunk->endReasons},
                {"specification",spec}}.dump()+"\n");
        }
        promise->set_value(chunk);return chunk;
    } catch(...) {
        promise->set_exception(std::current_exception());
        std::lock_guard lock(mutex_);chunks_.erase(key);
        throw;
    }
}

FiberChunkMatch matchFiberChunkPaths(const FiberChunkPaths& chunk,const std::vector<cv::Vec3d>& history,
    const cv::Vec3d& initialDirection,const FiberChunkTraceConfig& c,bool initial,
    const FiberChunkInitialRegion* region,
    const std::function<std::vector<float>(const std::vector<cv::Vec3d>&)>& widths,double widthTolerance)
{
    validate(c,1);
    FiberChunkMatch bestMatch;
    if(history.empty()) return bestMatch;
    for(const auto& p:history) if(!finite(p)) throw std::invalid_argument("nonfinite chunk history");
    if(initial) {
        if(!region||!finite(region->center)) throw std::invalid_argument("initial chunk match requires an ellipse");
        const auto& m=region->metric;
        for(int a=0;a<3;++a) for(int b=0;b<3;++b)
            if(!std::isfinite(m(a,b))||m(a,b)!=m(b,a)) throw std::invalid_argument("invalid initial metric");
        if(!(m(0,0)>0)||!(m(0,0)*m(1,1)-m(0,1)*m(0,1)>0)||!(cv::determinant(m)>0))
            throw std::invalid_argument("initial metric must be positive definite");
    }
    if(!initial&&c.joinWidthWeight&&(!widths||!std::isfinite(widthTolerance)||widthTolerance<=0))
        throw std::invalid_argument("width-aware join requires widths and positive tolerance");
    const auto dir=!initial&&history.size()>1?unit(history.back()-history[history.size()-2]):unit(initialDirection);
    if(!finite(dir)||cv::norm(dir)<.5) throw std::invalid_argument("invalid chunk match direction");
    for(size_t i=0;i<chunk.lines.size();++i) for(bool reverse:{false,true}) {
        auto line=chunk.lines[i];if(line.size()<2) continue;
        if(reverse) std::reverse(line.begin(),line.end());
        const auto end=project(line,initial?region->center:history.back(),dir,std::cos(c.joinAngleDegrees*pi/180),
            0,std::numeric_limits<double>::infinity(),initial?region:nullptr);
        if(end.distance==1e30) continue;
        const auto geometry=makePolylineArcGeometry(line);
        if(!initial&&(end.distance>c.joinDistanceBase||geometry.length()-end.arc<16)) continue;
        double overlap=0,error=end.distance,angle=1-end.tangent.dot(dir),lastArc=end.arc;
        size_t matched=1;
        std::vector<cv::Vec3d> widthPoints;
        if(!initial&&c.joinWidthWeight) widthPoints={history.back(),end.point};
        for(size_t j=history.size()-1;!initial&&j>0&&overlap<2*c.haloBase;--j) {
            if(!inside(history[j-1],chunk.lo,chunk.hi)) break;
            const auto tangent=unit(history[j]-history[j-1]);
            const double step=cv::norm(history[j]-history[j-1]);
            const int subdivisions=std::max(1,int(std::ceil(step/4)));
            bool valid=true;
            for(int n=1;n<=subdivisions;++n) {
                const auto point=history[j]+(history[j-1]-history[j])*(double(n)/subdivisions);
                const auto p=project(line,point,tangent,std::cos(c.joinAngleDegrees*pi/180),
                    std::max(0.,lastArc-2*step/subdivisions),lastArc);
                if(p.distance>c.joinDistanceBase||p.arc>=lastArc) {valid=false;break;}
                lastArc=p.arc;overlap+=step/subdivisions;error+=p.distance;
                if(c.joinWidthWeight) {widthPoints.push_back(point);widthPoints.push_back(p.point);}
                angle+=1-p.tangent.dot(tangent);++matched;
            }
            if(!valid) break;
        }
        if(!initial&&overlap<c.joinMinimumBase) continue;
        error/=matched;angle/=matched;
        double compatibility=1;
        if(!initial&&c.joinWidthWeight) {
            const auto values=widths(widthPoints);
            if(values.size()!=widthPoints.size()) throw std::runtime_error("chunk join width sample count mismatch");
            compatibility=0;
            for(size_t k=0;k<values.size();k+=2) {
                if(!std::isfinite(values[k])||!std::isfinite(values[k+1])||values[k]<=0||values[k+1]<=0) continue;
                compatibility+=fiberWidthPresenceScale(values[k+1],values[k],widthTolerance,true);
            }
            compatibility/=values.size()/2;
            if(compatibility<=0) continue;
        }
        const double weighted=overlap*compatibility;
        if(!bestMatch.valid||weighted>bestMatch.weightedOverlap||(weighted==bestMatch.weightedOverlap&&
            (error<bestMatch.error||(error==bestMatch.error&&angle<bestMatch.angleError))))
            bestMatch={i,end.segment,reverse,true,end.point,end.arc,overlap,error,angle,compatibility,weighted};
    }
    return bestMatch;
}

} // namespace vc::fiber_tracer
