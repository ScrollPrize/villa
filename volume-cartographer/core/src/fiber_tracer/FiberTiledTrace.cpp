#include "vc/fiber_tracer/FiberTiledTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/fiber_tracer/FiberReplayMetric.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace vc::fiber_tracer {
namespace {
using Json=nlohmann::json;
constexpr double eps=1e-6;
Json xyz(const cv::Vec3d& p){return Json::array({p[0],p[1],p[2]});}
struct End {std::size_t fragment, tile, source; int side; double arc; cv::Vec3d point, outward;};
double costBetween(const FiberletCropTraceLine& line,const PolylineArcGeometry& g,double a,double b){
    double result=0;
    for(const auto& segment:clippedPolylineArcSegments(g,a,b)) {
        const auto i=segment.segmentIndex;
        const double length=g.vertexArcs[i+1]-g.vertexArcs[i];
        if(length>0) result+=line.segmentMetricCosts[i]*(segment.endArc-segment.beginArc)/length;
    }
    return result;
}
}

FiberTiledTraceResult assembleFiberTraceTiles(std::vector<FiberTraceTile> tiles,
    const vc::lasagna::NormalSampler& normals,const FiberSeamConfig& config)
{
    if(!std::isfinite(config.minimumSupportBase) || config.minimumSupportBase<=0 ||
       !(config.minimumFraction>0 && config.minimumFraction<=1)) throw std::invalid_argument("invalid seam support");
    std::sort(tiles.begin(),tiles.end(),[](const auto&a,const auto&b){return a.indexXYZ<b.indexXYZ;});
    std::map<std::array<int,3>,std::size_t> byIndex;
    std::array<std::map<int,std::pair<double,double>>,3> axes;
    for(std::size_t t=0;t<tiles.size();++t){
        const auto& tile=tiles[t];
        if(!byIndex.emplace(tile.indexXYZ,t).second) throw std::invalid_argument("duplicate tile index");
        for(int a=0;a<3;++a)if(!std::isfinite(tile.coreMin[a]) || !std::isfinite(tile.coreMax[a]) ||
            !std::isfinite(tile.haloMin[a]) || !std::isfinite(tile.haloMax[a]) ||
            !(tile.coreMax[a]>tile.coreMin[a]) || tile.haloMin[a]>tile.coreMin[a] || tile.haloMax[a]<tile.coreMax[a])
            throw std::invalid_argument("invalid tile bounds");
        for(int a=0;a<3;++a){
            auto [it,inserted]=axes[a].emplace(tile.indexXYZ[a],std::pair(tile.coreMin[a],tile.coreMax[a]));
            if(!inserted && it->second!=std::pair(tile.coreMin[a],tile.coreMax[a]))throw std::invalid_argument("inconsistent tile grid");
        }
    }
    for(const auto& axis:axes)for(auto it=axis.begin();it!=axis.end();++it){
        const auto next=std::next(it);if(next==axis.end())break;
        if(it->second.second>next->second.first+eps ||
            (next->first==it->first+1 && std::abs(it->second.second-next->second.first)>eps))
            throw std::invalid_argument("overlapping or gapped adjacent tile cores");
    }
    FiberTiledTraceResult result;
    auto& meta=result.topology;
    meta={{"contract",1},{"tiles",Json::array()},{"fragments",Json::array()},{"candidates",Json::array()},
          {"cost_partition","linear_within_captured_step"},{"minimum_support_base",config.minimumSupportBase},
          {"minimum_fraction",config.minimumFraction},{"normal_radius",config.overlap.normalRadiusBase},
          {"tangential_radius",config.overlap.tangentialRadiusBase},{"angle_degrees",config.overlap.directionDegrees}};
    std::vector<std::vector<PolylineArcGeometry>> geometry(tiles.size());
    std::vector<std::vector<End>> ends(tiles.size());
    cv::Vec3d globalMin,globalMax;
    if(!tiles.empty()){globalMin=tiles[0].coreMin;globalMax=tiles[0].coreMax;}
    for(const auto& tile:tiles)for(int a=0;a<3;++a){globalMin[a]=std::min(globalMin[a],tile.coreMin[a]);globalMax[a]=std::max(globalMax[a],tile.coreMax[a]);}
    for(std::size_t t=0;t<tiles.size();++t){
        const auto& tile=tiles[t];
        meta["tiles"].push_back({{"index_xyz",tile.indexXYZ},{"core_min",xyz(tile.coreMin)},{"core_max",xyz(tile.coreMax)},
            {"halo_min",xyz(tile.haloMin)},{"halo_max",xyz(tile.haloMax)}});
        for(std::size_t s=0;s<tile.lines.size();++s){
            const auto& source=tile.lines[s];
            geometry[t].push_back(makePolylineArcGeometry(source.pointsBaseXYZ));
            const auto& g=geometry[t].back();
            if(source.segmentMetricCosts.size()+1!=source.pointsBaseXYZ.size())
                throw std::invalid_argument("tiled assembly requires --capture-step-costs on fresh tile traces");
            std::vector<std::pair<double,double>> arcs;
            const auto runs=clipPolylineToHalfOpenBox(source.pointsBaseXYZ,tile.coreMin,tile.coreMax,&arcs);
            for(std::size_t r=0;r<runs.size();++r){
                const auto [begin,end]=arcs[r];if(end-begin<=eps)continue;
                auto line=source;line.pointsBaseXYZ=runs[r];line.segmentMetricCosts.clear();line.segmentOriginalStepCosts.clear();
                line.originalSeedBaseXYZ=source.originalSeedBaseXYZ.value_or(source.seedBaseXYZ);
                line.seedBaseXYZ=(runs[r][0]+runs[r][1])*.5;
                line.pathLengthPredictionVoxels=source.pathLengthPredictionVoxels*(end-begin)/g.length();
                line.totalMetricCost=0;double arc=begin;
                for(std::size_t i=1;i<runs[r].size();++i){
                    const double next=arc+cv::norm(runs[r][i]-runs[r][i-1]);
                    const double cost=costBetween(source,g,arc,std::min(end,next));
                    line.segmentMetricCosts.push_back(cost);line.totalMetricCost+=cost;arc=next;
                }
                // Storage requires the ownership seed to be an explicit vertex.
                // A midpoint is interior even when both clipped endpoints lie on
                // core faces; split the captured first-step cost consistently.
                line.pointsBaseXYZ.insert(line.pointsBaseXYZ.begin()+1,line.seedBaseXYZ);
                const double firstCost=line.segmentMetricCosts.front();
                line.segmentMetricCosts.front()=firstCost*.5;
                line.segmentMetricCosts.insert(line.segmentMetricCosts.begin()+1,firstCost*.5);
                const std::size_t id=result.lines.size();
                const bool first=begin>eps,last=end<g.length()-eps;
                if(first)line.negativeTermination="tile_core_cut";
                if(last)line.positiveTermination="tile_core_cut";
                meta["fragments"].push_back({{"tile",t},{"source_line",s},{"source_run",r},
                    {"begin_arc",begin},{"end_arc",end},{"cut_start",first},{"cut_end",last},
                    {"external_start",false},{"external_end",false}});
                for(int side=0;side<2;++side)if(side?last:first){
                    auto sample=samplePolylineArc(g,side?end:begin);
                    const auto point=side?runs[r].back():runs[r].front(),outward=sample.tangent*(side?1.:-1.);
                    bool external=false;
                    for(int axis=0;axis<3;++axis)if((std::abs(point[axis]-globalMin[axis])<eps && outward[axis]<-eps) ||
                        (std::abs(point[axis]-globalMax[axis])<eps && outward[axis]>eps))external=true;
                    meta["fragments"].back()[side?"external_end":"external_start"]=external;
                    ends[t].push_back({id,t,s,side,side?end:begin,point,outward});
                }
                result.lines.push_back(std::move(line));
            }
        }
    }
    const double radius=std::max(config.overlap.normalRadiusBase,config.overlap.tangentialRadiusBase);
    const double cosine=std::cos(config.overlap.directionDegrees*std::acos(-1.)/180.);
    for(std::size_t t=0;t<tiles.size();++t)for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy)for(int dz=-1;dz<=1;++dz){
        auto idx=tiles[t].indexXYZ;idx[0]+=dx;idx[1]+=dy;idx[2]+=dz;
        const auto found=byIndex.find(idx);if(found==byIndex.end() || found->second<=t)continue;
        const auto u=found->second;
        cv::Vec3d low,high,boundLow,boundHigh;
        bool adjacent=true;
        for(int a=0;a<3;++a){
            low[a]=std::max(tiles[t].haloMin[a],tiles[u].haloMin[a]);
            high[a]=std::min(tiles[t].haloMax[a],tiles[u].haloMax[a]);
            boundLow[a]=std::max(tiles[t].coreMin[a],tiles[u].coreMin[a]);
            boundHigh[a]=std::min(tiles[t].coreMax[a],tiles[u].coreMax[a]);
            if(high[a]<=low[a] || boundLow[a]>boundHigh[a]+eps)adjacent=false;
        }
        if(!adjacent)continue;
        std::vector<End> matchingEnds;std::vector<FiberletCropTraceLine> support;std::vector<std::size_t> groups;
        for(auto tileId:{t,u})for(const auto& e:ends[tileId]){
            bool near=true;
            for(int a=0;a<3;++a)if(e.point[a]<boundLow[a]-radius || e.point[a]>boundHigh[a]+radius)near=false;
            if(!near)continue;
            // Only the contiguous shared-halo run containing THIS endpoint.
            const auto& g=geometry[tileId][e.source];std::vector<std::pair<double,double>> arcs;
            const auto runs=clipPolylineToHalfOpenBox(g.points,low,high,&arcs);
            for(std::size_t r=0;r<runs.size();++r)if(e.arc>=arcs[r].first-eps && e.arc<=arcs[r].second+eps){
                if(arcs[r].second-arcs[r].first<config.minimumSupportBase)break;
                FiberletCropTraceLine l;l.pointsBaseXYZ=runs[r];support.push_back(std::move(l));
                matchingEnds.push_back(e);groups.push_back(tileId);break;
            }
        }
        if(support.empty())continue;
        const auto graph=measureFiberPieceOverlap(support,normals,1.,config.overlap,groups,true);
        for(std::size_t i=0;i<support.size();++i)for(const auto& [j,forward]:graph.shorterPairs[i]){
            if(j<=i || forward<config.minimumFraction)continue;
            const auto& a=matchingEnds[i];const auto& b=matchingEnds[j];
            if(a.outward.dot(b.outward)>-cosine || cv::norm(a.point-b.point)>radius)continue;
            const auto& back=graph.shorterPairs[j];
            const auto foundBack=std::lower_bound(back.begin(),back.end(),i,[](const auto&p,std::size_t id){return p.first<id;});
            if(foundBack==back.end() || foundBack->first!=i || foundBack->second<config.minimumFraction)continue;
            const double length=std::min(forward*graph.lengthBase[i],foundBack->second*graph.lengthBase[j]);
            if(length<config.minimumSupportBase)continue;
            const auto measured=measureFiberReplayThreshold(a.point,b.point,normals,1.,config.overlap.normalRadiusBase,
                config.overlap.tangentialRadiusBase/config.overlap.normalRadiusBase);
            if(fiberReplayThresholdExceeded(measured,config.overlap.normalRadiusBase))continue;
            meta["candidates"].push_back({{"a",a.fragment},{"a_end",a.side},{"b",b.fragment},{"b_end",b.side},
                {"fraction_a",forward},{"fraction_b",foundBack->second},{"support_base",length},
                {"endpoint_distance_base",cv::norm(a.point-b.point)}});
        }
    }
    std::map<std::pair<std::size_t,int>,std::size_t> degrees;
    for(const auto& c:meta["candidates"]){++degrees[{c["a"],c["a_end"]}];++degrees[{c["b"],c["b_end"]}];}
    for(auto& c:meta["candidates"])c["unique"]=degrees[{c["a"],c["a_end"]}]==1 && degrees[{c["b"],c["b_end"]}]==1;
    std::set<std::pair<std::size_t,int>> uniqueEnds;
    for(const auto& c:meta["candidates"])if(c["unique"].get<bool>()){
        uniqueEnds.emplace(c["a"],c["a_end"]);uniqueEnds.emplace(c["b"],c["b_end"]);
    }
    std::size_t internal=0,external=0,unmatchedInternal=0,unmatchedExternal=0,ambiguous=0;
    for(const auto& list:ends)for(const auto& e:list){
        const bool outer=meta["fragments"][e.fragment][e.side?"external_end":"external_start"];
        if(outer)++external;else ++internal;
        const auto key=std::pair(e.fragment,e.side);
        if(!degrees.contains(key)){if(outer)++unmatchedExternal;else ++unmatchedInternal;}
        else if(!uniqueEnds.contains(key))++ambiguous;
    }
    meta["endpoint_counts"]={{"internal",internal},{"external",external},{"unique",uniqueEnds.size()},
        {"ambiguous",ambiguous},{"unmatched_internal",unmatchedInternal},{"unmatched_external",unmatchedExternal}};
    validateFiberTiledTopology(meta,result.lines.size());return result;
}

void validateFiberTiledTopology(const Json& meta,std::size_t count){
    if(meta.at("contract")!=1 || !meta.at("fragments").is_array() || meta.at("fragments").size()!=count)
        throw std::invalid_argument("invalid tiled topology contract/count");
    for(const auto& fragment:meta.at("fragments")){
        const double begin=fragment.at("begin_arc"),end=fragment.at("end_arc");
        if(fragment.at("tile").get<std::size_t>()>=meta.at("tiles").size() ||
           !std::isfinite(begin) || !std::isfinite(end) || begin<0 || end<=begin)
            throw std::invalid_argument("invalid tiled fragment provenance");
    }
    std::map<std::pair<std::size_t,int>,std::size_t> degrees;
    std::set<std::tuple<std::size_t,int,std::size_t,int>> seen;
    for(const auto& c:meta.at("candidates")){
        for(const char* key:{"fraction_a","fraction_b","support_base","endpoint_distance_base"}){
            const double value=c.at(key);if(!std::isfinite(value) || value<0)throw std::invalid_argument("invalid seam score");
        }
        if(c.at("fraction_a").get<double>()>1 || c.at("fraction_b").get<double>()>1)throw std::invalid_argument("invalid seam fraction");
        const auto a=c.at("a").get<std::size_t>(),b=c.at("b").get<std::size_t>();
        const int x=c.at("a_end"),y=c.at("b_end");
        if(a>=count || b>=count || a==b || x<0 || x>1 || y<0 || y>1 ||
           !meta["fragments"][a].at(x?"cut_end":"cut_start").get<bool>() ||
           !meta["fragments"][b].at(y?"cut_end":"cut_start").get<bool>())throw std::invalid_argument("invalid seam endpoints");
        auto key=a<b?std::tuple(a,x,b,y):std::tuple(b,y,a,x);
        if(!seen.insert(key).second)throw std::invalid_argument("duplicate seam link");
        ++degrees[{a,x}];++degrees[{b,y}];
    }
    for(const auto& c:meta.at("candidates"))if(c.at("unique").get<bool>() !=
        (degrees[{c["a"],c["a_end"]}]==1 && degrees[{c["b"],c["b_end"]}]==1))
        throw std::invalid_argument("incorrect seam ambiguity annotation");
}

void requireUnambiguousFiberTiledTopology(const Json& meta){
    for(const auto& c:meta.at("candidates"))if(!c.at("unique").get<bool>())
        throw std::invalid_argument("tiled trace has unresolved seam alternatives; pairwise BP cannot represent alternative-choice factors");
    if(meta.at("endpoint_counts").at("unmatched_internal").get<std::size_t>()>0)
        throw std::invalid_argument("tiled trace has unmatched internal cuts; cannot silently treat them as real BP breaks");
}

void addFiberTileContinuities(FiberTraceConstraintReport& report,const Json& meta,
    std::span<const std::size_t> originalTraceIds){
    requireUnambiguousFiberTiledTopology(meta);
    std::map<std::size_t,std::pair<std::size_t,std::size_t>> endpoints;
    for(std::size_t i=0;i<report.pieces.size();++i){
        const auto& p=report.pieces[i];if(p.traceIndex>=originalTraceIds.size())throw std::invalid_argument("seam trace map mismatch");
        const auto id=originalTraceIds[p.traceIndex];auto [it,inserted]=endpoints.emplace(id,std::pair(i,i));
        if(!inserted){if(p.beginArcBaseVoxels<report.pieces[it->second.first].beginArcBaseVoxels)it->second.first=i;
            if(p.endArcBaseVoxels>report.pieces[it->second.second].endArcBaseVoxels)it->second.second=i;}
    }
    for(const auto& c:meta.at("candidates")){
        const auto a=endpoints.find(c["a"]),b=endpoints.find(c["b"]);
        if(a==endpoints.end() || b==endpoints.end())throw std::invalid_argument("filter removed linked tiled trace; cannot silently drop seam");
        FiberTraceConstraint link;link.pieceA=c["a_end"].get<int>()?a->second.second:a->second.first;
        link.pieceB=c["b_end"].get<int>()?b->second.second:b->second.first;
        const auto& pa=report.pieces[link.pieceA];const auto& pb=report.pieces[link.pieceB];
        link.arcABaseVoxels=c["a_end"].get<int>()?pa.endArcBaseVoxels:pa.beginArcBaseVoxels;
        link.arcBBaseVoxels=c["b_end"].get<int>()?pb.endArcBaseVoxels:pb.beginArcBaseVoxels;
        link.pointABaseXYZ=c["a_end"].get<int>()?pa.samplePointsBaseXYZ.back():pa.samplePointsBaseXYZ.front();
        link.pointBBaseXYZ=c["b_end"].get<int>()?pb.samplePointsBaseXYZ.back():pb.samplePointsBaseXYZ.front();
        link.hardContinuity=true;link.artificialTileSeam=true;report.constraints.push_back(link);++report.hardConstraints;
    }
}
}
