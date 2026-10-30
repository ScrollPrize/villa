#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <set>

namespace vc::fiber_tracer {
using Json=nlohmann::json;
constexpr double pi=3.14159265358979323846;
nlohmann::json filterFiberChunkConflicts(FiberChunkPaths& chunk,const FiberChunkTraceConfig& cfg,
    const vc::lasagna::NormalSampler& normals,double scale,const FiberChunkParallelFor& parallel,
    const FiberPredictionSource* readinessSource,const std::vector<size_t>* auditQueries)
{
    if(!(std::isfinite(cfg.conflictNormalBase) && std::isfinite(cfg.conflictTangentBase) && std::isfinite(scale) &&
         cfg.conflictNormalBase>0 && cfg.conflictTangentBase>0 && scale>0 &&
         std::isfinite(cfg.conflictAgreementBase) && cfg.conflictAgreementBase>0 &&
         std::isfinite(cfg.conflictSeparationBase) && cfg.conflictSeparationBase>0))
        throw std::invalid_argument("conflict ellipse and scale must be positive");
    const size_t agreementSamples=size_t(std::ceil(cfg.conflictAgreementBase/8));
    const size_t separationSamples=size_t(std::ceil(cfg.conflictSeparationBase/8));
    for(const auto& line:chunk.lines) for(const auto& p:line) for(int a=0;a<3;++a)
        if(!std::isfinite(p[a]) || std::abs(p[a])>1e10)throw std::invalid_argument("invalid conflict path coordinate");
    const auto started=std::chrono::steady_clock::now();
    struct Line {std::vector<cv::Vec3d> points;std::vector<double> arcs;};
    std::vector<Line> lines(chunk.lines.size());
    std::set<size_t> queries;
    if(auditQueries){queries.insert(auditQueries->begin(),auditQueries->end());for(auto i:queries)if(i>=lines.size())throw std::invalid_argument("invalid audit query ID");}
    using Cell=std::array<int,3>;
    std::map<Cell,std::vector<size_t>> grid;
    const double cell=64, radius=std::max(cfg.conflictNormalBase,cfg.conflictTangentBase);
    auto key=[&](const cv::Vec3d&p){return Cell{int(std::floor(p[0]/cell)),int(std::floor(p[1]/cell)),int(std::floor(p[2]/cell))};};
    for(size_t i=0;i<lines.size();++i) {
        auto& l=lines[i];
        for(const auto&p:chunk.lines[i])if(l.points.empty() || cv::norm(p-l.points.back())>1e-8)l.points.push_back(p);
        l.arcs.push_back(0);
        for(size_t j=1;j<l.points.size();++j) {
            l.arcs.push_back(l.arcs.back()+cv::norm(l.points[j]-l.points[j-1]));
            cv::Vec3d lo,hi;for(int k=0;k<3;++k){lo[k]=std::min(l.points[j][k],l.points[j-1][k])-radius;hi[k]=std::max(l.points[j][k],l.points[j-1][k])+radius;}
            auto a=key(lo),b=key(hi);
            for(int x=a[0];x<=b[0];++x)for(int y=a[1];y<=b[1];++y)for(int z=a[2];z<=b[2];++z)grid[{x,y,z}].push_back(i);
        }
    }
    for(auto&[k,v]:grid){std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());}
    auto samples=[&](const Line& l) {
        std::vector<std::pair<cv::Vec3d,cv::Vec3d>> result;
        if(l.points.size()<2 || l.arcs.back()<cfg.conflictAgreementBase+cfg.conflictSeparationBase)return result;
        size_t segment=0;
        for(double a=0;a<=l.arcs.back();a+=8) {
            while(segment+2<l.points.size() && l.arcs[segment+1]<a)++segment;
            auto t=(l.points[segment+1]-l.points[segment])/(l.arcs[segment+1]-l.arcs[segment]);
            result.emplace_back(l.points[segment]+t*(a-l.arcs[segment]),t);
        }
        return result;
    };
    // Readiness belongs to the calling case, not to child worker threads.
    // Preflight exact points before concurrent normal sampling.
    if(readinessSource) {
        auto* field=dynamic_cast<const FiberPredictionField*>(readinessSource);
        auto* nf=dynamic_cast<const vc::lasagna::LasagnaNormalSampler*>(&normals);
        if(!field || !nf)throw std::runtime_error("conflict readiness requires native predicted normal field");
        std::vector<cv::Vec3f> batch;
        auto flush=[&]{if(batch.empty())return;vc::lasagna::LasagnaCornerBatch corners;
            if(!field->sampleCornerBatchWithNormals(*nf,batch,&corners,nullptr))throw std::runtime_error("conflict normal preflight failed");batch.clear();};
        for(const auto&l:lines)for(const auto&[p,t]:samples(l)) {
            // Cover both sides of float rounding without changing double sampling.
            cv::Vec3f lower(p/scale),upper(lower);
            for(int k=0;k<3;++k){lower[k]=std::nextafter(lower[k],-std::numeric_limits<float>::infinity());upper[k]=std::nextafter(upper[k],std::numeric_limits<float>::infinity());}
            batch.push_back(lower);batch.push_back(upper);if(batch.size()>=4096)flush();
        }
        flush();
    }
    const auto& lookup=grid;
    std::vector<std::vector<size_t>> partners(lines.size());
    std::vector<size_t> invalidNormals(lines.size());
    auto work=[&](size_t i) {
        const auto& l=lines[i];if(l.points.size()<2 || l.arcs.back()<cfg.conflictAgreementBase+cfg.conflictSeparationBase)return;
        std::vector<cv::Vec3d> points,tangents,normal;
        std::vector<bool> valid;
        std::set<size_t> candidates;
        for(const auto& [p,t]:samples(l)) {
            auto n=normals.sampleNormal(p/scale);
            auto nv=n.normal-t*n.normal.dot(t);double length=cv::norm(nv);
            bool good=n.valid && std::isfinite(length) && length>1e-8;
            normal.push_back(good?nv/length:cv::Vec3d(0,0,0));valid.push_back(good);
            invalidNormals[i]+=!good;points.push_back(p);tangents.push_back(t);
            auto it=lookup.find(key(p));if(it!=lookup.end())candidates.insert(it->second.begin(),it->second.end());
        }
        for(size_t j:candidates) {
            if(j==i || lines[j].points.size()<2)continue;
            if(auditQueries && !queries.count(i) && !queries.count(j))continue;
            const auto& other=lines[j];
            std::vector<double> projection;
            std::vector<std::pair<size_t,size_t>> shared,split;
            int beginShared=-1,beginSplit=-1;
            auto close=[](int& start,size_t end,size_t minimum,auto& runs){
                if(start>=0 && end-size_t(start)>=minimum)runs.emplace_back(size_t(start),end);
                start=-1;
            };
            for(size_t q=0;q<points.size();++q) {
                double best=std::numeric_limits<double>::infinity(),arc=0;cv::Vec3d delta,dir;
                for(size_t s=1;s<other.points.size();++s) {
                    auto v=other.points[s]-other.points[s-1];double len=other.arcs[s]-other.arcs[s-1];
                    double t=std::clamp((points[q]-other.points[s-1]).dot(v)/(len*len),0.,1.);
                    auto d=points[q]-(other.points[s-1]+t*v);double d2=d.dot(d);
                    if(d2<best){best=d2;delta=d;dir=v/len;arc=other.arcs[s-1]+t*len;}
                }
                projection.push_back(arc);
                double dn=delta.dot(normal[q]);
                double ratio=dn*dn/(cfg.conflictNormalBase*cfg.conflictNormalBase)+
                    std::max(0.,best-dn*dn)/(cfg.conflictTangentBase*cfg.conflictTangentBase);
                bool interior=valid[q] && arc>=16 && arc<=other.arcs.back()-16;
                bool agree=interior && ratio<=1 && std::abs(dir.dot(tangents[q]))>=std::cos(25*pi/180);
                bool depart=interior && ratio>1;
                if(agree){if(beginShared<0)beginShared=int(q);}else if(q)close(beginShared,q-1,agreementSamples,shared);
                if(depart){if(beginSplit<0)beginSplit=int(q);}else if(q)close(beginSplit,q-1,separationSamples,split);
            }
            close(beginShared,points.size()-1,agreementSamples,shared);close(beginSplit,points.size()-1,separationSamples,split);
            bool conflict=false;
            for(auto[a,b]:shared)for(auto[c,d]:split) {
                if(!(b<c || d<a))continue;
                bool positive=false,negative=false,continuous=true;
                for(size_t q=std::min(a,c)+1;q<=std::max(b,d);++q){double diff=projection[q]-projection[q-1];positive|=diff>2;negative|=diff< -2;if(std::abs(diff)>32)continuous=false;}
                if(continuous && !(positive && negative))conflict=true;
            }
            if(conflict)partners[i].push_back(j);
        }
    };
    if(parallel && normals.supportsConcurrentSampling())parallel(lines.size(),work);else for(size_t i=0;i<lines.size();++i)work(i);
    std::set<size_t> rejected;Json pairs=Json::array();size_t invalid=0;
    for(size_t i=0;i<lines.size();++i){invalid+=invalidNormals[i];for(size_t j:partners[i]){rejected.insert(i);rejected.insert(j);pairs.push_back({i,j});}}
    for(size_t i:rejected)chunk.lines[i].clear();
    return Json{{"key",chunk.key},{"total_paths",lines.size()},{"rejected",rejected},{"directed_conflict_pairs",pairs},
        {"invalid_normal_samples",invalid},{"normal_radius_base",cfg.conflictNormalBase},{"tangent_radius_base",cfg.conflictTangentBase},
        {"agreement_base",cfg.conflictAgreementBase},{"separation_base",cfg.conflictSeparationBase},
        {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}};
}


} // namespace vc::fiber_tracer
