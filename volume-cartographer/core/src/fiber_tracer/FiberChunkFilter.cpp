#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include "vc/fiber_tracer/detail/PolylineSegmentIndex.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <set>
#include <numeric>

namespace vc::fiber_tracer {
using Json=nlohmann::json;
constexpr double pi=3.14159265358979323846;
nlohmann::json inspectFiberChunkConflicts(const FiberChunkPaths& chunk,const FiberChunkTraceConfig& cfg,
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
    const double radius=std::max(cfg.conflictNormalBase,cfg.conflictTangentBase);
    for(size_t i=0;i<lines.size();++i) {
        auto& l=lines[i];
        for(const auto&p:chunk.lines[i])if(l.points.empty() || cv::norm(p-l.points.back())>1e-8)l.points.push_back(p);
        l.arcs.push_back(0);
        for(size_t j=1;j<l.points.size();++j) {
            l.arcs.push_back(l.arcs.back()+cv::norm(l.points[j]-l.points[j-1]));
        }
    }
    const auto indexStarted=std::chrono::steady_clock::now();
    const detail::ChunkSegmentIndex lookup(lines,[](const Line& l)->const auto& {return l.points;});
    const double indexSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-indexStarted).count();
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
            if(!field->sampleCornerBatchWithNormals(*nf,batch,1,&corners,nullptr))throw std::runtime_error("conflict normal preflight failed");batch.clear();};
        for(const auto&l:lines)for(const auto&[p,t]:samples(l)) {
            // Cover both sides of float rounding without changing double sampling.
            cv::Vec3f lower(p/scale),upper(lower);
            for(int k=0;k<3;++k){lower[k]=std::nextafter(lower[k],-std::numeric_limits<float>::infinity());upper[k]=std::nextafter(upper[k],std::numeric_limits<float>::infinity());}
            batch.push_back(lower);batch.push_back(upper);if(batch.size()>=4096)flush();
        }
        flush();
    }
    std::vector<std::vector<size_t>> partners(lines.size());
    std::vector<size_t> invalidNormals(lines.size());
    auto work=[&](size_t i) {
        const auto& l=lines[i];if(l.points.size()<2 || l.arcs.back()<cfg.conflictAgreementBase+cfg.conflictSeparationBase)return;
        std::vector<cv::Vec3d> points,tangents,normal;
        std::vector<bool> valid;
        std::set<size_t> candidates;
        std::vector<size_t> lastSeen(lines.size(),std::numeric_limits<size_t>::max()),run(lines.size());
        for(const auto& [p,t]:samples(l)) {
            auto n=normals.sampleNormal(p/scale);
            auto nv=n.normal-t*n.normal.dot(t);double length=cv::norm(nv);
            bool good=n.valid && std::isfinite(length) && length>1e-8;
            normal.push_back(good?nv/length:cv::Vec3d(0,0,0));valid.push_back(good);
            invalidNormals[i]+=!good;points.push_back(p);tangents.push_back(t);
            const size_t q=points.size()-1;
            lookup.nearbyPaths(p,radius,[&](size_t j) {
                if(j==i || lastSeen[j]==q)return;
                run[j]=(q>0 && lastSeen[j]==q-1)?run[j]+1:1;
                lastSeen[j]=q;
                // The existing close() requires end-start >= agreementSamples:
                // an actual shared run needs at least agreementSamples+1 samples.
                // Box overlap is only a necessary condition; exact ellipse,
                // direction, interior and correspondence tests below are unchanged.
                if(run[j]>agreementSamples)candidates.insert(j);
            });
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
    const auto matchStarted=std::chrono::steady_clock::now();
    if(parallel && normals.supportsConcurrentSampling())parallel(lines.size(),work);else for(size_t i=0;i<lines.size();++i)work(i);
    const double matchSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-matchStarted).count();
    std::set<size_t> rejected;Json pairs=Json::array();size_t invalid=0;
    for(size_t i=0;i<lines.size();++i){invalid+=invalidNormals[i];for(size_t j:partners[i]){rejected.insert(i);rejected.insert(j);pairs.push_back({i,j});}}
    return Json{{"key",chunk.key},{"total_paths",lines.size()},{"rejected",rejected},{"directed_conflict_pairs",pairs},
        {"invalid_normal_samples",invalid},{"normal_radius_base",cfg.conflictNormalBase},{"tangent_radius_base",cfg.conflictTangentBase},
        {"agreement_base",cfg.conflictAgreementBase},{"separation_base",cfg.conflictSeparationBase},
        {"index_seconds",indexSeconds},{"match_seconds",matchSeconds},
        {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}};
}

nlohmann::json filterFiberChunkConflicts(FiberChunkPaths& chunk,const FiberChunkTraceConfig& cfg,
    const vc::lasagna::NormalSampler& normals,double scale,const FiberChunkParallelFor& parallel,
    const FiberPredictionSource* readinessSource,const std::vector<size_t>* auditQueries)
{
    auto report=inspectFiberChunkConflicts(chunk,cfg,normals,scale,parallel,readinessSource,auditQueries);
    for(const auto& i:report.at("rejected"))chunk.lines[i.get<size_t>()].clear();
    return report;
}

nlohmann::json FiberChunkPathReduction::json() const {
    Json result={{"algorithm","terminal-complete-link-v2"},{"representative",representative},
        {"end_reasons",endReasons},{"eligible",eligible},{"terminal_pairs",terminalPairs},
        {"equivalent_pairs",equivalentPairs},{"fused_paths",fusedPaths},{"ambiguous_paths",ambiguousPaths}};
    // Omitted mode means historical strict behavior; retain existing resume bytes.
    if(!rejectAmbiguousPaths){result["reject_ambiguous_paths"]=false;result["retained_ambiguous_paths"]=retainedAmbiguousPaths;}
    return result;
}
void FiberChunkPathReduction::apply(FiberChunkPaths& chunk) const {
    if(chunk.lines.size()!=representative.size()||chunk.lines.size()!=endReasons.size())
        throw std::runtime_error("path reduction does not match source IDs");
    for(size_t i=0;i<representative.size();++i){
        const auto target=representative[i];
        if(target>=0&&(size_t(target)>=representative.size()||representative[target]!=target))
            throw std::runtime_error("invalid path reduction representative");
        if(target!=int64_t(i))chunk.lines[i].clear();
    }
    chunk.endReasons=endReasons;
}

FiberChunkPathReduction reduceFiberChunkPaths(const FiberChunkPaths& raw,const FiberChunkTraceConfig& cfg,
    const std::function<std::vector<float>(const std::vector<cv::Vec3d>&)>& widths,double widthTolerance,bool rejectAmbiguousPaths)
{
    validateFiberChunkConflictAnnotation(raw,cfg);
    if(raw.endReasons.size()!=raw.lines.size())throw std::invalid_argument("path reduction requires endpoint reasons");
    FiberChunkPathReduction result;result.rejectAmbiguousPaths=rejectAmbiguousPaths;
    result.representative.assign(raw.lines.size(),-1);result.endReasons=raw.endReasons;
    auto chunk=raw;std::vector<double> lengths(chunk.lines.size());
    for(size_t i=0;i<chunk.lines.size();++i){
        auto& line=chunk.lines[i];
        line.erase(std::unique(line.begin(),line.end(),[](const auto& a,const auto& b){return cv::norm(a-b)<1e-9;}),line.end());
        if(line.size()<2||(cfg.conflictNormalBase>0&&chunk.conflictFlags[i])){line.clear();continue;}
        lengths[i]=makePolylineArcGeometry(line).length();
        result.representative[i]=i;++result.eligible;
    }
    const detail::ChunkSegmentIndex index(chunk.lines,[](const auto& points)->const auto&{return points;});
    std::set<std::pair<size_t,size_t>> relations;
    for(size_t i=0;i<chunk.lines.size();++i)if(result.representative[i]>=0)for(bool reverse:{false,true}){
        auto history=chunk.lines[i];if(reverse)std::reverse(history.begin(),history.end());
        std::set<size_t> hits;
        index.nearbyPaths(history.back(),cfg.joinDistanceBase,[&](size_t j){if(j!=i&&result.representative[j]>=0)hits.insert(j);});
        if(hits.empty())continue;
        const std::vector<size_t> selected(hits.begin(),hits.end());
        const auto tangent=history.back()-history[history.size()-2];
        for(const auto& match:findFiberChunkMatches(chunk,history,tangent,cfg,false,nullptr,widths,widthTolerance,
                &selected,FiberChunkMatchPurpose::TerminalRelation))
            relations.emplace(std::min(i,match.line),std::max(i,match.line));
    }
    result.terminalPairs=relations.size();
    std::vector<size_t> parent(chunk.lines.size());std::iota(parent.begin(),parent.end(),0);
    const auto root=[&](size_t id){while(parent[id]!=id){parent[id]=parent[parent[id]];id=parent[id];}return id;};
    for(auto [a,b]:relations){a=root(a);b=root(b);if(a!=b)parent[std::max(a,b)]=std::min(a,b);}
    std::map<size_t,std::vector<size_t>> components;
    for(size_t i=0;i<parent.size();++i)if(result.representative[i]>=0)components[root(i)].push_back(i);
    // Each unordered proof is evaluated once. Full matching has no 128-vx cap;
    // a partial success of the ordinary matcher is NOT proof of equivalence.
    const auto fullOrientations=[&](size_t a,size_t b){
        const auto& history=chunk.lines[a];const std::vector<size_t> selected{b};std::set<bool> orientations;
        for(const auto& match:findFiberChunkMatches(chunk,history,history.back()-history[history.size()-2],cfg,false,
                nullptr,widths,widthTolerance,&selected,FiberChunkMatchPurpose::FullEquivalence)){
            constexpr double tolerance=1e-6; // base-voxel numerical coverage tolerance, not a geometric gate
            const auto& target=chunk.lines[b];
            const auto begin=match.reverse?target.back():target.front();
            const auto end=match.reverse?target.front():target.back();
            if(std::abs(match.overlap-lengths[a])<=tolerance&&
               cv::norm(history.front()-begin)<=cfg.joinDistanceBase&&
               cv::norm(history.back()-end)<=cfg.joinDistanceBase)orientations.insert(match.reverse);
        }
        return orientations;
    };
    for(const auto& [_,members]:components){
        if(members.size()<2)continue;
        bool equivalent=true;std::map<std::pair<size_t,size_t>,bool> orientations;
        for(size_t a=0;a<members.size()&&equivalent;++a)for(size_t b=a+1;b<members.size();++b)
            if(!relations.contains({members[a],members[b]})){equivalent=false;break;}
        for(size_t a=0;a<members.size()&&equivalent;++a)for(size_t b=a+1;b<members.size();++b){
            const auto i=members[a],j=members[b];const auto ij=fullOrientations(i,j),ji=fullOrientations(j,i);
            std::vector<bool> common;for(bool reverse:ij)if(ji.contains(reverse))common.push_back(reverse);
            if(common.size()!=1){equivalent=false;break;}
            orientations[{i,j}]=common.front();++result.equivalentPairs;
        }
        const auto reversed=[&](size_t a,size_t b){return a==b?false:orientations.at({std::min(a,b),std::max(a,b)});};
        if(equivalent)for(size_t a=0;a<members.size()&&equivalent;++a)for(size_t b=a+1;b<members.size();++b)
            if(reversed(members[a],members[b])!=(reversed(members[0],members[a])!=reversed(members[0],members[b]))){equivalent=false;break;}
        if(!equivalent){
            if(rejectAmbiguousPaths)for(auto i:members){result.representative[i]=-2;++result.ambiguousPaths;}
            else result.retainedAmbiguousPaths+=members.size();
            continue;
        }
        size_t representative=members.front();
        for(auto i:members)if(lengths[i]>lengths[representative])representative=i;
        for(auto i:members)result.representative[i]=representative;
        result.fusedPaths+=members.size()-1;
        // Never turn a member's terminal stop into a newly traversable end.
        for(size_t end=0;end<2;++end){
            std::optional<std::array<int,3>> destination;
            for(auto i:members){
                const size_t memberEnd=end^size_t(reversed(representative,i));
                const auto& line=chunk.lines[i];const auto point=memberEnd?line.back():line.front();
                const auto previous=memberEnd?line[line.size()-2]:line[1];
                const auto next=nextFiberChunk(chunk,point,previous,chunk.endReasons[i][memberEnd],cfg);
                if(!next.stop.empty()){
                    result.endReasons[representative][end]="fused_member_stop:"+next.stop;break;
                }
                if(destination&&*destination!=next.key){
                    result.endReasons[representative][end]="fused_member_stop:chunk_neighbor_disagreement";break;
                }
                destination=next.key;
            }
        }
    }
    return result;
}

} // namespace vc::fiber_tracer
