#include "FiberConstraintChunks.hpp"
#include "FiberCrossingAxes.hpp"
#include "FiberNeighbourNormals.hpp"
#include "vc/fiber_tracer/FiberGapFalloff.hpp"
#include "vc/fiber_tracer/FiberStripNormals.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "ChunkSaturationWindow.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/fiber_tracer/FiberNormalOrientation.hpp"
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <vc_object_store/objects/InitialImport.hpp>
#include <vc_object_store/objects/FiberGraphSolveExtension.hpp>
#include <algorithm>
#include <fstream>
#include <atomic>
#include <chrono>
#include <bit>
#include <cstring>
#include <ctime>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>

namespace {
using namespace vc::fiber_tracer;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
using Cell=std::array<int64_t,3>;
double elapsed(Clock::time_point a){return std::chrono::duration<double>(Clock::now()-a).count();}
std::optional<double> threadCpu() {
#ifdef CLOCK_THREAD_CPUTIME_ID
    timespec t{};
    if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0)return double(t.tv_sec)+double(t.tv_nsec)*1e-9;
#endif
    return {};
}

struct Frame {
    size_t source=0,ordinal=0;
    FiberNormalPieceFrame ends;
    bool arrived=false;
    signed char sign=2; // 2=pending; 0=explicitly unresolved; +/-1=resolved
    double outwardSum=0.,outwardWeight=0.;
};
struct Job {
    Cell cell;
    std::vector<size_t> ownedFrames;
    std::vector<size_t> encounters;
    std::vector<FiberTraceConstraint> events;
    FiberConstraintNormalSamples normals;
    Json metrics;
};
struct ChunkBenchmarkComplete:std::exception {
    const char* what()const noexcept override{return "intentional saturated chunk benchmark cutoff";}
};

// One regular arc grid per ORIGINAL fiber. Endpoints are evaluated against the
// preceding regular normal without changing that grid's local gauge. Thus two
// jobs referencing the same piece obtain the same frame independently of which
// attachments they own. Spatial cuts add identical boundary references, not
// scientific fiber endpoints or independent resampled curve geometries.
// Geometry-only strip normals per original fiber vertex (normal_source
// "geometry"): the sampled normal at an arc is the unit interpolation of the
// two vertex normals of its segment. The fiber's whole-fiber sign is left to
// the later sign stage, exactly like a Lasagna normal's.
using StripNormals=std::vector<std::vector<cv::Vec3d>>;
vc::lasagna::NormalSample stripNormalAt(const StripNormals& strips,size_t source,const PolylineArcGeometry& g,double arc)
{
    vc::lasagna::NormalSample sample;
    const auto& normals=strips[source];
    if(normals.size()!=g.points.size()||normals.size()<2){sample.reason="fiber too short for strip normals";return sample;}
    const auto at=samplePolylineArc(g,arc);
    const size_t s=std::min(at.segmentIndex,normals.size()-2);
    const double span=g.vertexArcs[s+1]-g.vertexArcs[s];
    const double f=span>0?std::clamp((arc-g.vertexArcs[s])/span,0.,1.):0.;
    cv::Vec3d n=normals[s]*(1.-f)+normals[s+1]*f;
    const double length=cv::norm(n);
    if(!(length>1e-9)){n=normals[f<.5?s:s+1];}
    else n/=length;
    sample.normal=n;sample.valid=true;return sample;
}

void sampleJob(Job& job,const std::vector<size_t>& owned,
    const std::vector<Frame>& frames,const std::vector<size_t>& offsets,
    const std::vector<PolylineArcGeometry>& geometries,
    const std::vector<PolylineGridPartition>& partitions,
    const vc::lasagna::LasagnaNormalSampler& reader,const StripNormals* strips,double spacing,
    const vc::core::util::Umbilicus& axis,
    std::vector<std::pair<size_t,Frame>>& completed)
{
    const auto started=Clock::now();
    std::map<size_t,std::vector<size_t>> attachments;
    for(size_t i=0;i<job.events.size();++i)for(int side=0;side<2;++side) {
        const auto& c=job.events[i];const auto source=side?c.pieceB:c.pieceA;
        const auto arc=side?c.arcBBaseVoxels:c.arcABaseVoxels;
        const auto& p=partitions[source];
        auto end=std::upper_bound(p.boundaries.begin(),p.boundaries.end(),arc,
            [](double a,const auto& b){return a<b.arc;});
        const auto ordinal=std::min(p.pieces.size()-1,size_t(end-p.boundaries.begin()-1));
        const auto frame=offsets[source]+ordinal;
        attachments[frame].push_back(2*i+side);
    }
    for(auto f:owned)attachments.try_emplace(f);
    struct Sample {double arc;cv::Vec3d point;size_t slot;double weight;};
    constexpr auto regular=std::numeric_limits<size_t>::max();
    struct Piece {size_t id,first,last;};
    std::vector<Sample> samples;std::vector<Piece> pieces;
    for(const auto& [id,slots]:attachments) {
        const auto& frame=frames[id];const auto source=frame.source;
        const auto& g=geometries[source];const auto& p=partitions[source];
        const auto& piece=p.pieces[frame.ordinal];
        const auto& a=p.boundaries[piece.beginBoundary];const auto& b=p.boundaries[piece.endBoundary];
        const auto first=samples.size();
        samples.push_back({a.arc,a.point,regular,0.});
        // Multiplication defines a stable, common original-fiber grid. Never
        // restart arc sampling at a spatial cut or repeatedly accumulate steps.
        const auto begin=size_t(std::ceil(a.arc/spacing));
        for(size_t k=begin;;++k) {
            const auto arc=double(k)*spacing;if(!(arc<b.arc))break;
            if(arc<a.arc)continue;
            const auto pnt=samplePolylineArc(g,arc).point;
            const double left=k?double(k-1)*spacing:0.;
            const double right=std::min(g.length(),double(k+1)*spacing);
            const auto weight=(right-left)*.5;
            if(arc==a.arc)samples.back().weight=weight;
            else samples.push_back({arc,pnt,regular,weight});
        }
        const double endWeight=b.arc==g.length()?
            (g.length()-std::max(0.,(std::ceil(g.length()/spacing)-1.)*spacing))*.5:0.;
        samples.push_back({b.arc,b.point,regular,endWeight});
        for(auto slot:slots) {
            const auto& c=job.events[slot/2];
            samples.push_back({slot%2?c.arcBBaseVoxels:c.arcABaseVoxels,
                slot%2?c.pointBBaseXYZ:c.pointABaseXYZ,slot,0.});
        }
        std::stable_sort(samples.begin()+first,samples.end(),[](const auto& a,const auto& b){return a.arc<b.arc;});
        pieces.push_back({id,first,samples.size()});
    }
    std::vector<cv::Vec3d> points;points.reserve(samples.size());
    for(const auto& sample:samples)points.push_back(sample.point);
    std::vector<vc::lasagna::NormalSampleWithDerivative> values;
    if(strips){
        values.resize(samples.size());
        for(const auto& piece:pieces){
            const auto source=frames[piece.id].source;
            for(size_t i=piece.first;i<piece.last;++i)values[i].sample=stripNormalAt(*strips,source,geometries[source],samples[i].arc);
        }
    } else (void)reader.sampleNormalBatch(points,false,1,values);
    if(values.size()!=samples.size())throw std::runtime_error("Chunk normal sample count differs");
    job.normals.endpoints.resize(2*job.events.size());job.normals.valid.resize(2*job.events.size());
    size_t invalid=0;
    for(const auto& piece:pieces) {
        Frame frame;frame.source=frames[piece.id].source;frame.ordinal=frames[piece.id].ordinal;
        const auto& partition=partitions[frame.source];const auto& range=partition.pieces[frame.ordinal];
        frame.ends.beginBoundary=range.beginBoundary;frame.ends.endBoundary=range.endBoundary;
        FiberNormalOrientationSweep sweep;
        for(size_t i=piece.first;i<piece.last;++i) {
            auto n=values[i].sample.normal;const bool valid=values[i].sample.valid;
            if(valid && (!std::isfinite(n.dot(n)) || std::abs(n.dot(n)-1.)>1e-6))
                throw std::runtime_error("Non-unit valid chunk normal");
            if(samples[i].slot==regular) {
                sweep.orient(n,valid);
                if(samples[i].arc==partition.boundaries[range.beginBoundary].arc) {
                    frame.ends.beginNormal=n;frame.ends.beginValid=valid;
                }
                if(samples[i].arc==partition.boundaries[range.endBoundary].arc) {
                    frame.ends.endNormal=n;frame.ends.endValid=valid;
                }
                if(valid && samples[i].weight>0.) {
                    const cv::Vec3d radial(-axis.vector_to_umbilicus(cv::Vec3f(samples[i].point)));
                    const auto length=cv::norm(radial);
                    if(length>0.) {frame.outwardSum+=samples[i].weight*n.dot(radial/length);
                        frame.outwardWeight+=samples[i].weight;}
                }
            } else {
                auto atEndpoint=sweep;atEndpoint.orient(n,valid);
                const auto slot=samples[i].slot;job.normals.endpoints[slot]=n;job.normals.valid[slot]=valid;
                if(!valid)job.normals.invalidReasons.emplace(slot,values[i].sample.reason);
            }
            invalid+=!valid;
        }
        if(std::binary_search(owned.begin(),owned.end(),piece.id))completed.emplace_back(piece.id,std::move(frame));
    }
    job.metrics["normal_samples"]=samples.size();job.metrics["invalid_samples"]=invalid;
    job.metrics["normal_seconds"]=elapsed(started);
}
}

Json extractFiberConstraintChunks(const Json& cfg,Json& metadata,
    const std::vector<std::string>& ids,const std::vector<FiberletCropTraceLine>& lines,
    const FiberTraceConstraintConfig& extraction,
    const vc::lasagna::LasagnaNormalSampler& normals,const vc::lasagna::LasagnaNormalSampler& windings,
    const vc::core::util::Umbilicus& axis,FiberConstraintDatabaseOptions databaseOptions,
    std::function<Json()> memorySnapshot,const vc::lasagna::LasagnaChannelBinding* presence)
{
    const auto started=Clock::now();const double size=cfg.value("chunk_size_base_voxels",defaultFiberConstraintChunkSizeBaseVoxels);
    const double spacing=cfg.value("normal_sample_spacing",4.);
    const bool profile=cfg.value("profile_stages",false);
    const bool benchmark=cfg.value("saturated_chunk_benchmark",false);
    if(benchmark)databaseOptions.packedOptions.memoryStats=std::make_shared<vc::objects::PackedStoreMemoryStats>();
    const size_t workers=std::max<size_t>(1,extraction.parallelThreads);
    if(!(size>0.)||!std::isfinite(size)||!(spacing>0.)||!std::isfinite(spacing))
        throw std::invalid_argument("Invalid spatial chunk dimensions or normal spacing");
    if(ids.size()!=lines.size() || extraction.crossQueryPrefix || extraction.extraPieceCutsBaseVoxels.size())
        throw std::invalid_argument("Chunk pipeline requires ordinary whole source fibers");
    // Original fiber pieces supply normal frames, not geometry search halos.
    std::vector<PolylineArcGeometry> geometries(lines.size());
    std::vector<PolylineGridPartition> partitions(lines.size());
    std::atomic<size_t> next{0};std::vector<std::future<void>> preparation;
    for(size_t w=0;w<std::min(workers,lines.size());++w)preparation.push_back(std::async(std::launch::async,[&] {
        for(size_t i;(i=next.fetch_add(1))<lines.size();) {
            geometries[i]=makePolylineArcGeometry(lines[i].pointsBaseXYZ);
            partitions[i]=partitionPolylineArcGrid(geometries[i],size);
        }
    }));
    std::exception_ptr preparationFailure;
    for(auto& task:preparation)try{task.get();}catch(...){if(!preparationFailure)preparationFailure=std::current_exception();}
    if(preparationFailure)std::rethrow_exception(preparationFailure);
    std::unique_ptr<StripNormals> strips;
    const auto normalSource=cfg.value("normal_source",std::string("lasagna"));
    if(normalSource=="geometry"||normalSource=="fused"){
        const auto stripStarted=Clock::now();
        vc::fiber_tracer::FiberStripNormalOptions options;
        options.twistWeight=cfg.value("strip_twist_weight",options.twistWeight);
        options.lateralWeight=cfg.value("strip_lateral_weight",options.lateralWeight);
        options.maxIterations=cfg.value("strip_iterations",options.maxIterations);
        options.smoothingSigma=cfg.value("strip_smoothing_sigma",options.smoothingSigma);
        options.resampleSpacing=cfg.value("strip_resample_spacing",options.resampleSpacing);
        std::vector<std::vector<cv::Vec3d>> polylines(lines.size());
        for(size_t i=0;i<lines.size();++i)polylines[i]=lines[i].pointsBaseXYZ;
        // "fused": the predicted sheet axis at every chain node, weighted by the
        // prediction's presence between presence_floor (weight 0) and
        // presence_full (weight 1); nodes without a prediction keep weight 0.
        std::unique_ptr<std::vector<std::vector<vc::fiber_tracer::FiberStripAxisTarget>>> targets;
        Json fusion;
        if(normalSource=="fused"){
            if(!presence)throw std::invalid_argument("normal_source fused requires the prediction's presence channel");
            options.axisWeight=cfg.value("fused_axis_weight",options.axisWeight);
            const double floor=cfg.value("fused_presence_floor",0.);
            const double full=cfg.value("fused_presence_full",128.);
            if(!(full>floor)||floor<0.)throw std::invalid_argument("fused_presence_full must exceed fused_presence_floor >= 0");
            // The prediction is sampled on the traced vertices themselves (they lie in
            // the sheet; the smoothed chain may not) and gathered per chain node as
            // the principal axis of the presence-weighted outer products of the
            // vertex axes assigned to the node (nearest arc).
            // Chains (smoothing + resampling) per fiber in parallel; the solve recomputes them identically.
            const auto chainStarted=Clock::now();
            std::vector<vc::fiber_tracer::FiberStripChain> chains(lines.size());
            cv::parallel_for_(cv::Range(0,int(lines.size())),[&](const cv::Range& range){
                for(int i=range.start;i<range.end;++i)chains[size_t(i)]=vc::fiber_tracer::fiberStripChain(polylines[size_t(i)],options);
            },int(workers));
            const double chainSeconds=elapsed(chainStarted);
            std::vector<cv::Vec3d> vertices;std::vector<std::pair<size_t,size_t>> owner;
            size_t vertexTotal=0;for(const auto& line:polylines)vertexTotal+=line.size();
            vertices.reserve(vertexTotal);owner.reserve(vertexTotal);
            targets=std::make_unique<std::vector<std::vector<vc::fiber_tracer::FiberStripAxisTarget>>>(lines.size());
            for(size_t i=0;i<lines.size();++i){
                (*targets)[i].resize(chains[i].points.size());
                for(size_t j=0;j<polylines[i].size();++j){vertices.push_back(polylines[i][j]);owner.push_back({i,j});}
            }
            // Axes, sampled per slice of whole fibers on worker threads: chunk
            // decoding happens on the calling thread, so one big batch would decode
            // the whole prediction serially.
            const auto axesStarted=Clock::now();
            std::vector<vc::lasagna::NormalSampleWithDerivative> axes(vertices.size());
            {
                const size_t slices=std::max<size_t>(1,std::min(workers*8,lines.size()));
                std::vector<size_t> sliceFirstVertex(slices+1,0);
                for(size_t sIdx=0;sIdx<slices;++sIdx){
                    const size_t fiberEnd=(sIdx+1)*lines.size()/slices;
                    size_t v=0;for(size_t i=0;i<fiberEnd;++i)v+=polylines[i].size();
                    sliceFirstVertex[sIdx+1]=v;
                }
                cv::parallel_for_(cv::Range(0,int(slices)),[&](const cv::Range& range){
                    for(int sIdx=range.start;sIdx<range.end;++sIdx){
                        const size_t begin=sliceFirstVertex[size_t(sIdx)],end=sliceFirstVertex[size_t(sIdx)+1];
                        if(begin>=end)continue;
                        std::vector<cv::Vec3d> slice(vertices.begin()+begin,vertices.begin()+end);
                        std::vector<vc::lasagna::NormalSampleWithDerivative> values;
                        (void)normals.sampleNormalBatch(slice,false,1,values);
                        for(size_t j=0;j<values.size();++j)axes[begin+j]=std::move(values[j]);
                    }
                },int(workers));
            }
            const double axesSeconds=elapsed(axesStarted);
            // Presence, sampled on `workers` slices in parallel (the vector batch itself is serial).
            const auto presenceStarted=Clock::now();
            std::vector<cv::Vec3f> verticesF(vertices.begin(),vertices.end());
            std::vector<std::vector<float>> presenceValues(verticesF.size());
            {
                const size_t slices=std::max<size_t>(1,std::min(workers*4,verticesF.size()));
                const size_t per=(verticesF.size()+slices-1)/slices;
                cv::parallel_for_(cv::Range(0,int(slices)),[&](const cv::Range& range){
                    for(int sIdx=range.start;sIdx<range.end;++sIdx){
                        const size_t begin=size_t(sIdx)*per,end=std::min(verticesF.size(),begin+per);
                        if(begin>=end)continue;
                        std::vector<cv::Vec3f> slice(verticesF.begin()+begin,verticesF.begin()+end);
                        std::vector<std::vector<float>> values;
                        vc::lasagna::sampleLasagnaVectorBatch({*presence},slice,values,false);
                        for(size_t j=0;j<values.size();++j)presenceValues[begin+j]=std::move(values[j]);
                    }
                },int(workers));
            }
            const double presenceSeconds=elapsed(presenceStarted);
            std::cout<<"fused sampling chain_seconds="<<chainSeconds<<" axes_seconds="<<axesSeconds<<" presence_seconds="<<presenceSeconds<<" vertices="<<vertices.size()<<std::endl;
            // Per fiber in parallel: gather the vertex axes into each chain node's
            // presence-weighted scatter matrix and take its principal axis.
            const auto gatherStarted=Clock::now();
            std::vector<size_t> fiberFirstVertex(lines.size()+1,0);
            for(size_t i=0;i<lines.size();++i)fiberFirstVertex[i+1]=fiberFirstVertex[i]+polylines[i].size();
            std::atomic<size_t> verticesWithAxisCount{0},withAxisCount{0};std::atomic<double> weightSumTotal{0.};
            cv::parallel_for_(cv::Range(0,int(lines.size())),[&](const cv::Range& range){
                for(int fi=range.start;fi<range.end;++fi){
                    const size_t i=size_t(fi);const auto& chain=chains[i];
                    if(chain.points.empty())continue;
                    std::vector<cv::Matx33d> scatter(chain.points.size(),cv::Matx33d::zeros());
                    std::vector<double> weightSums(chain.points.size(),0.);std::vector<size_t> counts(chain.points.size(),0);
                    size_t local=0;
                    for(size_t vertex=0;vertex<polylines[i].size();++vertex){
                        const size_t j=fiberFirstVertex[i]+vertex;
                        const double arc=chain.vertexArc[vertex];
                        const auto it=std::lower_bound(chain.arc.begin(),chain.arc.end(),arc);
                        size_t k=size_t(it-chain.arc.begin());
                        if(k==chain.arc.size())k=chain.arc.size()-1;
                        else if(k>0&&arc-chain.arc[k-1]<chain.arc[k]-arc)--k;
                        ++counts[k];
                        if(!axes[j].sample.valid||presenceValues[j].empty())continue;
                        const double w=std::clamp((double(presenceValues[j][0])-floor)/(full-floor),0.,1.);
                        if(!(w>0.))continue;
                        const cv::Vec3d a=axes[j].sample.normal;
                        scatter[k]+=w*cv::Matx33d(a[0]*a[0],a[0]*a[1],a[0]*a[2],a[1]*a[0],a[1]*a[1],a[1]*a[2],a[2]*a[0],a[2]*a[1],a[2]*a[2]);
                        weightSums[k]+=w;++local;
                    }
                    verticesWithAxisCount.fetch_add(local);
                    size_t nodesWithAxis=0;double fiberWeight=0.;
                    for(size_t k=0;k<chain.points.size();++k){
                        if(!(weightSums[k]>0.)||!counts[k])continue;
                        cv::Vec3d eigenvalues;cv::Matx33d eigenvectors;
                        cv::eigen(scatter[k],eigenvalues,eigenvectors);
                        auto& target=(*targets)[i][k];
                        target.axis=cv::Vec3d(eigenvectors(0,0),eigenvectors(0,1),eigenvectors(0,2));
                        target.weight=weightSums[k]/double(counts[k]);
                        ++nodesWithAxis;fiberWeight+=target.weight;
                    }
                    withAxisCount.fetch_add(nodesWithAxis);
                    double expected=weightSumTotal.load();while(!weightSumTotal.compare_exchange_weak(expected,expected+fiberWeight)){}
                }
            },int(workers));
            size_t nodes=0;for(const auto& chain:chains)nodes+=chain.points.size();
            const size_t verticesWithAxis=verticesWithAxisCount.load(),withAxis=withAxisCount.load();const double weightSum=weightSumTotal.load();
            const double gatherSeconds=elapsed(gatherStarted);
            std::cout<<"fused gather seconds="<<gatherSeconds<<" nodes="<<nodes<<" nodes_with_axis="<<withAxis<<std::endl;
            fusion=Json{{"axis_weight",options.axisWeight},{"presence_floor",floor},{"presence_full",full},
                {"chain_seconds",chainSeconds},{"axes_seconds",axesSeconds},{"presence_seconds",presenceSeconds},{"gather_seconds",gatherSeconds},
                {"vertices",vertices.size()},{"vertices_with_axis",verticesWithAxis},
                {"chain_nodes",nodes},{"nodes_with_axis",withAxis},{"mean_axis_weight",withAxis?weightSum/withAxis:0.}};
            // Signed anchors transported from an independently resolved fiber
            // volume: {"<fiber uuid>": [[arc_base, nx, ny, nz, weight], ...]}. They
            // replace the free axis at their nearest chain node and fix its sign.
            if(cfg.contains("normal_anchors")){
                const double anchorWeight=cfg.value("fused_anchor_weight",100.);
                std::ifstream in(cfg.at("normal_anchors").get<std::string>());
                if(!in)throw std::runtime_error("cannot read normal_anchors");
                const Json anchors=Json::parse(in);
                std::map<std::string,size_t> index;for(size_t i=0;i<ids.size();++i)index[ids[i]]=i;
                std::vector<std::vector<cv::Vec3d>> sums(lines.size());std::vector<std::vector<double>> anchorWeights(lines.size());
                for(size_t i=0;i<lines.size();++i){sums[i].assign(chains[i].points.size(),cv::Vec3d(0,0,0));anchorWeights[i].assign(chains[i].points.size(),0.);}
                size_t used=0,unknown=0;
                for(const auto& [id,list]:anchors.items()){
                    const auto it=index.find(id);
                    if(it==index.end()){++unknown;continue;}
                    const size_t i=it->second;const auto& chain=chains[i];
                    if(chain.points.empty())continue;
                    for(const auto& entry:list){
                        const double arc=entry.at(0).get<double>();
                        cv::Vec3d a(entry.at(1).get<double>(),entry.at(2).get<double>(),entry.at(3).get<double>());
                        const double w=entry.size()>4?entry.at(4).get<double>():1.;
                        const double length=cv::norm(a);
                        if(!(length>0.)||!(w>0.))continue;
                        const auto pos=std::lower_bound(chain.arc.begin(),chain.arc.end(),arc);
                        size_t k=size_t(pos-chain.arc.begin());
                        if(k==chain.arc.size())k=chain.arc.size()-1;
                        else if(k>0&&arc-chain.arc[k-1]<chain.arc[k]-arc)--k;
                        sums[i][k]+=a*(w/length);anchorWeights[i][k]+=w;++used;
                    }
                }
                size_t anchoredNodes=0;
                for(size_t i=0;i<lines.size();++i)for(size_t k=0;k<chains[i].points.size();++k){
                    if(!(anchorWeights[i][k]>0.)||!(cv::norm(sums[i][k])>0.))continue;
                    auto& target=(*targets)[i][k];
                    target.axis=sums[i][k]/cv::norm(sums[i][k]);target.weight=anchorWeight*anchorWeights[i][k];target.fixed=true;++anchoredNodes;
                }
                fusion["anchors"]=Json{{"weight",anchorWeight},{"anchors_used",used},{"anchors_for_unknown_fibers",unknown},{"anchored_nodes",anchoredNodes}};
            }
            // Sign-free axes from crossing fibers (own tangent x crossing tangent),
            // read from a crossing axes file (FiberCrossingAxes). Each spreads to the
            // chain nodes within 2 sigma of its arc with a Gaussian falloff, scaled
            // by a falloff of its 3D offset to the crossing fiber
            // (crossing_gap_falloff: none, inverse, inverse_square, gaussian, cauchy,
            // inverse_smooth). The offset is scaled per axis by crossing_gap_scales
            // (own tangent, side, normal) in crossing_gap_frame: isotropic (|offset|
            // / first scale), crossing (own tangent and crossing axis) or prediction
            // (per node: the prediction axis and the own tangent perpendicular to
            // it), and by |axis . prediction axis|^crossing_prediction_power (entries
            // below crossing_prediction_min are dropped). A node takes the principal
            // axis of the weighted outer products and the weight crossing_weight *
            // min(1, summed weight) * coherence^crossing_coherence_power, where
            // coherence = (l1 - l2) / (l1 + l2 + l3) of the scatter eigenvalues.
            if(cfg.contains("crossing_axes")){
                const double crossingWeight=cfg.value("crossing_weight",100.),sigma=cfg.value("crossing_sigma",64.);
                if(!(sigma>0.)||crossingWeight<0.)throw std::invalid_argument("crossing_sigma must be positive and crossing_weight >= 0");
                const std::string gapFalloff=cfg.value("crossing_gap_falloff",std::string("none")),gapFrame=cfg.value("crossing_gap_frame",std::string("isotropic"));
                const auto scales=cfg.value("crossing_gap_scales",std::vector<double>{64.,64.,64.});
                const double predictionPower=cfg.value("crossing_prediction_power",0.),
                    predictionMin=cfg.value("crossing_prediction_min",0.),coherencePower=cfg.value("crossing_coherence_power",0.);
                vc::fiber_tracer::FiberCrossingTargetOptions crossingOptions;
                crossingOptions.weight=crossingWeight;crossingOptions.sigma=sigma;
                crossingOptions.falloff=parseGapFalloff(gapFalloff);
                crossingOptions.frame=gapFrame=="isotropic"?vc::fiber_tracer::FiberCrossingGapFrame::Isotropic
                    :gapFrame=="crossing"?vc::fiber_tracer::FiberCrossingGapFrame::Crossing
                    :gapFrame=="prediction"?vc::fiber_tracer::FiberCrossingGapFrame::Prediction
                    :throw std::invalid_argument("crossing_gap_frame must be isotropic, crossing or prediction");
                if(scales.size()!=3)throw std::invalid_argument("crossing_gap_scales must be 3 positive values (own tangent, side, normal), crossing powers >= 0");
                crossingOptions.scales={scales[0],scales[1],scales[2]};
                crossingOptions.predictionPower=predictionPower;crossingOptions.predictionMin=predictionMin;crossingOptions.coherencePower=coherencePower;
                size_t unknown=0;
                const auto crossings=readFiberCrossingAxes(cfg.at("crossing_axes").get<std::string>(),ids,unknown);
                std::atomic<size_t> usedTotal{0},crossedTotal{0},predictionTotal{0},crossingTotal{0};
                cv::parallel_for_(cv::Range(0,int(crossings.size())),[&](const cv::Range& range){
                for(int fi=range.start;fi<range.end;++fi){
                    const size_t i=size_t(fi);
                    vc::fiber_tracer::FiberCrossingTargetCounts counts;
                    vc::fiber_tracer::setFiberCrossingTargets(chains[i],crossings[i],crossingOptions,(*targets)[i],&counts);
                    usedTotal+=counts.used;crossedTotal+=counts.crossedNodes;predictionTotal+=counts.predictionFrames;crossingTotal+=counts.crossingFrames;
                }
                },int(workers));
                const size_t used=usedTotal.load(),crossedNodes=crossedTotal.load(),predictionFrames=predictionTotal.load(),crossingFrames=crossingTotal.load();
                fusion["crossings"]=Json{{"weight",crossingWeight},{"sigma",sigma},{"gap_falloff",gapFalloff},{"gap_frame",gapFrame},{"gap_scales",scales},
                    {"prediction_frames",predictionFrames},{"crossing_frames",crossingFrames},
                    {"prediction_power",predictionPower},{"prediction_min",predictionMin},{"coherence_power",coherencePower},{"used",used},{"for_unknown_fibers",unknown},{"crossed_nodes",crossedNodes}};
            }
            if(cfg.contains("neighbour_normals")){
                // Signed normals of other fibers (neighbour_normals: strip samples of inferred
                // normals), gathered per chain node with a Gaussian of neighbour_sigma along
                // the arc: the signed mean direction, weight neighbour_weight * min(1, summed
                // Gaussian) * |mean| / summed Gaussian (agreement of the samples).
                const double neighbourWeight=cfg.value("neighbour_weight",100.),sigma=cfg.value("neighbour_sigma",32.);
                if(!(sigma>0.)||neighbourWeight<0.)throw std::invalid_argument("neighbour_sigma must be positive and neighbour_weight >= 0");
                size_t unknown=0;
                const auto inferred=readFiberStripSamples(cfg.at("neighbour_normals").get<std::string>(),ids,unknown);
                std::atomic<size_t> nodesTotal{0};
                cv::parallel_for_(cv::Range(0,int(lines.size())),[&](const cv::Range& range){
                    for(int fi=range.start;fi<range.end;++fi){
                        const size_t i=size_t(fi);
                        nodesTotal+=vc::fiber_tracer::setFiberNeighbourTargets(chains[i],inferred[i],neighbourWeight,sigma,(*targets)[i]);
                    }
                },int(workers));
                fusion["neighbours"]=Json{{"weight",neighbourWeight},{"sigma",sigma},{"nodes",nodesTotal.load()},{"for_unknown_fibers",unknown}};
            }
        }
        const auto solved=vc::fiber_tracer::solveFiberStripNormals(polylines,options,int(workers),targets.get());
        strips=std::make_unique<StripNormals>(lines.size());
        size_t degenerate=0,unconverged=0;double cost=0;
        for(size_t i=0;i<lines.size();++i){(*strips)[i]=solved[i].normals;degenerate+=solved[i].normals.empty();unconverged+=!solved[i].normals.empty()&&!solved[i].converged;cost+=solved[i].finalCost;}
        metadata["strip_normals"]=Json{{"source",normalSource},{"twist_weight",options.twistWeight},{"lateral_weight",options.lateralWeight},
            {"smoothing_sigma",options.smoothingSigma},{"resample_spacing",options.resampleSpacing},
            {"max_iterations",options.maxIterations},{"fibers",lines.size()},{"degenerate_fibers",degenerate},
            {"unconverged_fibers",unconverged},{"final_cost",cost},{"seconds",elapsed(stripStarted)}};
        if(!fusion.is_null())metadata["strip_normals"]["fusion"]=fusion;
        if(cfg.contains("strip_samples_output")){
            // The solved normals at a regular arc spacing, for neighbour-normal inference.
            const double step=cfg.value("strip_samples_spacing",16.);
            if(!(step>0.))throw std::invalid_argument("strip_samples_spacing must be positive");
            // With the fused prediction, each sample also carries the unit prediction axis
            // of its nearest chain node (zero where the node has none).
            std::vector<std::vector<FiberStripSample>> samples(lines.size());
            cv::parallel_for_(cv::Range(0,int(lines.size())),[&](const cv::Range& range){
                for(int fi=range.start;fi<range.end;++fi){
                    const size_t i=size_t(fi);const auto& g=geometries[i];
                    vc::fiber_tracer::FiberStripChain chain;
                    if(targets)chain=vc::fiber_tracer::fiberStripChain(polylines[i],options);
                    for(double arc=0.;arc<=g.length();arc+=step){
                        const auto n=stripNormalAt(*strips,i,g,arc);
                        if(!n.valid)continue;
                        cv::Vec3d prediction{0,0,0};
                        if(targets&&!chain.arc.empty()&&chain.arc.size()==(*targets)[i].size()){
                            const auto k=size_t(std::lower_bound(chain.arc.begin(),chain.arc.end(),arc)-chain.arc.begin());
                            const size_t near=k==0?0:k>=chain.arc.size()?chain.arc.size()-1:(arc-chain.arc[k-1]<chain.arc[k]-arc?k-1:k);
                            const auto& t=(*targets)[i][near];
                            if(t.weight>0.&&cv::norm(t.axis)>0.)prediction=t.axis/cv::norm(t.axis);
                        }
                        samples[i].push_back({arc,samplePolylineArc(g,arc).point,n.normal,prediction});
                    }
                }
            },int(workers));
            writeFiberStripSamples(cfg.at("strip_samples_output").get<std::string>(),ids,samples);
        }
        if(cfg.contains("strip_targets_output")){
            // Diagnostic: every chain node's targets next to the solved normal. Doubles per
            // node: fiber index, arc, point (3), solved normal (3, zero if invalid), prediction
            // axis (3) and weight, crossing axis (3) and weight, signed axis (3) and weight,
            // fixed flag. The fiber ids go to <output>.json in index order.
            if(!targets)throw std::invalid_argument("strip_targets_output needs fused axis targets");
            const std::string output=cfg.at("strip_targets_output").get<std::string>();
            std::vector<std::vector<double>> rows(lines.size());
            cv::parallel_for_(cv::Range(0,int(lines.size())),[&](const cv::Range& range){
                for(int fi=range.start;fi<range.end;++fi){
                    const size_t i=size_t(fi);const auto& g=geometries[i];const auto& list=(*targets)[i];
                    const auto chain=vc::fiber_tracer::fiberStripChain(polylines[i],options);
                    if(chain.arc.size()!=list.size())continue;
                    for(size_t k=0;k<list.size();++k){
                        const double arc=std::min(chain.arc[k],g.length());
                        const auto n=stripNormalAt(*strips,i,g,arc);
                        const cv::Vec3d normal=n.valid?n.normal:cv::Vec3d(0,0,0),point=chain.points[k];
                        const auto& t=list[k];
                        rows[i].insert(rows[i].end(),{double(i),arc,point[0],point[1],point[2],normal[0],normal[1],normal[2],
                            t.axis[0],t.axis[1],t.axis[2],t.weight,t.crossing[0],t.crossing[1],t.crossing[2],t.crossingWeight,
                            t.signedAxis[0],t.signedAxis[1],t.signedAxis[2],t.signedWeight,t.fixed?1.:0.});
                    }
                }
            },int(workers));
            std::ofstream file(output,std::ios::binary);
            if(!file)throw std::runtime_error("cannot write "+output);
            for(const auto& r:rows)file.write(reinterpret_cast<const char*>(r.data()),std::streamsize(r.size()*sizeof(double)));
            if(!file)throw std::runtime_error("write failed "+output);
            std::ofstream(output+".json")<<Json{{"record_doubles",21},{"ids",ids}}.dump();
        }
        std::cout<<"strip normals "<<metadata["strip_normals"].dump()<<std::endl;
    }
    std::map<Cell,size_t> cells;
    for(const auto& p:partitions)for(const auto& piece:p.pieces)cells.try_emplace(piece.cell,0);
    std::vector<Job> jobs(cells.size());size_t ordinal=0;
    for(auto& [cell,id]:cells){id=ordinal++;jobs[id].cell=cell;}
    std::vector<size_t> offsets(lines.size()+1);
    for(size_t f=0;f<lines.size();++f)offsets[f+1]=offsets[f]+partitions[f].pieces.size();
    std::vector<Frame> frames(offsets.back());
    for(size_t f=0;f<lines.size();++f)for(size_t p=0;p<partitions[f].pieces.size();++p) {
        const auto id=offsets[f]+p,owner=cells.at(partitions[f].pieces[p].cell);
        frames[id].source=f;frames[id].ordinal=p;jobs[owner].ownedFrames.push_back(id);
    }
    std::vector<FiberTraceConstraintPiece> pieces(lines.size());
    for(size_t i=0;i<lines.size();++i){pieces[i].traceIndex=i;pieces[i].endArcBaseVoxels=geometries[i].length();}
    const double planning=elapsed(started);
    std::cout<<"chunk catalog jobs="<<jobs.size()<<" pieces="<<frames.size()<<" workers="<<std::min(workers,jobs.size())
        <<" core="<<size<<" seconds="<<planning<<std::endl;
    std::mutex logMutex;std::atomic<size_t> nextChunk{0};std::atomic<bool> failed{false};
    std::vector<const FiberConstraintNormalSamples*> workerNormals(workers,nullptr);
    std::atomic<size_t> activeChunks{0},peakChunks{0};
    std::vector<Json> statistics(jobs.size());
    double poolSeconds=0.,poolCpu=0.,fusionSeconds=0.;
    double geometrySeconds=0.;FiberTraceConstraintReport geometryReport;
    std::optional<Json> benchmarkResult;
    databaseOptions.batchNormalSamples=[&](size_t worker){return workerNormals.at(worker);};
    std::promise<void> framesDone;auto framesReady=framesDone.get_future().share();
    std::vector<size_t> fusionInvalidated(databaseOptions.workers);
    // Per-piece frame bookkeeping is TEMP-only and removed by the transfer API.
    databaseOptions.finalizeMeasurements=[&](size_t shard,vc::objects::InitialShardImport& writer) {
        framesReady.get();
        return writer.transferStagedExtensionRows(vc::objects::kFiberConstraintTable,[&](vc::objects::ExtensionRow& row) {
            if(row.size()!=9)throw std::runtime_error("Missing transient normal-frame attachment payload");
            const auto& payload=std::get<std::vector<uint8_t>>(row.back());
            std::array<uint64_t,4> transient;
            if(payload.size()!=sizeof(transient))throw std::runtime_error("Invalid normal-frame attachment payload");
            std::memcpy(transient.data(),payload.data(),sizeof(transient));
            for(size_t side=0;side<2;++side) {
                const auto source=transient[side];const auto arc=std::bit_cast<double>(transient[2+side]);
                if(source>=partitions.size())throw std::runtime_error("Invalid staged fiber index");
                const auto& partition=partitions[source];
                const auto end=std::upper_bound(partition.boundaries.begin(),partition.boundaries.end(),arc,[](double a,const auto& b){return a<b.arc;});
                if(end==partition.boundaries.begin()||!std::isfinite(arc)||arc>partition.boundaries.back().arc)
                    throw std::runtime_error("Staged normal arc outside source fiber");
                const auto piece=std::min(partition.pieces.size()-1,size_t(end-partition.boundaries.begin()-1));
                const auto sign=frames[offsets[source]+piece].sign;
                auto& normal=std::get<std::vector<uint8_t>>(row.at(3+side));
                if(normal.size()!=3)throw std::runtime_error("Invalid staged compact normal");
                if(!sign) {
                    fusionInvalidated[shard]+=normal[2]!=0;normal={0,0,0};
                }else if(sign<0&&normal[2])normal[2]=normal[2]==1?255:1;
            }
        });
    };
    const auto producer=[&](const FiberConstraintBatchConsumer& consume) {
      bool framesCompleted=false;
      try {
        FiberTraceConstraintStream stages;
        stages.geometryReady=[&](std::span<const FiberTraceConstraintPiece> sourcePieces,
            std::span<const cv::Vec3d> positions,const FiberTraceConstraintStream::Measure& measure) {
        // Only finalized encounter locations determine ownership. No geometry
        // is copied, re-searched or re-refined in these spatial jobs.
        for(size_t i=0;i<positions.size();++i) {
            Cell cell;for(int k=0;k<3;++k)cell[k]=int64_t(std::floor(positions[i][k]/size));
            auto [it,inserted]=cells.try_emplace(cell,jobs.size());
            if(inserted){jobs.emplace_back();jobs.back().cell=cell;}
            jobs[it->second].encounters.push_back(i);
        }
        statistics.resize(jobs.size());
        std::cout<<"sampling partition encounters="<<positions.size()<<" jobs="<<jobs.size()<<std::endl;
        if(benchmark&&jobs.size()<workers) {
            benchmarkResult=Json{{"valid",false},{"reason","fewer_jobs_than_workers"},
                {"worker_count",workers},{"total_jobs",jobs.size()},
                {"chunk_size_base_voxels",size},{"total_geometric_encounters",positions.size()}};
            throw ChunkBenchmarkComplete{};
        }
        std::vector<size_t> execution(jobs.size());std::iota(execution.begin(),execution.end(),0);
        std::sort(execution.begin(),execution.end(),[&](size_t a,size_t b) {
            const auto wa=jobs[a].encounters.size()+jobs[a].ownedFrames.size();
            const auto wb=jobs[b].encounters.size()+jobs[b].ownedFrames.size();
            return wa!=wb?wa>wb:a<b;
        });
        const auto poolStart=Clock::now();const auto cpuStart=std::clock();
        const auto poolMonotonic=std::chrono::duration<double>(poolStart.time_since_epoch()).count();
        if(benchmark&&cfg.contains("benchmark_pool_start_file"))
            vcr::zarr::atomicWrite(cfg.at("benchmark_pool_start_file").get<std::string>(),Json{{"monotonic_seconds",poolMonotonic}}.dump()+"\n");
        std::atomic<size_t> liveJobs{0},finishedJobs{0};
        std::atomic<size_t> completedWorkers{0};
        std::array<std::atomic<uint64_t>,3> throughput{};
        ChunkSaturationWindow window(workers,cfg.value("benchmark_warmup_seconds",1.),
            cfg.value("benchmark_window_seconds",28.),cfg.value("benchmark_require_worker_completion",false));
        const auto snapshot=[&] {
            ChunkSaturationWindow::Sample sample;sample.seconds=elapsed(poolStart);
            sample.cpuSeconds=double(std::clock()-cpuStart)/CLOCKS_PER_SEC;
            sample.active=liveJobs.load();sample.remaining=jobs.size()-finishedJobs.load();
            sample.completedWorkers=completedWorkers.load();
            for(size_t i=0;i<throughput.size();++i)sample.counts[i]=throughput[i].load();
            return sample;
        };
        std::optional<std::jthread> monitor;
        if(benchmark)monitor.emplace([&](std::stop_token stop) {
            // Also bound a workload that never manages to occupy the pool.
            while(!stop.stop_requested()&&!window.done()) {
                window.observe(snapshot());
                if(elapsed(poolStart)>cfg.value("benchmark_warmup_seconds",1.)+cfg.value("benchmark_window_seconds",28.)+2.) {
                    window.abortUnstarted(snapshot());
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            // Record the window BEFORE waiting for in-flight synchronous
            // samplers to drain. The owning benchmark harness terminates the
            // process after this atomic checkpoint and cleans its temporary
            // staging directory; no incomplete graph is ever published.
            if(window.done()&&cfg.contains("benchmark_cutoff_file")) {
                auto measured=window.summary();measured["total_jobs"]=jobs.size();
                measured["pool_start_monotonic_seconds"]=poolMonotonic;
                measured["chunk_size_base_voxels"]=size;
                measured["total_geometric_encounters"]=positions.size();
                measured["final_persistence_seconds"]=0.;
                Json checkpoint={{"benchmark_cutoff_ready",true},{"config",cfg},
                    {"saturated_chunk_benchmark",measured},
                    {"selected_fibers",ids.size()},{"points",metadata.at("points")},
                    {"query_seconds",metadata.at("query_seconds")},{"load_seconds",metadata.at("load_seconds")},
                    {"chunk_pipeline",{{"geometry_stage_seconds",geometrySeconds},{"catalog_seconds",planning}}}};
                if(memorySnapshot)checkpoint["memory_snapshot"]=memorySnapshot();
                if(const auto& stats=databaseOptions.packedOptions.memoryStats)
                    checkpoint["memory_snapshot"]["packed_vfs"]={{"resident_bytes",stats->residentBytes.load()},
                        {"peak_resident_bytes",stats->peakResidentBytes.load()}};
                checkpoint["memory_budgets"]={{"packed_vfs_bytes",databaseOptions.packedOptions.residentBytes},
                    {"sqlite_page_cache_bytes",databaseOptions.sqliteCacheBytes*databaseOptions.workers}};
                vcr::zarr::atomicWrite(cfg.at("benchmark_cutoff_file").get<std::string>(),checkpoint.dump(2)+"\n");
                std::cout<<"SATURATED_WINDOW_CUTOFF "<<measured.dump()<<std::endl;
            }
        });
        std::vector<std::future<void>> tasks;
        for(size_t worker=0;worker<std::min(workers,jobs.size());++worker)tasks.push_back(std::async(std::launch::async,[&,worker] {
          try {
            bool completedOnce=false;
            for(size_t ticket;!failed.load()&&(!benchmark||!window.done())&&(ticket=nextChunk.fetch_add(1))<execution.size();) {
                const auto id=execution[ticket];auto& job=jobs[id];
                if(benchmark)liveJobs.fetch_add(1);
                const auto active=activeChunks.fetch_add(1)+1;
                auto peak=peakChunks.load();while(peak<active&&!peakChunks.compare_exchange_weak(peak,active)){}
                const auto jobStarted=Clock::now();
                const auto cpuStarted=threadCpu();
                const auto samplingStart=Clock::now();
                for(size_t begin=0;begin<job.encounters.size();begin+=extraction.candidateBatchSize) {
                    if(benchmark&&window.done())break;
                    const auto count=std::min(extraction.candidateBatchSize,job.encounters.size()-begin);
                    auto measured=measure(std::span<const size_t>(job.encounters).subspan(begin,count),worker);
                    if(benchmark)throughput[0].fetch_add(measured.size());
                    job.events.insert(job.events.end(),std::make_move_iterator(measured.begin()),std::make_move_iterator(measured.end()));
                }
                if(benchmark&&window.done()){liveJobs.fetch_sub(1);activeChunks.fetch_sub(1);break;}
                job.metrics={{"chunk",id},{"worker",worker},{"cell",job.cell},
                    {"candidates",job.encounters.size()},{"refined",0},
                    {"events",job.events.size()},{"winding_seconds",elapsed(samplingStart)},
                    {"start_seconds",std::chrono::duration<double>(jobStarted-started).count()}};
                const auto windingCpu=profile?threadCpu():std::optional<double>{};
                if(windingCpu&&cpuStarted)job.metrics["winding_thread_cpu_seconds"]=*windingCpu-*cpuStarted;
                for(auto& c:job.events) {
                    c.pieceA=sourcePieces[c.pieceA].traceIndex;
                    c.pieceB=sourcePieces[c.pieceB].traceIndex;
                }
                std::vector<std::pair<size_t,Frame>> completed;
                sampleJob(job,job.ownedFrames,frames,offsets,geometries,partitions,normals,strips.get(),spacing,axis,completed);
                if(benchmark)throughput[1].fetch_add(job.metrics.at("normal_samples").get<uint64_t>());
                if(benchmark&&window.done()){liveJobs.fetch_sub(1);activeChunks.fetch_sub(1);break;}
                job.metrics["compute_seconds"]=elapsed(jobStarted);
                if(const auto cpu=threadCpu();cpu&&cpuStarted)job.metrics["chunk_thread_cpu_seconds"]=*cpu-*cpuStarted;
                if(profile&&windingCpu)if(const auto cpu=threadCpu())job.metrics["normal_thread_cpu_seconds"]=*cpu-*windingCpu;
                activeChunks.fetch_sub(1);
                {std::lock_guard lock(logMutex);std::cout<<"CHUNK_COMPUTED "<<job.metrics.dump()<<std::endl;}
                for(auto& [f,value]:completed) {
                    frames[f].ends=value.ends;frames[f].outwardSum=value.outwardSum;
                    frames[f].outwardWeight=value.outwardWeight;frames[f].arrived=true;
                }
                // Import immediately in the deterministic local-piece gauge.
                // No job waits for another job's geometry, reads or signs.
                const auto writing=Clock::now();workerNormals[worker]=&job.normals;
                const auto outputCpu=profile?threadCpu():std::optional<double>{};
                if(id>0x7fffffffULL || job.events.size()>0xffffffffULL)throw std::overflow_error("Chunk event identifier exceeds int64");
                consume(worker,id*0x100000000ULL,job.events);workerNormals[worker]=nullptr;
                if(benchmark) {
                    throughput[2].fetch_add(job.events.size());liveJobs.fetch_sub(1);
                    const auto finished=finishedJobs.fetch_add(1)+1;
                    if(!completedOnce){completedOnce=true;completedWorkers.fetch_add(1);}
                    if(jobs.size()-finished<workers)window.observe(snapshot());
                }
                job.metrics["output_seconds"]=elapsed(writing);
                if(outputCpu)if(const auto cpu=threadCpu())job.metrics["output_thread_cpu_seconds"]=*cpu-*outputCpu;
                {std::lock_guard lock(logMutex);std::cout<<"CHUNK "<<job.metrics.dump()<<std::endl;}
                statistics[id]=std::move(job.metrics);
                std::vector<FiberTraceConstraint>().swap(job.events);job.normals={};
                std::vector<size_t>().swap(job.encounters);
            }
          } catch(...) {failed.store(true);throw;}
        }));
        std::exception_ptr error;
        for(auto& task:tasks)try{task.get();}catch(...){if(!error)error=std::current_exception();}
        if(error)std::rethrow_exception(error);
        if(benchmark) {
            monitor->request_stop();monitor->join();
            if(!window.done())window.observe(snapshot());
            benchmarkResult=window.summary();
            (*benchmarkResult)["total_jobs"]=jobs.size();(*benchmarkResult)["chunk_size_base_voxels"]=size;
            (*benchmarkResult)["total_geometric_encounters"]=positions.size();
            (*benchmarkResult)["completed_jobs_including_drain"]=finishedJobs.load();
            (*benchmarkResult)["pool_seconds_including_drain"]=elapsed(poolStart);
            (*benchmarkResult)["chunks"]=statistics;
            std::cout<<"SATURATED_CHUNK_RESULT "<<benchmarkResult->dump()<<std::endl;
            // Never publish an incomplete graph or perform final persistence.
            throw ChunkBenchmarkComplete{};
        }
        const auto fusion=Clock::now();
        for(size_t source=0;source<lines.size();++source)for(size_t f=offsets[source];f<offsets[source+1];++f) {
            auto& frame=frames[f];if(!frame.arrived)throw std::logic_error("Missing completed normal piece");
            if(f==offsets[source])frame.sign=1;
            else {
                const auto& previous=frames[f-1];
                const std::array<FiberNormalPieceFrame,2> pair{previous.ends,frame.ends};
                frame.sign=static_cast<signed char>(previous.sign*reconcileFiberNormalPieceFrames(pair)[1]);
            }
        }
        fusionSeconds=elapsed(fusion);
        std::cout<<"normal sign fusion pieces="<<frames.size()<<" seconds="<<fusionSeconds<<std::endl;
        framesDone.set_value();
        framesCompleted=true;
        poolSeconds=elapsed(poolStart);const auto cpuEnd=std::clock();
        if(cpuStart!=std::clock_t(-1)&&cpuEnd!=std::clock_t(-1))poolCpu=double(cpuEnd-cpuStart)/CLOCKS_PER_SEC;
        };
        const auto geometryStart=Clock::now();
        geometryReport=extractFiberTraceConstraints(lines,extraction,
            [&](const auto& a,const auto& b,double step){return windings.normalAlignedWindingDistance(a,b,step);},
            [&](const auto& pairs,double step,int inner){return windings.normalAlignedWindingDistancesBatch(pairs,step,inner);},
            {},nullptr,[&](const auto& progress) {
                if(profile && (progress.completed==0 || progress.completed==progress.total)) {
                    std::lock_guard lock(logMutex);
                    std::cout<<"GEOMETRY_PROFILE "<<Json{{"phase",progress.phase},
                        {"completed",progress.completed},{"total",progress.total},
                        {"seconds",elapsed(geometryStart)},{"process_cpu_seconds",double(std::clock())/CLOCKS_PER_SEC},
                        {"monotonic_seconds",std::chrono::duration<double>(Clock::now().time_since_epoch()).count()}}.dump()<<std::endl;
                }
                if(progress.phase=="geometry_complete") {
                    geometrySeconds=elapsed(geometryStart);
                    std::cout<<"global geometry seconds="<<geometrySeconds<<" candidates="<<progress.candidates<<std::endl;
                }
            },stages);
      } catch(...) {if(!framesCompleted)framesDone.set_exception(std::current_exception());throw;}
    };
    const auto processing=Clock::now();
    Json exported;
    try {exported=streamFiberConstraintDatabase(cfg.at("database").get<std::string>(),ids,pieces,{},
        normals,1,metadata,{},databaseOptions,0,producer,workers);}
    catch(...) {
        if(!benchmarkResult)throw;
        (*benchmarkResult)["partial_graph_discarded"]=true;
        (*benchmarkResult)["final_persistence_seconds"]=0.;
        return {{"benchmark",*benchmarkResult},{"geometry_stage_seconds",geometrySeconds},
            {"catalog_seconds",planning},{"seconds_including_cleanup",elapsed(started)}};
    }
    size_t newlyInvalidated=0;for(const auto count:fusionInvalidated)newlyInvalidated+=count;
    exported["invalid_endpoint_normals_before_piece_fusion"]=exported.at("invalid_endpoint_normals");
    exported["piece_fusion_invalidated_endpoint_normals"]=newlyInvalidated;
    exported["invalid_endpoint_normals"]=exported.at("invalid_endpoint_normals").get<size_t>()+newlyInvalidated;
    FiberConstraintNormalSamples priors;priors.outwardAgreement.resize(lines.size());priors.validArcWeight.resize(lines.size());
    size_t unresolved=0;
    for(size_t f=0;f<lines.size();++f) {
        double sum=0.,weight=0.;
        for(size_t p=offsets[f];p<offsets[f+1];++p) {
            const auto& frame=frames[p];unresolved+=frame.sign==0;
            if(frame.sign){sum+=frame.sign*frame.outwardSum;weight+=frame.outwardWeight;}
        }
        priors.outwardAgreement[f]=weight>0?sum/weight:0.;priors.validArcWeight[f]=weight;
    }
    priors.statistics={{"sample_spacing_base",spacing}};
    writeFiberConstraintOutwardPriors(cfg.at("normal_prior_output").get<std::string>(),ids,priors,exported.at("source_store"),metadata);
    // Global geometry counts must not be confused with sums of halo searches.
    return {{"export",exported},{"chunks",statistics},{"chunk_count",jobs.size()},{"chunk_size_base_voxels",size},
        {"chunk_workers",std::min(workers,jobs.size())},{"peak_active_chunks",peakChunks.load()},{"normal_pieces",frames.size()},
        {"pool_seconds",poolSeconds},{"pool_process_cpu_seconds",poolCpu},
        {"sign_fusion_seconds",fusionSeconds},
        {"geometry_stage_seconds",geometrySeconds},{"initial_candidates",geometryReport.measuredCandidates},
        {"refined_candidates",geometryReport.localRefinedCandidates},
        {"geometry_accepted_encounters",geometryReport.localCertifiedEvents},
        {"measured_encounters",geometryReport.localRetainedCertifiedEvents},
        {"pool_process_effective_cores",poolSeconds>0?poolCpu/poolSeconds:0.},
        {"unresolved_pieces",unresolved},{"catalog_seconds",planning},
        {"processing_including_persistence_seconds",elapsed(processing)},{"seconds",elapsed(started)},
        {"scheduling","global parallel geometry once; halo-free spatial sampling/import jobs; final piece sign fusion"}};
}
