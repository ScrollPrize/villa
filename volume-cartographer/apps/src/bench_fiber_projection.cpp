// Geometry-only diagnostic, not an alternative extraction/solve workflow.
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include <vc_object_store/objects/AfvSource.hpp>
#include <vc_object_store/geometry/ArrayBytes.hpp>
#include <vc_object_store/geometry/SegmentClip.hpp>
#include <vc_reader/zarr_write.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc,char** argv) {
    try {
        using namespace vc::fiber_tracer;
        using Clock=std::chrono::steady_clock;
        using Json=nlohmann::json;
        if(argc!=2)throw std::invalid_argument("usage: bench_fiber_projection CONFIG.json");
        Json cfg;std::ifstream input(argv[1]);input>>cfg;
        std::optional<std::array<double,6>> box;
        if(cfg.contains("bounds")) {
            box=cfg.at("bounds").get<std::array<double,6>>();
            for(int axis=0;axis<3;++axis)
                if(!std::isfinite((*box)[axis])||!std::isfinite((*box)[axis+3])||!((*box)[axis]<(*box)[axis+3]))
                    throw std::invalid_argument("Invalid spatial bounds");
        }
        const auto start=Clock::now();
        const auto seconds=[&](auto since){return std::chrono::duration<double>(Clock::now()-since).count();};
        vc::objects::AfvSource source(cfg.at("source").get<std::string>());
        std::vector<FiberletCropTraceLine> lines;std::size_t points=0,readFibers=0;
        vc::objects::AfvFiber fiber;
        while(source.next(fiber)) {
            ++readFibers;
            if(box&&fiber.detail.bbox) {
                bool intersects=true;
                for(int axis=0;axis<3;++axis)
                    intersects&=(*fiber.detail.bbox)[axis]<=(*box)[axis+3]&&(*fiber.detail.bbox)[axis+3]>=(*box)[axis];
                if(!intersects)continue;
            }
            const auto xyz=vc::objects::xyzFromBytes<double>(fiber.xyz);
            bool intersects=!box;
            for(std::size_t i=1;box&&i<xyz.size()&&!intersects;++i) {
                double first,last;intersects=vc::geometry::clipSegment(xyz[i-1],xyz[i],*box,first,last);
            }
            if(!intersects)continue;
            auto& line=lines.emplace_back();line.pointsBaseXYZ.reserve(xyz.size());
            for(const auto& p:xyz)line.pointsBaseXYZ.emplace_back(p[0],p[1],p[2]);
            points+=xyz.size();
        }
        if(lines.empty())throw std::runtime_error("No fibers intersect the requested region");
        const double load=seconds(start);
        FiberTraceConstraintConfig options;
        options.localEvents=true;options.localKeepWholeFibers=true;
        options.parallelThreads=cfg.value("threads",32u);
        if(options.parallelThreads<1)throw std::invalid_argument("Positive threads required");
        options.certifyWinding=false;options.enforceMaximumWindingDistance=false;
        options.closestParallelEstimator=false;options.localParallelReciprocal=false;
        options.localMovingProjectionWindows=true;options.localPerpendicularProjectionIterations=16;
        const auto solving=Clock::now();
        // Projection occurs BEFORE winding sampling in the production scorer.
        // Constant valid evidence enables refinement without any volume I/O.
        // These outputs are NOT physical winding measurements or a solve.
        const auto report=extractFiberTraceConstraints(lines,options,
            [](const auto&,const auto&,double){return .5;},
            [](const auto& pairs,double,int){return std::vector<double>(pairs.size(),.5);});
        Json result={{"mode","geometry-only projection diagnostic"},{"source_fibers",readFibers},
            {"selected_fibers",lines.size()},{"selected_points",points},
            {"curve_selection","whole original fibers intersecting bounds; curves are NOT clipped"},
            {"normal_reads",0},{"refinement_evidence","constant valid placeholder; not measured winding evidence"},
            {"threads",options.parallelThreads},{"candidates",report.measuredCandidates},
            {"refined_candidates",report.localRefinedCandidates},{"geometry_events",report.constraints.size()},
            {"load_seconds",load},{"extraction_seconds",seconds(solving)},{"total_seconds",seconds(start)},
            {"config",cfg}};
        const auto path=std::filesystem::path(cfg.at("output").get<std::string>());
        std::filesystem::create_directories(path.parent_path());
        vcr::zarr::atomicWrite(path,result.dump(2)+"\n");std::cout<<result.dump()<<std::endl;
        return 0;
    }catch(const std::exception& error){std::cerr<<"bench_fiber_projection: "<<error.what()<<'\n';return 1;}
}
