#include <QtTest/QtTest>
#include <QTemporaryDir>
#include <QProcess>
#include "vc/constraints/ConstraintGraph.hpp"
#include "vc/fiber_tracer/FiberGeometry.hpp"
#include "vc/lasagna/LineModel.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "../ConstraintGraphView.hpp"
#include "../../../core/src/constraints/GraphFiberJunctions.hpp"
#include <fstream>
using namespace vc::constraints;
namespace {
struct Normals : vc::lasagna::NormalSampler {
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&)const override{return {{0,0,1},true,{}};}
};
std::shared_ptr<Node> plane(std::string id,double offset=0) {
    auto n=std::make_shared<Node>();n->id=id;n->rows=5;n->cols=5;
    for(int y=0;y<5;++y)for(int x=0;x<5;++x){n->xyz.push_back({100+x*20.0,100+y*20.0,100});n->initialUv.push_back({x*20.0+offset,y*20.0});}
    for(int y=0;y<4;++y)for(int x=0;x<4;++x){int a=y*5+x;n->triangles.push_back({a,a+1,a+5});n->triangles.push_back({a+6,a+5,a+1});}
    return n;
}
Graph base(){Graph g;auto n=plane("a");g.seed="a";g.nodes.emplace("a",Placement{n,n->initialUv});return g;}
}
class ConstraintGraphTest:public QObject {
    Q_OBJECT
private slots:
    void horizontalAnchorsLinkedVerticalGeometry(){
        Catalog catalog;catalog.components={{"h","v"}};
        Fiber h;h.id="h";h.direction='H';h.controls={0,10,20};
        Fiber v;v.id="v";v.direction='V';v.controls={0,10,20};
        for(int i=0;i<=20;++i){
            h.line.push_back({100.0+i*20,300,100});
            v.line.push_back({304,100.0+i*20,112});
        }
        Normals normals;Settings settings;
        catalog.fibers={{"h",h},{"v",v}};catalog.links={{"h","v",1,1}};
        Graph graph=seedGraph(catalog,0,settings,normals);
        auto unlinked=std::make_shared<Node>(*graph.nodes.at("v").source);unlinked->id="unlinked";
        graph.nodes.emplace("unlinked",Placement{unlinked,unlinked->initialUv});
        Link link{fiberLineLocation(*graph.nodes.at("h").source,10),fiberLineLocation(*graph.nodes.at("v").source,10)};
        graph.links={link};
        const auto original=graphToJson(graph);
        const detail::FiberJunctions junctions(graph);
        const auto& vertical=*graph.nodes.at("v").source;
        const auto& horizontal=*graph.nodes.at("h").source;
        const auto target=sourceLocation(horizontal,link.a.grid);
        QVERIFY(cv::norm(sourceLocation(vertical,link.b.grid)+junctions.offset(vertical,link.b.grid[0])-target)<1e-8);
        QCOMPARE(junctions.offset(vertical,link.b.grid[0]-4),cv::Vec3d(0,0,0));
        QCOMPARE(junctions.offset(vertical,link.b.grid[0]+4),cv::Vec3d(0,0,0));
        QVERIFY(cv::norm(junctions.offset(vertical,link.b.grid[0]+2)-cv::Vec3d(-2,0,-6))<1e-8);
        QVERIFY(junctions.points(horizontal)==horizontal.xyz);
        QVERIFY(junctions.points(*unlinked)==unlinked->xyz);
        QCOMPARE(graphToJson(graph).dump(),original.dump());
        std::swap(graph.links[0].a,graph.links[0].b);
        QVERIFY(detail::FiberJunctions(graph).points(vertical)==junctions.points(vertical));
        // A fractional control gets the full horizontal position, independent
        // of the coarse ribbon columns used by the chart metric.
        graph.links[0].a.grid[0]+=.3;
        const detail::FiberJunctions fractional(graph);
        const auto& vl=graph.links[0].a;const auto& hl=graph.links[0].b;
        QVERIFY(cv::norm(sourceLocation(vertical,vl.grid)+fractional.offset(vertical,vl.grid[0])-sourceLocation(horizontal,hl.grid))<1e-8);
        graph.nodes.erase("unlinked");
        const auto fitted=fit(graph);QVERIFY2(fitted.valid,fitted.message.c_str());
        QVERIFY(fitted.correspondenceRms<.01);
        QVERIFY(fitted.graph.nodes.at("v").source==graph.nodes.at("v").source);
        const auto sample=fitted.surface->sampleAtSurface(placedLocation(fitted.graph,hl));
        QVERIFY(sample.valid());
        QVERIFY(cv::norm(cv::Vec3d(sample.volume)-sourceLocation(horizontal,hl.grid))<.02);
        QTemporaryDir dir;const auto path=std::filesystem::path(dir.path().toStdString())/"graph.json";
        saveSession(fitted,path);const auto restored=loadSession(path);
        QVERIFY2(restored.valid,restored.message.c_str());
        QCOMPARE(cv::norm(*restored.surface->rawPointsPtr(),*fitted.surface->rawPointsPtr(),cv::NORM_INF),0.0);
    }
    void verticalJunctionsBlendNearbyLinks(){
        auto h=plane("h"),v=plane("v");h->fiber=v->fiber=true;h->direction='H';v->direction='V';
        for(auto& xyz:v->xyz)xyz[2]+=10;
        Graph graph;graph.seed="h";graph.nodes={{"h",{h,h->initialUv}},{"v",{v,v->initialUv}}};
        graph.links={{{"h",{1,2}},{"v",{1,2}}},{{"h",{3,2}},{"v",{3,2}}}};
        const detail::FiberJunctions junctions(graph);
        // Equal neighboring corrections stay constant between links; they
        // must not add together or fade to the back between close anchors.
        QCOMPARE(junctions.offset(*v,2),cv::Vec3d(0,0,-10));
        const auto before=graphToJson(graph);
        auto fitted=fit(graph);QVERIFY2(fitted.valid,fitted.message.c_str());
        QVERIFY(fitted.correspondenceRms<.01);
        const auto& grid=*fitted.surface->rawPointsPtr();
        const auto uv=placedLocation(fitted.graph,{"h",{2,2}});
        const auto at=(uv-fitted.origin)/graph.settings.spacing;
        const auto xyz=grid(int(std::lround(at[1])),int(std::lround(at[0])));
        QVERIFY(std::abs(xyz[2]-100)<.2);
        QCOMPARE(graphToJson(graph).dump(),before.dump());
        graph.links.push_back({{"h",{0,2}},{"v",{1,2}}});
        QVERIFY_EXCEPTION_THROWN((void)detail::FiberJunctions(graph),std::runtime_error);
    }
    void incrementalVerticalJunctionKeepsDistantSurface(){
        Catalog catalog;catalog.components={{"h","v"}};
        Fiber h;h.id="h";h.direction='H';h.controls={0,10,20};
        Fiber v;v.id="v";v.direction='V';v.controls={0,100,200};
        for(int i=0;i<=20;++i)h.line.push_back({100.0+i*20,2100,100});
        for(int i=0;i<=200;++i)v.line.push_back({300,100.0+i*20,112});
        catalog.fibers={{"h",h},{"v",v}};catalog.links={{"h","v",1,1}};
        Normals normals;const auto committed=fit(seedGraph(catalog,0,Settings{},normals));
        QVERIFY2(committed.valid,committed.message.c_str());
        auto draft=committed.graph;
        draft.links[0].a=fiberLineLocation(*draft.nodes.at("h").source,10);
        draft.links[0].a.grid[0]+=.1;
        const auto fitted=fitIncremental(draft,committed);
        QVERIFY2(fitted.valid,fitted.message.c_str());QVERIFY(fitted.localChartVertices>0);
        const auto center=placedLocation(committed.graph,committed.graph.links[0].b);
        const auto& old=*committed.surface->rawPointsPtr();const auto& grid=*fitted.surface->rawPointsPtr();
        const auto shift=(committed.origin-fitted.origin)/draft.settings.spacing;
        size_t retained=0;
        for(int y=0;y<old.rows;++y)for(int x=0;x<old.cols;++x) {
            if(old(y,x)[0]==-1)continue;
            const auto uv=committed.origin+cv::Vec2d(x,y)*draft.settings.spacing;
            if(cv::norm(uv-center)<400)continue;
            QCOMPARE(grid(y+int(std::lround(shift[1])),x+int(std::lround(shift[0]))),old(y,x));++retained;
        }
        QVERIFY(retained>100);
    }
    void patchFitsAllLocalFibers(){
        QTemporaryDir dir;auto path=std::filesystem::path(dir.path().toStdString())/"patch";
        cv::Mat_<cv::Vec3f> grid(5,5);
        for(int y=0;y<5;++y)for(int x=0;x<5;++x)grid(y,x)={100+x*20.0f,100+y*20.0f,100};
        QuadSurface(grid,{.05f,.05f}).save(path,"patch");
        Catalog catalog;catalog.patches.push_back({"patch",path,{100,100,100},{180,180,100}});
        Graph graph;graph.seed="first";Normals normals;
        auto add=[&](std::string id,double y,double z,double chartOffset){
            Fiber f;f.id=id;f.direction='H';f.controls={0,80};
            for(int i=0;i<=80;++i)f.line.push_back({100.0+i,y,z});
            auto n=makeFiberNode(f,graph.settings,normals);auto uv=n->initialUv;
            for(auto& p:uv)p[1]+=chartOffset;
            graph.nodes.emplace(id,Placement{n,uv});
        };
        add("first",120,100,0);add("second",160,100,52);
        add("outside",160,105,40);add("other_traversal",160,100,4000);
        Candidate candidate;candidate.id="patch";
        Location anchor{"first"};anchor.lineIndex=40;
        candidate.contacts.push_back({anchor,{"patch",{2,1}}});
        auto proposal=propose(catalog,graph,candidate,false,normals);
        std::map<std::string,size_t> counts;std::set<std::pair<std::string,int64_t>> unique;
        for(const auto& link:proposal.links)if(link.b.node=="patch"){
            ++counts[link.a.node];QVERIFY(unique.emplace(link.a.node,link.a.lineIndex).second);
        }
        QCOMPARE(counts["first"],size_t(81));QCOMPARE(counts["second"],size_t(81));
        QCOMPARE(counts["outside"],size_t(0));QCOMPARE(counts["other_traversal"],size_t(0));
        auto manualCandidate=candidate;
        manualCandidate.contacts.push_back({{"second",{2,3}},{"patch",{2,3}},true});
        auto manual=propose(catalog,graph,manualCandidate,false,normals);
        QCOMPARE(std::count_if(manual.links.begin(),manual.links.end(),[](const auto& link){
            return link.a.node=="second"&&link.b.node=="patch";
        }),1);
        // Fit just the local connected geometry; unrelated nodes are tested
        // above for exclusion, not made part of this connected fit fixture.
        proposal.nodes.erase("outside");proposal.nodes.erase("other_traversal");
        const auto fitted=fit(proposal);QVERIFY2(fitted.valid,fitted.message.c_str());
        for(const auto& link:fitted.graph.links)
            QVERIFY(cv::norm(placedLocation(fitted.graph,link.a)-placedLocation(fitted.graph,link.b))<.2);
        const auto original=graphToJson(fitted.graph).dump();
        auto fibers=renderSourceLayer(fitted.graph,true),patches=renderSourceLayer(fitted.graph,false);
        QVERIFY2(fibers.valid,fibers.message.c_str());QVERIFY2(patches.valid,patches.message.c_str());
        QCOMPARE(fibers.graph.nodes.size(),size_t(2));QCOMPARE(patches.graph.nodes.size(),size_t(1));
        QCOMPARE(graphToJson(fitted.graph).dump(),original);
        for(const auto& [id,p]:fibers.graph.nodes)QVERIFY(p.uv==fitted.graph.nodes.at(id).uv);
        QVERIFY(patches.graph.nodes.at("patch").uv==fitted.graph.nodes.at("patch").uv);
        std::atomic_bool cancel{true};
        QVERIFY_EXCEPTION_THROWN(propose(catalog,graph,candidate,false,normals,&cancel),std::runtime_error);
    }
    void localAdditionPreservesDistantGeometry(){
        Fiber fiber;fiber.id="long";fiber.direction='H';fiber.controls={0,200};
        for(int i=0;i<=200;++i)fiber.line.push_back({100+i*20.0,100,100});
        Normals normals;Graph graph;auto source=makeFiberNode(fiber,graph.settings,normals);
        graph.seed=source->id;graph.nodes.emplace(source->id,Placement{source,source->initialUv});
        const auto committed=fit(graph);QVERIFY2(committed.valid,committed.message.c_str());
        auto draft=committed.graph;auto patch=plane("patch",2000);
        for(auto& xyz:patch->xyz)xyz+=cv::Vec3d{2000,-40,0};
        for(auto& uv:patch->initialUv)uv[1]-=40;
        draft.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        for(int i:{0,2,4})draft.links.push_back({fiberLineLocation(*source,100+i),{"patch",{double(i),2}}});
        auto local=fitIncremental(draft,committed);QVERIFY2(local.valid,local.message.c_str());
        QVERIFY(local.localChartVertices>0);QVERIFY(local.localChartVertices<source->xyz.size()/2);
        QVERIFY(local.localSurfaceSamples>0);QVERIFY(local.correspondenceRms<.1);QVERIFY(local.stretchP95<1.02);
        const auto& before=committed.graph.nodes.at("long").uv;
        const auto& after=local.graph.nodes.at("long").uv;
        for(size_t i=0;i<before.size();++i)if(before[i][0]<1700||before[i][0]>2400)QVERIFY(before[i]==after[i]);
        const auto& oldGrid=*committed.surface->rawPointsPtr();const auto& newGrid=*local.surface->rawPointsPtr();
        const auto offset=(committed.origin-local.origin)/graph.settings.spacing;
        size_t retained=0;
        for(int y=0;y<oldGrid.rows;++y)for(int x=0;x<oldGrid.cols;++x) {
            double u=committed.origin[0]+x*graph.settings.spacing;
            if(u>1700&&u<2400)continue;
            QVERIFY(oldGrid(y,x)==newGrid(y+int(std::lround(offset[1])),x+int(std::lround(offset[0]))));++retained;
        }
        QVERIFY(retained>100);
        const auto global=fit(draft);QVERIFY(global.valid);QVERIFY(std::abs(local.correspondenceRms-global.correspondenceRms)<.1);
        QTemporaryDir dir;auto path=std::filesystem::path(dir.path().toStdString())/"local.json";
        saveSession(local,path);const auto restored=loadSession(path);QVERIFY(restored.valid);
        QCOMPARE(cv::norm(*local.surface->rawPointsPtr(),*restored.surface->rawPointsPtr(),cv::NORM_INF),0.0);
        // A second nearby patch is optimized with the overlapping accepted patch.
        auto overlap=local.graph;auto second=plane("second",2040);
        for(auto& xyz:second->xyz)xyz+=cv::Vec3d{2040,-40,0};
        for(auto& uv:second->initialUv)uv[1]-=40;
        overlap.nodes.emplace(second->id,Placement{second,second->initialUv});
        overlap.links.push_back({{"patch",{2.1,2}},{"second",{0,2}},true});
        overlap.links.push_back({{"patch",{4,2}},{"second",{2,2}}});
        auto joint=fitIncremental(overlap,local);QVERIFY2(joint.valid,joint.message.c_str());
        QVERIFY(joint.localChartVertices>0);QVERIFY(joint.correspondenceRms<graph.settings.radius);
        QVERIFY(cv::norm(placedLocation(joint.graph,{"patch",{2.1,2}})-placedLocation(joint.graph,{"second",{0,2}}))<.02);
        QVERIFY(joint.graph.nodes.at("patch").uv!=local.graph.nodes.at("patch").uv);
        auto strained=draft;
        strained.links[1].a=fiberLineLocation(*source,103);
        strained.links[2].a=fiberLineLocation(*source,106);
        std::vector<std::string> progress;
        auto expanded=fitIncremental(strained,committed,nullptr,[&](const auto& message){progress.push_back(message);});
        QVERIFY2(expanded.valid,expanded.message.c_str());
        QVERIFY(expanded.localExpansions>0);QVERIFY(expanded.localChartVertices>0);
        // Corrupting the saved origin invalidates the bound grid checksum.
        std::ifstream input(path);auto document=nlohmann::json::parse(input);input.close();
        document["origin"][0]=document["origin"][0].get<double>()+20;
        std::ofstream(path)<<document;
        const auto recovered=loadSession(path);QVERIFY(recovered.valid);
        const auto rebuilt=render(local.graph);
        QVERIFY(recovered.origin==rebuilt.origin);
        QCOMPARE(cv::norm(*recovered.surface->rawPointsPtr(),*rebuilt.surface->rawPointsPtr(),cv::NORM_INF),0.0);
        // Global setting changes explicitly use the global path.
        auto changed=draft;changed.settings.stiffness=2;
        auto full=fitIncremental(changed,committed);QVERIFY(full.valid);QCOMPARE(full.localChartVertices,size_t(0));
        std::atomic_bool cancel{true};QVERIFY_EXCEPTION_THROWN(fitIncremental(draft,committed,&cancel),std::runtime_error);
    }
    void initialPlacementPreservesChartsAndFinalFit(){
        auto graph=base();const auto original=graphToJson(graph);
        auto placement=renderPlacement(graph);QVERIFY2(placement.valid,placement.message.c_str());
        QCOMPARE(graphToJson(placement.graph).dump(),original.dump());
        auto final=fit(graph),afterPreview=fit(placement.graph);
        QVERIFY(final.valid);QVERIFY(afterPreview.valid);
        QCOMPARE(graphToJson(final.graph).dump(),graphToJson(afterPreview.graph).dump());
        QCOMPARE(cv::norm(*final.surface->rawPointsPtr(),*afterPreview.surface->rawPointsPtr(),cv::NORM_INF),0.0);
        auto patch=plane("b",100);graph.nodes.emplace("b",Placement{patch,patch->initialUv});
        auto source=renderProposalSources(graph,base(),"b");QVERIFY(source.valid);
        QCOMPARE(source.graph.nodes.size(),size_t(1));QVERIFY(source.graph.nodes.contains("b"));
        std::atomic_bool cancel{true};QVERIFY_EXCEPTION_THROWN(renderPlacement(graph,&cancel),std::runtime_error);
    }
    void surfaceExpansionKeepsSettledChart(){
        Fiber fiber;fiber.id="long";fiber.direction='H';fiber.controls={0,200};
        for(int i=0;i<=200;++i)fiber.line.push_back({100+i*20.0,100,100});
        Normals normals;Graph graph;auto source=makeFiberNode(fiber,graph.settings,normals);
        graph.seed=source->id;graph.nodes.emplace(source->id,Placement{source,source->initialUv});
        auto committed=fit(graph);QVERIFY2(committed.valid,committed.message.c_str());
        // An accepted XYZ grid can carry earlier surface relaxation beyond
        // the next edit's chart footprint. Its boundary must settle too.
        auto& grid=*committed.surface->rawPointsPtr();
        for(int y=0;y<grid.rows;++y)for(int x=0;x<grid.cols;++x) {
            const double u=committed.origin[0]+x*graph.settings.spacing;
            if(grid(y,x)[0]!=-1&&u>=1740&&u<=2340)grid(y,x)[2]+=2;
        }
        auto draft=committed.graph;auto patch=plane("patch",2000);
        for(auto& xyz:patch->xyz)xyz+=cv::Vec3d{2000,-40,0};
        for(auto& uv:patch->initialUv)uv[1]-=40;
        draft.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        draft.links.push_back({fiberLineLocation(*source,102),{"patch",{2,2}}});
        size_t charts=0,surfaces=0;
        const auto result=fitIncremental(draft,committed,nullptr,[&](const std::string& message){
            charts+=message.starts_with("Fitting local charts:");
            surfaces+=message.starts_with("Updating local tifxyz region:");
        });
        QVERIFY2(result.valid,result.message.c_str());QVERIFY(result.localChartVertices>0);
        QCOMPARE(charts,size_t(1));QVERIFY(surfaces>1);
        QVERIFY(result.correspondenceRms<.01);QVERIFY(result.stretchP95<1.01);
        const auto shift=(committed.origin-result.origin)/graph.settings.spacing;
        const auto& after=*result.surface->rawPointsPtr();
        for(int y=0;y<grid.rows;++y)for(int x=0;x<grid.cols;++x) {
            const double u=committed.origin[0]+x*graph.settings.spacing;
            if(u<1400||u>2700)QCOMPARE(after(y+int(std::lround(shift[1])),x+int(std::lround(shift[0]))),grid(y,x));
        }
    }
    void initialViewUsesSeedAtInspectionScale(){
        auto g=base();
        auto far=plane("far",1000000);g.nodes.emplace("far",Placement{far,far->initialUv});
        g.links={{{"a",{1,1}},{"far",{1,1}},true},{{"a",{2,2}},{"far",{2,2}},true}};
        auto view=vc3d::constraint_graph::initialView(g);
        QCOMPARE(view.scale,.5);QVERIFY(cv::norm(view.center-cv::Vec2d{40,40})<1e-8);
        g.nodes.erase("far");g.links.clear();
        auto small=vc3d::constraint_graph::initialView(g);
        QCOMPARE(small.scale,view.scale);QVERIFY(cv::norm(small.center-view.center)<1e-8);
        g.settings.spacing=10;QCOMPARE(vc3d::constraint_graph::initialView(g).scale,1.0);
    }
    void windingCycleAudit(){
        const std::vector<cv::Vec3d> centers{{0,0,-100},{0,0,100}};
        Catalog c;
        auto fiber=[](const std::string& id,double sweep){
            Fiber f;f.id=id;f.fullyTraced=true;f.controls={0,100};
            for(int i=0;i<=100;++i){double a=.2+sweep*i/100;f.line.push_back({100*std::cos(a),100*std::sin(a),i*.1});}
            return f;
        };
        c.fibers.emplace("a",fiber("a",.2));c.fibers.emplace("b",fiber("b",.1));
        c.links={{"a","b",0,0},{"a","b",1,1}};
        auto audit=auditFiberCycles(c,centers,1);
        QVERIFY(audit.available);QCOMPARE(audit.cycles,size_t(1));QCOMPARE(audit.evaluatedCycles,size_t(1));QCOMPARE(audit.checkedCycles,size_t(1));QVERIFY(audit.issues.empty());
        QVERIFY(audit.summary.find("no inconsistencies found")!=std::string::npos);
        c.fibers.at("a").fullyTraced=false;
        audit=auditFiberCycles(c,centers,1);
        QCOMPARE(audit.uncertainCycles,size_t(1));QVERIFY(audit.issues.empty());
        QVERIFY(audit.summary.find("no inconsistencies found")!=std::string::npos);
        // The endpoints differ in XYZ, yet the cycle is winding-consistent.
        QVERIFY(cv::norm(c.fibers.at("a").line.back()-c.fibers.at("b").line.back())>4);
        c.fibers.at("a")=fiber("a",2*std::acos(-1.0));
        audit=auditFiberCycles(c,centers,1);
        QCOMPARE(audit.issues.size(),size_t(1));QCOMPARE(std::abs(audit.issues[0].turns),1LL);
        QVERIFY(audit.summary.find("nonzero winding closures found")!=std::string::npos);
        QVERIFY(audit.issues[0].reliable);QCOMPARE(audit.issues[0].links.size(),size_t(2));
        // Reverse traversal and the stored link direction: same diagnosis.
        auto& a=c.fibers.at("a");std::reverse(a.line.begin(),a.line.end());
        for(auto& l:c.links){l.controlA=1-l.controlA;std::swap(l.a,l.b);std::swap(l.controlA,l.controlB);}
        audit=auditFiberCycles(c,centers,1);QCOMPARE(audit.issues.size(),size_t(1));QCOMPARE(std::abs(audit.issues[0].turns),1LL);
        a.fullyTraced=false;audit=auditFiberCycles(c,centers,1);
        QCOMPARE(audit.evaluatedCycles,size_t(1));QCOMPARE(audit.checkedCycles,size_t(0));QCOMPARE(audit.uncertainCycles,size_t(1));QVERIFY(!audit.issues[0].reliable);
        c.links[0].pending=true;audit=auditFiberCycles(c,centers,1);
        QCOMPARE(audit.evaluatedCycles,size_t(0));QCOMPARE(audit.skippedLinks,size_t(1));QCOMPARE(audit.uncertainCycles,size_t(1));QVERIFY(audit.issues.empty());
        QVERIFY(audit.summary.find("no cycles could be evaluated")!=std::string::npos);
        audit=auditFiberCycles(c,{},1);QVERIFY(!audit.available);QCOMPARE(audit.cycles,size_t(1));
        c.links[0].pending=false;audit=auditFiberCycles(c,centers,200);QCOMPARE(audit.skippedLinks,size_t(2));
        std::atomic_bool cancel{true};QVERIFY_EXCEPTION_THROWN(auditFiberCycles(c,centers,1,&cancel),std::runtime_error);
    }
    void linkMetadataAudit(){
        QTemporaryDir dir;Settings s;s.fiberFolder=dir.path().toStdString();s.fiberScale=1;
        nlohmann::json a={{"type","vc3d_fiber"},{"version",1},{"line_points",{{100,100,100},{140,100,100}}},
            {"control_points",{{100,100,100},{140,100,100}}},{"branches",nlohmann::json::array()}};
        auto b=a;
        a["branches"].push_back({{"branch_file","b.json"},{"control_point_index",0},{"branch_control_point_index",1},
            {"control_point_position",{100,100,100}},{"branch_control_point_position",{140,100,100}}});
        b["branches"].push_back({{"branch_file","a.json"},{"control_point_index",1},{"branch_control_point_index",0}});
        auto write=[&]{std::ofstream(s.fiberFolder/"a.json")<<a;std::ofstream(s.fiberFolder/"b.json")<<b;};write();
        auto c=loadCatalog(s);QVERIFY(c.warnings.empty());QCOMPARE(c.links.size(),size_t(1));QVERIFY(!c.cycleAudit.available);
        a["branches"][0]["control_point_position"]={101,100,100};b["branches"]=nlohmann::json::array();write();
        c=loadCatalog(s);QCOMPARE(c.warnings.size(),size_t(2));QCOMPARE(c.links.size(),size_t(1));
        QVERIFY(c.warnings[0].find("Stale link")!=std::string::npos);
        QVERIFY(c.warnings[1].find("Missing reciprocal")!=std::string::npos);
    }
    void trimAndResample(){
        std::vector<cv::Vec3d> line{{100,100,100},{103,100,100},{118,100,100},{141,100,100},{183,100,100},{190,100,100}};
        auto trimmed=vc::fiber_tracer::linePointsBetweenOuterControlPoints(line,{line[1],line[4]});QCOMPARE(trimmed.size(),size_t(4));
        Fiber f;f.id="fiber:a";f.line=trimmed;f.controls={0,3};f.direction='H';Normals normals;Settings settings;
        auto n=makeFiberNode(f,settings,normals);QCOMPARE(n->rows,7);QCOMPARE(n->cols,5);
        QCOMPARE(n->columnArclength.front(),0.0);QCOMPARE(n->columnArclength.back(),80.0);
        auto loc=fiberLineLocation(*n,1);QVERIFY(std::abs(loc.grid[0]-.75)<1e-6);
        QVERIFY(cv::norm(sourceLocation(*n,loc.grid)-trimmed[1])<1e-5);
    }
    void isotropyAndOrigin(){
        auto r=fit(base());QVERIFY2(r.valid,r.message.c_str());QVERIFY(r.stretchP95<1.001);QVERIFY(r.correspondenceRms<1e-6);
        QCOMPARE(r.surface->scale()[0],0.05f);auto origin=r.surface->gridToSurface({0,0});QVERIFY(cv::norm(origin-r.origin)<1e-6);
        auto l=pick(r.graph,{30,30});QVERIFY(l);QVERIFY(cv::norm(sourceLocation(*r.graph.nodes.at(l->node).source,l->grid)-cv::Vec3d{130,130,100})<1e-5);
    }
    void jointKeypointsAndRoundtrip(){
        auto g=base();auto n=plane("b",7);g.nodes.emplace("b",Placement{n,n->initialUv});
        g.links={{{"a",{0,0}},{"b",{0,0}},true},{{"a",{4,4}},{"b",{4,4}},true}};
        auto before=graphToJson(g);auto r=fit(g);QVERIFY2(r.valid,r.message.c_str());QVERIFY(r.correspondenceRms<.02);QCOMPARE(graphToJson(g),before);
        auto reload=graphFromJson(graphToJson(r.graph));QCOMPARE(graphToJson(reload),graphToJson(r.graph));
        QTemporaryDir dir;auto path=std::filesystem::path(dir.path().toStdString())/"graph.json";saveSession(r,path);auto opened=loadSession(path);QVERIFY2(opened.valid,opened.message.c_str());
        QCOMPARE(graphToJson(opened.graph),graphToJson(r.graph));
    }
    void invalidAndCancelled(){
        auto g=base();g.nodes.at("a").uv[12]={-100,-100};auto r=render(g);QVERIFY(!r.valid);
        std::atomic_bool cancel{true};QVERIFY_EXCEPTION_THROWN(fit(base(),&cancel),std::runtime_error);
        auto json=graphToJson(base());json["nodes"][0]["rows"]=100;QVERIFY_EXCEPTION_THROWN(graphFromJson(json),std::runtime_error);
        g=base();auto other=plane("disconnected");g.nodes.emplace(other->id,Placement{other,other->initialUv});
        QVERIFY(!fit(g).valid);
    }
    void incompatibleKeypointsStayRejected(){
        auto graph=base();auto patch=plane("b",100);
        graph.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        // These pairs demand a reflection, which a rigid rotation cannot
        // satisfy. The no-flip barrier must still prevent committing the fit.
        graph.links={{{"a",{0,0}},{"b",{0,0}},true},
                     {{"a",{4,0}},{"b",{0,4}},true},
                     {{"a",{0,4}},{"b",{4,0}},true}};
        const auto result=fit(graph);
        QVERIFY(!result.valid);
        QVERIFY2(result.message.find("Keypoint constraints conflict")!=std::string::npos ||
                 result.message.find("inverted quad")!=std::string::npos,result.message.c_str());
    }
    void rejectedChartStepsStayQuiet(){
        // Capture the real solver's stderr in a child rather than changing
        // process-wide glog settings or installing a global log filter.
        QProcess child;
        child.start(QCoreApplication::applicationFilePath(),{"incompatibleKeypointsStayRejected"});
        QVERIFY(child.waitForFinished(30000));
        QCOMPARE(child.exitStatus(),QProcess::NormalExit);
        const auto output=child.readAllStandardOutput()+child.readAllStandardError();
        QVERIFY2(child.exitCode()==0,output.constData());
        QVERIFY2(!output.contains("Step failed to evaluate"),output.constData());
    }
    void linkedRibbonHandedness(){
        Catalog catalog;catalog.components={{"a","b","c"}};
        for(int j=0;j<3;++j){
            Fiber f;f.id=std::string(1,char('a'+j));f.direction='H';
            for(int i=0;i<=20;++i)f.line.push_back({100.0+(j==1?20-i:i)*4,100.0+j*20,100});
            f.controls={0,20};catalog.fibers.emplace(f.id,f);
        }
        catalog.links={{"a","b",0,1},{"b","c",1,0},{"a","c",1,1}};
        Normals normals;Settings s;auto graph=seedGraph(catalog,0,s,normals);
        auto fitted=fit(graph);QVERIFY2(fitted.valid,fitted.message.c_str());
        QVERIFY(fitted.correspondenceRms<.01);QVERIFY(fitted.stretchP95<1.01);
        auto a=placedLocation(fitted.graph,{"a",{},0});
        auto c=placedLocation(fitted.graph,{"c",{},0});
        QVERIFY(std::abs(cv::norm(c-a)-40)<.1);
    }
    void fibersStayInRowsAndColumns(){
        Catalog catalog;catalog.components={{"h","v"}};
        for(const auto& [id,angle]:std::vector<std::pair<std::string,double>>{{"h",.45},{"v",1.15}}) {
            Fiber f;f.id=id;f.direction=id=="h"?'H':'V';
            for(int i=0;i<=100;++i) {
                const double distance=(i-50)*20.0;
                f.line.push_back({3000+distance*std::cos(angle),3000+distance*std::sin(angle),100});
            }
            f.controls={0,50,100};catalog.fibers.emplace(id,f);
        }
        catalog.links={{"h","v",1,1}};
        Normals normals;auto graph=seedGraph(catalog,0,Settings{},normals);
        // Also exercise correction of a tilted seed in an existing session.
        for(auto& [id,p]:graph.nodes)for(auto& uv:p.uv)uv={.8*uv[0]-.6*uv[1],.6*uv[0]+.8*uv[1]};
        // Large initial branch error must settle before imposing the bands.
        for(auto& uv:graph.nodes.at("v").uv)uv+=cv::Vec2d{500,700};
        const auto fitted=fit(graph);QVERIFY2(fitted.valid,fitted.message.c_str());
        for(const auto& [id,p]:fitted.graph.nodes) {
            const auto& n=*p.source;const int axis=n.direction=='V'?0:1;
            double low=INFINITY,high=-INFINITY;
            for(int x=0;x<n.cols;++x){double value=p.uv[(n.rows/2)*n.cols+x][axis];low=std::min(low,value);high=std::max(high,value);}
            QVERIFY2(high-low<81,id.c_str());
        }
        QVERIFY(fitted.correspondenceRms<.1);
        QVERIFY(fitted.stretchP95<1.05);
    }
    void rowConstraintPreservesLocalWiggle(){
        auto n=std::make_shared<Node>();n->id="wavy";n->fiber=true;n->direction='H';n->rows=3;n->cols=101;
        for(int y=0;y<n->rows;++y)for(int x=0;x<n->cols;++x) {
            const double bend=15*std::sin(x*std::acos(-1.0)/10);
            n->xyz.push_back({100+x*20.0,100+y*20.0+bend,100});
            n->initialUv.push_back({x*20.0,y*20.0+bend});
        }
        for(int y=0;y+1<n->rows;++y)for(int x=0;x+1<n->cols;++x) {
            const int a=y*n->cols+x;
            n->triangles.push_back({a,a+1,a+n->cols});
            n->triangles.push_back({a+n->cols+1,a+n->cols,a+1});
        }
        Graph g;g.seed=n->id;g.nodes.emplace(n->id,Placement{n,n->initialUv});
        const auto result=fit(g);QVERIFY2(result.valid,result.message.c_str());
        const auto& uv=result.graph.nodes.at(n->id).uv;
        double low=INFINITY,high=-INFINITY;
        for(int x=0;x<n->cols;++x){low=std::min(low,uv[n->cols+x][1]);high=std::max(high,uv[n->cols+x][1]);}
        QVERIFY(high-low>25);QVERIFY(high-low<35);
        QVERIFY(result.stretchP95<1.01);
    }
    void discoveryStride(){
        QTemporaryDir dir;const auto root=std::filesystem::path(dir.path().toStdString());
        std::filesystem::create_directory(root/"fibers");std::filesystem::create_directory(root/"patches");
        nlohmann::json points=nlohmann::json::array();
        for(int x=100;x<=230;++x)points.push_back({x,110,100});
        const nlohmann::json fiber={{"type","vc3d_fiber"},{"version",1},{"line_points",points},
            {"control_points",{{100,110,100},{230,110,100}}},{"branches",nlohmann::json::array()}};
        for(const char* name:{"a.json","b.json"})std::ofstream(root/"fibers"/name)<<fiber;
        cv::Mat_<cv::Vec3f> grid(3,8);
        for(int y=0;y<grid.rows;++y)for(int x=0;x<grid.cols;++x)grid(y,x)={100+x*20.0f,100+y*20.0f,104};
        QuadSurface patch(grid,{.05f,.05f});patch.save(root/"patches"/"one","one");
        Settings settings;settings.fiberFolder=root/"fibers";settings.patchFolder=root/"patches";settings.fiberScale=1;
        const auto catalog=loadCatalog(settings);Normals normals;const auto graph=seedGraph(catalog,0,settings,normals);
        const auto before=graphToJson(graph).dump();
        for(int stride:{1,4,8}) {
            const auto forward=discover(catalog,graph,graph.seed,nullptr,{},{},stride);
            QCOMPARE(forward.size(),size_t(1));
            auto moved=graph;for(auto& uv:moved.nodes.at(graph.seed).uv)uv+=cv::Vec2d{500,700};
            const auto cached=discover(catalog,moved,graph.seed,nullptr,{},{},stride);
            QCOMPARE(cached.size(),forward.size());QCOMPARE(cached[0].distance,forward[0].distance);
            QCOMPARE(cached[0].contacts.size(),forward[0].contacts.size());
            for(size_t i=0;i<forward[0].contacts.size();++i) {
                QCOMPARE(cached[0].contacts[i].a.lineIndex,forward[0].contacts[i].a.lineIndex);
                QCOMPARE(cached[0].contacts[i].b.grid,forward[0].contacts[i].b.grid);
            }
            // Replacing an immutable source must invalidate an otherwise
            // identical request list; chart movement alone must not.
            auto replacement=std::make_shared<Node>(*graph.nodes.at(graph.seed).source);
            replacement->line[0][2]+=.5;moved.nodes.at(graph.seed).source=replacement;
            const auto changed=discover(catalog,moved,graph.seed,nullptr,{},{},stride);
            QCOMPARE(changed.size(),size_t(1));QVERIFY(changed[0].distance<forward[0].distance);
            moved=graph;moved.settings.patchScale=1.001;
            QVERIFY(discover(catalog,moved,graph.seed,nullptr,{},{},stride).empty());
            std::vector<int> expected;
            for(int i=0;i<=130;i+=stride)expected.push_back(i);
            if(expected.back()!=130)expected.push_back(130);
            QCOMPARE(forward[0].contacts.size(),expected.size());
            const auto proposal=propose(catalog,graph,forward[0],true,normals);
            const auto reverse=discover(catalog,proposal,"patch:one",nullptr,{},{},stride);
            QCOMPARE(reverse.size(),size_t(1));QVERIFY(reverse[0].fiber);
            QCOMPARE(reverse[0].contacts.size(),expected.size());
            for(size_t i=0;i<expected.size();++i) {
                const auto& a=forward[0].contacts[i];const auto& b=reverse[0].contacts[i];
                QCOMPARE(a.a.lineIndex,expected[i]);QCOMPARE(b.b.lineIndex,expected[i]);
                QVERIFY(a.b.grid==b.a.grid);
                // Every published contact retains the original 4-vx proximity check.
                const auto xyz=cv::Vec3d{100+20*a.b.grid[0],100+20*a.b.grid[1],104};
                QVERIFY(cv::norm(xyz-catalog.fibers.at(graph.seed).line.at(expected[i]))<=4.00001);
            }
            auto outside=graph;outside.settings.radius=3.99;
            QVERIFY(discover(catalog,outside,graph.seed,nullptr,{},{},stride).empty());
            auto outsideReverse=proposal;outsideReverse.settings.radius=3.99;
            QVERIFY(discover(catalog,outsideReverse,"patch:one",nullptr,{},{},stride).empty());
        }
        QCOMPARE(discover(catalog,graph,graph.seed)[0].contacts.size(),size_t(34));
        QCOMPARE(graphToJson(graph).dump(),before);
        QVERIFY_EXCEPTION_THROWN(discover(catalog,graph,graph.seed,nullptr,{},{},0),std::invalid_argument);
        QVERIFY_EXCEPTION_THROWN(discover(catalog,graph,graph.seed,nullptr,{},{},-1),std::invalid_argument);
    }
    void discoveryAndComponents(){
        QTemporaryDir dir;auto root=std::filesystem::path(dir.path().toStdString());std::filesystem::create_directory(root/"fibers");std::filesystem::create_directory(root/"patches");
        auto writeFiber=[&](const char* name,const char* peer){nlohmann::json j={{"type","vc3d_fiber"},{"version",1},
            {"line_points",{{90,100,100},{100,100,100},{120,100,100},{140,100,100},{150,100,100}}},
            {"control_points",{{100,100,100},{140,100,100}}},{"branches",nlohmann::json::array()}};
            if(peer)j["branches"].push_back({{"branch_file",peer},{"control_point_index",0},{"branch_control_point_index",0}});
            std::ofstream(root/"fibers"/name)<<j;};
        writeFiber("a.json","b.json");writeFiber("b.json","a.json");writeFiber("c.json","d.json");writeFiber("d.json","c.json");
        cv::Mat_<cv::Vec3f> grid(3,3);for(int y=0;y<3;++y)for(int x=0;x<3;++x)grid(y,x)={100+x*20.0f,100+y*20.0f,104};
        QuadSurface patch(grid,{.05f,.05f});patch.save(root/"patches"/"one","one");
        Settings s;s.fiberFolder=root/"fibers";s.patchFolder=root/"patches";s.fiberScale=1;
        auto catalog=loadCatalog(s);QCOMPARE(catalog.components.size(),size_t(2));QCOMPARE(catalog.components[0].size(),size_t(2));QCOMPARE(catalog.links.size(),size_t(2));
        Normals normals;auto g=seedGraph(catalog,0,s,normals);auto candidates=discover(catalog,g,"fiber:a.json");QCOMPARE(candidates.size(),size_t(1));QVERIFY(std::abs(candidates[0].distance-4)<1e-4);
        g.settings.radius=3.99;QVERIFY(discover(catalog,g,"fiber:a.json").empty());g.settings.radius=4;
        std::vector<Candidate> published;
        auto streamed=discover(catalog,g,"fiber:a.json",nullptr,{},[&](auto batch){published.insert(published.end(),batch.begin(),batch.end());});
        QCOMPARE(published.size(),candidates.size());QCOMPARE(streamed.size(),candidates.size());
        for(size_t i=0;i<candidates.size();++i){
            QCOMPARE(published[i].id,candidates[i].id);QCOMPARE(published[i].distance,candidates[i].distance);
            QCOMPARE(published[i].contacts.size(),candidates[i].contacts.size());
            for(size_t j=0;j<candidates[i].contacts.size();++j){
                QCOMPARE(published[i].contacts[j].a.lineIndex,candidates[i].contacts[j].a.lineIndex);
                QVERIFY(published[i].contacts[j].b.grid==candidates[i].contacts[j].b.grid);
            }
        }
        std::atomic_bool interrupted{false};size_t batches=0;
        // Publication includes only verified contacts.
        discover(catalog,g,"fiber:a.json",&interrupted,{},[&](auto batch){++batches;QVERIFY(!batch.empty());});QCOMPARE(batches,size_t(1));
        auto proposal=propose(catalog,g,candidates[0],true,normals);QCOMPARE(proposal.nodes.size(),size_t(3));QCOMPARE(g.nodes.size(),size_t(2));
        auto reverse=discover(catalog,proposal,"patch:one");QCOMPARE(reverse.size(),size_t(2));QCOMPARE(reverse[0].id,std::string("fiber:c.json"));
        auto grown=propose(catalog,proposal,reverse[0],true,normals);
        auto fitted=fit(grown);QVERIFY2(fitted.valid,fitted.message.c_str());QCOMPARE(fitted.graph.nodes.size(),size_t(5));
        // A second real patch must share overlap constraints with the first.
        patch.save(root/"patches"/"two","two");catalog=loadCatalog(s);
        interrupted=false;
        QVERIFY_EXCEPTION_THROWN(discover(catalog,g,"fiber:a.json",&interrupted,{},[&](auto){interrupted=true;}),std::runtime_error);
        auto more=discover(catalog,fitted.graph,"fiber:a.json");QCOMPARE(more.size(),size_t(1));
        auto overlap=propose(catalog,fitted.graph,more[0],true,normals);
        QVERIFY(std::any_of(overlap.links.begin(),overlap.links.end(),[](const auto& l){return l.a.node=="patch:one"&&l.b.node=="patch:two";}));
        auto final=fit(overlap);QVERIFY2(final.valid,final.message.c_str());QVERIFY(final.correspondenceRms<.1);
    }
};
QTEST_GUILESS_MAIN(ConstraintGraphTest)
#include "test_constraint_graph.moc"
