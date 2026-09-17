#include <QtTest/QtTest>
#include <QTemporaryDir>
#include <QProcess>
#include "vc/constraints/ConstraintGraph.hpp"
#include "vc/fiber_tracer/FiberGeometry.hpp"
#include "vc/lasagna/LineModel.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "../ConstraintGraphView.hpp"
#include "../../../core/src/constraints/GraphFiberJunctions.hpp"
#include "../../../core/src/constraints/GraphFitCosts.hpp"
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
    void progressReportsIterationsWithoutChangingTheFit(){
        auto graph=base();std::vector<std::string> messages;
        const auto reported=fit(graph,nullptr,[&](const auto& message){messages.push_back(message);});
        const auto quiet=fit(graph);
        QVERIFY(reported.valid);QVERIFY(quiet.valid);
        QVERIFY(std::any_of(messages.begin(),messages.end(),[](const auto& message){return message.starts_with("Fitting flattened layout: iteration");}));
        QCOMPARE(graphToJson(reported.graph).dump(),graphToJson(quiet.graph).dump());
        QCOMPARE(cv::norm(*reported.surface->rawPointsPtr(),*quiet.surface->rawPointsPtr(),cv::NORM_INF),0.0);
        std::atomic_bool stop{false};
        QVERIFY_EXCEPTION_THROWN(fit(graph,&stop,[&](const auto& message){
            if(message.starts_with("Fitting flattened layout: iteration"))stop=true;
        }),std::runtime_error);
    }
    void graphSnapshotsAreIndependent(){
        auto graph=base();graph.links.push_back({{"a",{0,0}},{"a",{1,0}}});
        auto snapshot=graph;
        QVERIFY(graph.nodes.at("a").uv.sharesStorage(snapshot.nodes.at("a").uv));
        QVERIFY(graph.links.sharesStorage(snapshot.links));
        graph.nodes.at("a").uv[0]={50,60};graph.links[0].a.grid={2,2};
        QCOMPARE(std::as_const(snapshot.nodes.at("a").uv)[0],cv::Vec2d(0,0));
        QCOMPARE(std::as_const(snapshot.links)[0].a.grid,cv::Vec2d(0,0));
        auto other=snapshot;other.links.push_back({{"a",{2,2}},{"a",{3,2}}});
        QCOMPARE(snapshot.links.size(),size_t(1));QCOMPARE(other.links.size(),size_t(2));
    }
    void analyticChartMetricMatchesAutodiff(){
        for(double orientation:{1.0,-1.0})for(double scale:{.3,1.0,4.0}) {
            const detail::ChartMetric metric{.05,-.017,.04,1.3,orientation};
            detail::ChartMetricCost cost(metric);
            ceres::AutoDiffCostFunction<detail::ChartMetric,8,2,2,2> reference(new detail::ChartMetric(metric));
            double a[2]={3,7},b[2]={3+20*scale,7+2*scale},c[2]={8,7+orientation*30*scale};
            const double* p[3]={a,b,c};double actual[8],expected[8],ja[3][16],je[3][16];
            double* jacActual[3]={ja[0],ja[1],ja[2]};double* jacExpected[3]={je[0],je[1],je[2]};
            QVERIFY(cost.Evaluate(p,actual,jacActual));QVERIFY(reference.Evaluate(p,expected,jacExpected));
            for(int r=0;r<8;++r)QCOMPARE(actual[r],expected[r]);
            for(int i=0;i<3;++i)for(int k=0;k<16;++k)
                QVERIFY(std::abs(ja[i][k]-je[i][k])<1e-12*(1+std::abs(je[i][k])));
            jacActual[1]=nullptr;QVERIFY(cost.Evaluate(p,actual,jacActual));
            QVERIFY(cost.Evaluate(p,actual,nullptr));
            std::swap(p[1],p[2]);QVERIFY(!cost.Evaluate(p,actual,jacActual));
            p[1]=p[0];QVERIFY(!cost.Evaluate(p,actual,nullptr));
        }
    }
    void analyticMatchJacobian(){
        detail::MatchCost cost({.25,.75,-1},1.5);
        double a[2]={2,3},b[2]={4,7},c[2]={5,9},residuals[2],jac[3][4];
        const double* p[3]={a,b,c};double* j[3]={jac[0],jac[1],jac[2]};
        QVERIFY(cost.Evaluate(p,residuals,j));
        QCOMPARE(residuals[0],1.5*(.25*2+.75*4-5));
        QCOMPARE(residuals[1],1.5*(.25*3+.75*7-9));
        for(int i=0;i<3;++i)for(int d=0;d<2;++d) {
            double* parameter=i==0?a:i==1?b:c;
            double plus[2],minus[2];parameter[d]+=1e-5;cost.Evaluate(p,plus,nullptr);
            parameter[d]-=2e-5;cost.Evaluate(p,minus,nullptr);parameter[d]+=1e-5;
            for(int r=0;r<2;++r)QVERIFY(std::abs((plus[r]-minus[r])/2e-5-jac[i][2*r+d])<1e-9);
        }
        j[1]=nullptr;QVERIFY(cost.Evaluate(p,residuals,j));
    }
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
        auto h=plane("h"),upper=plane("upper"),v=plane("v");
        h->fiber=upper->fiber=v->fiber=true;h->direction=upper->direction='H';v->direction='V';
        for(auto& xyz:upper->xyz)xyz[1]+=40;
        for(auto& uv:upper->initialUv)uv[1]+=40;
        // A vertical crosses two parallel horizontals. Its source columns must
        // actually run vertically now that endpoint alignment is enforced.
        for(auto& xyz:v->xyz){const double x=xyz[0];xyz[0]=280-xyz[1];xyz[1]=x;xyz[2]+=10;}
        for(auto& uv:v->initialUv){const double x=uv[0];uv[0]=80-uv[1];uv[1]=x;}
        Graph graph;graph.seed="h";graph.nodes={{"h",{h,h->initialUv}},{"upper",{upper,upper->initialUv}},{"v",{v,v->initialUv}}};
        graph.links={{{"h",{2,2}},{"v",{2,2}}},{{"upper",{2,2}},{"v",{4,2}}}};
        const detail::FiberJunctions junctions(graph);
        // Equal neighboring corrections stay constant between links; they
        // must not add together or fade to the back between close anchors.
        QCOMPARE(junctions.offset(*v,3),cv::Vec3d(0,0,-10));
        const auto before=graphToJson(graph);
        auto fitted=fit(graph);QVERIFY2(fitted.valid,fitted.message.c_str());
        QVERIFY(fitted.correspondenceRms<.01);
        const auto& grid=*fitted.surface->rawPointsPtr();
        const auto uv=placedLocation(fitted.graph,{"h",{2,2}});
        const auto at=(uv-fitted.origin)/graph.settings.spacing;
        const auto xyz=grid(int(std::lround(at[1])),int(std::lround(at[0])));
        QVERIFY(std::abs(xyz[2]-100)<.2);
        QCOMPARE(graphToJson(graph).dump(),before.dump());
        // Junction corrections can reverse the local normal relative to the
        // original vertical ribbon. Reopening an accepted grid must retain it.
        for(auto& xyz:h->xyz)xyz[2]=300;
        for(int y=0;y<v->rows;++y)for(int x=0;x<v->cols;++x)v->xyz[y*v->cols+x][2]=100+40*x;
        for(const auto& n:{h,upper,v}) {
            for(int x=0;x<n->cols;++x) {
                n->line.push_back(n->xyz[(n->rows/2)*n->cols+x]);
                n->lineArclength.push_back(x==0?0:n->lineArclength.back()+cv::norm(n->line[x]-n->line[x-1]));
            }
            n->columnArclength=n->lineArclength;n->controls={0,size_t(n->cols-1)};
        }
        const auto corrected=fit(graph);QVERIFY2(corrected.valid,corrected.message.c_str());
        QTemporaryDir dir;const auto path=std::filesystem::path(dir.path().toStdString())/"corrected.json";
        saveSession(corrected,path);const auto restored=loadSession(path);
        QVERIFY2(restored.valid,restored.message.c_str());
        QCOMPARE(cv::norm(*restored.surface->rawPointsPtr(),*corrected.surface->rawPointsPtr(),cv::NORM_INF),0.0);
        graph.links.push_back({{"h",{0,2}},{"v",{2,2}}});
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
        const auto edit=graphEdit(prepareGraph(committed),draft);
        const auto preparedFit=fitEdit(edit);
        QVERIFY2(preparedFit.valid,preparedFit.message.c_str());
        QCOMPARE(preparedFit.origin,fitted.origin);
        QVERIFY(cv::norm(*preparedFit.surface->rawPointsPtr(),*fitted.surface->rawPointsPtr(),cv::NORM_INF)<1e-4);
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
    void fiberComponentFitsAllLocalPatches(){
        auto graph=base();
        for(const auto& id:{"b","outside","other_traversal"}) {
            auto patch=plane(id);
            if(patch->id=="outside")for(auto& p:patch->xyz)p[2]+=5;
            if(patch->id=="other_traversal")for(auto& p:patch->initialUv)p[0]+=4000;
            graph.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        }
        graph.links.push_back({{"a",{2,2}},{"b",{2,2}}});
        Catalog catalog;catalog.components={{"h0","h1"}};catalog.links={{"h0","h1",0,0}};
        for(int j=0;j<2;++j) {
            Fiber f;f.id=j==0?"h0":"h1";f.direction='H';f.controls={0,80};
            for(int i=0;i<=80;++i)f.line.push_back({100.0+i,120.0+j*40,100});
            catalog.fibers.emplace(f.id,f);
        }
        Normals normals;Candidate candidate;candidate.id="h0";candidate.fiber=true;
        candidate.contacts.push_back({{"a",{2,1}},{"h0",{},40}});
        auto proposal=propose(catalog,graph,candidate,true,normals);
        auto count=[&](const Graph& g,const std::string& patch,const std::string& fiber){
            return std::count_if(g.links.begin(),g.links.end(),[&](const auto& l){
                return (l.a.node==patch&&l.b.node==fiber)||(l.b.node==patch&&l.a.node==fiber);
            });
        };
        for(const auto& fiber:{"h0","h1"}) {
            QCOMPARE(count(proposal,"a",fiber),81);QCOMPARE(count(proposal,"b",fiber),81);
            QCOMPARE(count(proposal,"outside",fiber),0);QCOMPARE(count(proposal,"other_traversal",fiber),0);
        }
        auto manualCandidate=candidate;
        manualCandidate.contacts.push_back({{"b",{2,3}},{"h1",{},40},true});
        auto manual=propose(catalog,graph,manualCandidate,true,normals);
        QCOMPARE(count(manual,"b","h1"),1);QCOMPARE(count(manual,"a","h1"),81);
        proposal.nodes.erase("outside");proposal.nodes.erase("other_traversal");
        const auto result=fit(proposal);QVERIFY2(result.valid,result.message.c_str());
        QVERIFY(result.correspondenceRms<.1);QVERIFY(result.stretchP95<1.01);
        std::atomic_bool cancel{true};
        QVERIFY_EXCEPTION_THROWN(propose(catalog,graph,candidate,true,normals,&cancel),std::runtime_error);
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
        const auto prepared=prepareGraph(committed);
        const auto edit=graphEdit(prepared,draft);
        QCOMPARE(edit.addedNodes.size(),size_t(1));QCOMPARE(edit.addedLinks.size(),size_t(3));QVERIFY(edit.removedLinks.empty());
        const auto preparedFit=fitEdit(edit);QVERIFY2(preparedFit.valid,preparedFit.message.c_str());
        QCOMPARE(preparedFit.origin,local.origin);
        QCOMPARE(preparedFit.localChartVertices,local.localChartVertices);
        QVERIFY(cv::norm(*preparedFit.surface->rawPointsPtr(),*local.surface->rawPointsPtr(),cv::NORM_INF)<1e-4);
        QVERIFY(std::abs(preparedFit.correspondenceRms-local.correspondenceRms)<1e-6);
        QVERIFY(std::abs(preparedFit.stretchP95-local.stretchP95)<1e-6);
        QCOMPARE(graphToJson(materialize(edit)).dump(),graphToJson(draft).dump());
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
        const auto preparedPath=path.parent_path()/"prepared.json";
        saveSession(preparedFit,preparedPath);const auto preparedRestored=loadSession(preparedPath);
        QVERIFY(preparedRestored.valid);
        QCOMPARE(cv::norm(*preparedFit.surface->rawPointsPtr(),*preparedRestored.surface->rawPointsPtr(),cv::NORM_INF),0.0);
        auto disconnected=edit;disconnected.addedLinks.clear();
        QVERIFY(!fitEdit(disconnected).valid);
        auto outOfRange=edit;outOfRange.removedLinks.push_back(committed.graph.links.size());
        QVERIFY_EXCEPTION_THROWN(fitEdit(outOfRange),std::runtime_error);
        std::atomic_bool stopped{true};QVERIFY_EXCEPTION_THROWN(fitEdit(edit,&stopped),std::runtime_error);
        QVERIFY_EXCEPTION_THROWN(prepareGraph(committed,&stopped),std::runtime_error);
        // A second nearby patch is optimized with the overlapping accepted patch.
        auto overlap=local.graph;auto second=plane("second",2040);
        for(auto& xyz:second->xyz)xyz+=cv::Vec3d{2040,-40,0};
        for(auto& uv:second->initialUv)uv[1]-=40;
        overlap.nodes.emplace(second->id,Placement{second,second->initialUv});
        overlap.links.push_back({{"patch",{2.1,2}},{"second",{0,2}},true});
        overlap.links.push_back({{"patch",{4,2}},{"second",{2,2}}});
        auto joint=fitIncremental(overlap,local);QVERIFY2(joint.valid,joint.message.c_str());
        const auto jointEdit=graphEdit(prepareGraph(local),overlap);
        const auto preparedJoint=fitEdit(jointEdit);QVERIFY2(preparedJoint.valid,preparedJoint.message.c_str());
        QCOMPARE(preparedJoint.origin,joint.origin);
        QVERIFY(cv::norm(*preparedJoint.surface->rawPointsPtr(),*joint.surface->rawPointsPtr(),cv::NORM_INF)<1e-4);
        const auto nextPrepared=prepareGraph(preparedFit,edit);
        auto nextDraft=preparedFit.graph;nextDraft.nodes.emplace(second->id,Placement{second,second->initialUv});
        auto nextEdit=graphEdit(nextPrepared,nextDraft);
        addKeypoint(nextEdit,{{"patch",{2.1,2}},{"second",{0,2}},true});
        nextEdit.addedLinks.push_back({{"patch",{4,2}},{"second",{2,2}}});
        const auto reused=fitEdit(nextEdit);QVERIFY2(reused.valid,reused.message.c_str());
        auto rebuiltEdit=nextEdit;rebuiltEdit.base=prepareGraph(preparedFit);
        const auto rebuiltFit=fitEdit(rebuiltEdit);QVERIFY2(rebuiltFit.valid,rebuiltFit.message.c_str());
        QCOMPARE(reused.origin,rebuiltFit.origin);
        QVERIFY(cv::norm(*reused.surface->rawPointsPtr(),*rebuiltFit.surface->rawPointsPtr(),cv::NORM_INF)<1e-4);
        auto pairEdit=edit;
        const Link manual{edit.addedLinks.front().a,edit.addedLinks.front().b,true};
        addKeypoint(pairEdit,manual);QCOMPARE(pairEdit.addedLinks.size(),size_t(1));
        QVERIFY(pairEdit.addedLinks.front().keypoint);
        removeKeypoint(pairEdit,manual);QVERIFY(pairEdit.addedLinks.empty());
        QVERIFY_EXCEPTION_THROWN(removeKeypoint(pairEdit,manual),std::runtime_error);
        GraphEdit acceptedPairEdit{nextPrepared};
        const auto acceptedAutomatic=std::as_const(preparedFit.graph.links).front();
        addKeypoint(acceptedPairEdit,acceptedAutomatic);
        QCOMPARE(acceptedPairEdit.removedLinks.size(),preparedFit.graph.links.size());
        auto pairedGraph=materialize(acceptedPairEdit);
        QCOMPARE(pairedGraph.links.size(),size_t(1));QVERIFY(std::as_const(pairedGraph.links).front().keypoint);
        QVERIFY(!std::as_const(preparedFit.graph.links).front().keypoint);
        auto invalidSource=edit;invalidSource.addedNodes.begin()->second.source.reset();
        QVERIFY_EXCEPTION_THROWN(materialize(invalidSource),std::runtime_error);
        QVERIFY_EXCEPTION_THROWN(fitEdit(invalidSource),std::runtime_error);
        QVERIFY(joint.localChartVertices>0);QVERIFY(joint.correspondenceRms<graph.settings.radius);
        QVERIFY(cv::norm(placedLocation(joint.graph,{"patch",{2.1,2}})-placedLocation(joint.graph,{"second",{0,2}}))<.02);
        QVERIFY(joint.graph.nodes.at("patch").uv!=local.graph.nodes.at("patch").uv);
        auto strained=draft;
        strained.links[1].a=fiberLineLocation(*source,103);
        strained.links[2].a=fiberLineLocation(*source,106);
        std::vector<std::string> progress;
        auto expanded=fitIncremental(strained,committed,nullptr,[&](const auto& message){progress.push_back(message);});
        QVERIFY2(expanded.valid,expanded.message.c_str());QVERIFY(expanded.localChartVertices>0);
        // Contacts that disagree with the fiber's spacing stretch the added
        // patch; the committed fiber is the reference and barely moves.
        double fiberMoved=0;
        for(size_t i=0;i<before.size();++i)fiberMoved=std::max(fiberMoved,cv::norm(expanded.graph.nodes.at("long").uv[i]-before[i]));
        QVERIFY2(fiberMoved<.5,std::to_string(fiberMoved).c_str());
        const auto& stretched=expanded.graph.nodes.at("patch").uv;
        QVERIFY(cv::norm(stretched[2*5+4]-stretched[2*5])>cv::norm(patch->initialUv[2*5+4]-patch->initialUv[2*5])*1.2);
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
    void addedFiberDeformsCommittedPatch(){
        // A committed chart can still contain substantial UV distortion.
        // Its new fiber crosses only part of it: freezing the rest at a small
        // halo used to exhaust the budget and freeze the whole patch instead.
        Normals normals;Graph graph;
        Fiber reference;reference.id="reference";reference.direction='H';reference.controls={0,200};
        for(int x=0;x<=200;++x)reference.line.push_back({100+x*20.0,100,100});
        auto frame=makeFiberNode(reference,graph.settings,normals);
        graph.seed=frame->id;graph.nodes.emplace(frame->id,Placement{frame,frame->initialUv});
        auto patch=std::make_shared<Node>();patch->id="patch";patch->rows=81;patch->cols=121;
        for(int y=0;y<patch->rows;++y)for(int x=0;x<patch->cols;++x){
            patch->xyz.push_back({1500+x*20.0,100+y*20.0,100});
            patch->initialUv.push_back({1400+x*20.0,y*20.0});
        }
        detail::triangulate(*patch);
        auto warped=patch->initialUv;
        for(int y=0;y<patch->rows;++y)for(int x=0;x<patch->cols;++x)
            warped[y*patch->cols+x][0]+=400*std::sin(x*std::acos(-1.0)/120)*y/80;
        graph.nodes.emplace(patch->id,Placement{patch,warped});
        for(int x=0;x<patch->cols;++x)
            graph.links.push_back({fiberLineLocation(*frame,70+x),{patch->id,{double(x),0}}});
        const auto committed=renderPlacement(graph);QVERIFY2(committed.valid,committed.message.c_str());
        Fiber fiber;fiber.id="addition";fiber.direction='H';fiber.controls={0,10};
        for(int x=0;x<=10;++x)fiber.line.push_back({1580+x*20.0,1300,100});
        auto source=makeFiberNode(fiber,graph.settings,normals);
        auto draft=graph;auto uv=source->initialUv;
        for(auto& p:uv)p+=cv::Vec2d{1480,1200};
        draft.nodes.emplace(source->id,Placement{source,uv});
        for(int x=0;x<=10;++x)
            draft.links.push_back({fiberLineLocation(*source,x),{patch->id,{double(x+4),60}}});
        const auto result=fitEdit(graphEdit(prepareGraph(committed),draft));
        QVERIFY2(result.valid,result.message.c_str());QVERIFY(result.localChartVertices>source->xyz.size());
        const auto first=fiberLineLocation(*source,0),last=fiberLineLocation(*source,10);
        const double length=cv::norm(placedLocation(result.graph,last)-placedLocation(result.graph,first));
        QVERIFY2(std::abs(length-200)<1,std::to_string(length).c_str());
        QVERIFY(result.correspondenceRms<.1);
        QVERIFY(result.graph.nodes.at(patch->id).uv!=committed.graph.nodes.at(patch->id).uv);
        QCOMPARE(result.graph.nodes.at(source->id).source.get(),source.get());
        QCOMPARE(result.graph.nodes.at(patch->id).source.get(),patch.get());
        const auto& referenceAfter=result.graph.nodes.at(frame->id).uv;
        for(size_t i=0;i<frame->initialUv.size();++i)
            if(frame->initialUv[i][0]<1000)QCOMPARE(referenceAfter[i],frame->initialUv[i]);
    }
    void farReachingLocalEditKeepsExpandingInsteadOfRefittingEverything(){
        // A committed strip attached to the fiber only at its ends can bend
        // along its whole length. A small patch that asks its middle to sit
        // closer to the fiber moves it far beyond the fourth doubled margin.
        // That used to trigger the global solver, which re-lays out every
        // chart; the local region must keep growing instead.
        Fiber fiber;fiber.id="long";fiber.direction='H';fiber.controls={0,600};
        for(int i=0;i<=600;++i)fiber.line.push_back({100+i*20.0,100,100});
        Normals normals;Graph graph;auto source=makeFiberNode(fiber,graph.settings,normals);
        graph.seed=source->id;graph.nodes.emplace(source->id,Placement{source,source->initialUv});
        // The strip's first row lies on the fiber line; its middle row is
        // 40 vx above it, attached to the fiber only through the strip ends.
        auto strip=std::make_shared<Node>();strip->id="strip";strip->rows=5;strip->cols=81;
        for(int y=0;y<5;++y)for(int x=0;x<81;++x){strip->xyz.push_back({6100+x*20.0,100+y*20.0,100});strip->initialUv.push_back({6000+x*20.0,y*20.0});}
        for(int y=0;y<4;++y)for(int x=0;x<80;++x){int a=y*81+x;strip->triangles.push_back({a,a+1,a+81});strip->triangles.push_back({a+82,a+81,a+1});}
        graph.nodes.emplace(strip->id,Placement{strip,strip->initialUv});
        for(int x:{0,80})graph.links.push_back({fiberLineLocation(*source,300+x),{"strip",{double(x),0}}});
        const auto committed=fit(graph);QVERIFY2(committed.valid,committed.message.c_str());
        // The added patch sits below the fiber line and claims the strip's
        // middle column lies 30 vx along the fiber from where it is, so the
        // strip must slide between its pinned ends.
        auto draft=committed.graph;auto patch=plane("patch",6800);
        for(auto& xyz:patch->xyz)xyz+=cv::Vec3d{6800,-80,0};
        for(auto& uv:patch->initialUv)uv[1]-=80;
        draft.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        for(int i:{0,2,4})draft.links.push_back({fiberLineLocation(*source,340+i),{"patch",{double(i),4}}});
        draft.links.push_back({{"strip",{40,0}},{"patch",{1.5,4}}});
        std::vector<std::string> progress;
        const auto result=fitIncremental(draft,committed,nullptr,[&](const std::string& message){progress.push_back(message);});
        QVERIFY2(result.valid,result.message.c_str());
        QVERIFY(result.localChartVertices>0);QVERIFY(result.localChartVertices<source->xyz.size()/2);
        QVERIFY2(result.localExpansions>=4,std::to_string(result.localExpansions).c_str());
        QVERIFY(result.correspondenceRms<graph.settings.radius);
        QVERIFY(std::none_of(progress.begin(),progress.end(),[](const auto& m){return m=="Fitting full graph";}));
        QVERIFY(std::any_of(progress.begin(),progress.end(),[](const auto& m){return m.find("boundary moved")!=std::string::npos;}));
        const auto& before=committed.graph.nodes.at("long").uv;
        const auto& after=result.graph.nodes.at("long").uv;
        double fiberMoved=0;size_t unchanged=0;
        for(size_t i=0;i<before.size();++i) {
            fiberMoved=std::max(fiberMoved,cv::norm(before[i]-after[i]));
            if(std::abs(before[i][0]-6800)>5000){QVERIFY(before[i]==after[i]);++unchanged;}
        }
        QVERIFY(unchanged>500);QVERIFY2(fiberMoved<.5,std::to_string(fiberMoved).c_str());
        progress.clear();
        const auto bounded=fitEdit(graphEdit(prepareGraph(committed),draft),nullptr,[&](const auto& message){progress.push_back(message);});
        QVERIFY(bounded.valid||bounded.needsRefit);
        QVERIFY(std::none_of(progress.begin(),progress.end(),[](const auto& m){return m=="Fitting full graph";}));
        if(bounded.needsRefit){
            QVERIFY(!bounded.valid);QVERIFY(!bounded.surface);
            QCOMPARE(graphToJson(bounded.graph).dump(),graphToJson(draft).dump());
        }
    }
    void foldedPatchCellsArePrunedInsteadOfRejectingTheFit(){
        // A patch folded back over itself produces grid cells whose fitted
        // normal opposes the source normals. Those cells are omitted like
        // unsupported ones; the fit stays local and is not rejected wholesale.
        Fiber fiber;fiber.id="long";fiber.direction='H';fiber.controls={0,200};
        for(int i=0;i<=200;++i)fiber.line.push_back({100+i*20.0,100,100});
        Normals normals;Graph graph;auto source=makeFiberNode(fiber,graph.settings,normals);
        graph.seed=source->id;graph.nodes.emplace(source->id,Placement{source,source->initialUv});
        const auto committed=fit(graph);QVERIFY2(committed.valid,committed.message.c_str());
        auto draft=committed.graph;auto patch=plane("patch",2000);
        for(auto& xyz:patch->xyz)xyz+=cv::Vec3d{2000,-40,0};
        for(int y=0;y<5;++y)patch->xyz[y*5+4][0]=patch->xyz[y*5+2][0];
        for(auto& uv:patch->initialUv)uv[1]-=40;
        draft.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        for(int i:{0,1,2})draft.links.push_back({fiberLineLocation(*source,100+i),{"patch",{double(i),2}}});
        std::vector<std::string> progress;
        const auto result=fitIncremental(draft,committed,nullptr,[&](const std::string& message){progress.push_back(message);});
        QVERIFY2(result.valid,result.message.c_str());QVERIFY(result.localChartVertices>0);
        QVERIFY2(result.message.find("unsupported boundary samples omitted")!=std::string::npos,result.message.c_str());
        QVERIFY(std::none_of(progress.begin(),progress.end(),[](const auto& m){return m=="Fitting full graph";}));
        // The omitted cells lie under the folded columns, not elsewhere.
        const auto& grid=*result.surface->rawPointsPtr();
        const auto foldLow=placedLocation(result.graph,{"patch",{3,0}}),foldHigh=placedLocation(result.graph,{"patch",{4,4}});
        size_t holesInFold=0,holesOutside=0;
        for(int y=0;y<grid.rows;++y)for(int x=0;x<grid.cols;++x) {
            if(grid(y,x)[0]!=-1)continue;
            const auto uv=result.origin+cv::Vec2d{double(x),double(y)}*graph.settings.spacing;
            const bool nearPatch=uv[1]>-70&&uv[1]<70&&uv[0]>1900&&uv[0]<2200;
            if(!nearPatch)continue;
            if(uv[0]>=std::min(foldLow[0],foldHigh[0])-25&&uv[0]<=std::max(foldLow[0],foldHigh[0])+25)++holesInFold;else ++holesOutside;
        }
        QVERIFY(holesInFold>0);QCOMPARE(holesOutside,size_t(0));
    }
    void addedChartIsPlacedBeforeChoosingTheLocalRegion(){
        // An unplaced source chart says nothing about where an edit lands.
        // The first attempt must already enclose the rigidly aligned chart.
        Fiber fiber;fiber.id="long";fiber.direction='H';fiber.controls={0,200};
        for(int i=0;i<=200;++i)fiber.line.push_back({100+i*20.0,100,100});
        Normals normals;Graph graph;auto source=makeFiberNode(fiber,graph.settings,normals);
        graph.seed=source->id;graph.nodes.emplace(source->id,Placement{source,source->initialUv});
        const auto committed=fit(graph);QVERIFY2(committed.valid,committed.message.c_str());
        auto draft=committed.graph;auto patch=plane("patch",-500000);
        for(auto& xyz:patch->xyz)xyz+=cv::Vec3d{2000,-40,0};
        draft.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        for(int i:{0,2,4})draft.links.push_back({fiberLineLocation(*source,100+i),{"patch",{double(i),2}}});
        std::vector<std::string> progress;
        const auto result=fitIncremental(draft,committed,nullptr,[&](const std::string& message){progress.push_back(message);});
        QVERIFY2(result.valid,result.message.c_str());QVERIFY(result.localChartVertices>0);
        QVERIFY(std::none_of(progress.begin(),progress.end(),[](const auto& m){return m.find("leaves the local region")!=std::string::npos;}));
        QVERIFY(std::none_of(progress.begin(),progress.end(),[](const auto& m){return m=="Fitting full graph";}));
        QVERIFY(cv::norm(placedLocation(result.graph,{"patch",{2,2}})-placedLocation(result.graph,fiberLineLocation(*source,102)))<.1);
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
    void boundariesFollowProposalCommitAndUndo(){
        auto committed=base();
        auto proposal=committed;
        for(const auto& id:{"patch:new","fiber:new","fiber:linked"}){
            auto node=plane(id);node->fiber=std::string(id).starts_with("fiber:");
            proposal.nodes.emplace(id,Placement{node,node->initialUv});
        }
        auto visible=[&](bool showAll){
            std::set<std::string> ids;
            for(const auto& [id,placement]:proposal.nodes)
                if(vc3d::constraint_graph::showBoundary(id,committed,showAll))ids.insert(id);
            return ids;
        };
        const std::set<std::string> proposals{"patch:new","fiber:new","fiber:linked"};
        const std::set<std::string> all{"a","patch:new","fiber:new","fiber:linked"};
        QVERIFY(visible(false)==proposals);
        QVERIFY(visible(true)==all);
        const auto previous=committed;
        committed=proposal;
        QVERIFY(visible(false).empty());
        QVERIFY(visible(true)==all);
        QVERIFY(visible(false).empty());
        committed=previous;
        QVERIFY(visible(false)==proposals);
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
    void rigidRegistrationSkipsUnneededDeformation(){
        auto graph=base();auto patch=plane("b",1000);
        graph.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        graph.links={{{"a",{0,0}},{"b",{0,0}}},{{"a",{4,4}},{"b",{4,4}}}};
        size_t solves=0;
        const auto result=fit(graph,nullptr,[&](const auto& stage){solves+=stage=="Fitting connected graph";});
        QVERIFY2(result.valid,result.message.c_str());QCOMPARE(solves,size_t(1));
        QVERIFY(result.correspondenceRms<.01);QVERIFY(result.stretchP95<1.01);
        graph.seed="missing";QVERIFY(!fit(graph).valid);
    }
    void patchAttachmentsKeepFibersLevel(){
        // Two measured fibers have opposite small slopes across the same
        // patch. Dense attachment constraints must not override their axes.
        auto patch=std::make_shared<Node>();patch->id="patch";patch->rows=15;patch->cols=21;
        for(int y=0;y<patch->rows;++y)for(int x=0;x<patch->cols;++x) {
            patch->xyz.push_back({100+x*20.0,100+y*20.0,100});
            patch->initialUv.push_back({x*20.0,y*20.0});
        }
        detail::triangulate(*patch);
        Graph graph;graph.seed=patch->id;graph.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        Normals normals;
        for(int j=0;j<2;++j) {
            Fiber fiber;fiber.id=j==0?"h0":"h1";fiber.direction='H';fiber.controls={0,20};
            for(int x=0;x<=20;++x)fiber.line.push_back({100+x*20.0,180+j*120+(j==0?1:-1)*(x-10)*2.0,100});
            auto n=makeFiberNode(fiber,graph.settings,normals);
            // Start in the patch's frame, as a selected attachment would.
            auto uv=n->initialUv;
            const double angle=(j==0?1:-1)*std::atan(.1);
            for(auto& p:uv)p={std::cos(angle)*p[0]-std::sin(angle)*p[1],std::sin(angle)*p[0]+std::cos(angle)*p[1]+80+j*120};
            graph.nodes.emplace(n->id,Placement{n,uv});
            for(int x=0;x<=20;++x)graph.links.push_back({fiberLineLocation(*n,x),{patch->id,{double(x),(fiber.line[x][1]-100)/20}}});
        }
        const auto result=fit(graph);QVERIFY2(result.valid,result.message.c_str());
        for(const auto& [id,p]:result.graph.nodes)if(p.source->fiber) {
            const auto& n=*p.source;const int first=(n.rows/2)*n.cols;
            QVERIFY2(std::abs(p.uv[first+n.cols-1][1]-p.uv[first][1])<.1,id.c_str());
        }
        QVERIFY(result.correspondenceRms<.1);QVERIFY(result.stretchP95<1.2);
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
