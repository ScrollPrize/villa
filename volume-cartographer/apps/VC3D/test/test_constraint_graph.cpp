#include <QtTest/QtTest>
#include <QTemporaryDir>
#include <QProcess>
#include "vc/constraints/ConstraintGraph.hpp"
#include "vc/constraints/FiberReview.hpp"
#include "vc/fiber_tracer/FiberGeometry.hpp"
#include "vc/lasagna/LineModel.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "../ConstraintGraphView.hpp"
#include "../../../core/src/constraints/GraphFiberJunctions.hpp"
#include "../../../core/src/constraints/GraphFitCosts.hpp"
#include <fstream>
#include <bit>
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
Graph legacyFiberGraph() {
    Graph g;auto n=std::make_shared<Node>();n->id="fiber:legacy";n->fiber=true;n->direction='H';n->rows=7;n->cols=5;n->controls={0,4};
    for(int c=0;c<5;++c){n->line.push_back({100.+20*c,100,100});n->lineArclength.push_back(20*c);n->columnArclength.push_back(20*c);}
    for(int r=0;r<7;++r)for(int c=0;c<5;++c){n->xyz.push_back({100.+20*c,100.+20*(r-3),100});n->initialUv.push_back({20.*c,20.*(r-3)});}
    for(int r=0;r<6;++r)for(int c=0;c<4;++c){int a=r*5+c;n->triangles.push_back({a,a+1,a+5});n->triangles.push_back({a+6,a+5,a+1});}
    g.seed=n->id;auto uv=n->initialUv;for(auto& p:uv)p+=cv::Vec2d{12,34};g.nodes.emplace(n->id,Placement{n,std::move(uv)});return g;
}
std::filesystem::path quarantineFixture(const std::filesystem::path& root) {
    const auto directory=root/"project",patches=directory/"patches";
    std::filesystem::create_directories(patches);
    nlohmann::json sources=nlohmann::json::array();
    std::ofstream geometry(directory/"layout.bin",std::ios::binary);uint64_t offset=0;
    for(const auto& id:{"a:0","a:1","b:0","c:0"}) {
        const auto source=patches/std::string(1,id[0]);std::filesystem::create_directories(source);
        std::ofstream(source/"marker.txt")<<id[0];auto node=plane(id);
        for(size_t i=0;i<node->xyz.size();++i) {
            const std::array<double,5> values{node->xyz[i][0],node->xyz[i][1],100.+(id[0]-'a'),node->initialUv[i][0],node->initialUv[i][1]};
            for(double v:values){const auto bits=std::bit_cast<uint64_t>(v);for(int b=0;b<8;++b)geometry.put(char(bits>>(8*b)));}
        }
        sources.push_back({{"id",id},{"path",source.string()},{"fiber",false},{"rows",5},{"cols",5},
            {"byte_offset",offset},{"quad_runs",nlohmann::json::array({{0,16}})}});offset+=25*40;
    }
    geometry.close();auto settings=graphToJson(base()).at("settings");settings["patch_folder"]=patches.string();
    nlohmann::json document={{"type","vc3d_winding_layout"},{"version",2},{"seed","a:0"},
        {"geometry_format","little_endian_float64_xyzuv"},{"geometry_file","layout.bin"},{"geometry_bytes",offset},
        {"settings",settings},{"layout",{{"read_only",true}}},{"sources",sources}};
    const auto path=directory/"layout.workspace.json";std::ofstream(path)<<document.dump(2);return path;
}
std::string readFile(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);return {std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
}
std::filesystem::path reviewQuarantineFixture(const std::filesystem::path& root) {
    FiberReview review;review.settings.patchFolder=root/"project/patches";review.selected="fiber:a";
    for(const auto* name:{"one","two"}){
        std::filesystem::create_directories(review.settings.patchFolder/name);
        std::ofstream(review.settings.patchFolder/name/"marker.txt")<<name;
    }
    Normals normals;
    for(const auto* id:{"fiber:a","fiber:b"}){
        Fiber fiber;fiber.id=id;fiber.direction='H';fiber.controls={0,4};
        for(int i=0;i<5;++i)fiber.line.push_back({100.+20*i,100,100});
        auto node=makeFiberNode(fiber,review.settings,normals);
        auto& f=review.fibers[id];f.path=id;f.direction='H';f.discovered=true;
        f.base.settings=review.settings;f.base.seed=id;f.base.nodes.emplace(id,Placement{node,node->initialUv});
        for(const auto* patchId:{"patch:one","patch:two","patch:one-component"}){
            auto patch=plane(patchId);patch->path=review.settings.patchFolder/(std::string(patchId)=="patch:two"?"two":"one");
            auto graph=f.base;graph.nodes.emplace(patchId,Placement{patch,patch->initialUv});
            graph.links.push_back({{id,{},0},{patchId,{0,0}},true});
            Candidate c;c.id=patchId;c.contacts=graph.links.read();f.proposals.push_back(c);
            if(std::string(id)=="fiber:b")f.decisions[patchId]={ReviewStatus::Accepted,review.nextOrder++,graph};
            else if(std::string(patchId)=="patch:one")f.pending=graph;
        }
        f.current=0;
    }
    const auto path=root/"project/review.json";saveReview(review,path);return path;
}

}
class ConstraintGraphTest:public QObject {
    Q_OBJECT
private slots:
    void robustStripUpgradePreservesPatchPlacementAndContacts(){
        Normals normals;auto graph=legacyFiberGraph();auto patch=plane("patch:test");graph.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        graph.links.push_back({{graph.seed,{1.3,2.7}},{patch->id,{1,1}},true});
        const auto location=placedLocation(graph,graph.links[0].a);const auto oldPatch=graph.nodes.at(patch->id);
        auto legacy=graphToJson(graph);for(auto& n:legacy["nodes"])n.erase("strip_generation");graph=graphFromJson(legacy);
        QVERIFY(regenerateFiberStrips(graph,normals));QCOMPARE(graph.nodes.at(graph.seed).source->stripGeneration,7);
        QCOMPARE(graph.nodes.at(patch->id).source->xyz,oldPatch.source->xyz);QCOMPARE(graph.nodes.at(patch->id).uv,oldPatch.uv);
        QVERIFY(cv::norm(placedLocation(graph,graph.links[0].a)-location)<1e-8);
        QCOMPARE(graphToJson(graphFromJson(graphToJson(graph))),graphToJson(graph));
        const auto upgraded=graphToJson(graph);QVERIFY(!regenerateFiberStrips(graph,normals));QCOMPARE(graphToJson(graph),upgraded);
    }
    void robustReviewUpgradePreservesDecisionsAndPendingEdits(){
        Normals normals;ReviewFiber f;f.base=legacyFiberGraph();f.direction='H';f.reverse=true;f.discovered=true;f.current=0;
        auto graph=f.base;auto patch=plane("patch:test");graph.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        graph.links.push_back({{graph.seed,{1.25,3}},{patch->id,{1,1}},true});
        f.decisions[patch->id]={ReviewStatus::Accepted,42,graph};f.pending=graph;
        Candidate c;c.id=patch->id;c.contacts=graph.links.read();f.proposals.push_back(c);
        const auto patchUv=graph.nodes.at(patch->id).uv;const auto beforeContact=placedLocation(graph,graph.links[0].a);
        QVERIFY(regenerateReviewFiber(f,normals));QCOMPARE(f.current,0);QVERIFY(f.reverse);QVERIFY(f.discovered);
        const auto& decision=f.decisions.at(patch->id);QCOMPARE(decision.order,uint64_t(42));QCOMPARE(decision.status,ReviewStatus::Accepted);
        QCOMPARE(decision.alignment.nodes.at(patch->id).uv,patchUv);QCOMPARE(f.pending->nodes.at(patch->id).uv,patchUv);
        QCOMPARE(f.base.nodes.at(f.base.seed).source,decision.alignment.nodes.at(f.base.seed).source);
        QCOMPARE(f.base.nodes.at(f.base.seed).uv,decision.alignment.nodes.at(f.base.seed).uv);
        QCOMPARE(f.base.nodes.at(f.base.seed).uv,f.pending->nodes.at(f.base.seed).uv);
        QVERIFY(cv::norm(placedLocation(f.base,f.proposals[0].contacts[0].a)-beforeContact)<1e-8);
        FiberReview review;review.selected=f.base.seed;review.fibers.emplace(review.selected,f);
        const auto saved=reviewToJson(review);QCOMPARE(reviewToJson(reviewFromJson(saved)),saved);
        QVERIFY(!regenerateReviewFiber(f,normals));
    }
    void robustStripUpgradeCancellationIsTransactional(){
        Normals normals;auto graph=legacyFiberGraph();const auto before=graphToJson(graph);std::atomic_bool cancel=true;
        QVERIFY_EXCEPTION_THROWN(regenerateFiberStrips(graph,normals,&cancel),std::runtime_error);QCOMPARE(graphToJson(graph),before);
    }
    void robustStripUpgradeRemapsContactsWhenFinerGeometryIsRequired(){
        Normals normals;auto graph=legacyFiberGraph();
        auto source=std::make_shared<Node>(*graph.nodes.at(graph.seed).source);
        for(int c=0;c<source->cols;++c){
            source->line[c][0]=100+1.25*c;source->lineArclength[c]=1.25*c;source->columnArclength[c]=1.25*c;
            for(int r=0;r<source->rows;++r)source->xyz[r*source->cols+c][0]=100+1.25*c;
        }
        graph.nodes.at(graph.seed).source=source;
        auto patch=plane("patch:test");graph.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        graph.links.push_back({{graph.seed,{1.3,2.7}},{patch->id,{1,1}},true});
        const auto before=placedLocation(graph,graph.links[0].a);
        const auto patchUv=graph.nodes.at(patch->id).uv;
        QVERIFY(regenerateFiberStrips(graph,normals));
        const auto& upgraded=*graph.nodes.at(graph.seed).source;
        QCOMPARE(upgraded.rows,61);QCOMPARE(upgraded.stripGeneration,7);
        QVERIFY(cv::norm(placedLocation(graph,graph.links[0].a)-before)<1e-8);
        QCOMPARE(graph.nodes.at(patch->id).uv,patchUv);
        QVERIFY(std::abs(upgraded.initialUv.front()[1]-upgraded.initialUv[(upgraded.rows/2)*upgraded.cols][1])==60);
    }
    void fiberReviewPrunesMissingProposalsAndPreservesEmbeddedWork(){
        QTemporaryDir temp;const auto path=reviewQuarantineFixture(temp.path().toStdString());
        auto review=loadReview(path);auto& f=review.fibers.at("fiber:a");
        const auto accepted=reviewToJson(review).at("fibers").at("fiber:b");
        const auto pending=graphToJson(*f.pending);
        // The pending patch survives even though its source is unavailable.
        QCOMPARE(pruneUnavailableReviewProposals(f,{"patch:two"}),size_t(1));
        QCOMPARE(f.current,0);QCOMPARE(graphToJson(*f.pending),pending);
        QCOMPARE(pruneUnavailableReviewProposals(review.fibers.at("fiber:b"),{}),size_t(0));
        QCOMPARE(reviewToJson(review).at("fibers").at("fiber:b"),accepted);
        f.pending.reset();f.proposals.push_back(f.proposals.front());
        f.decisions["patch:one"]={ReviewStatus::Skipped,review.nextOrder++,{}};
        QCOMPARE(pruneUnavailableReviewProposals(f,{"patch:two"}),size_t(2));
        QCOMPARE(f.current,0);QCOMPARE(f.proposals.front().id,std::string("patch:two"));
        QCOMPARE(f.decisions.at("patch:one").status,ReviewStatus::Skipped);
        QCOMPARE(pruneUnavailableReviewProposals(f,{"patch:two"}),size_t(0));
        saveReview(review,path);QCOMPARE(reviewToJson(loadReview(path)),reviewToJson(review));
        QCOMPARE(pruneUnavailableReviewProposals(f,{}),size_t(1));
        QCOMPARE(f.current,-1);QVERIFY(f.proposals.empty());QCOMPARE(f.decisions.size(),size_t(1));
    }
    void fiberReviewPruningRemapsSelectionAndWraps(){
        ReviewFiber f;
        for(const char* id:{"missing","first","decided","last"}){Candidate c;c.id=id;f.proposals.push_back(c);}
        f.current=3;f.decisions["decided"]={ReviewStatus::Rejected};
        QCOMPARE(pruneUnavailableReviewProposals(f,{"first","decided","last"}),size_t(1));
        QCOMPARE(f.current,2);QCOMPARE(f.proposals[f.current].id,std::string("last"));
        QCOMPARE(pruneUnavailableReviewProposals(f,{"first","decided"}),size_t(1));
        QCOMPARE(f.current,0);QCOMPARE(f.proposals[f.current].id,std::string("first"));
        QCOMPARE(pruneUnavailableReviewProposals(f,{"decided"}),size_t(1));
        QCOMPARE(f.current,-1);QCOMPARE(f.decisions.at("decided").status,ReviewStatus::Rejected);
    }
    void fiberReviewQuarantineRemovesAllAssociationsAndPersists(){
        QTemporaryDir temp;const auto path=reviewQuarantineFixture(temp.path().toStdString());
        auto before=loadReview(path);const auto original=readFile(path);
        const auto retained=graphToJson(before.fibers.at("fiber:b").decisions.at("patch:two").alignment);
        before.proposalOpacity=37; // Unsaved review edits are published too.
        const PatchEntry patch{"patch:one",before.settings.patchFolder/"one",{}, {}};
        const auto result=quarantineReviewPatch(before,path,patch);
        QVERIFY(!std::filesystem::exists(patch.path));
        QCOMPARE(readFile(quarantineDirectory(path)/"one/marker.txt"),std::string("one"));
        QVERIFY(result.quarantined.contains("patch:one"));QVERIFY(result.quarantined.contains("patch:one-component"));
        const auto& a=result.fibers.at("fiber:a");QCOMPARE(a.current,0);QVERIFY(!a.pending);
        QCOMPARE(a.proposals.size(),size_t(1));QCOMPARE(a.proposals.front().id,std::string("patch:two"));
        const auto& b=result.fibers.at("fiber:b");QCOMPARE(acceptedPatchCount(b),size_t(1));
        QCOMPARE(graphToJson(b.decisions.at("patch:two").alignment),retained);
        QCOMPARE(graphToJson(a.base),graphToJson(before.fibers.at("fiber:a").base));
        const auto reopened=loadReview(path);QCOMPARE(reviewToJson(reopened),reviewToJson(result));
        QCOMPARE(reopened.sessionRevision,result.sessionRevision);QCOMPARE(reopened.proposalOpacity,37);
        size_t backups=0;for(const auto& file:std::filesystem::directory_iterator(path.parent_path()))
            if(file.path().filename().string().find("before-quarantine-")!=std::string::npos){++backups;QCOMPARE(readFile(file.path()),original);}
        QCOMPARE(backups,size_t(1));
        QVERIFY_EXCEPTION_THROWN(quarantineReviewPatch(before,path,{"patch:two",before.settings.patchFolder/"two",{}, {}}),std::runtime_error);
        // Quarantining the final available patch leaves a usable empty review.
        const auto complete=quarantineReviewPatch(result,path,{"patch:two",result.settings.patchFolder/"two",{}, {}});
        QCOMPARE(nextReviewProposal(complete.fibers.at("fiber:a")),-1);
        QCOMPARE(acceptedPatchCount(complete.fibers.at("fiber:b")),size_t(0));
        QCOMPARE(reviewToJson(loadReview(path)),reviewToJson(complete));
    }
    void fiberReviewQuarantineFailurePreservesWorkspaceAndFiles(){
        QTemporaryDir temp;const auto path=reviewQuarantineFixture(temp.path().toStdString());
        const auto before=loadReview(path);const auto original=readFile(path);
        const PatchEntry patch{"patch:one",before.settings.patchFolder/"one",{}, {}};
        QVERIFY_EXCEPTION_THROWN(quarantineReviewPatch(before,path,patch,[&](const auto& progress){
            if(progress.starts_with("Saving the workspace"))std::filesystem::create_directories(quarantineDirectory(path)/"one");
        }),std::runtime_error);
        QCOMPARE(readFile(path),original);QCOMPARE(readFile(patch.path/"marker.txt"),std::string("one"));
        QVERIFY(!std::filesystem::exists(path.string()+".quarantine-lock"));
        QVERIFY_EXCEPTION_THROWN(quarantineReviewPatch(before,path,{"patch:outside",temp.path().toStdString(),{}, {}}),std::runtime_error);
        QCOMPARE(readFile(path),original);
    }

    void fiberReviewOrderingAndPairDecisions(){
        Normals normals;Fiber fiber;fiber.id="fiber";fiber.direction='H';fiber.controls={0,4};
        for(int i=0;i<5;++i)fiber.line.push_back({100.+20*i,100,100});
        ReviewFiber review;auto node=makeFiberNode(fiber,Settings{},normals);
        review.base.seed=fiber.id;review.base.nodes.emplace(fiber.id,Placement{node,node->initialUv});
        auto candidate=[&](std::string patch,int i){Candidate c;c.id=patch;c.contacts.push_back({{fiber.id,{},i},{patch,{0,0}}});return c;};
        review.proposals={candidate("c",4),candidate("a",0),candidate("b",0),candidate("a",3)};
        orderReviewProposals(review);QCOMPARE(review.proposals[0].id,std::string("a"));QCOMPARE(review.proposals[1].id,std::string("b"));
        QCOMPARE(nextReviewProposal(review),0);
        review.decisions["a"]={ReviewStatus::Rejected};QCOMPARE(nextReviewProposal(review),1);
        review.decisions["b"]={ReviewStatus::Accepted};review.decisions["c"]={ReviewStatus::Rejected};QCOMPARE(nextReviewProposal(review),-1);QCOMPARE(acceptedPatchCount(review),size_t(1));
        auto other=review;other.decisions.clear();QCOMPARE(nextReviewProposal(other),0);QCOMPARE(nextReviewProposal(other,-1,-1),3);
        review.decisions.erase("a");QCOMPARE(nextReviewProposal(review),0);
        review.reverse=true;orderReviewProposals(review);QCOMPARE(review.proposals.front().id,std::string("c"));QCOMPARE(nextReviewProposal(review),1);
        review.proposals.clear();QCOMPARE(nextReviewProposal(review),-1);
    }

    void fiberReviewSkipIsDistinctPersistentAndLocal(){
        QTemporaryDir temp;const auto path=reviewQuarantineFixture(temp.path().toStdString());
        auto review=loadReview(path);auto& fiber=review.fibers.at("fiber:a");
        const auto otherBefore=reviewToJson(review).at("fibers").at("fiber:b");
        fiber.proposals.push_back(fiber.proposals.front());
        fiber.decisions["patch:one"]={ReviewStatus::Skipped,review.nextOrder++,{}};fiber.pending.reset();
        QCOMPARE(nextReviewProposal(fiber,0),1);QCOMPARE(nextReviewProposal(fiber,3),1);
        QCOMPARE(acceptedPatchCount(fiber),size_t(0));
        fiber.current=nextReviewProposal(fiber,0);
        saveReview(review,path);auto reopened=loadReview(path);const auto& restored=reopened.fibers.at("fiber:a");
        QCOMPARE(restored.decisions.at("patch:one").status,ReviewStatus::Skipped);
        QCOMPARE(nextReviewProposal(restored),1);QCOMPARE(acceptedPatchCount(restored),size_t(0));
        const auto document=reviewToJson(reopened);QCOMPARE(document.at("version"),2);
        QCOMPARE(document.at("fibers").at("fiber:a").at("decisions").at("patch:one").at("status"),"skipped");
        QCOMPARE(document.at("fibers").at("fiber:b"),otherBefore);
        QVERIFY(std::filesystem::exists(review.settings.patchFolder/"one/marker.txt"));QVERIFY(reopened.quarantined.empty());
        reopened.fibers.at("fiber:a").decisions.erase("patch:one");
        QCOMPARE(nextReviewProposal(reopened.fibers.at("fiber:a")),0);
        auto invalid=document;invalid["fibers"]["fiber:a"]["decisions"]["patch:one"]["status"]="unknown";
        QVERIFY_EXCEPTION_THROWN(reviewFromJson(invalid),std::runtime_error);
    }
    void fiberReviewRoundTripAndSharedSources(){
        Normals normals;Fiber fiber;fiber.id="fiber";fiber.direction='H';fiber.controls={0,4};
        for(int i=0;i<5;++i)fiber.line.push_back({100.+20*i,100,100});
        FiberReview review;review.settings.radius=1.5;review.selected="fiber";
        auto& f=review.fibers["fiber"];f.path="missing/fiber.json";f.direction='H';f.reverse=true;f.discovered=true;
        auto node=makeFiberNode(fiber,review.settings,normals);f.base.settings=review.settings;f.base.seed=fiber.id;f.base.nodes.emplace(fiber.id,Placement{node,node->initialUv});
        auto patch=plane("patch");auto graph=f.base;graph.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        graph.links.push_back({{fiber.id,{},0},{patch->id,{0,0}},true});
        f.decisions[patch->id]={ReviewStatus::Accepted,1,graph};f.decisions["rejected"]={ReviewStatus::Rejected,2,{}};
        Candidate c;c.id="patch";c.contacts=graph.links.read();f.proposals.push_back(c);f.current=0;f.pending=graph;
        review.nextOrder=3;review.committedOpacity=17;review.proposalOpacity=61;review.boundaries=true;
        review.fiberLineOpacity=23;review.fiberLineWidth=1.5;review.committedTint=27;
        const auto json=reviewToJson(review);QCOMPARE(json.at("sources").size(),size_t(2));
        const auto restored=reviewFromJson(json);const auto& saved=restored.fibers.at("fiber");
        QCOMPARE(reviewToJson(restored),json);QCOMPARE(saved.base.nodes.at("fiber").source,saved.decisions.at("patch").alignment.nodes.at("fiber").source);
        QCOMPARE(saved.pending->nodes.at("patch").source,saved.decisions.at("patch").alignment.nodes.at("patch").source);
        QTemporaryDir dir;auto path=std::filesystem::path(dir.path().toStdString())/"review.json";
        saveReview(review,path);QCOMPARE(reviewToJson(loadReview(path)),json);
        auto broken=json;broken["fibers"]["fiber"]["current"]=100;QVERIFY_EXCEPTION_THROWN(reviewFromJson(broken),std::runtime_error);
        broken=json;broken["version"]=999;QVERIFY_EXCEPTION_THROWN(reviewFromJson(broken),std::runtime_error);
        auto legacy=json;legacy["version"]=1;
        for(auto& f:legacy["fibers"])for(auto& d:f["decisions"]){d["accepted"]=d["status"]=="accepted";d.erase("status");}
        QCOMPARE(reviewToJson(reviewFromJson(legacy)),json);
        legacy.erase("committed_tint");QCOMPARE(reviewFromJson(legacy).committedTint,4);
        for(int tint:{-1,101}){broken=json;broken["committed_tint"]=tint;QVERIFY_EXCEPTION_THROWN(reviewFromJson(broken),std::runtime_error);}
        legacy.erase("fiber_line_opacity");legacy.erase("fiber_line_width");
        QCOMPARE(reviewFromJson(legacy).fiberLineOpacity,FiberReview{}.fiberLineOpacity);
        QCOMPARE(reviewFromJson(legacy).fiberLineWidth,FiberReview{}.fiberLineWidth);
        for(int opacity:{-1,101}){broken=json;broken["fiber_line_opacity"]=opacity;QVERIFY_EXCEPTION_THROWN(reviewFromJson(broken),std::runtime_error);}
        for(double width:{0.,11.}){broken=json;broken["fiber_line_width"]=width;QVERIFY_EXCEPTION_THROWN(reviewFromJson(broken),std::runtime_error);}
        // Rejection-only and not-yet-built fibers can be saved too.
        f.base={};f.pending.reset();f.proposals.clear();f.current=-1;f.decisions.erase("patch");
        QCOMPARE(acceptedPatchCount(reviewFromJson(reviewToJson(review)).fibers.at("fiber")),size_t(0));
    }

    void fiberReviewVerticalTraversalStartsAtTop(){
        Normals normals;Fiber fiber;fiber.id="fiber";fiber.direction='H';fiber.controls={0,4};
        for(int i=0;i<5;++i)fiber.line.push_back({100.+20*i,100,100});
        auto node=std::make_shared<Node>(*makeFiberNode(fiber,Settings{},normals));node->direction='V';
        auto uv=node->initialUv;for(auto& p:uv)p={-p[1],p[0]};
        ReviewFiber review;review.base.seed=fiber.id;review.base.nodes.emplace(fiber.id,Placement{node,uv});
        for(int i:{0,4,2}){Candidate c;c.id=std::to_string(i);c.contacts.push_back({{fiber.id,{},i},{c.id,{0,0}}});review.proposals.push_back(c);}
        orderReviewProposals(review);QCOMPARE(review.proposals[0].id,std::string("4"));QCOMPARE(review.proposals[2].id,std::string("0"));
        review.reverse=true;orderReviewProposals(review);QCOMPARE(review.proposals[0].id,std::string("0"));
    }

    void fixedReferenceReviewDoesNotMoveFiber(){
        Normals normals;Fiber fiber;fiber.id="fiber";fiber.direction='H';fiber.controls={0,4};
        for(int i=0;i<5;++i)fiber.line.push_back({100.+20*i,100,100});
        Graph graph;graph.seed=fiber.id;auto node=makeFiberNode(fiber,graph.settings,normals);graph.nodes.emplace(node->id,Placement{node,node->initialUv});
        auto baseResult=renderPlacement(graph);QVERIFY2(baseResult.valid,baseResult.message.c_str());
        auto patch=plane("patch");auto draft=graph;draft.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        for(int i=0;i<5;++i)draft.links.push_back({{fiber.id,{},i},{patch->id,{double(i),0}}});
        auto edit=graphEdit(prepareGraph(baseResult),draft);
        const auto result=fitEdit(edit,nullptr,{},FitPolicy::FixedReference);QVERIFY2(result.valid,result.message.c_str());
        QCOMPARE(result.graph.nodes.at("fiber").uv,graph.nodes.at("fiber").uv);
        QCOMPARE(result.graph.nodes.at("fiber").source->xyz,node->xyz);
        QCOMPARE(result.graph.nodes.at("patch").source->xyz,patch->xyz);
        // Repeating the same pair never incorporates independently stored decisions.
        const auto repeated=fitEdit(edit,nullptr,{},FitPolicy::FixedReference);QVERIFY(repeated.valid);
        QCOMPARE(result.graph.nodes.at("patch").uv,repeated.graph.nodes.at("patch").uv);
        std::atomic_bool cancelled{true};QVERIFY_EXCEPTION_THROWN(fitEdit(edit,&cancelled,{},FitPolicy::FixedReference),std::runtime_error);
        // An imperfect automatic correspondence should still produce a review
        // overlay, while the strict full-graph attachment policy stays intact.
        auto imperfect=edit;imperfect.addedLinks.back().b=imperfect.addedLinks.front().b;
        QVERIFY(!fitEdit(imperfect,nullptr,{},FitPolicy::FixedReference).valid);
        const auto best=fitEdit(imperfect,nullptr,{},FitPolicy::FiberReview);
        QVERIFY2(best.valid,best.message.c_str());QVERIFY(best.correspondenceRms>graph.settings.radius);
        QCOMPARE(best.graph.nodes.at("fiber").uv,graph.nodes.at("fiber").uv);
        QVERIFY(renderSourceLayer(best.graph,false).valid);
        QVERIFY_EXCEPTION_THROWN(fitEdit(edit,&cancelled,{},FitPolicy::FiberReview),std::runtime_error);
    }

    void fiberReviewFitsAcceptedOverlaps(){
        Normals normals;Fiber fiber;fiber.id="fiber";fiber.direction='H';fiber.controls={0,4};
        for(int i=0;i<5;++i)fiber.line.push_back({100.+20*i,100,100});
        ReviewFiber review;auto& graph=review.base;graph.seed=fiber.id;
        auto node=makeFiberNode(fiber,graph.settings,normals);graph.nodes.emplace(node->id,Placement{node,node->initialUv});
        const auto baseResult=renderPlacement(graph);QVERIFY(baseResult.valid);
        auto patch=plane("proposal"),reference=plane("accepted");
        auto accepted=graph;auto uv=reference->initialUv;
        for(auto& p:uv)p[0]+=.15*p[1]; // Agrees at the fiber, shears away from it.
        accepted.nodes.emplace(reference->id,Placement{reference,uv});
        for(int i=0;i<5;++i)accepted.links.push_back({{fiber.id,{},i},{reference->id,{double(i),0}}});
        review.decisions[reference->id]={ReviewStatus::Accepted,1,accepted};
        auto draft=graph;draft.nodes.emplace(patch->id,Placement{patch,patch->initialUv});
        for(int i=0;i<5;++i)draft.links.push_back({{fiber.id,{},i},{patch->id,{double(i),0}}});
        auto edit=graphEdit(prepareGraph(baseResult),draft);
        const auto alone=fitEdit(edit,nullptr,{},FitPolicy::FiberReview);QVERIFY2(alone.valid,alone.message.c_str());
        addReviewOverlapTargets(edit,review);QCOMPARE(edit.fixedTargets.size(),size_t(25));
        auto error=[&](const Graph& g){double sum=0;for(const auto& t:edit.fixedTargets){const auto e=placedLocation(g,t.point)-t.uv;sum+=e.dot(e);}return std::sqrt(sum/edit.fixedTargets.size());};
        const auto fitted=fitEdit(edit,nullptr,{},FitPolicy::FiberReview);QVERIFY2(fitted.valid,fitted.message.c_str());
        QVERIFY(error(fitted.graph)<error(alone.graph)*.25);
        QCOMPARE(fitted.graph.nodes.size(),size_t(2));
        QCOMPARE(fitted.graph.nodes.at(fiber.id).uv,graph.nodes.at(fiber.id).uv);
        QCOMPARE(review.decisions.at(reference->id).alignment.nodes.at(reference->id).uv.read(),uv);
        QCOMPARE(fitted.graph.nodes.at(patch->id).source->xyz,patch->xyz);
        // Targets are derived again on reopen; do not persist reference nodes
        // in an independently editable two-node association.
        FiberReview workspace;workspace.selected=fiber.id;workspace.fibers[fiber.id]=review;
        Candidate c;c.id=patch->id;c.contacts=draft.links.read();
        auto& saved=workspace.fibers.at(fiber.id);saved.proposals={c};saved.current=0;saved.pending=fitted.graph;
        const auto restored=reviewFromJson(reviewToJson(workspace));
        auto reopened=graphEdit(prepareGraph(renderPlacement(restored.fibers.at(fiber.id).base)),*restored.fibers.at(fiber.id).pending);
        addReviewOverlapTargets(reopened,restored.fibers.at(fiber.id));
        QCOMPARE(reopened.fixedTargets.size(),edit.fixedTargets.size());
        for(size_t i=0;i<edit.fixedTargets.size();++i)QCOMPARE(reopened.fixedTargets[i].uv,edit.fixedTargets[i].uv);
        const auto repeated=fitEdit(edit,nullptr,{},FitPolicy::FiberReview);QVERIFY(repeated.valid);
        QCOMPARE(repeated.graph.nodes.at(patch->id).uv,fitted.graph.nodes.at(patch->id).uv);
        // Neither rejected/skipped decisions nor the reopened proposal itself
        // can constrain the fit. Refresh also removes stale overlap targets.
        review.decisions[patch->id]={ReviewStatus::Accepted,2,fitted.graph};
        for(auto status:{ReviewStatus::Rejected,ReviewStatus::Skipped}) {
            review.decisions.at(reference->id).status=status;addReviewOverlapTargets(edit,review);QVERIFY(edit.fixedTargets.empty());
        }
        review.decisions.at(reference->id).status=ReviewStatus::Accepted;
        auto far=std::make_shared<Node>(*reference);for(auto& p:far->xyz)p[2]+=100;
        review.decisions.at(reference->id).alignment.nodes.at(reference->id).source=far;
        addReviewOverlapTargets(edit,review);QVERIFY(edit.fixedTargets.empty());
        // Crossing sheets are not same-surface matches, even at zero distance.
        auto crossing=std::make_shared<Node>(*reference);for(auto& p:crossing->xyz){p[2]+=p[1]-100;p[1]=100;}
        review.decisions.at(reference->id).alignment.nodes.at(reference->id).source=crossing;
        addReviewOverlapTargets(edit,review);QVERIFY(edit.fixedTargets.empty());
        std::atomic_bool cancel{true};QVERIFY_EXCEPTION_THROWN(addReviewOverlapTargets(edit,review,&cancel),std::runtime_error);
    }

    void fiberReviewRefinesDistortedPlacementDespiteBadOverlap(){
        Normals normals;Fiber fiber;fiber.id="fiber";fiber.direction='H';fiber.controls={0,4};
        for(int i=0;i<5;++i)fiber.line.push_back({100.+20*i,100,100});
        Graph graph;graph.seed=fiber.id;auto node=makeFiberNode(fiber,graph.settings,normals);
        graph.nodes.emplace(node->id,Placement{node,node->initialUv});
        auto patch=plane("patch");auto uv=patch->initialUv;
        uv.back()+=cv::Vec2d{600,600}; // A previously committed spike, with no flipped triangles.
        auto draft=graph;draft.nodes.emplace(patch->id,Placement{patch,uv});
        for(int i=0;i<5;++i)draft.links.push_back({{fiber.id,{},i},{patch->id,{double(i),0}}});
        auto edit=graphEdit(prepareGraph(renderPlacement(graph)),draft);
        for(int i=1;i<4;++i)edit.fixedTargets.push_back({{patch->id,{double(i),3}},{20.*i,60}});
        edit.fixedTargets.push_back({{patch->id,{4,4}},{1000,1000}});
        const auto fit=fitEdit(edit,nullptr,{},FitPolicy::FiberReview);QVERIFY2(fit.valid,fit.message.c_str());
        const auto& placed=fit.graph.nodes.at(patch->id);
        for(const auto& t:patch->triangles)for(int i=0;i<3;++i){
            const int a=t[i],b=t[(i+1)%3];
            QVERIFY(cv::norm(placed.uv[b]-placed.uv[a])<2*cv::norm(patch->xyz[b]-patch->xyz[a]));
        }
        for(int i=1;i<4;++i)QVERIFY(cv::norm(placedLocation(fit.graph,{patch->id,{double(i),3}})-cv::Vec2d{20.*i,60})<1);
        QVERIFY(cv::norm(placed.uv.back()-cv::Vec2d{1000,1000})>1000);
        QCOMPARE(fit.graph.nodes.at(fiber.id).uv,graph.nodes.at(fiber.id).uv);
        QCOMPARE(placed.source->xyz,patch->xyz);QCOMPARE(draft.nodes.at(patch->id).uv.read(),uv);
        // Consistent manual constraints can still require an unusable chart:
        // pin two source voxels to an 80-voxel fiber span. Even with a nearly
        // disabled shape penalty, this must report distortion rather than Ready.
        auto fixed=graph;fixed.settings.stiffness=1e-12;
        auto impossible=fixed;auto stretched=patch->initialUv;
        for(auto& p:stretched)p[0]*=40;
        impossible.nodes.emplace(patch->id,Placement{patch,stretched});
        for(int i=0;i<5;++i)impossible.links.push_back({{fiber.id,{},i},{patch->id,{.025*i,0}},true});
        const auto failed=fitEdit(graphEdit(prepareGraph(renderPlacement(fixed)),impossible),nullptr,{},FitPolicy::FiberReview);
        QVERIFY(!failed.valid);
        QVERIFY2(failed.message.find("Excessive local chart distortion")!=std::string::npos,failed.message.c_str());
    }

    void fiberReviewRobustInitializationAndDiagnostics(){
        Normals normals;Fiber fiber;fiber.id="fiber";fiber.direction='H';fiber.controls={0,4};
        for(int i=0;i<5;++i)fiber.line.push_back({100.+20*i,100,100});
        Graph graph;graph.seed=fiber.id;auto node=makeFiberNode(fiber,graph.settings,normals);
        graph.nodes.emplace(node->id,Placement{node,node->initialUv});
        auto patch=plane("patch");auto uv=patch->initialUv;
        // A half-turn plus a distant translation used to stall rotation and/or
        // trip the local region boundary although the complete patch is free.
        for(auto& p:uv)p=-p+cv::Vec2d{10000,10000};
        auto draft=graph;draft.nodes.emplace(patch->id,Placement{patch,uv});
        for(int i=0;i<5;++i)draft.links.push_back({{fiber.id,{},i},{patch->id,{double(i),0}}});
        auto edit=graphEdit(prepareGraph(renderPlacement(graph)),draft);
        const auto fit=fitEdit(edit,nullptr,{},FitPolicy::FiberReview);QVERIFY2(fit.valid,fit.message.c_str());
        QVERIFY(fit.correspondenceRms<.01);QCOMPARE(fit.graph.nodes.at(fiber.id).uv,graph.nodes.at(fiber.id).uv);
        // Keep genuine user-keypoint conflicts strict, and expose their cause.
        edit.addedLinks.front().keypoint=true;
        edit.addedLinks.push_back({{fiber.id,{},4},{patch->id,{0,0}},true});
        const auto conflict=fitEdit(edit,nullptr,{},FitPolicy::FiberReview);
        QVERIFY(!conflict.valid);QVERIFY2(conflict.message.find("Keypoint constraints conflict")!=std::string::npos,conflict.message.c_str());
        QVERIFY(!conflict.needsRefit);
    }

    void fiberReviewUsesWholeFiberAndTwoProximityRadii(){
        QTemporaryDir dir;const auto root=std::filesystem::path(dir.path().toStdString());
        std::filesystem::create_directory(root/"fibers");std::filesystem::create_directory(root/"patches");
        nlohmann::json points=nlohmann::json::array();
        for(int x=100;x<=300;x+=20)points.push_back({x,120,100});
        std::ofstream(root/"fibers/a.json")<<nlohmann::json{{"type","vc3d_fiber"},{"version",1},{"line_points",points},
            {"control_points",{points.front(),points.back()}},{"branches",nlohmann::json::array()}};
        for(const auto* name:{"two","one","none"}) {
            cv::Mat_<cv::Vec3f> grid(3,11);
            for(int y=0;y<grid.rows;++y)for(int x=0;x<grid.cols;++x) {
                float distance=x==5?4:2;
                if(x==0&&std::string(name)!="none")distance=0;
                if(x==10&&std::string(name)=="two")distance=0;
                grid(y,x)={100+x*20.f,100+y*20.f,100+distance};
            }
            // The initial chart is much longer than the fiber, so many useful
            // correspondences start over 80 vx apart in the flattened view.
            QuadSurface(grid,{.01f,.05f}).save(root/"patches"/name,name);
        }
        Settings settings;settings.fiberFolder=root/"fibers";settings.patchFolder=root/"patches";
        settings.fiberScale=1;settings.radius=1.5;
        const auto catalog=loadCatalog(settings);Normals normals;
        const auto graph=seedGraph(catalog,0,settings,normals);
        const auto baseline=discover(catalog,graph,graph.seed,nullptr,{},{},1);
        QCOMPARE(std::count_if(baseline.begin(),baseline.end(),[](const auto& c){return c.id=="patch:two";}),2);
        std::vector<Candidate> published;
        const auto proposals=discover(catalog,graph,graph.seed,nullptr,{},[&](auto batch){published.insert(published.end(),batch.begin(),batch.end());},8,{},ContactPolicy::FiberReview);
        QCOMPARE(proposals.size(),size_t(1));QCOMPARE(published.size(),size_t(1));
        QCOMPARE(proposals[0].id,std::string("patch:two"));QCOMPARE(proposals[0].contacts.size(),size_t(2));
        QCOMPARE(proposals[0].contacts.front().a.lineIndex,int64_t(0));QCOMPARE(proposals[0].contacts.back().a.lineIndex,int64_t(10));
        const auto rendered=renderPlacement(graph);QVERIFY2(rendered.valid,rendered.message.c_str());
        const auto prepared=prepareGraph(rendered);
        const auto local=proposeEdit(catalog,prepared,proposals[0],false,normals);
        const auto edit=proposeEdit(catalog,prepared,proposals[0],false,normals,nullptr,ContactPolicy::FiberReview);
        QCOMPARE(local.addedLinks.size(),size_t(2));QCOMPARE(edit.addedLinks.size(),size_t(10));
        std::set<int64_t> indices;
        for(const auto& link:edit.addedLinks)indices.insert(link.a.lineIndex);
        QVERIFY(indices.contains(9));QVERIFY(!indices.contains(5));QCOMPARE(indices.size(),edit.addedLinks.size());
        const auto fitted=fitEdit(edit,nullptr,{},FitPolicy::FiberReview);QVERIFY2(fitted.valid,fitted.message.c_str());
        QCOMPARE(fitted.graph.nodes.at(graph.seed).uv,graph.nodes.at(graph.seed).uv);
        QCOMPARE(fitted.graph.links.size(),edit.addedLinks.size());QVERIFY(std::isfinite(fitted.correspondenceRms));
        QVERIFY(renderSourceLayer(fitted.graph,false).valid);
        // Eligibility still uses r, not 2r, and exclusions apply to review too.
        QVERIFY(discover(catalog,graph,graph.seed,nullptr,{},{},1,{"patch:two"},ContactPolicy::FiberReview).empty());
        auto wider=graph;wider.settings.radius=2.1;
        QCOMPARE(discover(catalog,wider,graph.seed,nullptr,{},{},1,{},ContactPolicy::FiberReview).size(),size_t(3));
        std::atomic_bool cancelled{true};
        QVERIFY_EXCEPTION_THROWN(proposeEdit(catalog,prepared,proposals[0],false,normals,&cancelled,ContactPolicy::FiberReview),std::runtime_error);
    }

    void fiberReviewRefreshesLegacyContactsWithoutLosingDecisions(){
        QTemporaryDir temp;const auto path=reviewQuarantineFixture(temp.path().toStdString());
        auto document=reviewToJson(loadReview(path));
        for(auto& fiber:document["fibers"])fiber.erase("contact_policy");
        const auto restored=reviewFromJson(document);
        QVERIFY(!restored.fibers.at("fiber:a").discovered);QVERIFY(!restored.fibers.at("fiber:b").discovered);
        QVERIFY(restored.fibers.at("fiber:a").pending); // Manual keypoint retained.
        QCOMPARE(reviewToJson(restored)["fibers"]["fiber:b"]["decisions"],document["fibers"]["fiber:b"]["decisions"]);
        // Automatic pending fits are regenerated; accepted decisions stay intact.
        auto automatic=loadReview(path);automatic.fibers.at("fiber:a").contactPolicyVersion=0;
        for(auto& link:automatic.fibers.at("fiber:a").pending->links)link.keypoint=false;
        const auto migrated=reviewFromJson(reviewToJson(automatic));
        QVERIFY(!migrated.fibers.at("fiber:a").pending);QVERIFY(!migrated.fibers.at("fiber:a").discovered);
        QCOMPARE(migrated.fibers.at("fiber:a").current,automatic.fibers.at("fiber:a").current);
    }

    void windingLayoutLoadsWithoutSourcesAndCannotRefit(){
        QTemporaryDir temp;QVERIFY(temp.isValid());
        const auto directory=std::filesystem::path(temp.path().toStdString());
        const auto binary=directory/"layout.bin",manifest=directory/"layout.workspace.json";
        auto n=plane("patch");
        std::ofstream stream(binary,std::ios::binary);
        for(size_t i=0;i<n->xyz.size();++i){
            const std::array<double,5> values{n->xyz[i][0],n->xyz[i][1],n->xyz[i][2],n->initialUv[i][0]-400,n->initialUv[i][1]+7000};
            for(double v:values){auto bits=std::bit_cast<uint64_t>(v);for(int b=0;b<8;++b)stream.put(char(bits>>(8*b)));}
        }
        stream.close();
        auto settings=graphToJson(base()).at("settings");
        nlohmann::json document={{"type","vc3d_winding_layout"},{"version",1},{"seed","patch"},
            {"geometry_format","little_endian_float64_xyzuv"},{"geometry_file","layout.bin"},
            {"geometry_bytes",std::filesystem::file_size(binary)},{"settings",settings},
            {"layout",{{"read_only",true}}},
            {"sources",nlohmann::json::array({{{"id","patch"},{"path","missing.tifxyz"},
                {"fiber",false},{"rows",5},{"cols",5},{"byte_offset",0}}})}};
        auto write=[&]{std::ofstream output(manifest);output<<document;};write();
        auto result=loadSession(manifest);QVERIFY2(result.valid,result.message.c_str());
        QCOMPARE(result.graph.nodes.at("patch").uv[0],cv::Vec2d(-400,7000));
        QVERIFY(!result.graph.windingLayout.is_null());
        QVERIFY_EXCEPTION_THROWN(fit(result.graph),std::runtime_error);
        QVERIFY_EXCEPTION_THROWN(prepareGraph(result),std::runtime_error);
        QVERIFY_EXCEPTION_THROWN(saveSession(result,directory/"saved.json"),std::runtime_error);
        // The raster uses source XYZ without a surface-refinement solve.
        const auto& grid=*result.surface->rawPointsPtr();
        for(int y=0;y<grid.rows;++y)for(int x=0;x<grid.cols;++x)if(grid(y,x)[0]!=-1){
            const auto uv=result.origin+cv::Vec2d(x*20.,y*20.);
            QCOMPARE(grid(y,x)[0],float(uv[0]+500));QCOMPARE(grid(y,x)[1],float(uv[1]-6900));
            QCOMPARE(grid(y,x)[2],100.f);
        }
        // All vertices exist, including around an explicitly unaccepted cell.
        // The importer must preserve the cell mask instead of filling its hole.
        document["version"]=2;
        document["sources"][0]["quad_runs"]={{0,5},{6,10}};write();
        auto masked=loadSession(manifest);QVERIFY2(masked.valid,masked.message.c_str());
        QCOMPARE(masked.graph.nodes.at("patch").source->triangles.size(),size_t(30));
        QVERIFY_EXCEPTION_THROWN(sourceLocation(*masked.graph.nodes.at("patch").source,{1.5,1.5}),std::runtime_error);
        // A one-cell-wide strip still has faces and remains continuous.
        document["sources"][0]["quad_runs"]={{0,1},{4,1},{8,1},{12,1}};write();
        auto strip=loadSession(manifest);QVERIFY2(strip.valid,strip.message.c_str());
        QCOMPARE(strip.graph.nodes.at("patch").source->triangles.size(),size_t(8));
        document["sources"][0]["quad_runs"]={{0,17}};write();
        QVERIFY_EXCEPTION_THROWN(loadSession(manifest),std::runtime_error);
        document["sources"][0]["quad_runs"]={{0,5},{4,1}};write();
        QVERIFY_EXCEPTION_THROWN(loadSession(manifest),std::runtime_error);
        document["version"]=1;document["sources"][0].erase("quad_runs");
        document["geometry_bytes"]=1;write();QVERIFY_EXCEPTION_THROWN(loadSession(manifest),std::runtime_error);
        document["geometry_bytes"]=std::filesystem::file_size(binary);
        document["geometry_file"]="../layout.bin";write();QVERIFY_EXCEPTION_THROWN(loadSession(manifest),std::runtime_error);
        document["geometry_file"]="layout.bin";document["sources"][0]["byte_offset"]=40;write();
        QVERIFY_EXCEPTION_THROWN(loadSession(manifest),std::runtime_error);
    }
    void fixedOverlapRetainsCoverageAndMeasuredPositions(){
        auto graph=base();graph.settings.radius=1;graph.windingLayout={{"read_only",true}};
        const auto baseline=renderPlacement(graph);QVERIFY(baseline.valid);
        auto other=plane("b");
        for(auto& xyz:other->xyz)xyz[2]+=3;
        graph.nodes.emplace("b",Placement{other,other->initialUv});
        const auto overlap=renderPlacement(graph);QVERIFY2(overlap.valid,overlap.message.c_str());
        QCOMPARE(cv::norm(*baseline.surface->rawPointsPtr(),*overlap.surface->rawPointsPtr(),cv::NORM_INF),0.0);
        QVERIFY(overlap.message.find("overlap samples resolved by source ownership")!=std::string::npos);
        // Duplicate sources cannot change either the winner or its position.
        auto duplicate=std::make_shared<Node>(*other);duplicate->id="c";
        graph.nodes.emplace("c",Placement{duplicate,duplicate->initialUv});
        const auto repeated=renderPlacement(graph);QVERIFY(repeated.valid);
        QCOMPARE(cv::norm(*overlap.surface->rawPointsPtr(),*repeated.surface->rawPointsPtr(),cv::NORM_INF),0.0);
    }
    void fixedOverlapTransitionsDoNotBridgeUnrelatedSheets(){
        auto graph=base();graph.settings.radius=1;graph.windingLayout={{"read_only",true}};
        auto other=plane("b",40);
        for(auto& xyz:other->xyz){xyz[0]+=40;xyz[2]+=3;}
        graph.nodes.emplace("b",Placement{other,other->initialUv});
        auto result=renderPlacement(graph);QVERIFY2(result.valid,result.message.c_str());
        auto quads=[](const FitResult& result){
            const auto& grid=*result.surface->rawPointsPtr();size_t count=0;
            for(int y=0;y+1<grid.rows;++y)for(int x=0;x+1<grid.cols;++x)
                count+=grid(y,x)[0]!=-1&&grid(y,x+1)[0]!=-1&&grid(y+1,x)[0]!=-1&&grid(y+1,x+1)[0]!=-1;
            return count;
        };
        QCOMPARE(quads(result),size_t(24));
        QVERIFY(result.message.find("0 discontinuous transition cells")!=std::string::npos);
        auto far=std::make_shared<Node>(*other);for(auto& xyz:far->xyz)xyz[2]+=100;
        graph.nodes.at("b").source=far;
        result=renderPlacement(graph);QVERIFY2(result.valid,result.message.c_str());
        QVERIFY(quads(result)>0&&quads(result)<24);
        const auto& grid=*result.surface->rawPointsPtr();
        for(const auto& xyz:grid)if(xyz[0]!=-1)QVERIFY(xyz[2]==100||xyz[2]==203);
    }
    void renderedPickingUsesTheVisibleOwnerAndRejectsGaps(){
        auto graph=base();graph.windingLayout={{"read_only",true}};
        auto large=plane("z");
        for(auto& xyz:large->xyz){xyz[0]=100+(xyz[0]-100)*2;xyz[1]=100+(xyz[1]-100)*2;}
        for(auto& uv:large->initialUv)uv*=2;
        graph.nodes.emplace("z",Placement{large,large->initialUv});
        const auto result=renderPlacement(graph);QVERIFY2(result.valid,result.message.c_str());
        QCOMPARE(pick(graph,{55,55})->node,std::string("a"));
        const auto visible=pickRendered(result,{55,55});QVERIFY(visible);
        QCOMPARE(visible->node,std::string("z"));
        QVERIFY(!pickRendered(result,{-100,-100}));
        QVERIFY(!pickRendered(result,{100000,100000}));
        QVERIFY(!pickRendered(result,{NAN,0}));
        QVERIFY(pickRendered(result,{160,60}));
    }
    void selectedPatchBoundaryIncludesAllItsComponents(){
        auto graph=base();auto a=std::make_shared<Node>(*graph.nodes.at("a").source);a->path="patches/shared";graph.nodes.at("a").source=a;
        auto component=plane("another-component");component->path=a->path;graph.nodes.emplace(component->id,Placement{component,component->initialUv});
        auto other=plane("other");other->path="patches/other";graph.nodes.emplace(other->id,Placement{other,other->initialUv});
        QVERIFY(vc3d::constraint_graph::showBoundary("a",graph,false,"a"));
        QVERIFY(vc3d::constraint_graph::showBoundary("another-component",graph,false,"a"));
        QVERIFY(!vc3d::constraint_graph::showBoundary("other",graph,false,"a"));
        QVERIFY(!vc3d::constraint_graph::showBoundary("a",graph,false));
    }
    void quarantineMovesWholePatchAndPersistsWithoutChangingGeometry(){
        QTemporaryDir temp;const auto path=quarantineFixture(temp.path().toStdString());
        const auto original=readFile(path),binary=readFile(path.parent_path()/"layout.bin");
        const auto before=loadSession(path);QVERIFY(before.valid);
        QCOMPARE(pickRendered(before,{30,30})->node,std::string("a:0"));
        const auto result=quarantinePatches(before,path,{"a:0"});QVERIFY(result.valid);
        QVERIFY(!result.graph.nodes.contains("a:0"));QVERIFY(!result.graph.nodes.contains("a:1"));
        QCOMPARE(result.graph.nodes.size(),size_t(2));QCOMPARE(result.graph.seed,std::string("b:0"));
        QCOMPARE(readFile(path.parent_path()/"layout.bin"),binary);
        QVERIFY(!std::filesystem::exists(path.parent_path()/"patches/a"));
        QCOMPARE(readFile(quarantineDirectory(path)/"a/marker.txt"),std::string("a"));
        QVERIFY(!std::filesystem::exists(path.string()+".quarantine-lock"));
        size_t backups=0;for(const auto& file:std::filesystem::directory_iterator(path.parent_path()))
            if(file.path().filename().string().find("before-quarantine-")!=std::string::npos){++backups;QCOMPARE(readFile(file.path()),original);}
        QCOMPARE(backups,size_t(1));
        auto reopened=loadSession(path);QVERIFY(reopened.valid);
        QCOMPARE(reopened.graph.nodes.size(),size_t(2));QCOMPARE(reopened.sessionRevision,result.sessionRevision);
        QCOMPARE(pickRendered(reopened,{30,30})->node,std::string("b:0"));
        QCOMPARE(cv::norm(*reopened.surface->rawPointsPtr(),*result.surface->rawPointsPtr(),cv::NORM_INF),0.0);
        // The old snapshot cannot publish over this newer manifest.
        QVERIFY_EXCEPTION_THROWN(quarantinePatches(before,path,{"b:0"}),std::runtime_error);
        QVERIFY(std::filesystem::exists(path.parent_path()/"patches/b"));
    }
    void quarantineFailureRestoresEarlierMovesAndPreservesManifest(){
        QTemporaryDir temp;const auto path=quarantineFixture(temp.path().toStdString());
        const auto original=readFile(path);const auto before=loadSession(path);
        QVERIFY_EXCEPTION_THROWN(quarantinePatches(before,path,{"a:0","b:0"},[&](const auto& progress){
            if(progress.starts_with("Saving the workspace")) {
                std::filesystem::create_directories(quarantineDirectory(path)/"b");
                std::ofstream(quarantineDirectory(path)/"b/existing.txt")<<"keep";
            }
        }),std::runtime_error);
        QCOMPARE(readFile(path),original);
        QVERIFY(std::filesystem::exists(path.parent_path()/"patches/a/marker.txt"));
        QVERIFY(std::filesystem::exists(path.parent_path()/"patches/b/marker.txt"));
        QVERIFY(!std::filesystem::exists(quarantineDirectory(path)/"a"));
        QCOMPARE(readFile(quarantineDirectory(path)/"b/existing.txt"),std::string("keep"));
        QVERIFY(!std::filesystem::exists(path.string()+".quarantine-lock"));
        QVERIFY_EXCEPTION_THROWN(quarantinePatches(before,path,{"a:0","b:0","c:0"}),std::runtime_error);
        QCOMPARE(readFile(path),original);
    }
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
    void reviewDirichletJacobianAndBarrier(){
        for(double orientation:{1.,-1.})for(double scale:{.3,1.,4.}) {
            const detail::ChartMetric metric{.05,-.017,.04,1.3,orientation};
            detail::ReviewChartMetricCost cost(metric);detail::ChartMetricCost original(metric);
            double p[3][2]={{3,7},{3+20*scale,7+2*scale},{8,7+orientation*30*scale}};
            const double* blocks[3]={p[0],p[1],p[2]};double r[9],reference[8],jac[3][18];double* j[3]={jac[0],jac[1],jac[2]};
            QVERIFY(cost.Evaluate(blocks,r,j));QVERIFY(original.Evaluate(blocks,reference,nullptr));
            for(int k=0;k<8;++k)QVERIFY(std::abs(r[k]-reference[k])<1e-12);
            for(int i=0;i<3;++i)for(int axis=0;axis<2;++axis){
                const auto saved=p[i][axis];double plus[9],minus[9];
                p[i][axis]=saved+1e-5;QVERIFY(cost.Evaluate(blocks,plus,nullptr));
                p[i][axis]=saved-1e-5;QVERIFY(cost.Evaluate(blocks,minus,nullptr));p[i][axis]=saved;
                for(int k=0;k<9;++k)QVERIFY(std::abs((plus[k]-minus[k])/2e-5-jac[i][2*k+axis])<1e-6*(1+std::abs(jac[i][2*k+axis])));
            }
            j[1]=nullptr;QVERIFY(cost.Evaluate(blocks,r,j));
            std::swap(blocks[1],blocks[2]);QVERIFY(!cost.Evaluate(blocks,r,j));
            blocks[1]=blocks[0];QVERIFY(!cost.Evaluate(blocks,r,nullptr));
        }
        detail::ReviewChartMetricCost identity({1,0,1,2,1});
        double a[2]={0,0},b[2]={1,0},c[2]={0,1};const double* p[3]={a,b,c};double r[9];
        QVERIFY(identity.Evaluate(p,r,nullptr));QCOMPARE(r[8],0.);
        QCOMPARE(identity.distortion(p),1.);
        b[0]=5;QCOMPARE(identity.distortion(p),5.);
        b[0]=.2;QVERIFY(std::abs(identity.distortion(p)-5)<1e-12);
        b[0]=0;QVERIFY(!std::isfinite(identity.distortion(p)));
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
        // Exclusions must bypass both cached contacts and source loading,
        // including when a previously indexed patch has been moved away.
        const auto freshCatalog=loadCatalog(settings);
        std::filesystem::rename(root/"patches/one",root/"quarantined-one");
        for(const auto* c:{&catalog,&freshCatalog})
            QVERIFY(discover(*c,graph,graph.seed,nullptr,{},{},1,{"patch:one"}).empty());
    }
    void sourceChartGroupsDoNotRequireAFusedRaster(){
        auto graph=base();
        for(const auto& [id,offset]:std::vector<std::pair<std::string,double>>{{"b",100},{"c",200}}) {
            auto node=plane(id,offset);graph.nodes.emplace(id,Placement{node,node->initialUv});
        }
        for(const auto& [a,b]:std::vector<std::pair<std::string,std::string>>{{"a","b"},{"b","c"}})
            for(const auto& xy:std::vector<cv::Vec2d>{{0,0},{4,0},{0,4},{4,4}})
                graph.links.push_back({{a,xy},{b,xy}});
        const auto before=graphToJson(graph).dump();
        for(bool rigid:{false,true}) {
            const auto result=fitSourceCharts(graph,nullptr,{},rigid);
            QVERIFY2(result.valid,result.message.c_str());QVERIFY(!result.surface);
            for(const auto& link:graph.links)
                QVERIFY(cv::norm(placedLocation(result.graph,link.a)-placedLocation(result.graph,link.b))<1e-3);
            for(const auto& [id,p]:graph.nodes)QVERIFY(result.graph.nodes.at(id).source==p.source);
        }
        QCOMPARE(graphToJson(graph).dump(),before);
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
