#include <iostream>
#include <QApplication>
#include "overlays/FiberOverlayCoordinates.hpp"
// Built only with VC_TESTING. Runs the actual controller publication path.
int runObjectStoreControllerTests()
{
    using Controller = LineAnnotationController;
    using namespace vc::objects;
    auto require = [](bool value, const char* message) { if (!value) throw std::runtime_error(message); };
    const auto path = std::filesystem::temp_directory_path() / ("vc3d-store-test-"+vc::util::newUuid()+".vcobj");
    try {
        OpenOptions options; options.author = "vc3d-controller-test";
        { auto db = ObjectStore::create(path,options); }
        CState state;
        Controller controller(&state,nullptr,nullptr,nullptr);
        controller._errorDialogsSuppressed = true;
        state.attachObjectStore(QString::fromStdString(path.string()),"vc3d-controller-test");
        Controller::StoredFiber a;
        a.id = controller.nextFiberId(); a.fileName = "same_name.json";
        a.username = "test"; a.startedAt = "20260101"; a.sequence = 1;
        a.coordinateBaseShapeZYX = std::array<size_t,3>{100,100,100};
        for (int x : {0,10,20}) { a.controlPoints.emplace_back(cv::Vec3d{x*1.,0,1}); a.linePoints.emplace_back(x*1.,0,1); }
        auto b = a; b.uuid = vc::util::newUuid(); b.id = controller.nextFiberId();
        std::vector<Controller::FiberSaveSnapshot> batch{controller.makeFiberSaveSnapshot(a),controller.makeFiberSaveSnapshot(b)};
        controller.commitFiberStoreAction(batch);
        a = batch[0].fiber; b = batch[1].fiber;
        const auto first = a.storePin;
        controller.loadFibersForCurrentPackage();
        require(controller._fibers.size()==2,"load both fibers with same name");
        require(std::none_of(controller._fibers.begin(), controller._fibers.end(), [](const auto& fiber) { return fiber.linePointsLoaded; }), "catalog does not materialize dense lines");
        (void)controller.fiberSummaries(); (void)controller.fiberSnapshots(false);
        require(std::none_of(controller._fibers.begin(), controller._fibers.end(), [](const auto& fiber) { return fiber.linePointsLoaded; }), "catalog and control overlays keep line geometry lazy");
        controller.loadFiberDetails(controller._fibers[0].id);
        require(controller._fibers[0].linePointsLoaded && !controller._fibers[1].linePointsLoaded, "expanding a fiber loads only its dense geometry");
        const auto details = controller.fiberSummaries();
        const auto detail = std::find_if(details.begin(),details.end(),[&](const auto& summary) { return summary.id == controller._fibers[0].id; });
        require(detail != details.end() && detail->spans.size() == 2 && detail->annotatedLengthVx == 20.0, "on-demand spans retain exact lengths and control ranges");
        require(controller._fibers[0].id != controller._fibers[1].id,"UUID runtime identity");
        auto resolve = [&](const std::string& uuid) {
            const auto it = std::find_if(controller._fibers.begin(),controller._fibers.end(),[&](const auto& f) { return f.uuid==uuid; });
            if (it == controller._fibers.end()) throw std::runtime_error("fiber missing"); controller.ensureFiberLines(*it); return *it;
        };
        a=resolve(a.uuid);b=resolve(b.uuid);
        Controller::FiberBranchRef link;
        link.controlPointIndex=0;link.branchControlPointIndex=0;link.branchFiberId=b.id;link.branchFileName=b.fileName;
        link.controlPointPosition=a.controlPoints[0];link.branchControlPointPosition=b.controlPoints[0];
        link.controlPointDirection={1,0,0};link.branchControlPointDirection={1,0,0};
        a.branches.push_back(link);
        std::swap(link.controlPointPosition,link.branchControlPointPosition); link.branchFiberId=a.id;link.branchFileName=a.fileName;
        b.branches.push_back(link);
        batch={controller.makeFiberSaveSnapshot(a),controller.makeFiberSaveSnapshot(b)};
        controller.commitFiberStoreAction(batch);
        controller.loadFibersForCurrentPackage();a=resolve(a.uuid);b=resolve(b.uuid);
        require(a.branches.size()==1 && b.branches.size()==1,"shared branch roundtrip");
        const auto transferPaths = controller.fiberTransportPaths({a.id});
        require(transferPaths.size() == 2, "transport includes both linked endpoints");
        auto readDocument = [](const auto& file) { std::ifstream input(file); return nlohmann::json::parse(input); };
        const auto aWire = readDocument(transferPaths.at(a.uuid));
        const auto bWire = readDocument(transferPaths.at(b.uuid));
        require(aWire["branches"][0]["branch_file"] == b.uuid + ".json", "transport serializes branch target UUID");
        require(bWire["branches"][0]["branch_file"] == a.uuid + ".json", "transport serializes reciprocal branch");
        require(aWire["name"] == bWire["name"] && aWire["filename"] != bWire["filename"], "same-name fibers have distinct transport identities");
        {
            QTemporaryDir transfers;
            const auto bundle = fs::path(transfers.path().toStdString()) / "network.json";
            QString error; int exported = 0;
            require(controller.exportFibersToPath(bundle, &error, &exported) && exported == 2, "export linked same-name fibers");
            const auto target = fs::path(transfers.path().toStdString()) / "target.vcobj";
            { auto db = ObjectStore::create(target, options); }
            CState destination;
            Controller imported(&destination, nullptr, nullptr, nullptr);
            imported._errorDialogsSuppressed = true;
            destination.attachObjectStore(QString::fromStdString(target.string()), "roundtrip-test");
            int count = 0;
            if (!imported.importFibersFromPath(bundle, &error, &count)) throw std::runtime_error(error.toStdString());
            require(count == 2 && imported._fibers.size() == 2, "duplicate display names import by UUID");
            for (const auto& fiber : imported._fibers) {
                require(fiber.fileName == "same_name.json", "display name survives transfer");
                require(fiber.branches.size() == 1, "shared branch survives transfer");
                require(fiber.branches.front().edgeUuid == a.branches.front().edgeUuid, "edge UUID survives transfer");
            }
        }
        struct OverlayChain { uint64_t id; std::vector<cv::Vec3f> points; std::vector<int> links; };
        const std::vector<OverlayChain> chains{{1, {{8,16,24}}, {2}}, {2, {{8,16,24}}, {1}}, {3, {{8,16,24}}, {}}};
        const auto visible = vc3d::boundFiberChains(chains, [](uint64_t id) -> std::optional<double> {
            if (id == 1) return .25;
            if (id == 2) return .5;
            return {};
        });
        require(visible.size() == 2 && visible[0].points[0] == cv::Vec3f(2,4,6) &&
                visible[1].points[0] == cv::Vec3f(4,8,12), "each overlay uses its own binding and excludes unbound fibers");
        require(visible[0].id == 1 && visible[0].links == chains[0].links, "transformed hit-test identity and link markers remain aligned");
        std::vector<std::string> retiredNotifications;
        QObject::connect(&controller, &Controller::fiberRetired, &controller,
            [&](const QString& uuid) { retiredNotifications.push_back(uuid.toStdString()); });
        const auto edge=a.branches.front().edgeUuid;
        a.controlPoints[0][0]=-1; a.linePoints[0][0]=-1;
        batch={controller.makeFiberSaveSnapshot(a)};
        controller.canonicalizeFiberSaveSnapshots(batch);controller.commitFiberStoreAction(batch);
        controller.loadFibersForCurrentPackage();a=resolve(a.uuid);
        require(a.controlPoints.front()[0]==-1,"sparse move persists");
        require(a.branches.front().edgeUuid==edge,"branch identity survives move");
        const auto id=a.controlPoints[1].storeId;
        a.controlPoints.erase(a.controlPoints.begin()+1); a.linePoints.erase(a.linePoints.begin()+1);
        batch={controller.makeFiberSaveSnapshot(a)};
        controller.commitFiberStoreAction(batch);
        controller.loadFibersForCurrentPackage();a=resolve(a.uuid);
        require(a.controlPoints.size()==2 && a.controlPoints.back().storeId==2,"structural deletion preserves control identities");
        a.controlPoints.insert(a.controlPoints.begin()+1,vc3d::line_annotation::StoredControlPoint{cv::Vec3d{10,0,1}});
        a.linePoints.insert(a.linePoints.begin()+1,cv::Vec3d{10,0,1});
        batch={controller.makeFiberSaveSnapshot(a)};controller.commitFiberStoreAction(batch);
        controller.loadFibersForCurrentPackage();a=resolve(a.uuid);
        require(a.controlPoints[1].storeId!=id,"deleted point identity is not reused");
        a.tags={"reviewed"}; batch={controller.makeFiberSaveSnapshot(a)};controller.commitFiberStoreAction(batch);
        a=batch.front().fiber;
        controller.scheduleFiberSave(a);
        controller.scheduleFiberSave(a);
        batch={controller.makeFiberSaveSnapshot(a)};
        const auto stale=batch;
        controller.loadFibersForCurrentPackage();a=resolve(a.uuid);
        { auto db=ObjectStore::open(path,options);auto txn=db.write({{a.uuid,a.storePin->objectRevision}});db.setMeta(txn,a.uuid,"external","true");txn.commit(); }
        bool conflict=false;try { auto old=stale;controller.commitFiberStoreAction(old); } catch(...) { conflict=true; }
        require(conflict,"stale controller edits reject");
        controller.loadFibersForCurrentPackage();a=resolve(a.uuid);
        auto outcome=controller.deleteFibers({a.id});require(outcome.deletedIds.size()==1,"retire through controller");
        require(retiredNotifications == std::vector<std::string>{a.uuid}, "retirement publishes object UUID");
        {
        auto db=ObjectStore::open(path,options);auto txn=db.read();
        require(db.objectByUuid(txn,a.uuid)->retired,"retirement persisted");
        require(db.objectRevision(txn,first->objectRevision).has_value(),"historical revision retained");
        require(db.edgesByKind(txn,"fiber_branch").empty(),"retirement and branch removal atomic");
        }

        namespace la=vc3d::line_annotation;
        Controller::StoredFiber parent;
        parent.id=controller.nextFiberId(); parent.fileName="parent.json";
        parent.username="test"; parent.startedAt="20260101"; parent.sequence=3;
        parent.coordinateBaseShapeZYX=std::array<size_t,3>{100,100,100};
        for(int x:{30,40,50,60}) {parent.controlPoints.emplace_back(cv::Vec3d{x*1.,0,1});parent.linePoints.emplace_back(x*1.,0,1);}
        batch={controller.makeFiberSaveSnapshot(parent)};
        const auto frame=nlohmann::json{{"volume_uuid",vc::util::newUuid()},{"scale_factor",4.0}};
        batch.front().coordinateIdentity["coordinates"]=frame;
        controller.commitFiberStoreAction(batch);parent=batch.front().fiber;
        controller.loadFibersForCurrentPackage();parent=resolve(parent.uuid);
        require(controller.fiberToJson(parent).at("coordinates") == frame, "current transfer format preserves explicit coordinate identity");
        require(controller.fiberToJson(parent).at("version") == 4, "current transfer format is independent of source ingestion version");
        const auto identity=[&](const char* name) {la::NewFiberIdentity result;result.id=controller.nextFiberId();result.username="test";result.startedAt="20260101";result.sequence=result.id;result.fileName=name;return result;};
        std::string error;
        const auto split=la::planFiberSplit(parent,1,identity("prefix.json"),identity("suffix.json"),true,&error);
        if(!split) throw std::runtime_error(error);
        Controller::StructuralEditRequest splitRequest;splitRequest.verb="split";splitRequest.newFibers={split->prefix,split->suffix};splitRequest.originals={parent};
        bool committed=false;
        auto splitResult=controller.commitStructuralEdit(std::move(splitRequest),&committed);
        if(!splitResult.ok) throw std::runtime_error("split: "+splitResult.error);
        require(committed,"split committed");
        controller.loadFibersForCurrentPackage();
        auto prefix=resolve(split->prefix.uuid),suffix=resolve(split->suffix.uuid);
        require(prefix.controlPoints.front().storeId==parent.controlPoints.front().storeId,"split retains control identity");
        require(prefix.branches.size()==1 && suffix.branches.size()==1,"split shared branch persists");
        {auto db=ObjectStore::open(path,options);auto txn=db.read();require(nlohmann::json::parse(db.objectByUuid(txn,prefix.uuid)->meta)["coordinates"]==frame,"split inherits explicit frame");}
        const auto merge=la::planFiberMerge(prefix,int(prefix.controlPoints.size()-1),suffix,0,parent.optimizationMode,identity("merged.json"),&error);
        if(!merge) throw std::runtime_error(error);
        Controller::StructuralEditRequest mergeRequest;mergeRequest.verb="merge";mergeRequest.newFibers={merge->merged};mergeRequest.originals={prefix,suffix};
        auto mergeResult=controller.commitStructuralEdit(std::move(mergeRequest),&committed);
        if(!mergeResult.ok) throw std::runtime_error("merge: "+mergeResult.error);
        controller.loadFibersForCurrentPackage();const auto merged=resolve(merge->merged.uuid);
        require(merged.controlPoints.size()==parent.controlPoints.size(),"merge persists controls");
        {
            auto live=std::make_shared<Controller::LineAnnotationSession>();
            live->surfaceName="store-test-interaction";live->fiberId=merged.id;live->fiberUuid=merged.uuid;
            live->fiberFileName=merged.fileName;live->fiberUsername=merged.username;
            live->fiberStartedAt=merged.startedAt;live->fiberSequence=merged.sequence;
            live->storePin=merged.storePin;live->coordinateBaseShapeZYX=merged.coordinateBaseShapeZYX;
            live->fiberOptimizationMode=merged.optimizationMode;live->suppressGeneratedViews=true;live->suppressErrorDialogs=true;
            for(size_t i=0;i<merged.controlPoints.size();++i) {
                la::LineControlPoint control(double(i),merged.controlPoints[i],i==0,int(i));
                control.storeId=merged.controlPoints[i].storeId;control.segmentToNext=merged.controlPoints[i].segmentToNext;
                live->controlPoints.push_back(control);
            }
            for(const auto& point:merged.linePoints) {vc::lasagna::LinePoint line;line.position=point;live->optimizedLine.points.push_back(line);}
            Controller::PaneRecord pane;pane.surfaceName=live->surfaceName;pane.session=live;controller._panes.push_back(pane);
            live->controlPoints[0].volumePoint[0]-=1;live->optimizedLine.points[0].position[0]-=1;
            const auto failures=controller._fiberSaveFailureCount;
            controller.saveSessionAsFiber(*live,false);
            require(controller._fiberSaveFailureCount==failures,"completed manual interaction commits without solver");
            require(live->storePin->objectRevision!=merged.storePin->objectRevision,"interaction advances pin immediately");
            const auto savedPin=live->storePin;
            controller.saveOpenFibers();
            require(live->storePin==savedPin,"Save barrier does not create another revision");
            controller._panes.clear();
            auto db=ObjectStore::open(path,options);auto txn=db.read();
            const auto saved=std::get<vc::geometry::Fiber>(vc::geometry::decode(db.readGeometryData(txn,*savedPin->geometryRevision)));
            require(saved.controlPoints[0][0]==live->controlPoints[0].volumePoint[0],"manual edit survives independent read");
        }
        const auto artifact=controller.fiberTransportPaths({merged.id}).at(merged.uuid);
        require(std::filesystem::exists(artifact) && controller.fiberIdForFilePath(artifact)==merged.id,"pinned Spiral transport retains UUID");
        {
            CState child(nullptr,false,&state);
            require(child.objectStoreSession()==state.objectStoreSession(),"secondary workspaces share session");
        }
        require(state.objectStoreSession()->attachment().has_value(),"secondary destruction keeps main attachment");
        state.detachObjectStore();
        require(controller._fibers.empty(),"detach clears views");
        {
            // Store-valid control-only geometry must not empty the editor catalog.
            auto db = ObjectStore::open(path, options);
            ObjectRecord object; object.uuid = vc::util::newUuid(); object.kind = "fiber"; object.name = "controls-only";
            auto txn = db.write({{object.uuid,{}}});
            object.formatVersion = 1;
            db.upsertObject(txn, object);
            db.putGeometry(txn, object.uuid, vc::geometry::encode(vc::geometry::Fiber{{{0,0,0},{1,0,0}}, {}}));
            txn.commit();
            ObjectRecord missing; missing.uuid = vc::util::newUuid(); missing.kind = "fiber"; missing.name = "geometry-pending";
            { auto pending = db.write({{missing.uuid,{}}}); db.upsertObject(pending,missing); pending.commit(); }
            state.attachObjectStore(QString::fromStdString(path.string()), {}, true);
            const auto unavailable = std::find_if(controller._fibers.begin(),controller._fibers.end(),[&](const auto& fiber) { return fiber.uuid == missing.uuid; });
            require(unavailable != controller._fibers.end() && !unavailable->editorError.empty(), "unavailable geometry stays in catalog without blocking other fibers");
            require(controller._fibers.size() >= 2, "control-only record preserves the full catalog");
            const auto controlsOnly = resolve(object.uuid);
            require(controlsOnly.controlPoints.size() == 2 && controlsOnly.linePoints.empty(), "control-only geometry preserved without synthesizing lines");
            require(controller.fiberSummaries().size() == controller._fibers.size(), "read-only store fibers remain listed");
            auto pending = std::vector<Controller::FiberSaveSnapshot>{controller.makeFiberSaveSnapshot(controlsOnly)};
            bool rejected = false; try { controller.commitFiberStoreAction(pending); } catch (...) { rejected = true; }
            require(rejected, "read-only controller cannot publish changes");
            auto transfer = controller.fiberToJson(controlsOnly); transfer["version"] = 1;
            rejected = false; try { (void)controller.loadFiberJson(transfer, "legacy.json"); } catch (...) { rejected = true; }
            require(rejected, "legacy file import is rejected without repair");
            state.detachObjectStore();
        }
        if(const char* smoke=std::getenv("VC3D_OBJECT_STORE_SMOKE_DB")) {
            state.attachObjectStore(QString::fromUtf8(smoke),"vc3d-real-smoke");
            require(!controller._fibers.empty(),"real database catalog loads");
            const size_t count=controller._fibers.size();
            auto fiber=resolve(controller._fibers.front().uuid);
            if(const char* ct=std::getenv("VC3D_OBJECT_STORE_SMOKE_CT")) {
                state.setCurrentVolume(Volume::New(ct));
                require(!controller.fiberCoordinateScale(fiber.id,false),"unbound real fiber requires explicit assignment");
                QTimer::singleShot(0,[] {
                    for(auto* widget:QApplication::topLevelWidgets()) if(auto* input=qobject_cast<QInputDialog*>(widget)) {
                        input->setTextValue("4");input->accept();
                    }
                });
                controller._errorDialogsSuppressed=false;
                const auto factor=controller.fiberCoordinateScale(fiber.id,true);
                controller._errorDialogsSuppressed=true;
                require(factor && *factor==0.25,"explicit scale-2 binding persists");
                fiber=resolve(fiber.uuid);
                const auto display = controller.fiberInVolumeCoordinates(fiber);
                require(display.controlPoints.front() == fiber.controlPoints.front() * .25 && display.linePoints.front() == fiber.linePoints.front() * .25, "display geometry uses explicit binding");
                require(resolve(fiber.uuid).linePoints == fiber.linePoints, "display transform does not change stored geometry");
                const auto point=fiber.controlPoints.front()*(*factor);
                Array3D<uint8_t> crop({2,2,2});
                require(state.currentVolume()->readZYX(crop,{int(point[2]),int(point[1]),int(point[0])}),"read real CT at scale 2");
                std::cerr << "CT smoke: " << ct << " scale 2, voxel size " << state.currentVolume()->voxelSize() << " um, sampled XYZ " << point << "\n";
            }
            const auto originalPin=fiber.storePin;
            fiber.tags.push_back("vc3d_smoke");
            controller.scheduleFiberSave(fiber);
            state.detachObjectStore();state.attachObjectStore(QString::fromUtf8(smoke),"vc3d-real-smoke");
            require(controller._fibers.size()==count,"real database reopens");
            const auto restored=resolve(fiber.uuid);
            require(std::find(restored.tags.begin(),restored.tags.end(),"vc3d_smoke")!=restored.tags.end(),"real database edit persists");
            require(restored.storePin->objectRevision!=originalPin->objectRevision,"real database revision advances");
            auto realSplit=la::planFiberSplit(restored,restored.controlPoints.size()/2,
                identity("real-prefix.json"),identity("real-suffix.json"),true,&error);
            if(!realSplit) throw std::runtime_error("real split plan: "+error);
            Controller::StructuralEditRequest realRequest;realRequest.verb="split";
            realRequest.newFibers={realSplit->prefix,realSplit->suffix};realRequest.originals={restored};
            realRequest.sources.push_back({restored.id,restored.fileName,nullptr,[&](int index)->std::optional<la::RedirectTarget> {
                const auto mapped=la::remappedSplitControlPointIndex(realSplit->plan,index);
                if(!mapped)return std::nullopt;
                const auto& half=mapped->first ? realSplit->suffix : realSplit->prefix;
                return la::RedirectTarget{half.id,half.fileName,mapped->second};
            }});
            const auto realSplitResult=controller.commitStructuralEdit(std::move(realRequest),&committed);
            if(!realSplitResult.ok) throw std::runtime_error("real split: "+realSplitResult.error);
            controller.loadFibersForCurrentPackage();
            auto realPrefix=resolve(realSplit->prefix.uuid),realSuffix=resolve(realSplit->suffix.uuid);
            auto realMerge=la::planFiberMerge(realPrefix,int(realPrefix.controlPoints.size()-1),realSuffix,0,
                restored.optimizationMode,identity("real-merged.json"),&error);
            if(!realMerge) throw std::runtime_error("real merge plan: "+error);
            realRequest={};realRequest.verb="merge";realRequest.newFibers={realMerge->merged};realRequest.originals={realPrefix,realSuffix};
            realRequest.sources.push_back({realPrefix.id,realPrefix.fileName,nullptr,[&](int index) {
                return la::RedirectTarget{realMerge->merged.id,realMerge->merged.fileName,realMerge->clickedRemap(index)};
            }});
            realRequest.sources.push_back({realSuffix.id,realSuffix.fileName,nullptr,[&](int index) {
                return la::RedirectTarget{realMerge->merged.id,realMerge->merged.fileName,realMerge->farRemap(index)};
            }});
            const auto realMergeResult=controller.commitStructuralEdit(std::move(realRequest),&committed);
            if(!realMergeResult.ok) throw std::runtime_error("real merge: "+realMergeResult.error);
            state.detachObjectStore();state.attachObjectStore(QString::fromUtf8(smoke),"vc3d-real-smoke");
            const auto roundtrip=resolve(realMerge->merged.uuid);
            require(roundtrip.controlPoints.size()==restored.controlPoints.size(),"real split/merge control count");
            require(roundtrip.linePoints==realMerge->merged.linePoints,"real split/merge preserves planner numerical geometry");
            require(roundtrip.branches.size()==restored.branches.size(),"real split/merge preserves external branches");
            std::cerr << "real-data smoke: " << count << " fibers; edit, shared link, split/merge and reopen; "
                      << roundtrip.controlPoints.size() << " controls and " << roundtrip.linePoints.size() << " planned line points preserved\n";
            state.detachObjectStore();
        }
        std::cerr << "vc3d object store controller tests passed\n";
    } catch(const std::exception& error) { std::cerr << "vc3d object store test: " << error.what() << '\n'; return 1; }
    std::filesystem::remove(path);
    return 0;
}
