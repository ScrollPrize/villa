#include <iostream>
#include <QApplication>
#include <QElapsedTimer>
#include <QMainWindow>
#include <QListWidget>
#include <QSpinBox>
#include <QThread>
#include "FiberCollectionController.hpp"
#include "SpiralWorkspace.hpp"
#include "SpiralServiceManager.hpp"
#include "overlays/FiberOverlayCoordinates.hpp"
// Built only with VC_TESTING. Runs the actual controller publication path.
int runObjectStoreControllerTests()
{
    using Controller = LineAnnotationController;
    using namespace vc::objects;
    auto require = [](bool value, const char* message) { if (!value) throw std::runtime_error(message); };
    const auto path = std::filesystem::temp_directory_path() / ("vc3d-store-test-"+vc::util::newUuid()+".vcobj");
    try {
        if (const char* source = std::getenv("VC3D_OBJECT_STORE_READ_ONLY_BENCH")) {
            CState state;
            Controller controller(&state,nullptr,nullptr,nullptr);
            QMainWindow window;
            FiberCollectionController collection(&state,nullptr,&window,&controller);
            auto* list = window.findChild<QListWidget*>("fiberCollectionList");
            auto* limit = window.findChild<QSpinBox*>("fiberCollectionMaxDisplayed");
            require(list && limit, "fiber catalog controls exist");
            require(limit->value() == 5000, "clean settings default to 5000 displayed fibers");
            int heartbeats = 0;
            QTimer heartbeat; heartbeat.setInterval(10);
            QObject::connect(&heartbeat, &QTimer::timeout, [&] { ++heartbeats; });
            heartbeat.start();
            QElapsedTimer timer; timer.start();
            state.attachObjectStore(QString::fromUtf8(source), {}, true);
            const auto attachedMs = timer.elapsed();
            require(controller._fibers.empty(), "attachment loads no editor geometry");
            while (list->count() == 0 && timer.elapsed() < 60000) {
                QApplication::processEvents(); QThread::msleep(1);
            }
            require(list->count() > 0 && list->count() <= 200, "first catalog page is bounded");
            require(controller._fibers.empty(), "catalog page loads no editor geometry");
            require(heartbeats > 0, "UI heartbeat runs while catalog loads");
            const auto pageMs = timer.elapsed();
            const auto id = list->item(0)->data(Qt::UserRole).toULongLong();
            controller.ensureStoreFiber(id);
            require(!controller._fibers.empty() && !controller._fiberCatalogComplete,
                    "selection loads resident editor fibers without loading the full catalog");
            const auto selected = std::find_if(controller._fibers.begin(),controller._fibers.end(),
                [&](const auto& fiber) { return fiber.id == id; });
            require(selected != controller._fibers.end() && selected->editorError.empty(), "selected controls decode");
            require(!selected->linePointsLoaded, "selection controls do not eagerly decode sampled line");
            controller.ensureFiberLines(*selected);
            const auto residentCount = controller._fibers.size();
            controller.ensureAllFiberLines();
            require(controller._fibers.size() == residentCount && !controller._fiberCatalogComplete,
                    "editor helpers do not promote resident reads to a full-store load");
            require(selected->linePointsLoaded, "selected line decodes on demand");
            std::cout << "read-only attach_ms=" << attachedMs << " first_page_ms=" << pageMs
                      << " selected_ms=" << timer.elapsed()-pageMs << " cached_editors=" << controller._fibers.size()
                      << " heartbeats=" << heartbeats << std::endl;
            state.detachObjectStore();
            require(controller._fibers.empty() && controller._registeredStoreFibers.empty(), "detach clears resident identities");
            return 0;
        }
        {
            QTemporaryDir volumeDir;
            Volume::ZarrCreateOptions create; create.shapeZYX = {4,4,4};
            create.chunkShapeZYX = {4,4,4}; create.compressor = "none"; create.uuid = "20260411134726";
            const auto volume = Volume::New(fs::path(volumeDir.path().toStdString())/"ct.zarr", create);
            const auto binding = vc3d::coordinateBindingForVolume(*volume, 4);
            require(binding.at("coordinates").at("volume_uuid") == "20260411134726", "VC3D retains catalog volume ID verbatim");
            require(vc3d::geometryScaleForVolume(binding, *volume) == .25, "VC3D matches the authoritative volume ID");
        }
        OpenOptions options; options.author = "vc3d-controller-test";
        { auto db = ObjectStore::create(path,options); }
        CState state;
        Controller controller(&state,nullptr,nullptr,nullptr);
        controller._errorDialogsSuppressed = true;
        state.attachObjectStore(QString::fromStdString(path.string()),"vc3d-controller-test");
        auto transferJson = [&](const Controller::StoredFiber& fiber) {
            nlohmann::json result;
            state.objectStoreSession()->read(fiber.storePin->attachment, [&](const auto& db, const auto& txn, const auto&) {
                result = nlohmann::json::parse(fiberRevisionJson(db, txn, fiber.storePin->objectRevision));
            });
            return result;
        };
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
            controller.loadFibersForCurrentPackage(std::vector<std::string>{});
            require(controller._fibers.empty(), "export starts with no resident editor geometry");
            QString error; int exported = 0;
            require(controller.exportFibersToPath(bundle, &error, &exported) && exported == 2, "export linked same-name fibers");
            require(controller._fibers.empty(), "export does not populate editor geometry");
            controller.loadFibersForCurrentPackage();
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
        {
            QTemporaryDir directory;
            const auto target = fs::path(directory.path().toStdString()) / "service.vcobj";
            { auto db = ObjectStore::create(target, options); }
            CState destination;
            Controller imported(&destination, nullptr, nullptr, nullptr);
            imported._errorDialogsSuppressed = true;
            destination.attachObjectStore(QString::fromStdString(target.string()), "service-import-test");
            SpiralWorkspace workspace(&destination);
            workspace.setLineAnnotationController(&imported);
            auto* service = workspace.findChild<SpiralServiceManager*>();
            service->inputEditorRequested(QJsonObject{{"kind", "fiber"}, {"id", QString::fromStdString(a.uuid)}},
                QString::fromStdString(transferPaths.at(a.uuid).string()));
            require(imported._fibers.size() == 2 && service->inputDraftStatus().size() == 2,
                    "service-only component is imported into attached store before editing");
            auto document = aWire;
            const auto uuid = vc::util::newUuid(); document["uuid"] = uuid;
            document["branches"] = nlohmann::json::array(); document["adjacent_branches"] = nlohmann::json::array();
            document.erase("optimization_mode"); document["custom_annotation"] = {{"note", "retain"}};
            const auto volumeUuid = vc::util::newUuid();
            document["coordinates"] = {{"volume_uuid", volumeUuid}, {"scale_factor", 4}};
            document["coordinate_binding"] = {{"active_volume_id", "display"}};
            QString error;
            const auto id = imported.importFiberJson(document, "optional-mode.json", &error);
            if (!id) throw std::runtime_error(error.toStdString());
            auto found = std::find_if(imported._fibers.begin(), imported._fibers.end(), [&](const auto& fiber) { return fiber.id == id; });
            require(found != imported._fibers.end() && !found->readOnly && found->controlPoints.size() == 3,
                    "absent optional optimization mode keeps geometry editable");
            const auto bundle = fs::path(directory.path().toStdString()) / "export.json";
            require(imported.exportFibersToPath(bundle, &error), "shared UI exporter accepts absent mode");
            const auto exportedDocument = readDocument(bundle);
            for (const auto& entry : exportedDocument.at("point_collections")) if (entry.at("uuid") == uuid) {
                require(!entry.contains("optimization_mode"), "export does not invent absent mode");
                require(entry.at("custom_annotation") == document.at("custom_annotation"), "UI export preserves unknown source annotations");
                require(entry.at("coordinates") == document.at("coordinates"), "UI export preserves authoritative coordinates");
                require(entry.at("coordinate_binding") == document.at("coordinate_binding"), "binding travels with coordinates");
            }
            auto edited = *found; edited.tags.push_back("edited"); imported.commitFibers({&edited});
            const auto paths = imported.fiberTransportPaths({id});
            require(readDocument(paths.at(uuid)).at("custom_annotation") == document.at("custom_annotation"),
                    "editing preserves unknown source annotations");
        }
        {
            SpiralWorkspace workspace(&state);
            workspace.setLineAnnotationController(&controller);
            auto* service = workspace.findChild<SpiralServiceManager*>();
            require(service, "Spiral service exists");
            require(workspace.stageFiberInputs({a.id}), "stage a newly saved fiber");
            auto rows = service->inputDraftStatus();
            require(rows.size() == 2, "staging includes linked fibers");
            for (const auto& row : rows)
                require(row.toObject()["applied_revision"].toInteger() == 0, "staging does not apply to fit");
            workspace.refreshTrackedFibers();
            require(service->inputDraftStatus() == rows, "unchanged fibers are not restaged");
            // Reopening a service input registers it even in a fresh workspace.
            SpiralWorkspace reopened(&state);
            reopened.setLineAnnotationController(&controller);
            auto* reopenedService = reopened.findChild<SpiralServiceManager*>();
            reopenedService->inputEditorRequested(QJsonObject{{"kind", "fiber"}, {"id", QString::fromStdString(a.uuid)}},
                QString::fromStdString(transferPaths.at(a.uuid).string()));
            require(reopenedService->inputDraftStatus().size() == 2, "reopened service fibers are tracked");
            auto changed = a; changed.tags.push_back("spiral-draft-edit");
            controller.commitFibers({&changed});
            workspace.refreshTrackedFibers(); reopened.refreshTrackedFibers();
            require(service->inputDraftStatus() != rows, "saved edits refresh tracked drafts");
            for (const auto& row : service->inputDraftStatus())
                require(row.toObject()["applied_revision"].toInteger() == 0, "refresh does not auto-apply");
        }
        a = resolve(a.uuid); b = resolve(b.uuid);
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
        require(controller.generatedBranchLinkMarkers(a.branches, 1.0).empty(), "unbound peers have no branch markers");
        {
            auto changedSession = std::make_shared<Controller::LineAnnotationSession>();
            changedSession->fiberId = a.id; changedSession->fiberUuid = a.uuid; changedSession->storePin = a.storePin;
            auto untouched = std::make_shared<Controller::LineAnnotationSession>();
            untouched->fiberId = b.id; untouched->fiberUuid = b.uuid; untouched->storePin = b.storePin;
            untouched->fiberTags = {"unsaved-work"};
            untouched->runningSolveCancel = std::make_shared<std::atomic_bool>(false);
            Controller::PaneRecord changedPane, keptPane;
            changedPane.surfaceName = "changed-test-pane"; changedPane.session = changedSession;
            keptPane.surfaceName = "kept-test-pane"; keptPane.session = untouched;
            controller._panes.push_back(changedPane); controller._panes.push_back(keptPane);
            { auto db = ObjectStore::open(path, options); auto txn = db.write({{a.uuid, a.storePin->objectRevision}});
              db.setMeta(txn, a.uuid, "external_note", "true"); txn.commit(); }
            controller.refreshChangedStoreFibers();
            require(controller._panes.size() == 1 && controller._panes.front().session == untouched,
                    "refresh removes only changed panes");
            require(untouched->fiberTags == std::vector<std::string>{"unsaved-work"} && !untouched->runningSolveCancel->load(),
                    "unrelated pending edits and solver are preserved");
            controller.discardFiberPanes({b.id});
            a = resolve(a.uuid); b = resolve(b.uuid);
        }
        std::vector<std::string> retiredNotifications;
        QObject::connect(&controller, &Controller::fiberRetired, &controller,
            [&](const QString& uuid) { retiredNotifications.push_back(uuid.toStdString()); });
        {
            auto invalid = a;
            invalid.controlPoints.front()[0] -= 2;
            invalid.linePoints.front()[0] -= 2;
            invalid.branches.front().controlPointIndex = 999;
            auto failed = std::vector<Controller::FiberSaveSnapshot>{controller.makeFiberSaveSnapshot(invalid)};
            bool rejected = false;
            try { controller.commitFiberStoreAction(failed); } catch (...) { rejected = true; }
            require(rejected, "invalid relationship rejects after geometry edits start");
            require(failed.front().fiber.storePin->objectRevision == invalid.storePin->objectRevision, "failed transaction leaves working pins unchanged");
            auto db = ObjectStore::open(path, options); auto txn = db.read();
            require(db.currentObjectRevision(txn, a.uuid)->uuid == a.storePin->objectRevision,
                    "failed relationship rolls back geometry and metadata");
        }
        const auto edge=a.branches.front().edgeUuid;
        a.controlPoints[0][0]=-1; a.linePoints[0][0]=-1;
        batch={controller.makeFiberSaveSnapshot(a)};
        vc3d::line_annotation::canonicalizeFiberSaveSnapshots(batch);controller.commitFiberStoreAction(batch);
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
        controller.commitFibers({&a});
        controller.commitFibers({&a});
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
        {
            Controller::StoredFiber firstSeed;
            firstSeed.id = controller.nextFiberId(); firstSeed.fileName = "duplicate-seed.json";
            firstSeed.controlPoints.emplace_back(cv::Vec3d{1,2,3});
            firstSeed.tags = {la::kNeedsReoptimizationTag};
            auto secondSeed = firstSeed;
            secondSeed.id = controller.nextFiberId(); secondSeed.uuid = vc::util::newUuid();
            std::vector<Controller::FiberSaveSnapshot> seeds{
                controller.makeFiberSaveSnapshot(firstSeed), controller.makeFiberSaveSnapshot(secondSeed)};
            controller.commitFiberStoreAction(seeds);
            controller.loadFibersForCurrentPackage();
            controller.reoptimizeMergedFibers({secondSeed.id});
            controller.loadFibersForCurrentPackage();
            require(resolve(firstSeed.uuid).tags == firstSeed.tags, "same-name non-target keeps reoptimization tag");
            require(resolve(secondSeed.uuid).tags.empty(), "reoptimization selects exact runtime identity");
        }

        Controller::StoredFiber parent;
        parent.id=controller.nextFiberId(); parent.fileName="parent.json";
        parent.username="test"; parent.startedAt="20260101"; parent.sequence=3;
        parent.coordinateBaseShapeZYX=std::array<size_t,3>{100,100,100};
        for(int x:{30,40,50,60}) {parent.controlPoints.emplace_back(cv::Vec3d{x*1.,0,1});parent.linePoints.emplace_back(x*1.,0,1);}
        batch={controller.makeFiberSaveSnapshot(parent)};
        const auto frame=nlohmann::json{{"volume_uuid",vc::util::newUuid()},{"scale_factor",4.0}};
        batch.front().fiber.coordinateIdentity["coordinates"]=frame;
        controller.commitFiberStoreAction(batch);parent=batch.front().fiber;
        controller.loadFibersForCurrentPackage();parent=resolve(parent.uuid);
        require(transferJson(parent).at("coordinates") == frame, "current transfer format preserves explicit coordinate identity");
        require(transferJson(parent).at("version") == 4, "current transfer format is independent of source ingestion version");
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
            require(controller._fibers.empty(), "read-only attachment defers controls");
            auto registerObject = [&](const std::string& uuid) {
                auto read = db.read();
                const auto revision = db.currentObjectRevision(read, uuid);
                const auto attachment = *state.objectStoreSession()->attachment();
                return controller.registerStoreFiber({attachment, {attachment.storeUuid, uuid}, revision->uuid, revision->geometryRevision});
            };
            const auto missingId = registerObject(missing.uuid);
            const auto controlsId = registerObject(object.uuid);
            require(controller._fibers.empty(), "registering catalog identities does not load controls");
            controller.ensureStoreFiber(missingId);
            controller.ensureStoreFiber(controlsId);
            const auto unavailable = std::find_if(controller._fibers.begin(),controller._fibers.end(),[&](const auto& fiber) { return fiber.uuid == missing.uuid; });
            require(unavailable != controller._fibers.end() && !unavailable->editorError.empty(), "unavailable geometry stays in catalog without blocking other fibers");
            require(controller._fibers.size() >= 2, "control-only record preserves the full catalog");
            const auto controlsOnly = resolve(object.uuid);
            require(controlsOnly.controlPoints.size() == 2 && controlsOnly.linePoints.empty(), "control-only geometry preserved without synthesizing lines");
            require(controller.fiberSummaries().size() == controller._fibers.size(), "read-only store fibers remain listed");
            QTemporaryDir transferDir;
            const auto transferPath = fs::path(transferDir.path().toStdString()) / "controls.json";
            const auto exportedControls = transferJson(controlsOnly);
            require(!exportedControls["control_points"][0].contains("segment_to_next"), "export does not invent tracing provenance");
            writeJsonAtomic(transferPath, exportedControls);
            const auto target = fs::path(transferDir.path().toStdString()) / "controls.vcobj";
            { auto targetDb = ObjectStore::create(target, options); }
            CState destination;
            Controller imported(&destination, nullptr, nullptr, nullptr);
            imported._errorDialogsSuppressed = true;
            destination.attachObjectStore(QString::fromStdString(target.string()), "control-only-roundtrip");
            QString importError; int importedCount = 0;
            if (!imported.importFibersFromPath(transferPath, &importError, &importedCount))
                throw std::runtime_error(importError.toStdString());
            require(importedCount == 1, "control-only transfer imports");
            imported.ensureAllFiberLines();
            require(imported._fibers.front().controlPoints.size() == 2 && imported._fibers.front().linePoints.empty(),
                    "control-only transfer preserves geometry without synthesizing a line");
            auto pending = std::vector<Controller::FiberSaveSnapshot>{controller.makeFiberSaveSnapshot(controlsOnly)};
            bool rejected = false; try { controller.commitFiberStoreAction(pending); } catch (...) { rejected = true; }
            require(rejected, "read-only controller cannot publish changes");
            auto transfer = transferJson(controlsOnly); transfer["version"] = 1;
            rejected = false; try { (void)vc::objects::decodeFiberJson(transfer.dump()); } catch (...) { rejected = true; }
            require(rejected, "legacy file import is rejected without repair");
            state.detachObjectStore();
        }
        if(const char* smoke=std::getenv("VC3D_OBJECT_STORE_SMOKE_DB")) {
            state.attachObjectStore(QString::fromUtf8(smoke),"vc3d-real-smoke");controller.loadFibersForCurrentPackage();
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
                auto mismatched = fiber.coordinateIdentity;
                mismatched["coordinates"]["volume_uuid"] = vc::util::newUuid();
                require(!vc3d::geometryScaleForVolume(mismatched, *state.currentVolume()),
                        "matching display ID cannot override conflicting volume UUID");
                const auto display = controller.fiberInVolumeCoordinates(fiber);
                require(display.controlPoints.front() == fiber.controlPoints.front() * .25 && display.linePoints.front() == fiber.linePoints.front() * .25, "display geometry uses explicit binding");
                require(resolve(fiber.uuid).linePoints == fiber.linePoints, "display transform does not change stored geometry");
                require(!fiber.branches.empty(), "real fixture has a linked peer");
                const auto branch = fiber.branches.front();
                auto peer = std::find_if(controller._fibers.begin(), controller._fibers.end(), [&](const auto& entry) { return entry.id == branch.branchFiberId; });
                require(peer != controller._fibers.end(), "linked peer is present");
                const auto peerFrame = peer->coordinateIdentity;
                peer->coordinateIdentity = coordinateBindingForVolume(*state.currentVolume(), 2.0);
                const auto markers = controller.generatedBranchLinkMarkers({branch}, .25);
                require(markers.size() == 1 && markers[0].localControlPoint == toVec3f(branch.controlPointPosition) * .25f &&
                    markers[0].linkedControlPoint == toVec3f(branch.branchControlPointPosition) * .5f,
                    "linked markers use both coordinate frames");
                Controller::LineAnnotationSession preview;
                preview.branches = {branch}; preview.fiberBaseToVolumeScale = .25; preview.showLinkedLineOverlays = true;
                controller.ensureFiberLines(*peer);
                const auto native = la::linePointsBetweenOuterControlPoints(peer->linePoints, la::storedControlPointPositions(peer->controlPoints));
                const auto overlays = controller.generatedBranchLinePointsForSession(preview);
                require(overlays.size() == 1 && overlays.front().front() == toVec3f(native.front()) * .5f,
                    "linked polyline uses its own frame");
                peer->coordinateIdentity = peerFrame;
                const auto point=fiber.controlPoints.front()*(*factor);
                Array3D<uint8_t> crop({2,2,2});
                require(state.currentVolume()->readZYX(crop,{int(point[2]),int(point[1]),int(point[0])}),"read real CT at scale 2");
                std::cerr << "CT smoke: " << ct << " scale 2, voxel size " << state.currentVolume()->voxelSize() << " um, sampled XYZ " << point << "\n";
            }
            const auto originalPin=fiber.storePin;
            fiber.tags.push_back("vc3d_smoke");
            controller.commitFibers({&fiber});
            state.detachObjectStore();state.attachObjectStore(QString::fromUtf8(smoke),"vc3d-real-smoke");controller.loadFibersForCurrentPackage();
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
            state.detachObjectStore();state.attachObjectStore(QString::fromUtf8(smoke),"vc3d-real-smoke");controller.loadFibersForCurrentPackage();
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
