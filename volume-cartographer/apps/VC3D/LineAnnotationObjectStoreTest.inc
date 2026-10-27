#include <iostream>
#include <QApplication>
#include <QElapsedTimer>
#include <QThread>
#include "FiberCollectionController.hpp"
#include "SpiralWorkspace.hpp"
#include "SpiralServiceManager.hpp"
#include <vc_object_store/geometry/Geometry.hpp>
// Built only with VC_TESTING. Runs the actual controller publication path.
int runObjectStoreControllerTests()
{
    using Controller = LineAnnotationController;
    using namespace vc::objects;
    auto require = [](bool value, const char* message) { if (!value) throw std::runtime_error(message); };
    const auto path = std::filesystem::temp_directory_path() / ("vc3d-store-test-"+vc::objects::util::newUuid()+".vcobj");
    auto checkHubStatus = [&](CState& state, Controller& controller,
                              FiberCollectionController& collection,
                              const std::function<bool()>& catalogReady) {
        const auto generation = controller._fiberDataGeneration;
        const auto snapshots = controller._sideStripStoredSnapshotCache.size();
        int catalogUpdates = 0, fiberUpdates = 0, selectionResets = 0;
        QObject observer;
        QObject::connect(&collection, &FiberCollectionController::catalogChanged, &observer,
            [&](const auto&, bool) { ++catalogUpdates; });
        QObject::connect(&collection, &FiberCollectionController::selectionChanged, &observer,
            [&](const auto& selection) { if (selection.id == 0) ++selectionResets; });
        QObject::connect(&controller, &Controller::fibersChanged, &observer, [&] { ++fiberUpdates; });
        std::vector<double> elapsedMs;
        for (int i = 0; i < 20; ++i) {
            QElapsedTimer timer; timer.start();
            state.objectStoreChanged({StoreSessionEvent::Kind::Hub,
                *state.objectStoreAttachment(), 0, {}});
            QApplication::processEvents();
            while (!catalogReady() && timer.elapsed() < 10000) {
                QApplication::processEvents(); QThread::msleep(1);
            }
            require(catalogReady(), "hub status test catalog settles");
            elapsedMs.push_back(timer.nsecsElapsed() / 1.e6);
        }
        std::sort(elapsedMs.begin(), elapsedMs.end());
        double total = 0;
        for (const auto ms : elapsedMs) total += ms;
        std::cout << "hub status events=20 mean_ms=" << total / elapsedMs.size()
                  << " median_ms=" << (elapsedMs[9] + elapsedMs[10]) / 2
                  << " p95_ms=" << elapsedMs[18] << " catalog_updates=" << catalogUpdates
                  << " fiber_updates=" << fiberUpdates << " selection_resets=" << selectionResets << std::endl;
        require(catalogUpdates == 0 && fiberUpdates == 0 && selectionResets == 0,
                "hub status does not reload the catalog, redraw fibers or clear selection");
        require(controller._fiberDataGeneration == generation &&
                controller._sideStripStoredSnapshotCache.size() == snapshots,
                "hub status preserves fiber generation and warm snapshots");
    };
    try {
        if (const char* source = std::getenv("VC3D_OBJECT_STORE_READ_ONLY_BENCH")) {
            CState state;
            Controller controller(&state,nullptr,nullptr,nullptr);
            FiberCollectionController collection(&state,nullptr,nullptr,&controller);
            require(collection.options().maxDisplayed == 5000, "clean settings default to 5000 displayed fibers");
            std::vector<FiberCollectionController::CatalogRow> page;
            QObject::connect(&collection, &FiberCollectionController::catalogChanged,
                [&](const std::vector<FiberCollectionController::CatalogRow>& rows, bool) { page = rows; });
            int heartbeats = 0;
            QTimer heartbeat; heartbeat.setInterval(10);
            QObject::connect(&heartbeat, &QTimer::timeout, [&] { ++heartbeats; });
            heartbeat.start();
            QElapsedTimer timer; timer.start();
            state.attachObjectStore(QString::fromUtf8(source), {}, true);
            const auto attachedMs = timer.elapsed();
            while (page.empty() && timer.elapsed() < 60000) {
                QApplication::processEvents(); QThread::msleep(1);
            }
            require(!page.empty() && page.size() <= 200, "first catalog page is bounded");
            require(heartbeats > 0, "UI heartbeat runs while catalog loads");
            const auto pageMs = timer.elapsed();
            const auto id = controller.fiberIdForUuid(page.front().record.uuid);
            auto selected = controller.readStoredFiber(id);
            require(selected && selected->editorError.empty(), "selected controls decode from the store");
            require(!selected->linePointsLoaded, "selection controls do not eagerly decode sampled line");
            controller.ensureFiberLines(*selected);
            require(selected->linePointsLoaded, "selected line decodes on demand");
            std::cout << "read-only attach_ms=" << attachedMs << " first_page_ms=" << pageMs
                      << " selected_ms=" << timer.elapsed()-pageMs << " heartbeats=" << heartbeats << std::endl;
            require(!controller.fiberSnapshotsForSideStripQuery({*selected}).empty(), "warm real fiber snapshot");
            collection.selectFiber(id, false, false);
            checkHubStatus(state, controller, collection, [&] { return !page.empty(); });
            state.detachObjectStore();
            require(controller.fiberUuidForId(id).empty() && !controller.readStoredFiber(id), "detach clears runtime identities");
            return 0;
        }
        {
            QTemporaryDir volumeDir;
            Volume::ZarrCreateOptions create; create.shapeZYX = {4,4,4};
            create.chunkShapeZYX = {4,4,4}; create.compressor = "none"; create.uuid = "20260411134726";
            const auto volume = Volume::New(fs::path(volumeDir.path().toStdString())/"ct.zarr", create);
            const auto binding = vc3d::coordinateBindingForVolume(*volume, 4);
            require(binding.at("coordinates").at("volume_uuid") == "20260411134726", "VC3D retains catalog volume ID verbatim");
            require(vc::objects::geometryScaleToVolume(binding, volume->id()) == .25, "VC3D matches the authoritative volume ID");
        }
        {
            // The umbilicus is the store's, bound to a volume; its frame was set when stored.
            QTemporaryDir dir;
            Volume::ZarrCreateOptions create; create.shapeZYX = {16,8,8};
            create.chunkShapeZYX = {16,8,8}; create.compressor = "none"; create.uuid = "umbilicus-volume";
            const auto volume = Volume::New(fs::path(dir.path().toStdString())/"ct.zarr", create);
            const auto storePath = fs::path(dir.path().toStdString())/"objects.vcobj";
            OpenOptions umbilicusOptions; umbilicusOptions.author = "vc3d-umbilicus-test";
            { auto db = ObjectStore::create(storePath, umbilicusOptions); }
            CState state;
            Controller controller(&state,nullptr,nullptr,nullptr);
            state.attachObjectStore(QString::fromStdString(storePath.string()), "vc3d-umbilicus-test");
            state.setCurrentVolume(volume);
            int changed = 0, fibersChanged = 0;
            QObject::connect(&state, &CState::umbilicusChanged, [&] { ++changed; });
            QObject::connect(&controller, &Controller::fibersChanged, [&] { ++fibersChanged; });
            const auto attachment = *state.objectStoreAttachment();
            require(!vc::objects::volumeUmbilicus(*state.objectStoreSession(), attachment, volume->id()) &&
                        !controller.umbilicusStatus().available,
                    "a volume without a stored umbilicus has none");
            auto add = [&](const std::string& name) {
                const auto uuid = vc::objects::util::newUuid();
                state.objectStoreSession()->act(attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
                    ObjectRecord record; record.uuid = uuid; record.kind = "umbilicus"; record.name = name;
                    record.meta = vc3d::coordinateBindingForVolume(*volume, 2).dump();
                    db.upsertObject(txn, record);
                    db.putGeometry(txn, uuid, vc::geometry::encode(vc::geometry::Umbilicus{{{8, 6, 0}, {8, 6, 30}}}));
                });
                state.pollObjectStoreChanges();
                return uuid;
            };
            const auto first = add("scroll");
            const auto stored = vc::objects::volumeUmbilicus(*state.objectStoreSession(), attachment, volume->id());
            require(changed > 0, "committing an umbilicus announces it");
            require(fibersChanged == 0, "a commit touching no fiber leaves the fiber caches alone");
            require(stored && stored->uuid == first &&
                    stored->points == std::vector<vc::geometry::Point>{{4, 3, 0}, {4, 3, 15}},
                    "the stored umbilicus is read in the volume's voxels");
            require(controller.umbilicusStatus().available && controller.umbilicusStatus().text == "scroll",
                    "the fiber map status names the stored umbilicus");
            const auto snapshot = controller.umbilicusForSnapshot(controller.annotationFrame());
            require(snapshot.message.isEmpty() && snapshot.centers.size() == 2, "the fiber map reads the stored umbilicus");
            const auto token = controller.umbilicusFingerprint();
            add("second");
            require(controller.umbilicusStatus().text.contains("several") &&
                    controller.umbilicusFingerprint() != token,
                    "several umbilici bound to one volume are refused");
            state.detachObjectStore();
        }
        {
            // Fibers bound by scale or through a registration are read and edited
            // in the open volume's voxels and saved back to their stored frame.
            QTemporaryDir dir;
            Volume::ZarrCreateOptions create; create.shapeZYX = {16,16,16};
            create.chunkShapeZYX = {16,16,16}; create.compressor = "none"; create.uuid = "frame-volume";
            const auto volume = Volume::New(fs::path(dir.path().toStdString())/"ct.zarr", create);
            const auto storePath = fs::path(dir.path().toStdString())/"objects.vcobj";
            OpenOptions frameOptions; frameOptions.author = "vc3d-frame-test";
            VolumeAffine shear; shear(0,0)=2; shear(0,1)=.5; shear(0,3)=1; shear(2,3)=3;
            std::string scaled, registered;
            const std::vector<vc::geometry::Point> controls{{2,4,6},{10,4,6}}, line{{2,4,6},{6,4,6},{10,4,6}};
            {
                auto db = ObjectStore::create(storePath, frameOptions);
                auto txn = db.write();
                (void)db.registerVolumeTransform(txn, "other-volume", "frame-volume", shear, "scan registration");
                auto add = [&](const nlohmann::json& meta) {
                    ObjectRecord record; record.kind = "fiber"; record.meta = meta.dump();
                    vc::objects::FiberDetail detail; detail.width = 8; record.fiber = detail;
                    const auto uuid = db.upsertObject(txn, record);
                    vc::geometry::Fiber fiber; fiber.controlPoints = controls; fiber.linePoints = line;
                    db.putGeometry(txn, uuid, vc::geometry::encode(fiber));
                    return uuid;
                };
                // Two stored units per voxel: the volume sees half the coordinates.
                scaled = add(vc3d::coordinateBindingForVolume(*volume, 2));
                registered = add({{"coordinates", {{"volume_uuid", "other-volume"}, {"scale_factor", 1}}}});
                txn.commit();
            }
            CState state;
            Controller controller(&state,nullptr,nullptr,nullptr);
            controller._errorDialogsSuppressed = true;
            state.attachObjectStore(QString::fromStdString(storePath.string()), "vc3d-frame-test");
            state.setCurrentVolume(volume);
            auto read = [&](const std::string& uuid) {
                auto fibers = controller.readStoredFibers(std::vector<std::string>{uuid}, true);
                require(fibers.size() == 1 && fibers.front().editorError.empty(), "bound fiber decodes");
                return fibers.front();
            };
            auto stored = [&](const std::string& uuid) {
                vc::geometry::Fiber fiber;
                state.objectStoreSession()->read(*state.objectStoreAttachment(), [&](const auto& db, const auto& txn, const auto&) {
                    fiber = GeometrySnapshot(db, txn).fiber(uuid).load();
                });
                return fiber;
            };
            auto s = read(scaled), r = read(registered);
            require(s.frame && r.frame && controller.fiberInVolume(s) && controller.fiberInVolume(r),
                    "fibers bound by scale or registration are read in the volume");
            require(cv::Vec3d(s.controlPoints[1]) == cv::Vec3d{5,2,3} && s.width == 4, "scale-bound fiber in volume voxels");
            require(cv::Vec3d(r.controlPoints[0]) == cv::Vec3d{7,4,9}, "registered fiber in volume voxels");
            // Move one control point (and its line sample) of each; save through the controller.
            for (auto* fiber : {&s, &r}) {
                fiber->controlPoints[1][0] += 1;
                fiber->linePoints[2][0] += 1;
            }
            std::vector<Controller::FiberSaveSnapshot> edits{controller.makeFiberSaveSnapshot(s), controller.makeFiberSaveSnapshot(r)};
            controller.commitFiberStoreAction(edits);
            const auto savedScaled = stored(scaled), savedRegistered = stored(registered);
            require(savedScaled.controlPoints[0] == controls[0] && savedScaled.linePoints[1] == line[1],
                    "unchanged points keep their stored values");
            require(savedScaled.controlPoints[1] == vc::geometry::Point{12,4,6} && savedScaled.linePoints[2] == savedScaled.controlPoints[1],
                    "a moved point is saved in the stored frame");
            require(savedRegistered.controlPoints[0] == controls[0] && savedRegistered.linePoints[1] == line[1],
                    "unchanged points of a registered fiber keep their stored values");
            const auto moved = shear.apply(savedRegistered.controlPoints[1]);
            require(std::abs(moved[0] - (r.controlPoints[1][0])) < 1e-9 && savedRegistered.linePoints[2] == savedRegistered.controlPoints[1],
                    "a moved registered point maps back through the inverse");
            const auto reread = read(scaled);
            require(reread.width == 4 && cv::Vec3d(reread.controlPoints[1]) == cv::Vec3d{6,2,3}, "the saved fiber reads back in the volume");
            // A record-only edit of a fiber read without its line (tag, rename)
            // loads the line on save in the same frame and leaves it unchanged.
            for (const auto& uuid : {scaled, registered}) {
                const auto before = stored(uuid);
                auto lazy = controller.readStoredFibers(std::vector<std::string>{uuid}).front();
                require(!lazy.linePointsLoaded, "lines are loaded on demand");
                lazy.tags.push_back("reviewed");
                std::vector<Controller::FiberSaveSnapshot> tagEdit{controller.makeFiberSaveSnapshot(lazy)};
                controller.commitFiberStoreAction(tagEdit);
                const auto after = stored(uuid);
                require(after.linePoints == before.linePoints && after.controlPoints == before.controlPoints,
                        "a record-only edit keeps the stored geometry");
            }
            state.detachObjectStore();
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
        auto b = a; b.uuid = vc::objects::util::newUuid(); b.id = controller.nextFiberId();
        std::vector<Controller::FiberSaveSnapshot> batch{controller.makeFiberSaveSnapshot(a),controller.makeFiberSaveSnapshot(b)};
        controller.commitFiberStoreAction(batch);
        a = batch[0].fiber; b = batch[1].fiber;
        const auto first = a.storePin;
        {
            const auto all = controller.readStoredFibers(std::nullopt);
            require(all.size()==2,"store lists both fibers with same name");
            require(std::none_of(all.begin(), all.end(), [](const auto& fiber) { return fiber.linePointsLoaded; }), "editor reads do not materialize dense lines");
            require(all[0].id != all[1].id,"UUID runtime identity");
            const auto detail = controller.fiberSummary(a.id);
            require(detail && detail->spans.size() == 2 && detail->annotatedLengthVx == 20.0, "store summary retains exact lengths and control ranges");
        }
        // Every read resolves the store's current head; nothing is retained between reads.
        auto resolve = [&](const std::string& uuid) {
            auto fibers = controller.readStoredFibers(std::vector<std::string>{uuid}, true);
            if (fibers.empty()) throw std::runtime_error("fiber missing");
            return fibers.front();
        };
        a=resolve(a.uuid);b=resolve(b.uuid);
        Controller::FiberBranchRef link;
        link.controlPointIndex=0;link.branchControlPointIndex=0;link.branchFiberId=b.id;link.branchFileName=b.fileName;
        link.controlPointPosition=a.controlPoints[0];link.branchControlPointPosition=b.controlPoints[0];
        link.controlPointDirection={1,0,0};link.branchControlPointDirection={1,0,0};
        // An unreviewed automatic link ("Link A and B").
        link.pending=true;link.autoLink=true;
        a.branches.push_back(link);
        std::swap(link.controlPointPosition,link.branchControlPointPosition); link.branchFiberId=a.id;link.branchFileName=a.fileName;
        b.branches.push_back(link);
        // The panel's catalog refreshes from the store's own commit events.
        FiberCollectionController collection(&state, nullptr, nullptr, &controller);
        std::vector<FiberCollectionController::CatalogRow> catalog;
        QObject::connect(&collection, &FiberCollectionController::catalogChanged,
            [&](const std::vector<FiberCollectionController::CatalogRow>& rows, bool) { catalog = rows; });
        batch={controller.makeFiberSaveSnapshot(a),controller.makeFiberSaveSnapshot(b)};
        controller.commitFiberStoreAction(batch);
        a=resolve(a.uuid);b=resolve(b.uuid);
        require(a.branches.size()==1 && b.branches.size()==1,"shared branch roundtrip");
        require(a.branches[0].pending && a.branches[0].autoLink && b.branches[0].autoLink,
                "automatic mark is one shared edge state");
        {
            QElapsedTimer timer; timer.start();
            auto linked = [&] { return catalog.size() == 2 && std::all_of(catalog.begin(), catalog.end(),
                [](const auto& row) { return row.links == 1 && row.componentSize == 2; }); };
            while (!linked() && timer.elapsed() < 10000) { QApplication::processEvents(); QThread::msleep(1); }
            require(linked(), "catalog rows come from the store with shared links and components");
            require(std::none_of(catalog.begin(), catalog.end(), [](const auto& row) { return row.mapping.has_value(); }),
                    "fibers without a binding list without a volume length");
        }
        require(controller.fiberSnapshotsForSideStripQuery({a, b}).size() == 2, "warm linked fiber snapshots");
        collection.selectFiber(a.id, false, false);
        checkHubStatus(state, controller, collection, [&] { return catalog.size() == 2; });
        {
            const auto generation = controller._fiberDataGeneration;
            const auto snapshots = controller._sideStripStoredSnapshotCache.size();
            int catalogUpdates = 0, fiberUpdates = 0, selectionResets = 0;
            QObject observer;
            QObject::connect(&collection, &FiberCollectionController::catalogChanged, &observer,
                [&](const auto&, bool) { ++catalogUpdates; });
            QObject::connect(&collection, &FiberCollectionController::selectionChanged, &observer,
                [&](const auto& selection) { if (selection.id == 0) ++selectionResets; });
            QObject::connect(&controller, &Controller::fibersChanged, &observer, [&] { ++fiberUpdates; });
            const auto connection = state.objectStoreSession();
            const auto attachment = *state.objectStoreAttachment();
            const auto unrelated = vc::objects::util::newUuid();
            connection->act(attachment, {{unrelated, std::nullopt}}, [&](auto& db, auto& txn) {
                ObjectRecord record; record.uuid = unrelated; record.kind = "point_collection";
                db.upsertObject(txn, record);
            });
            state.pollObjectStoreChanges();
            // Old/foreign attachments must not invalidate this store's fibers.
            StoreSessionEvent foreign{StoreSessionEvent::Kind::Committed,
                {vc::objects::util::newUuid(), attachment.generation}, 0, {}};
            JournalEntry change; change.kind = "fiber"; foreign.changes.push_back(change);
            state.objectStoreChanged(foreign);
            QApplication::processEvents();
            require(catalogUpdates == 0 && fiberUpdates == 0 && selectionResets == 0,
                    "unrelated and foreign commits preserve catalog and selection");
            require(controller._fiberDataGeneration == generation &&
                    controller._sideStripStoredSnapshotCache.size() == snapshots,
                    "unrelated and foreign commits preserve warm fiber snapshots");
            foreign.attachment = attachment;
            foreign.changes.front().kind = kAttachmentEdgeKind;
            state.objectStoreChanged(foreign);
            require(controller._fiberDataGeneration > generation && controller._sideStripStoredSnapshotCache.empty(),
                    "attachment changes in the user store invalidate fiber snapshots");

        }
        const auto transferPaths = controller.fiberTransportPaths({a.id});
        require(transferPaths.size() == 2, "transport includes both linked endpoints");
        auto readDocument = [](const auto& file) { std::ifstream input(file); return nlohmann::json::parse(input); };
        const auto aWire = readDocument(transferPaths.at(a.uuid));
        const auto bWire = readDocument(transferPaths.at(b.uuid));
        require(aWire["branches"][0]["branch_file"] == b.uuid + ".json", "transport serializes branch target UUID");
        require(bWire["branches"][0]["branch_file"] == a.uuid + ".json", "transport serializes reciprocal branch");
        require(aWire["branches"][0].value("auto", false) && bWire["branches"][0].value("auto", false),
                "transport writes the automatic mark on both entries");
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
            const auto importedFibers = imported.readStoredFibers(std::nullopt);
            require(count == 2 && importedFibers.size() == 2, "duplicate display names import by UUID");
            for (const auto& fiber : importedFibers) {
                require(fiber.fileName == "same_name.json", "display name survives transfer");
                require(fiber.branches.size() == 1, "shared branch survives transfer");
                require(fiber.branches.front().edgeUuid == a.branches.front().edgeUuid, "edge UUID survives transfer");
                require(fiber.branches.front().autoLink, "automatic mark survives transfer");
            }
        }
        {
            // Approval clears the mark on the shared edge.
            auto approved = a;
            approved.branches[0].pending = false;
            const auto generation = controller._fiberDataGeneration;
            controller.commitFibers({&approved});
            state.pollObjectStoreChanges();
            require(controller._fiberDataGeneration > generation && controller._sideStripStoredSnapshotCache.empty(),
                    "real commits still invalidate fiber geometry snapshots after hub status events");
            QElapsedTimer timer; timer.start();
            const auto approvedCatalog = [&] {
                return catalog.size() == 2 && std::all_of(catalog.begin(), catalog.end(),
                    [](const auto& row) { return row.pendingLinks == 0; });
            };
            while (!approvedCatalog() && timer.elapsed() < 10000) { QApplication::processEvents(); QThread::msleep(1); }
            require(approvedCatalog(), "real commits still refresh catalog link state");
            const auto again = resolve(a.uuid);
            require(!again.branches[0].pending && !again.branches[0].autoLink, "approval clears the automatic mark");
            auto pending = again;
            pending.branches[0].pending = true;
            controller.commitFibers({&pending});
            const auto human = resolve(b.uuid);
            require(human.branches[0].pending && !human.branches[0].autoLink, "re-flagged link is a human pending link");
            a=resolve(a.uuid);b=resolve(b.uuid);
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
            require(imported.readStoredFibers(std::nullopt).size() == 2 && service->inputDraftStatus().size() == 2,
                    "service-only component is imported into attached store before editing");
            auto document = aWire;
            const auto uuid = vc::objects::util::newUuid(); document["uuid"] = uuid;
            document["branches"] = nlohmann::json::array(); document["adjacent_branches"] = nlohmann::json::array();
            document.erase("optimization_mode"); document["custom_annotation"] = {{"note", "retain"}};
            const auto volumeUuid = vc::objects::util::newUuid();
            document["coordinates"] = {{"volume_uuid", volumeUuid}, {"scale_factor", 4}};
            document["coordinate_binding"] = {{"active_volume_id", "display"}};
            QString error;
            const auto single = fs::path(directory.path().toStdString()) / "optional-mode.json";
            { std::ofstream out(single); out << document.dump(); }
            if (!imported.importFibersFromPath(single, &error)) throw std::runtime_error(error.toStdString());
            const auto id = imported.fiberIdForUuid(uuid);
            auto found = imported.readStoredFiber(id, true);
            require(found && !found->readOnly && found->controlPoints.size() == 3,
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
        require(controller.generatedBranchLinkMarkers(a.branches).empty(), "unbound peers have no branch markers");
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
        a=resolve(a.uuid);
        require(a.controlPoints.front()[0]==-1,"sparse move persists");
        require(a.branches.front().edgeUuid==edge,"branch identity survives move");
        const auto id=a.controlPoints[1].storeId;
        a.controlPoints.erase(a.controlPoints.begin()+1); a.linePoints.erase(a.linePoints.begin()+1);
        batch={controller.makeFiberSaveSnapshot(a)};
        controller.commitFiberStoreAction(batch);
        a=resolve(a.uuid);
        require(a.controlPoints.size()==2 && a.controlPoints.back().storeId==2,"structural deletion preserves control identities");
        a.controlPoints.insert(a.controlPoints.begin()+1,vc3d::line_annotation::StoredControlPoint{cv::Vec3d{10,0,1}});
        a.linePoints.insert(a.linePoints.begin()+1,cv::Vec3d{10,0,1});
        batch={controller.makeFiberSaveSnapshot(a)};controller.commitFiberStoreAction(batch);
        a=resolve(a.uuid);
        require(a.controlPoints[1].storeId!=id,"deleted point identity is not reused");
        a.tags={"reviewed"}; batch={controller.makeFiberSaveSnapshot(a)};controller.commitFiberStoreAction(batch);
        a=batch.front().fiber;
        controller.commitFibers({&a});
        controller.commitFibers({&a});
        batch={controller.makeFiberSaveSnapshot(a)};
        const auto stale=batch;
        a=resolve(a.uuid);
        { auto db=ObjectStore::open(path,options);auto txn=db.write({{a.uuid,a.storePin->objectRevision}});db.setMeta(txn,a.uuid,"external","true");txn.commit(); }
        bool conflict=false;try { auto old=stale;controller.commitFiberStoreAction(old); } catch(...) { conflict=true; }
        require(conflict,"stale controller edits reject");
        a=resolve(a.uuid);
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
            secondSeed.id = controller.nextFiberId(); secondSeed.uuid = vc::objects::util::newUuid();
            std::vector<Controller::FiberSaveSnapshot> seeds{
                controller.makeFiberSaveSnapshot(firstSeed), controller.makeFiberSaveSnapshot(secondSeed)};
            controller.commitFiberStoreAction(seeds);
            controller.reoptimizeMergedFibers({secondSeed.id});
            require(resolve(firstSeed.uuid).tags == firstSeed.tags, "same-name non-target keeps reoptimization tag");
            require(resolve(secondSeed.uuid).tags.empty(), "reoptimization selects exact runtime identity");
        }

        Controller::StoredFiber parent;
        parent.id=controller.nextFiberId(); parent.fileName="parent.json";
        parent.username="test"; parent.startedAt="20260101"; parent.sequence=3;
        parent.coordinateBaseShapeZYX=std::array<size_t,3>{100,100,100};
        for(int x:{30,40,50,60}) {parent.controlPoints.emplace_back(cv::Vec3d{x*1.,0,1});parent.linePoints.emplace_back(x*1.,0,1);}
        batch={controller.makeFiberSaveSnapshot(parent)};
        const auto frame=nlohmann::json{{"volume_uuid",vc::objects::util::newUuid()},{"scale_factor",4.0}};
        batch.front().fiber.coordinateIdentity["coordinates"]=frame;
        controller.commitFiberStoreAction(batch);parent=batch.front().fiber;
        parent=resolve(parent.uuid);
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
        auto prefix=resolve(split->prefix.uuid),suffix=resolve(split->suffix.uuid);
        require(prefix.controlPoints.front().storeId==parent.controlPoints.front().storeId,"split retains control identity");
        require(prefix.branches.size()==1 && suffix.branches.size()==1,"split shared branch persists");
        {auto db=ObjectStore::open(path,options);auto txn=db.read();require(nlohmann::json::parse(db.objectByUuid(txn,prefix.uuid)->meta)["coordinates"]==frame,"split inherits explicit frame");}
        const auto merge=la::planFiberMerge(prefix,int(prefix.controlPoints.size()-1),suffix,0,parent.optimizationMode,identity("merged.json"),&error);
        if(!merge) throw std::runtime_error(error);
        Controller::StructuralEditRequest mergeRequest;mergeRequest.verb="merge";mergeRequest.newFibers={merge->merged};mergeRequest.originals={prefix,suffix};
        auto mergeResult=controller.commitStructuralEdit(std::move(mergeRequest),&committed);
        if(!mergeResult.ok) throw std::runtime_error("merge: "+mergeResult.error);
        const auto merged=resolve(merge->merged.uuid);
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
        require(state.objectStoreAttachment().has_value(),"secondary destruction keeps main attachment");
        state.detachObjectStore();
        require(controller.readStoredFibers(std::nullopt).empty() && controller.fiberUuidForId(merged.id).empty(),
                "detach leaves nothing to read and no identities");
        {
            // Store-valid control-only geometry must not empty the editor catalog.
            auto db = ObjectStore::open(path, options);
            ObjectRecord object; object.uuid = vc::objects::util::newUuid(); object.kind = "fiber"; object.name = "controls-only";
            auto txn = db.write({{object.uuid,{}}});
            object.formatVersion = 1;
            db.upsertObject(txn, object);
            db.putGeometry(txn, object.uuid, vc::geometry::encode(vc::geometry::Fiber{{{0,0,0},{1,0,0}}, {}}));
            txn.commit();
            ObjectRecord missing; missing.uuid = vc::objects::util::newUuid(); missing.kind = "fiber"; missing.name = "geometry-pending";
            { auto pending = db.write({{missing.uuid,{}}}); db.upsertObject(pending,missing); pending.commit(); }
            state.attachObjectStore(QString::fromStdString(path.string()), {}, true);
            const auto everything = controller.readStoredFibers(std::nullopt);
            const auto unavailable = std::find_if(everything.begin(),everything.end(),[&](const auto& fiber) { return fiber.uuid == missing.uuid; });
            require(unavailable != everything.end() && !unavailable->editorError.empty(), "unavailable geometry stays listed without blocking other fibers");
            require(everything.size() >= 2, "control-only record preserves the full catalog");
            const auto controlsOnly = resolve(object.uuid);
            require(controlsOnly.controlPoints.size() == 2 && controlsOnly.linePoints.empty(), "control-only geometry preserved without synthesizing lines");
            require(controller.fiberSummaries(std::nullopt).size() == everything.size(), "read-only store fibers remain listed");
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
            const auto importedControls = imported.readStoredFibers(std::nullopt, true);
            require(importedControls.size() == 1 && importedControls.front().controlPoints.size() == 2 &&
                    importedControls.front().linePoints.empty(),
                    "control-only transfer preserves geometry without synthesizing a line");
            auto pending = std::vector<Controller::FiberSaveSnapshot>{controller.makeFiberSaveSnapshot(controlsOnly)};
            bool rejected = false; try { controller.commitFiberStoreAction(pending); } catch (...) { rejected = true; }
            require(rejected, "read-only controller cannot publish changes");
            auto transfer = transferJson(controlsOnly); transfer["version"] = 2;
            rejected = false; try { (void)vc::objects::decodeFiberJson(transfer.dump()); } catch (...) { rejected = true; }
            require(rejected, "unsupported fiber version is rejected");
            state.detachObjectStore();
        }
        if(const char* smoke=std::getenv("VC3D_OBJECT_STORE_SMOKE_DB")) {
            state.attachObjectStore(QString::fromUtf8(smoke),"vc3d-real-smoke");
            const auto everything=controller.readStoredFibers(std::nullopt);
            require(!everything.empty(),"real database catalog reads");
            const size_t count=everything.size();
            auto fiber=resolve(everything.front().uuid);
            if(const char* ct=std::getenv("VC3D_OBJECT_STORE_SMOKE_CT")) {
                state.setCurrentVolume(Volume::New(ct));
                require(!controller.ensureFiberInVolume(fiber.id,false) && !fiber.frame,"unbound real fiber requires explicit assignment");
                const auto stored = fiber;
                QTimer::singleShot(0,[] {
                    for(auto* widget:QApplication::topLevelWidgets()) if(auto* input=qobject_cast<QInputDialog*>(widget)) {
                        input->setTextValue("4");input->accept();
                    }
                });
                controller._errorDialogsSuppressed=false;
                const bool bound=controller.ensureFiberInVolume(fiber.id,true);
                controller._errorDialogsSuppressed=true;
                require(bound,"explicit scale-4 binding persists");
                fiber=resolve(fiber.uuid);
                auto mismatched = fiber.coordinateIdentity;
                mismatched["coordinates"]["volume_uuid"] = vc::objects::util::newUuid();
                require(!vc::objects::geometryScaleToVolume(mismatched, state.currentVolume()->id()),
                        "matching display ID cannot override conflicting volume UUID");
                require(fiber.frame && cv::Vec3d(fiber.controlPoints.front()) == cv::Vec3d(stored.controlPoints.front()) * .25 &&
                        fiber.linePoints.front() == stored.linePoints.front() * .25, "a bound fiber is read in the volume's voxels");
                require(!fiber.branches.empty(), "real fixture has a linked peer");
                const auto branch = fiber.branches.front();
                auto peer = controller.readStoredFiber(branch.branchFiberId, true);
                require(peer.has_value(), "linked peer is present");
                require(controller.fiberBoundToVolume(peer->id, state.currentVolume()), "linked peer is bound to the CT");
                const auto markers = controller.generatedBranchLinkMarkers({branch});
                require(markers.size() == 1 && markers[0].localControlPoint == toVec3f(branch.controlPointPosition) &&
                    markers[0].linkedControlPoint == toVec3f(branch.branchControlPointPosition),
                    "linked markers are in the volume's voxels");
                Controller::LineAnnotationSession preview;
                preview.branches = {branch}; preview.showLinkedLineOverlays = true;
                const auto line = la::linePointsBetweenOuterControlPoints(peer->linePoints, la::storedControlPointPositions(peer->controlPoints));
                const auto overlays = controller.generatedBranchLinePointsForSession(preview);
                require(overlays.size() == 1 && overlays.front().front() == toVec3f(line.front()),
                    "linked polyline is in the volume's voxels");
                const cv::Vec3d point=fiber.controlPoints.front();
                Array3D<uint8_t> crop({2,2,2});
                require(state.currentVolume()->readZYX(crop,{int(point[2]),int(point[1]),int(point[0])}),"read real CT at scale 2");
                std::cerr << "CT smoke: " << ct << " scale 2, voxel size " << state.currentVolume()->voxelSize() << " um, sampled XYZ " << point << "\n";
            }
            const auto originalPin=fiber.storePin;
            fiber.tags.push_back("vc3d_smoke");
            controller.commitFibers({&fiber});
            state.detachObjectStore();state.attachObjectStore(QString::fromUtf8(smoke),"vc3d-real-smoke");
            require(controller.readStoredFibers(std::nullopt).size()==count,"real database reopens");
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
