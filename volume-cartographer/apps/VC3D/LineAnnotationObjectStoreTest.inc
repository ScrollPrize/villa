#include <iostream>
#include <vc_reader/http.hpp>
#include <QApplication>
#include <QElapsedTimer>
#include <QThread>
#include "FiberCollectionController.hpp"
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/objects/AttachedDocument.hpp>
#include <vc_object_store/objects/ObjectCollection.hpp>
#include <vc_object_store/objects/Project.hpp>
#include "vc/core/types/Volume.hpp"
#include "volume_viewers/ViewerSampling.hpp"
#include "test/support/TestVolume.hpp"
#include <nlohmann/json.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <vc_reader/sftp.hpp>
#include <filesystem>
#include <fstream>
#include "fibers3d/Fiber3DCanvas.hpp"
#include "fibers3d/Fiber3DSliceControls.hpp"
#include <QCheckBox>
#include <QToolButton>
#include <QProcess>
#include <QRegularExpression>
#include <random>
// Built only with VC_TESTING. Runs the actual controller publication path.
namespace {
// Open a store as the state's project, as VC3D opens one.
void openStoreAsProject(CState& state, const QString& path, const QString& author, bool readOnly = false)
{
    auto projectPath = std::filesystem::path(path.toStdString());
    projectPath.replace_extension(".vcproj");
    if (!std::filesystem::exists(projectPath))
        vc::project::ProjectDocument::create(projectPath, "test", path.toStdString(), readOnly);
    auto document = vc::project::ProjectDocument::load(projectPath);
    auto data = document.data();
    data["stores"][0]["read_only"] = readOnly;
    document.replace(std::move(data));
    state.setVpkg(VolumePkg::open(projectPath));
}
}
int runObjectStoreControllerTests()
{
    using Controller = LineAnnotationController;
    using namespace vc::objects;
    auto require = [](bool value, const char* message) { if (!value) throw std::runtime_error(message); };
    const auto path = std::filesystem::temp_directory_path() / ("vc3d-store-test-"+vc::objects::util::newUuid()+".vcobj");
    // These tests exercise validation against the store's status; the
    // event-driven store view has its own section (which turns it on).
    Controller::s_fiberStoreViewDefault = false;
    try {
#include "LineAnnotationNearbyBenchmark.inc"
        if (const char* endpoint = std::getenv("VC3D_SSH_STORE_TEST")) {
            QTemporaryDir directory;
            const auto local = fs::path(directory.path().toStdString()) / "local.vcobj";
            OpenOptions options; options.author = "ssh-gui-test";
            { auto store = ObjectStore::create(local, options); }
            CState state;
            openStoreAsProject(state, QString::fromStdString(local.string()), options.author.c_str());
            int prompts = 0, heartbeats = 0;
            vcr::sftp::setAuthenticator([&](const vcr::sftp::Prompt& prompt) -> std::optional<std::string> {
                if (!prompt.confirmation) return std::nullopt;
                QMetaObject::invokeMethod(qApp, [&] { ++prompts; }, Qt::BlockingQueuedConnection);
                return "yes"; // disposable local sshd from run_ssh_test.py
            });
            const auto resetAuth = qScopeGuard([] { vcr::sftp::setAuthenticator({}); });
            QTimer heartbeat; heartbeat.setInterval(5);
            QObject::connect(&heartbeat, &QTimer::timeout, [&] {
                ++heartbeats;
                state.objectStoreSession()->finalize();
            });
            heartbeat.start();
            const auto hosted = state.attachHostedStore(endpoint, options.author.c_str());
            state.setStoreFor("fiber", hosted);
            const auto project = state.vpkg()->document().path();
            require(prompts == 1 && heartbeats > 0, "SSH prompts and GUI events run during attachment");
            require(state.objectStoreSession()->location(hosted) == endpoint, "project retains pasted SSH endpoint");
            state.closeProject();
            state.setVpkg(VolumePkg::open(project));
            require(state.storeFor("fiber") && state.objectStoreSession()->location(*state.storeFor("fiber")) == endpoint,
                    "project reopening reestablishes its SSH hosted store");
            auto invalid = state.vpkg()->document().data();
            invalid["stores"][1]["location"] = "sftp://host:8750/file.vcobj";
            bool refused = false;
            try { state.vpkg()->document().replace(invalid); } catch (...) { refused = true; }
            require(refused, "hosted endpoints reject SFTP file paths");
            state.closeProject();
        }
        if (const char* legacy = std::getenv("VC3D_UPGRADE_STORE_TEST")) {
            QTemporaryDir directory;
            const auto source = fs::path(directory.path().toStdString()) / "legacy.vcobj";
            fs::copy_file(legacy, source);
            if (fs::exists(std::string(legacy) + ".artifacts"))
                fs::copy(std::string(legacy) + ".artifacts", source.string() + ".artifacts", fs::copy_options::recursive);
            CState state;
            openStoreAsProject(state, QString::fromStdString(source.string()), "test", true);
            const auto original = state.vpkg()->document().data();
            require(!state.storeFor("fiber"), "read-only legacy store is not upgraded");
            state.closeProject();
            openStoreAsProject(state, QString::fromStdString(source.string()), "test");
            require(state.storeFor("fiber").has_value(), "writable legacy store upgrades and attaches automatically");
            const auto project = state.vpkg()->document().path();
            const auto data = state.vpkg()->document().data();
            const auto upgraded = state.vpkg()->document().resolve(data["stores"][0]["location"]);
            require(upgraded != source && fs::exists(source), "upgrade keeps the original database");
            require(data["selections"] == original["selections"], "upgrade preserves store routes");
            std::ifstream reportFile(upgraded.string() + ".report.json");
            const auto report = nlohmann::json::parse(reportFile);
            require(report["validation"] == "ok" && fs::exists(upgraded.string() + ".mapping.json"), "upgrade retains conversion report and revision mapping");
            state.closeProject();
            state.setVpkg(VolumePkg::open(project));
            require(state.storeFor("fiber") && state.vpkg()->document().data() == data, "reopening uses the upgraded store without another conversion");
            state.closeProject();
        }
#include "LineAnnotationHostedReadTest.inc"
#include "LineAnnotationCompoundTest.inc"
#include "LineAnnotationMetricsDatasetTest.inc"
#include "LineAnnotationFiberCacheTest.inc"
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
            openStoreAsProject(state, QString::fromUtf8(source), {}, true);
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
            state.closeProject();
            require(controller.fiberUuidForId(id).empty() && !controller.readStoredFiber(id), "detach clears runtime identities");
            return 0;
        }
        {
            QTemporaryDir volumeDir;
            const std::filesystem::path volumeRoot1 = fs::path(volumeDir.path().toStdString())/"ct.zarr";
            const vcr::zarr::CreateOptions levelZero1{.shape = {4, 4, 4}, .chunks = {4, 4, 4}, .compressor = "", .overwrite = true};
            vcr::zarr::Writer::create(volumeRoot1 / "0", levelZero1);
            std::ofstream(volumeRoot1 / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "20260411134726"}, {"name", "20260411134726"}, {"format", "zarr"}, {"width", 4}, {"height", 4}, {"slices", 4}, {"voxelsize", 1.0}}.dump();
            const auto volume = Volume::New(volumeRoot1);
            const auto binding = nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(volume->id(), 4)}};
            require(binding.at("coordinates").at("volume_uuid") == "20260411134726", "VC3D retains catalog volume ID verbatim");
            require(vc::objects::geometryScaleToVolume(binding, volume->id()) == .25, "VC3D matches the authoritative volume ID");
        }
        {
            // Annotation has its own viewer manager, but its CT controls must
            // find the main CT settings and place slices at the live fiber.
            using namespace vc3d::fibers3d;
            QTemporaryDir dir;
            vc3d::test::VolumeSpec spec;
            spec.shapeZYX = {32,32,32}; spec.chunkShapeZYX = {32,32,32};
            auto volume = vc3d::test::createVolume(fs::path(dir.path().toStdString())/"ct.zarr", spec);
            CState state;
            state.setCurrentVolume(volume);
            ViewerManager mainViews(&state, state.pointCollection());
            ViewerManager annotationViews(&state, state.pointCollection());
            QWidget host;
            host.resize(480, 360);
            const std::array<std::string,3> names{"xy plane", "seg xz", "seg yz"};
            const std::array<cv::Vec3f,3> normals{{{0,0,1}, {0,1,0}, {1,0,0}}};
            for (int i=0; i<3; ++i) {
                state.setSurface(names[i], std::make_shared<PlaneSurface>(cv::Vec3f{0,0,0}, normals[i]));
                auto* viewer = mainViews.createViewerInWidget(names[i], &host);
                auto settings = viewer->compositeRenderSettings();
                setCompositeMethod(settings.params, "max");
                settings.planeLayersFront = 3; settings.planeLayersBehind = 2;
                viewer->setCompositeRenderSettings(settings);
                dynamic_cast<QWidget*>(viewer)->hide();
            }
            // Sampling/control wiring needs no GL context. Keep the hidden
            // canvas out of the visible host's widget tree so Qt's offscreen
            // compositor does not try to create a GL backing store for it.
            auto canvas = std::make_unique<Fiber3DCanvas>();
            canvas->hide();
            auto* toggle = new QToolButton(&host);
            auto* controls = new Fiber3DSliceControls(&state, &annotationViews, canvas.get(), toggle, &host);
            controls->setCenter(Point{16,18,20});
            host.show();
            toggle->setChecked(true);
            auto settle = [&](const std::function<bool()>& done) {
                QElapsedTimer timer; timer.start();
                while (!done() && timer.elapsed()<15000) { QApplication::processEvents(); QThread::msleep(5); }
                require(done(), "annotation CT slice controls settle");
            };
            settle([&] { return !canvas->ctSlices().empty() && !canvas->ctSlices().front()->preview; });
            for (const auto& image : canvas->ctSlices()) {
                for (int axis=0; axis<3; ++axis)
                    require(std::abs(image->plane.origin[axis]+.5*image->plane.right[axis]+.5*image->plane.down[axis]-Point{16,18,20}[axis])<1e-5,
                            "annotation CT slices pass through the live fiber point, not the global plane origin");
            }
            auto* volumetric = controls->findChild<QCheckBox*>("fibers3DSliceVolumetric");
            volumetric->setChecked(true);
            settle([&] { return !canvas->ctSlices().empty() && canvas->ctSlices().front()->slab && !canvas->ctSlices().front()->preview; });
            require(volumetric->isChecked(), "annotation volumetric toggle stays enabled without a Composite View panel");
            require(canvas->ctSlices().front()->plane.front == 3 && canvas->ctSlices().front()->plane.behind == 2,
                    "annotation volumetric CT uses the linked main-view thickness");
            volumetric->setChecked(false);
            settle([&] { return !canvas->ctSlices().empty() && !canvas->ctSlices().front()->slab && !canvas->ctSlices().front()->preview; });
            controls->cancel();
            require(canvas->ctSlices().empty(), "closing CT controls removes their images");
        }
        {
            // The umbilicus is the store's, bound to a volume; its frame was set when stored.
            QTemporaryDir dir;
            const std::filesystem::path volumeRoot2 = fs::path(dir.path().toStdString())/"ct.zarr";
            const vcr::zarr::CreateOptions levelZero2{.shape = {16, 8, 8}, .chunks = {16, 8, 8}, .compressor = "", .overwrite = true};
            vcr::zarr::Writer::create(volumeRoot2 / "0", levelZero2);
            std::ofstream(volumeRoot2 / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "umbilicus-volume"}, {"name", "umbilicus-volume"}, {"format", "zarr"}, {"width", 8}, {"height", 8}, {"slices", 16}, {"voxelsize", 1.0}}.dump();
            const auto volume = Volume::New(volumeRoot2);
            const auto storePath = fs::path(dir.path().toStdString())/"objects.vcobj";
            OpenOptions umbilicusOptions; umbilicusOptions.author = "vc3d-umbilicus-test";
            { auto db = ObjectStore::create(storePath, umbilicusOptions); }
            CState state;
            Controller controller(&state,nullptr,nullptr,nullptr);
            openStoreAsProject(state, QString::fromStdString(storePath.string()), "vc3d-umbilicus-test");
            state.setCurrentVolume(volume);
            int changed = 0, fibersChanged = 0;
            QObject::connect(&state, &CState::umbilicusChanged, [&] { ++changed; });
            QObject::connect(&controller, &Controller::fibersChanged, [&] { ++fibersChanged; });
            const auto attachment = *state.storeFor("fiber");
            require(!vc::objects::volumeUmbilicus(*state.objectStoreSession(), attachment, {volume->id()}) &&
                        !controller.umbilicusStatus().available,
                    "a volume without a stored umbilicus has none");
            auto add = [&](const std::string& name) {
                const auto uuid = vc::objects::util::newUuid();
                ObjectRecord record; record.uuid = uuid; record.kind = "umbilicus"; record.name = name;
                record.meta = nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(volume->id(), 2)}}.dump();
                auto request = state.objectStoreSession()->newRequest(attachment);
                request.expectedObjects = {{uuid, std::nullopt}};
                request.operations.push_back(vc::objects::CreateObjectOp{vc::objects::objectRecordJson(record)});
                request.operations.push_back(vc::objects::PutGeometryOp{uuid,
                    vc::geometry::encode(vc::geometry::Umbilicus{{{8, 6, 0}, {8, 6, 30}}}), {}});
                state.objectStoreSession()->submit(attachment, request);
                state.pollObjectStoreChanges();
                return uuid;
            };
            const auto first = add("scroll");
            const auto stored = vc::objects::volumeUmbilicus(*state.objectStoreSession(), attachment, {volume->id()});
            require(changed > 0, "committing an umbilicus announces it");
            require(fibersChanged == 0, "a commit touching no fiber leaves the fiber caches alone");
            require(stored && stored->uuid == first &&
                    stored->points == std::vector<vc::geometry::Point>{{4, 3, 0}, {4, 3, 15}},
                    "the stored umbilicus is read in the volume's voxels");
            require(controller.umbilicusStatus().available && controller.umbilicusStatus().text == "scroll",
                    "the fiber map status names the stored umbilicus");
            const auto snapshot = controller.umbilicusForSnapshot(state.annotationVolume()->volume);
            require(snapshot.message.isEmpty() && snapshot.centers.size() == 2, "the fiber map reads the stored umbilicus");
            const auto token = controller.umbilicusDependency();
            add("second");
            require(controller.umbilicusStatus().text.contains("several") &&
                    controller.umbilicusDependency() != token,
                    "several umbilici bound to one volume are refused");
            state.closeProject();
        }
        {
            // A volume's point collections live in the store: edits reach it in a
            // debounced transaction, a new session loads them, and someone else's
            // commit reloads them; corrections (anchored) stay out.
            QTemporaryDir dir;
            const std::filesystem::path volumeRoot3 = fs::path(dir.path().toStdString())/"ct.zarr";
            const vcr::zarr::CreateOptions levelZero3{.shape = {16, 8, 8}, .chunks = {16, 8, 8}, .compressor = "", .overwrite = true};
            vcr::zarr::Writer::create(volumeRoot3 / "0", levelZero3);
            std::ofstream(volumeRoot3 / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "points-volume"}, {"name", "points-volume"}, {"format", "zarr"}, {"width", 8}, {"height", 8}, {"slices", 16}, {"voxelsize", 1.0}}.dump();
            const auto volume = Volume::New(volumeRoot3);
            const auto storePath = fs::path(dir.path().toStdString())/"objects.vcobj";
            OpenOptions pointOptions; pointOptions.author = "vc3d-points-test";
            { auto db = ObjectStore::create(storePath, pointOptions); }
            auto settle = [](const std::function<bool()>& done) {
                QElapsedTimer timer; timer.start();
                while (!done() && timer.elapsed() < 5000) { QApplication::processEvents(); QThread::msleep(5); }
                return done();
            };
            CState state;
            openStoreAsProject(state, QString::fromStdString(storePath.string()), "vc3d-points-test");
            state.setCurrentVolume(volume);
            auto* points = state.pointCollection();
            const auto first = points->addPoint("seeds", {1, 2, 3});
            points->addPoint("seeds", {4, 5, 6});
            const auto anchored = points->addCollection("correction");
            points->setCollectionAnchor2d(anchored, cv::Vec2f(1, 1));
            points->addPoint("correction", {7, 7, 7});
            const auto attachment = *state.storeFor("fiber");
            auto stored = [&] { return vc::objects::volumeCollections(*state.objectStoreSession(), attachment, {volume->id()}); };
            require(settle([&] { return stored().size() == 1 && stored()[0].points.points.size() == 2; }),
                    "an edited collection reaches the store after the debounce");
            require(stored()[0].name == "seeds", "the store holds the collection's name");
            auto moved = *points->getPoint({first.collectionId, first.id});
            moved.p = {9, 9, 9};
            points->updatePoint(moved);
            require(settle([&] { return stored()[0].points.points[0] == std::array<double, 3>{9, 9, 9}; }),
                    "a moved point is written back keeping its id");
            {
                CState other;
                openStoreAsProject(other, QString::fromStdString(storePath.string()), "vc3d-points-test");
                other.setCurrentVolume(volume);
                const auto& loaded = other.pointCollection()->getAllCollections();
                require(loaded.size() == 1 && loaded.begin()->second.name == "seeds" && loaded.begin()->second.points.size() == 2,
                        "a new session loads the volume's collections");
                other.closeProject();
            }
            auto edited = stored()[0];
            edited.points.points[1] = {3, 3, 3};
            vc::objects::writeVolumeCollections(*state.objectStoreSession(), attachment, {volume->id()}, {edited}, {});
            state.pollObjectStoreChanges();
            require(settle([&] {
                        const auto found = points->getAllCollections().find(*points->getCollectionId("seeds"));
                        return std::any_of(found->second.points.begin(), found->second.points.end(),
                                           [](const auto& p) { return p.second.p == cv::Vec3f(3, 3, 3); });
                    }), "someone else's commit reloads the collection");
            require(points->getCollectionId("correction").has_value(), "a reload keeps the corrections");
            points->clearCollection(*points->getCollectionId("seeds"));
            require(settle([&] { return stored().empty(); }), "a removed collection is retired");
            state.closeProject();
        }
        {
            // Fibers bound by scale or through a registration are read and edited
            // in the open volume's voxels and saved back to their stored frame.
            QTemporaryDir dir;
            const std::filesystem::path volumeRoot4 = fs::path(dir.path().toStdString())/"ct.zarr";
            const vcr::zarr::CreateOptions levelZero4{.shape = {16, 16, 16}, .chunks = {16, 16, 16}, .compressor = "", .overwrite = true};
            vcr::zarr::Writer::create(volumeRoot4 / "0", levelZero4);
            std::ofstream(volumeRoot4 / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "frame-volume"}, {"name", "frame-volume"}, {"format", "zarr"}, {"width", 16}, {"height", 16}, {"slices", 16}, {"voxelsize", 1.0}}.dump();
            const auto volume = Volume::New(volumeRoot4);
            const auto storePath = fs::path(dir.path().toStdString())/"objects.vcobj";
            OpenOptions frameOptions; frameOptions.author = "vc3d-frame-test";
            VolumeAffine shear; shear(0,0)=2; shear(0,1)=.5; shear(0,3)=1; shear(2,3)=3;
            std::string scaled, registered, scaledLong;
            const std::vector<vc::geometry::Point> controls{{2,4,6},{10,4,6}}, line{{2,4,6},{6,4,6},{10,4,6}};
            const std::vector<vc::geometry::Point> longControls{{2,4,6},{6,4,6},{10,4,6},{14,4,6}};
            {
                auto db = ObjectStore::create(storePath, frameOptions);
                auto txn = db.write();
                (void)db.registerVolumeTransform(txn, "other-volume", "frame-volume", shear, "scan registration");
                auto add = [&](const nlohmann::json& meta, const std::vector<vc::geometry::Point>& cps,
                               const std::vector<vc::geometry::Point>& samples) {
                    ObjectRecord record; record.kind = "fiber"; record.meta = meta.dump();
                    vc::objects::FiberDetail detail; detail.width = 8; record.fiber = detail;
                    const auto uuid = db.upsertObject(txn, record);
                    vc::geometry::Fiber fiber; fiber.controlPoints = cps; fiber.linePoints = samples;
                    db.putGeometry(txn, uuid, vc::geometry::encode(fiber));
                    return uuid;
                };
                // Two stored units per voxel: the volume sees half the coordinates.
                scaled = add(nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(volume->id(), 2)}}, controls, line);
                registered = add({{"coordinates", {{"volume_uuid", "other-volume"}, {"scale_factor", 1}}}}, controls, line);
                scaledLong = add(nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(volume->id(), 2)}},
                                 longControls, longControls);
                txn.commit();
            }
            CState state;
            Controller controller(&state,nullptr,nullptr,nullptr);
            controller._errorDialogsSuppressed = true;
            openStoreAsProject(state, QString::fromStdString(storePath.string()), "vc3d-frame-test");
            state.setCurrentVolume(volume);
            auto read = [&](const std::string& uuid) {
                auto fibers = controller.readStoredFibers(std::vector<std::string>{uuid}, true);
                require(fibers.size() == 1 && fibers.front().editorError.empty(), "bound fiber decodes");
                return fibers.front();
            };
            auto stored = [&](const std::string& uuid) {
                vc::geometry::Fiber fiber;
                state.objectStoreSession()->readStore(*state.storeFor("fiber"), [&](const StoreReader& reader, const auto&) {
                    fiber = GeometrySnapshot(reader).fiber(uuid).load();
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
            // An open session of a bound fiber is in the volume's voxels too. Its
            // snapshot carries the frame, so a save and the fibers split or merged
            // from it are written in the stored frame and stay bound to the volume.
            {
                namespace la = vc3d::line_annotation;
                const auto opened = read(scaledLong);
                auto live = std::make_shared<Controller::LineAnnotationSession>();
                live->surfaceName = "frame-test-session"; live->fiberId = opened.id; live->fiberUuid = opened.uuid;
                live->fiberFileName = opened.fileName; live->fiberUsername = opened.username;
                live->fiberStartedAt = opened.startedAt; live->fiberSequence = opened.sequence;
                live->storePin = opened.storePin; live->graphCommit = opened.graphCommit; live->coordinateIdentity = opened.coordinateIdentity; live->frame = opened.frame;
                live->fiberOptimizationMode = opened.optimizationMode; live->suppressGeneratedViews = true; live->suppressErrorDialogs = true;
                for (size_t i = 0; i < opened.controlPoints.size(); ++i) {
                    la::LineControlPoint control(double(i), opened.controlPoints[i], i == 0, int(i));
                    control.storeId = opened.controlPoints[i].storeId;
                    live->controlPoints.push_back(control);
                }
                for (const auto& point : opened.linePoints) { vc::lasagna::LinePoint sample; sample.position = point; live->optimizedLine.points.push_back(sample); }
                live->controlPoints[3].volumePoint[0] += 1; live->optimizedLine.points[3].position[0] += 1;
                auto snapshot = controller.makeStoredFiberSessionSnapshot(*live).fiber;
                require(snapshot.frame && snapshot.coordinateIdentity == opened.coordinateIdentity, "session snapshot carries the stored frame");
                std::vector<Controller::FiberSaveSnapshot> save{controller.makeFiberSaveSnapshot(snapshot)};
                controller.commitFiberStoreAction(save);
                require(stored(scaledLong).controlPoints[3] == vc::geometry::Point{16,4,6}, "a session save maps back to the stored frame");
                live->storePin = save.front().fiber.storePin;
                live->graphCommit = save.front().fiber.graphCommit;
                const auto parent = controller.makeStoredFiberSessionSnapshot(*live).fiber;
                const auto identity = [&](const char* name) {
                    la::NewFiberIdentity result; result.id = controller.nextFiberId(); result.username = "test";
                    result.startedAt = "20260101"; result.sequence = result.id; result.fileName = name; return result;
                };
                std::string error;
                const auto halves = la::planFiberSplit(parent, 1, identity("frame-prefix.json"), identity("frame-suffix.json"), true, &error);
                if (!halves) throw std::runtime_error(error);
                Controller::StructuralEditRequest split; split.verb = "split";
                split.newFibers = {halves->prefix, halves->suffix}; split.originals = {parent};
                bool committed = false;
                const auto splitResult = controller.commitStructuralEdit(std::move(split), &committed);
                if (!splitResult.ok) throw std::runtime_error("split: " + splitResult.error);
                const auto storedPrefix = stored(halves->prefix.uuid), storedSuffix = stored(halves->suffix.uuid);
                require(storedPrefix.controlPoints == std::vector<vc::geometry::Point>{{2,4,6},{6,4,6}} &&
                        storedSuffix.controlPoints == std::vector<vc::geometry::Point>{{10,4,6},{16,4,6}},
                        "split halves of an open session are written in the stored frame");
                const auto prefix = read(halves->prefix.uuid), suffix = read(halves->suffix.uuid);
                require(prefix.coordinateIdentity == opened.coordinateIdentity && controller.fiberInVolume(prefix) &&
                        controller.fiberInVolume(suffix), "split halves keep the parent's frame");
                const auto merge = la::planFiberMerge(prefix, 1, suffix, 0, parent.optimizationMode, identity("frame-merged.json"), &error);
                if (!merge) throw std::runtime_error(error);
                Controller::StructuralEditRequest join; join.verb = "merge";
                join.newFibers = {merge->merged}; join.originals = {prefix, suffix};
                const auto mergeResult = controller.commitStructuralEdit(std::move(join), &committed);
                if (!mergeResult.ok) throw std::runtime_error("merge: " + mergeResult.error);
                require(stored(merge->merged.uuid).controlPoints == std::vector<vc::geometry::Point>{{2,4,6},{6,4,6},{10,4,6},{16,4,6}},
                        "a merge of bound fibers is written in the stored frame");
            }
            state.closeProject();
        }
        OpenOptions options; options.author = "vc3d-controller-test";
        { auto db = ObjectStore::create(path,options); }
        CState state;
        Controller controller(&state,nullptr,nullptr,nullptr);
        controller._errorDialogsSuppressed = true;
        openStoreAsProject(state, QString::fromStdString(path.string()),"vc3d-controller-test");
        auto transferJson = [&](const Controller::StoredFiber& fiber) {
            nlohmann::json result;
            state.objectStoreSession()->readStore(fiber.storePin->attachment, [&](const StoreReader& reader, const auto&) {
                result = nlohmann::json::parse(fiberVersionJson(reader, fiber.storePin->reference()));
            });
            return result;
        };
        Controller::StoredFiber a;
        a.id = controller.nextFiberId(); a.fileName = "same_name.json";
        a.username = "test"; a.startedAt = "20260101"; a.sequence = 1;
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
        {
            auto beforeReset = std::vector<Controller::FiberSaveSnapshot>{controller.makeFiberSaveSnapshot(a)};
            { auto db = ObjectStore::open(path, options); db.startHistoryGeneration("restored backup"); }
            bool rejected = false;
            try { controller.commitFiberStoreAction(beforeReset); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "fiber opened before reset rejects edits");
            a = resolve(a.uuid); b = resolve(b.uuid);
            require(a.storePin->reference().generation != a.storePin->writeGeneration,
                    "reopened fiber retains its publication generation");
            const auto publication = a.storePin->reference();
            a.tags.push_back("after-reset");
            batch = {controller.makeFiberSaveSnapshot(a)};
            controller.commitFiberStoreAction(batch);
            a = batch.front().fiber;
            require(a.storePin->reference().version == publication.version + 1 &&
                    a.storePin->reference().generation == a.storePin->writeGeneration,
                    "reopened fiber saves in the active generation");
            a.tags.push_back("second-save");
            batch = {controller.makeFiberSaveSnapshot(a)};
            controller.commitFiberStoreAction(batch);
            a = batch.front().fiber;
            require(a.storePin->reference().version == publication.version + 2, "subsequent fiber saves use the new pin");
        }
        // The preceding saves advanced the graph baseline of a only. Reopen
        // both endpoints before the independent link-creation action.
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
        {
            const auto retained = controller._sideStripStoredSnapshotCache.at(b.id).snapshot.polyline;
            (void)controller.fiberSnapshotsForSideStripQuery({a});
            require(controller._sideStripStoredSnapshotCache.at(b.id).snapshot.polyline == retained,
                    "visiting a different candidate set retains immutable stored snapshots");
        }
        collection.selectFiber(a.id, false, false);
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
            const auto attachment = *state.storeFor("fiber");
            const auto unrelated = vc::objects::util::newUuid();
            {
                ObjectRecord record; record.uuid = unrelated; record.kind = "point_collection";
                auto request = connection->newRequest(attachment);
                request.expectedObjects = {{unrelated, std::nullopt}};
                request.operations.push_back(vc::objects::CreateObjectOp{vc::objects::objectRecordJson(record)});
                connection->submit(attachment, request);
            }
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
            openStoreAsProject(destination, QString::fromStdString(target.string()), "roundtrip-test");
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
            destination.closeProject();
            { auto db = ObjectStore::open(target, options); auto txn = db.write(); db.seal(txn); txn.commit(); }
            const auto overlay = fs::path(transfers.path().toStdString()) / "overlay.vcobj";
            { auto collection = ObjectCollection::create(overlay, {target}, options); }
            openStoreAsProject(destination, QString::fromStdString(overlay.string()), "inherited-graph-test");
            auto stale = imported.readStoredFibers(std::nullopt).front();
            const auto loaded = stale.storePin->reference();
            { auto db = ObjectStore::open(overlay, options); auto txn = db.write();
              db.setApproval(txn, stale.branches.front().edgeUuid, "approved"); txn.commit(); }
            require(destination.objectStoreSession()->currentObject(stale.storePin->attachment, stale.uuid)->reference() == loaded,
                    "overlay relationship changes leave inherited endpoint identity unchanged");
            bool rejected = false;
            try { imported.commitFibers({&stale}); } catch (const TransactionConflict&) { rejected = true; }
            require(rejected, "loaded graph baseline rejects unseen overlay relationship changes");
        }
        {
            // Approval clears the mark on the shared edge.
            a = resolve(a.uuid);
            auto approved = a;
            approved.branches[0].pending = false;
            const auto generation = controller._fiberDataGeneration;
            controller.commitFibers({&approved});
            state.pollObjectStoreChanges();
            require(controller._fiberDataGeneration > generation && controller._sideStripStoredSnapshotCache.empty(),
                    "real commits invalidate fiber geometry snapshots");
            QElapsedTimer timer; timer.start();
            const auto approvedCatalog = [&] {
                return catalog.size() == 2 && std::all_of(catalog.begin(), catalog.end(),
                    [](const auto& row) { return row.pendingLinks == 0; });
            };
            while (!approvedCatalog() && timer.elapsed() < 10000) { QApplication::processEvents(); QThread::msleep(1); }
            require(approvedCatalog(), "real commits still refresh catalog link state");
            const auto again = resolve(a.uuid);
            require(!again.branches[0].pending && !again.branches[0].autoLink, "approval clears the automatic mark");
            auto stale = a;
            bool staleRejected = false;
            try { controller.commitFibers({&stale}); } catch (const TransactionConflict&) { staleRejected = true; }
            require(staleRejected, "a loaded graph cannot overwrite a later branch approval even with unchanged geometry");
            auto pending = again;
            pending.branches[0].pending = true;
            controller.commitFibers({&pending});
            const auto human = resolve(b.uuid);
            require(human.branches[0].pending && !human.branches[0].autoLink, "re-flagged link is a human pending link");
            a=resolve(a.uuid);b=resolve(b.uuid);
        }
        {
            // An open pane's links stay current through this controller's
            // own commits that change them (the linked fiber's saves, a new
            // link, a removed one), so its next save is not refused; an
            // unseen commit by another writer still refuses it.
            QTemporaryDir directory;
            const auto target = fs::path(directory.path().toStdString()) / "panes.vcobj";
            { auto db = ObjectStore::create(target, options); }
            CState panesState;
            Controller saving(&panesState, nullptr, nullptr, nullptr);
            saving._errorDialogsSuppressed = true;
            openStoreAsProject(panesState, QString::fromStdString(target.string()), "panes-test");
            auto newFiber = [&](double y) {
                Controller::StoredFiber fiber;
                fiber.id = saving.nextFiberId(); fiber.uuid = vc::objects::util::newUuid();
                fiber.fileName = "panes.json"; fiber.username = "test"; fiber.startedAt = "20260101"; fiber.sequence = 1;
                for (int x : {0,10,20}) { fiber.controlPoints.emplace_back(cv::Vec3d{x*1.,y,1}); fiber.linePoints.emplace_back(x*1.,y,1); }
                return fiber;
            };
            auto p = newFiber(0), q = newFiber(5), r = newFiber(10);
            Controller::FiberBranchRef link;
            link.controlPointIndex = 2; link.branchControlPointIndex = 0;
            link.branchFiberId = q.id; link.branchFileName = q.fileName;
            link.controlPointPosition = p.controlPoints[2]; link.branchControlPointPosition = q.controlPoints[0];
            link.controlPointDirection = {1,0,0}; link.branchControlPointDirection = {1,0,0};
            link.pending = true;
            p.branches.push_back(link);
            q.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(link, p.id, p.fileName));
            {
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(p),
                    saving.makeFiberSaveSnapshot(q), saving.makeFiberSaveSnapshot(r)};
                saving.commitFiberStoreAction(created);
            }
            auto read = [&](const std::string& uuid) {
                auto fibers = saving.readStoredFibers(std::vector<std::string>{uuid}, true);
                if (fibers.empty()) throw std::runtime_error("fiber missing");
                return fibers.front();
            };
            auto position = [&] {
                int64_t at = 0;
                panesState.objectStoreSession()->readStore(*panesState.storeFor("fiber"),
                    [&](const StoreReader& reader, const auto&) { at = reader.status().commitSequence; });
                return at;
            };
            auto tagged = [&](const std::string& uuid, const std::string& tag) {
                const auto tags = read(uuid).tags;
                return std::find(tags.begin(), tags.end(), tag) != tags.end();
            };
            // Fibers open in panes, as VC3D opens them.
            const auto openPane = [&](const Controller::StoredFiber& original, const std::string& name, bool reversed = false) {
                auto pane = std::make_shared<Controller::LineAnnotationSession>();
                pane->surfaceName = name;
                pane->fiberId = original.id; pane->fiberUuid = original.uuid;
                pane->fiberFileName = original.fileName; pane->fiberUsername = original.username;
                pane->fiberStartedAt = original.startedAt; pane->fiberSequence = original.sequence;
                pane->storePin = original.storePin; pane->coordinateIdentity = original.coordinateIdentity;
                pane->graphCommit = original.graphCommit; pane->frame = original.frame;
                pane->fiberOptimizationMode = original.optimizationMode; pane->fiberTags = original.tags;
                pane->fiberWidth = original.width; pane->branches = original.branches;
                pane->suppressGeneratedViews = true; pane->suppressErrorDialogs = true;
                for (size_t i = 0; i < original.controlPoints.size(); ++i) {
                    const auto& cp = original.controlPoints[i];
                    vc3d::line_annotation::LineControlPoint control(double(i), cp, i == 0, int(i));
                    control.storeId = cp.storeId; control.direction = cp.direction;
                    pane->controlPoints.push_back(control);
                }
                // Session order is free (stored order is by line position):
                // a reversed pane holds the same controls at other indices.
                if (reversed) {
                    const int last = int(pane->controlPoints.size()) - 1;
                    std::reverse(pane->controlPoints.begin(), pane->controlPoints.end());
                    for (auto& branch : pane->branches) branch.controlPointIndex = last - branch.controlPointIndex;
                }
                for (const auto& point : original.linePoints) {
                    vc::lasagna::LinePoint linePoint; linePoint.position = point; pane->optimizedLine.points.push_back(linePoint);
                }
                Controller::PaneRecord record; record.surfaceName = name; record.session = pane;
                saving._panes.push_back(record);
                return pane;
            };
            const auto original = read(q.uuid);
            auto qPane = openPane(original, "panes-q");
            // A second pane of q, its controls in another session order.
            auto qMirror = openPane(original, "panes-q-reversed", true);
            const auto sessionIndexOf = [](const auto& pane, int64_t storeId) {
                for (size_t i = 0; i < pane->controlPoints.size(); ++i)
                    if (pane->controlPoints[i].storeId == storeId) return int(i);
                return -1;
            };
            // A pane's links against the store's: the same edges, each from the
            // same control (by store id) to the same linked control.
            const auto paneMatchesStore = [&](const auto& pane) {
                const auto stored = read(pane->fiberUuid);
                if (stored.branches.size() != pane->branches.size()) return false;
                for (const auto& link : stored.branches) {
                    const auto own = sessionIndexOf(pane, stored.controlPoints.at(size_t(link.controlPointIndex)).storeId);
                    if (own < 0) return false;
                    // The linked control: in the linked fiber's open pane's order
                    // when it is open (the last such pane), else in stored order.
                    int linkedIndex = link.branchControlPointIndex;
                    for (const auto& open : saving._panes) {
                        if (!open.session || open.session == pane || open.session->fiberId != link.branchFiberId) continue;
                        for (size_t i = 0; i < open.session->controlPoints.size(); ++i)
                            if (vc3d::line_annotation::pointsApproximatelyEqual(open.session->controlPoints[i].volumePoint,
                                                                              link.branchControlPointPosition)) {
                                linkedIndex = int(i);
                                break;
                            }
                    }
                    if (std::none_of(pane->branches.begin(), pane->branches.end(), [&](const auto& held) {
                            return held.edgeUuid == link.edgeUuid && held.controlPointIndex == own &&
                                held.controlPointPosition == pane->controlPoints[size_t(own)].volumePoint &&
                                vc3d::line_annotation::pointsApproximatelyEqual(held.branchControlPointPosition,
                                                                              link.branchControlPointPosition) &&
                                held.branchFiberId == link.branchFiberId && held.branchControlPointIndex == linkedIndex &&
                                held.pending == link.pending && held.adjacent == link.adjacent && held.autoLink == link.autoLink;
                        }))
                        return false;
                }
                return true;
            };
            require(qPane->branches.size() == 1, "the pane loads q's link");
            const auto saveQ = [&](const std::string& tag) {
                qPane->fiberTags.push_back(tag);
                auto snapshot = saving.makeStoredFiberSessionSnapshot(*qPane);
                saving.commitFibers({&snapshot.fiber});
                return snapshot.fiber;
            };
            // The linked fiber's save rewrites the shared link.
            { auto linked = read(p.uuid); linked.tags.push_back("p-tag"); saving.commitFibers({&linked}); }
            require(qPane->graphCommit == position() && qPane->branches.size() == 1,
                    "the linked fiber's save keeps the pane's links current");
            saveQ("after-linked-save");
            require(tagged(q.uuid, "after-linked-save") && read(q.uuid).branches.size() == 1,
                    "the pane saves past the linked fiber's save");
            // A new link to q from another fiber appears in the pane, and the
            // pane's save keeps it.
            {
                auto linking = read(r.uuid);
                Controller::FiberBranchRef added = link;
                added.controlPointIndex = 0; added.branchControlPointIndex = 2;
                added.branchFiberId = q.id; added.controlPointPosition = linking.controlPoints[0];
                added.branchControlPointPosition = original.controlPoints[2];
                linking.branches.push_back(added);
                saving.commitFibers({&linking});
            }
            require(qPane->graphCommit == position() && qPane->branches.size() == 2 && paneMatchesStore(qPane) &&
                    qMirror->graphCommit == position() && paneMatchesStore(qMirror),
                    "a new link to the pane's fiber appears in every pane of it, at each pane's own control");
            const auto toR = std::find_if(qPane->branches.begin(), qPane->branches.end(),
                                          [&](const auto& branch) { return branch.branchFiberId == r.id; });
            require(toR != qPane->branches.end() && toR->controlPointIndex == 2 && toR->branchControlPointIndex == 0 &&
                    !toR->edgeUuid.empty(), "the new link's pane end is the control it was made to");
            saveQ("after-new-link");
            require(tagged(q.uuid, "after-new-link") && read(q.uuid).branches.size() == 2,
                    "the pane's save keeps the new link");
            // The linked fiber inserts a control before its linked one: the
            // pane's link follows it to the new index.
            {
                auto linked = read(p.uuid);
                linked.controlPoints.insert(linked.controlPoints.begin() + 1, vc3d::line_annotation::StoredControlPoint{cv::Vec3d{5,0,1}});
                linked.linePoints.insert(linked.linePoints.begin() + 1, cv::Vec3d{5,0,1});
                for (auto& branch : linked.branches) ++branch.controlPointIndex;
                saving.commitFibers({&linked});
            }
            const auto toP = std::find_if(qPane->branches.begin(), qPane->branches.end(),
                                          [&](const auto& branch) { return branch.branchFiberId == p.id; });
            require(qPane->graphCommit == position() && toP != qPane->branches.end() && toP->branchControlPointIndex == 3,
                    "the pane's link follows the linked fiber's control to its new index");
            saveQ("after-insert");
            const auto fromP = read(p.uuid);
            require(tagged(q.uuid, "after-insert") && fromP.branches.size() == 1 &&
                    fromP.branches.front().controlPointIndex == 3, "the pane's save keeps the link at that control");
            // The other fiber removes its link: the pane drops it and its save
            // does not bring it back.
            { auto unlinking = read(r.uuid); unlinking.branches.clear(); saving.commitFibers({&unlinking}); }
            require(qPane->graphCommit == position() && qPane->branches.size() == 1 && paneMatchesStore(qPane) &&
                    qMirror->graphCommit == position() && paneMatchesStore(qMirror), "a removed link leaves the panes");
            saveQ("after-removal");
            require(tagged(q.uuid, "after-removal") && read(q.uuid).branches.size() == 1 && read(r.uuid).branches.empty(),
                    "the pane's save does not restore the removed link");
            // A link made from r's pane to q (open in its own pane) is in both
            // panes with its edge id, and survives q's later saves.
            {
                auto rPane = openPane(read(r.uuid), "panes-r");
                // What linkSessionControlToStoredControl commits (it also needs a
                // bound volume, which this store has none of): the pane's ref
                // without an edge id, the pane's fiber and a stored copy of q.
                {
                    const auto far = read(q.uuid);
                    Controller::FiberBranchRef localRef;
                    localRef.controlPointIndex = 0; localRef.branchControlPointIndex = 1;
                    localRef.branchFiberId = q.id; localRef.branchFileName = far.fileName;
                    localRef.controlPointPosition = rPane->controlPoints[0].volumePoint;
                    localRef.branchControlPointPosition = far.controlPoints[1];
                    localRef.controlPointDirection = {1,0,0}; localRef.branchControlPointDirection = {1,0,0};
                    localRef.pending = true;
                    rPane->branches.push_back(localRef);
                    auto localFiber = saving.storedFiberFromSession(*rPane);
                    auto farFiber = far;
                    const auto stored = std::find_if(localFiber.branches.begin(), localFiber.branches.end(),
                                                     [&](const auto& branch) { return branch.branchFiberId == q.id; });
                    require(stored != localFiber.branches.end(), "the pane's link serializes");
                    farFiber.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(*stored, localFiber.id, localFiber.fileName));
                    saving.commitFibers({&localFiber, &farFiber});
                }
                const auto toQ = std::find_if(rPane->branches.begin(), rPane->branches.end(),
                                              [&](const auto& branch) { return branch.branchFiberId == q.id; });
                const auto fromR = std::find_if(qPane->branches.begin(), qPane->branches.end(),
                                                [&](const auto& branch) { return branch.branchFiberId == r.id; });
                require(toQ != rPane->branches.end() && !toQ->edgeUuid.empty() && fromR != qPane->branches.end() &&
                        fromR->edgeUuid == toQ->edgeUuid && fromR->controlPointIndex == 1 && qPane->branches.size() == 2 &&
                        qPane->graphCommit == position() && rPane->graphCommit == position(),
                        "a link made in one pane is in the linked fiber's pane, with its edge id");
                { auto linked = read(p.uuid); linked.tags.push_back("after-pane-link"); saving.commitFibers({&linked}); }
                saveQ("after-pane-link");
                require(read(q.uuid).branches.size() == 2 && read(r.uuid).branches.size() == 1 &&
                        paneMatchesStore(qPane) && paneMatchesStore(qMirror) && paneMatchesStore(rPane) &&
                        qMirror->graphCommit == position(),
                        "the pane-made link survives the linked fiber's pane's save");
                std::erase_if(saving._panes, [&](const auto& pane) { return pane.session == rPane; });
            }
            // Approving a link in one pane of q reaches the other pane of it.
            {
                const auto toggle = [&](bool pending, const char* tag) {
                    for (auto& branch : qPane->branches) if (branch.branchFiberId == p.id) branch.pending = pending;
                    saveQ(tag);
                    const auto mirrored = std::find_if(qMirror->branches.begin(), qMirror->branches.end(),
                                                       [&](const auto& branch) { return branch.branchFiberId == p.id; });
                    return mirrored != qMirror->branches.end() && mirrored->pending == pending &&
                        qMirror->graphCommit == position() && paneMatchesStore(qMirror) && paneMatchesStore(qPane);
                };
                // The other pane has a restorable edit armed: its copy takes the
                // approval too, so restoring it does not bring the old state back.
                qMirror->branchesBeforeModeChange = qMirror->branches;
                qMirror->controlPointsBeforeModeChange = qMirror->controlPoints;
                require(toggle(false, "approved"), "an approval in one pane of a fiber reaches its other panes");
                require(std::all_of(qMirror->branchesBeforeModeChange->begin(), qMirror->branchesBeforeModeChange->end(),
                                    [&](const auto& branch) { return branch.branchFiberId != p.id || !branch.pending; }),
                        "an armed pane's restorable links take the approval");
                // The linked fiber moves its linked control: the restorable
                // copy follows it, so a restore brings back no older endpoint.
                {
                    auto moved = read(p.uuid);
                    const auto linkedAt = size_t(moved.branches.front().controlPointIndex);
                    moved.controlPoints[linkedAt][1] += 1; moved.linePoints[linkedAt][1] += 1;
                    moved.branches.front().controlPointPosition = moved.controlPoints[linkedAt];
                    saving.commitFibers({&moved});
                }
                require(qMirror->graphCommit == position() && paneMatchesStore(qMirror), "the armed pane follows the moved control");
                // What was approved is no longer what is stored: the link is
                // pending again, in the store and in every pane.
                const auto pendingToP = [&](const std::vector<Controller::FiberBranchRef>& branches) {
                    const auto toP = std::find_if(branches.begin(), branches.end(),
                                                  [&](const auto& branch) { return branch.branchFiberId == p.id; });
                    return toP != branches.end() && toP->pending;
                };
                require(pendingToP(read(q.uuid).branches) && pendingToP(qPane->branches) && pendingToP(qMirror->branches) &&
                        pendingToP(*qMirror->branchesBeforeModeChange), "an approved link whose control moved is pending again");
                qMirror->branches = *qMirror->branchesBeforeModeChange; // the restore
                qMirror->controlPoints = *qMirror->controlPointsBeforeModeChange;
                require(paneMatchesStore(qMirror), "the restored links are the stored ones");
                qMirror->branchesBeforeModeChange.reset(); qMirror->controlPointsBeforeModeChange.reset();
                // An approved link stays approved through saves that leave its
                // ends where they are.
                require(toggle(false, "approved-again"), "the link is approved again");
                { auto linked = read(p.uuid); linked.tags.push_back("approved-kept"); saving.commitFibers({&linked}); }
                saveQ("approved-kept");
                {
                    const auto stored = read(q.uuid);
                    const auto toP = std::find_if(stored.branches.begin(), stored.branches.end(),
                                                  [&](const auto& branch) { return branch.branchFiberId == p.id; });
                    require(toP != stored.branches.end() && !toP->pending && paneMatchesStore(qPane) && paneMatchesStore(qMirror),
                            "an approved link stays approved through saves that do not move it");
                }
                require(toggle(true, "re-flagged"), "re-flagging in one pane reaches its other panes");
            }
            // A pane of the linked fiber with a control it has not saved yet
            // (its indices are shifted): q's save still anchors the link at the
            // linked control of p as stored.
            {
                auto pPane = openPane(read(p.uuid), "panes-p");
                vc3d::line_annotation::LineControlPoint unsaved(-0.5, cv::Vec3d{-5,0,1}, false, -1);
                pPane->controlPoints.push_back(unsaved);
                saveQ("past-unsaved-peer-control");
                const auto fromP = read(p.uuid);
                require(fromP.branches.size() == 1 && fromP.branches.front().controlPointIndex == 3 &&
                        fromP.controlPoints.size() == 4, "the link stays at the linked control as stored");
                std::erase_if(saving._panes, [&](const auto& pane) { return pane.session == pPane; });
            }
            // An unlink made while a save of the same fiber is in flight (its
            // snapshot still has the link): installing that save does not bring
            // the link back into the pane, and the unlink's save removes it.
            {
                std::vector<Controller::FiberSaveSnapshot> inFlight{
                    saving.makeFiberSaveSnapshot(saving.makeStoredFiberSessionSnapshot(*qPane).fiber)};
                auto job = saving.prepareFiberCommit(inFlight, {});
                std::erase_if(qPane->branches, [&](const auto& branch) { return branch.branchFiberId == r.id; });
                auto result = Controller::executeFiberCommit(job, *panesState.objectStoreSession());
                saving.installFiberCommit(job, result);
                saving.applyCommittedFibers(job.edits, false);
                require(qPane->branches.size() == 1 && qPane->graphCommit == position(),
                        "a save in flight does not undo an unlink made meanwhile");
                saveQ("after-unlink");
                require(read(q.uuid).branches.size() == 1 && read(r.uuid).branches.empty() &&
                        paneMatchesStore(qPane) && paneMatchesStore(qMirror) && qMirror->graphCommit == position(),
                        "the unlink's save removes the link, from every pane of the fiber");
            }
            // A pane with a restorable edit armed (its links may be restored
            // over what is installed): a link saved to its fiber from elsewhere
            // reaches the pane and its restorable copy, so the restored links
            // are saved with the new link kept.
            {
                qMirror->branchesBeforeModeChange = qMirror->branches;
                qMirror->controlPointsBeforeModeChange = qMirror->controlPoints;
                auto linking = read(r.uuid);
                auto linked = read(q.uuid);
                Controller::FiberBranchRef added = link;
                added.controlPointIndex = 0; added.branchControlPointIndex = 2; added.branchFiberId = q.id;
                added.controlPointPosition = linking.controlPoints[0]; added.branchControlPointPosition = linked.controlPoints[2];
                linking.branches.push_back(added);
                linked.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(added, linking.id, linking.fileName));
                saving.commitFibers({&linking, &linked});
                require(qMirror->graphCommit == position() && paneMatchesStore(qMirror) && qPane->graphCommit == position() &&
                        paneMatchesStore(qPane) && qMirror->branchesBeforeModeChange->size() == qMirror->branches.size(),
                        "an armed pane and its restorable copy take a link saved to its fiber");
                qMirror->branches = *qMirror->branchesBeforeModeChange; // the restore
                qMirror->controlPoints = *qMirror->controlPointsBeforeModeChange;
                qMirror->branchesBeforeModeChange.reset(); qMirror->controlPointsBeforeModeChange.reset();
                require(paneMatchesStore(qMirror), "the restored links are the stored ones");
                auto restored = saving.makeStoredFiberSessionSnapshot(*qMirror).fiber;
                restored.tags.push_back("restored");
                saving.commitFibers({&restored});
                require(tagged(q.uuid, "restored") && read(q.uuid).branches.size() == 2 && read(r.uuid).branches.size() == 1 &&
                        paneMatchesStore(qPane), "the armed pane's restored links are saved with the new link kept");
                std::erase_if(saving._panes, [&](const auto& pane) { return pane.session == qMirror; });
                { auto unlinking = read(r.uuid); unlinking.branches.clear(); saving.commitFibers({&unlinking}); }
            }
            // A collapse's rollback that cannot take a link saved to the fiber
            // (its controls lack the one the link is on): the pane keeps its
            // loaded position, so restoring the copy cannot save the link away.
            {
                auto qArmed = openPane(read(q.uuid), "panes-q-armed");
                Controller::LineAnnotationSession::ControlPointCollapseRollback rollback;
                rollback.controlPoints = qArmed->controlPoints;
                rollback.branches = qArmed->branches;
                rollback.controlPoints.pop_back(); // the copy predates q's last control
                qArmed->controlPointCollapseRollback = rollback;
                // Its other restorable copy could take the link: neither does.
                qArmed->branchesBeforeModeChange = qArmed->branches;
                qArmed->controlPointsBeforeModeChange = qArmed->controlPoints;
                const auto armedAt = qArmed->graphCommit;
                auto linking = read(r.uuid);
                auto linked = read(q.uuid);
                Controller::FiberBranchRef added = link;
                added.controlPointIndex = 0; added.branchControlPointIndex = 2; added.branchFiberId = q.id;
                added.controlPointPosition = linking.controlPoints[0]; added.branchControlPointPosition = linked.controlPoints[2];
                linking.branches.push_back(added);
                linked.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(added, linking.id, linking.fileName));
                saving.commitFibers({&linking, &linked});
                require(qArmed->graphCommit == armedAt && qArmed->branches.size() == 1 &&
                        qArmed->controlPointCollapseRollback->branches.size() == 1 &&
                        qArmed->branchesBeforeModeChange->size() == 1 && qPane->graphCommit == position() &&
                        paneMatchesStore(qPane), "a pane whose rollback cannot take a saved link keeps its loaded position, untouched");
                std::erase_if(saving._panes, [&](const auto& pane) { return pane.session == qArmed; });
                { auto unlinking = read(r.uuid); unlinking.branches.clear(); saving.commitFibers({&unlinking}); }
            }
            // A control replaced (same position, same rank) while the pane's
            // save is in flight is not the saved control: it does not take its
            // store id, and a link saved to q from elsewhere then leaves the
            // pane at its loaded position.
            {
                std::vector<Controller::FiberSaveSnapshot> inFlight{
                    saving.makeFiberSaveSnapshot(saving.makeStoredFiberSessionSnapshot(*qPane).fiber)};
                auto job = saving.prepareFiberCommit(inFlight, {});
                const auto replacedId = qPane->controlPoints[1].storeId;
                auto replacement = qPane->controlPoints[1];
                replacement.storeId = -1; replacement.identity = 0;
                qPane->controlPoints[1] = replacement;
                auto result = Controller::executeFiberCommit(job, *panesState.objectStoreSession());
                saving.installFiberCommit(job, result);
                saving.applyCommittedFibers(job.edits, false);
                require(replacedId >= 0 && qPane->controlPoints[1].storeId == -1,
                        "a control replaced while its save was in flight does not take the saved control's id");
                const auto before = qPane->graphCommit;
                auto linking = read(r.uuid);
                auto linked = read(q.uuid);
                Controller::FiberBranchRef added = link;
                added.controlPointIndex = 0; added.branchControlPointIndex = 1; added.branchFiberId = q.id;
                added.controlPointPosition = linking.controlPoints[0]; added.branchControlPointPosition = linked.controlPoints[1];
                linking.branches.push_back(added);
                linked.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(added, linking.id, linking.fileName));
                saving.commitFibers({&linking, &linked});
                require(qPane->graphCommit == before, "a pane with a replaced control is not brought past a link saved to it");
                std::erase_if(saving._panes, [&](const auto& pane) { return pane.session == qPane; });
                qPane = openPane(read(q.uuid), "panes-q");
                { auto unlinking = read(r.uuid); unlinking.branches.clear(); saving.commitFibers({&unlinking}); }
                require(qPane->branches.size() == 1 && qPane->graphCommit == position(), "the reopened pane follows on");
            }
            // An unlink in q's pane while a save of p (alone) is in flight: the
            // save does not bring the link back into q's pane, the pane is
            // brought to it, and the unlink's save removes the link.
            {
                auto linked = read(p.uuid); linked.tags.push_back("in-flight");
                std::vector<Controller::FiberSaveSnapshot> inFlight{saving.makeFiberSaveSnapshot(linked)};
                auto job = saving.prepareFiberCommit(inFlight, {});
                std::erase_if(qPane->branches, [&](const auto& branch) { return branch.branchFiberId == p.id; });
                auto result = Controller::executeFiberCommit(job, *panesState.objectStoreSession());
                saving.installFiberCommit(job, result);
                saving.applyCommittedFibers(job.edits, false);
                require(std::none_of(qPane->branches.begin(), qPane->branches.end(),
                                     [&](const auto& branch) { return branch.branchFiberId == p.id; }) &&
                        qPane->graphCommit == position() && read(p.uuid).branches.size() == 1,
                        "an unlink in another pane is not undone by a save in flight");
                saveQ("unlinked-in-flight");
                require(tagged(q.uuid, "unlinked-in-flight") && read(p.uuid).branches.empty() &&
                        paneMatchesStore(qPane), "the pane's unlink is saved past the save that was in flight");
                // Linked again for what follows.
                auto relinking = read(p.uuid);
                auto relinked = read(q.uuid);
                Controller::FiberBranchRef again = link;
                again.controlPointIndex = int(relinking.controlPoints.size()) - 1; again.branchControlPointIndex = 0;
                again.branchFiberId = q.id; again.branchFileName = relinked.fileName;
                again.controlPointPosition = relinking.controlPoints.back(); again.branchControlPointPosition = relinked.controlPoints[0];
                relinking.branches.push_back(again);
                relinked.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(again, relinking.id, relinking.fileName));
                saving.commitFibers({&relinking, &relinked});
                require(qPane->graphCommit == position() && paneMatchesStore(qPane), "the pane takes the link made again");
            }
            // Another writer's commit that this controller has not seen: the
            // pane's links are not known to be current, and its save is refused.
            {
                auto db = ObjectStore::open(target, options); auto txn = db.write();
                ObjectRecord record; record.uuid = vc::objects::util::newUuid(); record.kind = "point_collection";
                db.upsertObject(txn, record); txn.commit();
            }
            { auto linked = read(p.uuid); linked.tags.push_back("after-other-writer"); saving.commitFibers({&linked}); }
            require(qPane->graphCommit && *qPane->graphCommit + 2 == position(), "an unseen commit leaves the pane's links as loaded");
            bool refused = false;
            try { saveQ("after-other-writer"); } catch (const TransactionConflict&) { refused = true; }
            require(refused && !tagged(q.uuid, "after-other-writer"), "the pane's save past an unseen commit is refused");
            saving._panes.clear();
            // A structural edit swaps prebuilt redirected links into its panes
            // itself (applyCommittedFibers with installLinks false): a pane is
            // brought to the edit's position only when they are exactly the
            // saved links (one missing would be deleted by its next save).
            {
                {
                    auto linking = read(r.uuid);
                    auto linked = read(q.uuid);
                    Controller::FiberBranchRef added = link;
                    added.controlPointIndex = 0; added.branchControlPointIndex = 1; added.branchFiberId = q.id;
                    added.controlPointPosition = linking.controlPoints[0]; added.branchControlPointPosition = linked.controlPoints[1];
                    linking.branches.push_back(added);
                    linked.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(added, linking.id, linking.fileName));
                    saving.commitFibers({&linking, &linked});
                }
                const auto stored = read(q.uuid);
                require(stored.branches.size() == 2, "q has two links");
                Controller::FiberSaveSnapshot structural; structural.fiber = stored;
                structural.fiber.graphCommit = *stored.graphCommit + 1;
                auto partial = openPane(stored, "panes-q-partial");
                partial->branches.pop_back(); // the redirect lacks one saved link
                const auto partialAt = partial->graphCommit;
                auto exact = openPane(stored, "panes-q-exact");
                saving.applyCommittedFibers({structural}, true, false);
                require(partial->graphCommit == partialAt, "a pane whose redirected links lack a saved one keeps its position");
                require(exact->graphCommit == structural.fiber.graphCommit, "a pane whose redirected links are the saved ones takes it");
                saving._panes.clear();
            }
            // Two controls of the linked fiber at (almost) one position, the
            // stored link on the later one (VC3D itself resolves a link end by
            // position, so the store is written directly here): a save of the
            // other end alone keeps the link on that control.
            {
                auto twin = newFiber(30), other = newFiber(40);
                twin.controlPoints = {vc3d::line_annotation::StoredControlPoint{cv::Vec3d{0,30,1}},
                                      vc3d::line_annotation::StoredControlPoint{cv::Vec3d{10,30,1}},
                                      vc3d::line_annotation::StoredControlPoint{cv::Vec3d{10 + 5e-7,30,1}},
                                      vc3d::line_annotation::StoredControlPoint{cv::Vec3d{20,30,1}}};
                twin.linePoints = {cv::Vec3d{0,30,1}, cv::Vec3d{10,30,1}, cv::Vec3d{10 + 5e-7,30,1}, cv::Vec3d{20,30,1}};
                Controller::FiberBranchRef onLater = link;
                onLater.controlPointIndex = 0; onLater.branchControlPointIndex = 2;
                onLater.branchFiberId = twin.id; onLater.branchFileName = twin.fileName;
                onLater.controlPointPosition = other.controlPoints[0]; onLater.branchControlPointPosition = twin.controlPoints[2];
                other.branches.push_back(onLater);
                twin.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(onLater, other.id, other.fileName));
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(twin), saving.makeFiberSaveSnapshot(other)};
                saving.commitFiberStoreAction(created);
                const auto laterId = read(twin.uuid).controlPoints.at(2).storeId;
                {
                    auto db = ObjectStore::open(target, options);
                    auto txn = db.write();
                    const auto edgeUuid = read(other.uuid).branches.front().edgeUuid;
                    auto edge = *db.edgeByUuid(txn, edgeUuid);
                    auto& end = edge.aUuid == twin.uuid ? edge.aAnchor : edge.bAnchor;
                    end.points = {{"control_point_ids", {laterId}}};
                    end.position = std::array<double, 3>{10 + 5e-7, 30, 1};
                    db.updateEdge(txn, edge);
                    txn.commit();
                }
                panesState.pollObjectStoreChanges();
                saving._storedFiberDecodeCache.clear(); saving._storedFiberDependencies.clear();
                const auto anchored = read(twin.uuid);
                require(anchored.branches.size() == 1 && anchored.branches.front().controlPointIndex == 2,
                        "the stored link is on the later of two coincident controls");
                { auto tagged = read(other.uuid); tagged.tags.push_back("coincident"); saving.commitFibers({&tagged}); }
                const auto after = read(twin.uuid);
                require(after.branches.size() == 1 && after.branches.front().controlPointIndex == 2 &&
                        after.controlPoints.at(2).storeId == laterId, "the link stays on the control it is anchored to");
                // With the coincident fiber open, a save that rewrites its link
                // does not bring its pane past it (which control the link is on
                // cannot be followed by position).
                auto twinPane = openPane(read(twin.uuid), "panes-twin");
                const auto twinAt = twinPane->graphCommit;
                { auto tagged = read(other.uuid); tagged.tags.push_back("coincident-open"); saving.commitFibers({&tagged}); }
                require(twinPane->graphCommit == twinAt, "a pane of a fiber with coincident controls is not brought past a link save");
                saving._panes.clear();
            }
            // Two links between the same fibers, one fiber's pane in another
            // session order: links installed in the other pane name its controls
            // in that order, so unlinking one there (the real handler) removes
            // exactly that link on both ends.
            {
                saving._panes.clear();
                auto u = newFiber(50), v = newFiber(60), w = newFiber(70);
                Controller::FiberBranchRef first = link, second = link;
                first.controlPointIndex = 0; first.branchControlPointIndex = 2;
                first.branchFiberId = u.id; first.branchFileName = u.fileName;
                first.controlPointPosition = v.controlPoints[0]; first.branchControlPointPosition = u.controlPoints[2];
                second.controlPointIndex = 1; second.branchControlPointIndex = 0;
                second.branchFiberId = u.id; second.branchFileName = u.fileName;
                second.controlPointPosition = v.controlPoints[1]; second.branchControlPointPosition = u.controlPoints[0];
                v.branches = {first, second};
                u.branches = {vc3d::line_annotation::reciprocalBranchRef(first, v.id, v.fileName),
                              vc3d::line_annotation::reciprocalBranchRef(second, v.id, v.fileName)};
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(u), saving.makeFiberSaveSnapshot(v),
                                                                   saving.makeFiberSaveSnapshot(w)};
                saving.commitFiberStoreAction(created);
                auto uPane = openPane(read(u.uuid), "panes-u", true);
                auto vPane = openPane(read(v.uuid), "panes-v");
                saving.syncBranchEndpointPositions(*uPane);
                saving.syncBranchEndpointPositions(*vPane);
                const auto namesUInPaneOrder = [&](const std::vector<Controller::FiberBranchRef>& branches) {
                    // u's pane is reversed: stored control k is its pane's control 2 - k.
                    const auto stored = read(u.uuid);
                    return std::all_of(branches.begin(), branches.end(), [&](const auto& branch) {
                        if (branch.branchFiberId != u.id) return true;
                        const auto at = std::find_if(stored.controlPoints.begin(), stored.controlPoints.end(), [&](const auto& cp) {
                            return vc3d::line_annotation::pointsApproximatelyEqual(cp, branch.branchControlPointPosition);
                        });
                        return at != stored.controlPoints.end() &&
                            branch.branchControlPointIndex == 2 - int(at - stored.controlPoints.begin());
                    });
                };
                // u saved alone while v's pane has a restorable edit armed: both
                // v's links and the copy name u's controls in u's pane order, so a
                // restore leaves them so.
                vPane->branchesBeforeModeChange = vPane->branches;
                vPane->controlPointsBeforeModeChange = vPane->controlPoints;
                { auto tagged = read(u.uuid); tagged.tags.push_back("u-alone"); saving.commitFibers({&tagged}); }
                require(vPane->graphCommit == position() && namesUInPaneOrder(vPane->branches) &&
                        namesUInPaneOrder(*vPane->branchesBeforeModeChange), "v's links and their restorable copy name u in its pane order");
                vPane->branches = *vPane->branchesBeforeModeChange; // the restore
                vPane->controlPoints = *vPane->controlPointsBeforeModeChange;
                vPane->branchesBeforeModeChange.reset(); vPane->controlPointsBeforeModeChange.reset();
                {
                    auto linking = read(w.uuid);
                    auto linked = read(v.uuid);
                    Controller::FiberBranchRef added = link;
                    added.controlPointIndex = 0; added.branchControlPointIndex = 2; added.branchFiberId = v.id;
                    added.branchFileName = v.fileName;
                    added.controlPointPosition = linking.controlPoints[0]; added.branchControlPointPosition = linked.controlPoints[2];
                    linking.branches.push_back(added);
                    linked.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(added, linking.id, linking.fileName));
                    saving.commitFibers({&linking, &linked});
                }
                require(vPane->branches.size() == 3 && vPane->graphCommit == position(), "v's pane takes w's link");
                const auto toUFirst = std::find_if(vPane->branches.begin(), vPane->branches.end(), [&](const auto& branch) {
                    return branch.branchFiberId == u.id && branch.controlPointIndex == 0;
                });
                require(toUFirst != vPane->branches.end() && toUFirst->branchControlPointIndex == 0,
                        "v's link names u's control in u's pane order (u's stored control 2 is its pane's control 0)");
                const auto removedEdge = toUFirst->edgeUuid;
                const auto keptEdge = std::find_if(vPane->branches.begin(), vPane->branches.end(), [&](const auto& branch) {
                    return branch.branchFiberId == u.id && branch.controlPointIndex == 1;
                })->edgeUuid;
                require(!removedEdge.empty() && !keptEdge.empty() && removedEdge != keptEdge, "two distinct links between u and v");
                saving.handleGeneratedControlPointUnlink("panes-v", 0, u.id, toUFirst->branchControlPointIndex);
                const auto storedU = read(u.uuid), storedV = read(v.uuid);
                const auto has = [](const auto& branches, const std::string& edge) {
                    return std::any_of(branches.begin(), branches.end(), [&](const auto& branch) { return branch.edgeUuid == edge; });
                };
                require(storedU.branches.size() == 1 && storedV.branches.size() == 2 && uPane->branches.size() == 1 &&
                        vPane->branches.size() == 2 && !has(storedU.branches, removedEdge) && !has(storedV.branches, removedEdge) &&
                        has(storedU.branches, keptEdge) && !has(uPane->branches, removedEdge) && has(uPane->branches, keptEdge),
                        "the unlink removes exactly that link on both ends");
                saving._panes.clear();
            }
            // Two panes of one fiber: one replaces a linked control at its
            // position and saves (the store gives it a new id); the other pane
            // takes that id and stays saveable.
            {
                saving._panes.clear();
                auto x = newFiber(80), y = newFiber(90);
                Controller::FiberBranchRef xy = link;
                xy.controlPointIndex = 1; xy.branchControlPointIndex = 1;
                xy.branchFiberId = y.id; xy.branchFileName = y.fileName;
                xy.controlPointPosition = x.controlPoints[1]; xy.branchControlPointPosition = y.controlPoints[1];
                x.branches.push_back(xy);
                y.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(xy, x.id, x.fileName));
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(x), saving.makeFiberSaveSnapshot(y)};
                saving.commitFiberStoreAction(created);
                auto xA = openPane(read(x.uuid), "panes-x-a");
                auto xB = openPane(read(x.uuid), "panes-x-b");
                const auto oldId = xA->controlPoints[1].storeId;
                auto replacement = xA->controlPoints[1];
                replacement.storeId = -1; replacement.identity = 0;
                xA->controlPoints[1] = replacement;
                auto fromA = saving.makeStoredFiberSessionSnapshot(*xA).fiber;
                saving.commitFibers({&fromA});
                const auto newId = read(x.uuid).controlPoints[1].storeId;
                require(newId >= 0 && newId != oldId && xA->controlPoints[1].storeId == newId, "the replaced control has a new id");
                require(xB->controlPoints[1].storeId == newId && xB->graphCommit == position() && paneMatchesStore(xB),
                        "the other pane of the fiber takes the new id");
                xB->fiberTags.push_back("from-b");
                auto fromB = saving.makeStoredFiberSessionSnapshot(*xB).fiber;
                saving.commitFibers({&fromB});
                require(tagged(x.uuid, "from-b") && read(x.uuid).controlPoints[1].storeId == newId &&
                        read(y.uuid).branches.size() == 1, "the other pane saves");
                saving._panes.clear();
            }
            // A metadata-only save (width) of a fiber whose pane has a control it
            // has not saved yet: the pane is brought to that commit, so the
            // autosave that follows stores the new control with an id of its own.
            {
                saving._panes.clear();
                auto z = newFiber(100);
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(z)};
                saving.commitFiberStoreAction(created);
                auto zPane = openPane(read(z.uuid), "panes-z");
                vc3d::line_annotation::LineControlPoint added(3.0, cv::Vec3d{30,100,1}, false, -1);
                zPane->controlPoints.push_back(added);
                vc::lasagna::LinePoint end; end.position = cv::Vec3d{30,100,1}; zPane->optimizedLine.points.push_back(end);
                zPane->fiberWidth += 0.5;
                const auto zBefore = position();
                saving.saveFiberDisplayAnnotations(*zPane); // and the autosave it schedules
                const auto stored = read(z.uuid);
                require(position() == zBefore + 2 && zPane->graphCommit == position() && stored.controlPoints.size() == 4 &&
                        zPane->controlPoints.back().storeId >= 0 &&
                        zPane->controlPoints.back().storeId == stored.controlPoints.back().storeId,
                        "after a metadata-only save the pane's geometry save stores its new control with an id of its own");
                saving._panes.clear();
            }
            // A link anchor written again after its direction was normalized once
            // more (which need not give the same doubles) has not moved; one on
            // another control, or at another position, has.
            {
                const auto once = vc3d::line_annotation::normalizedOrZero(cv::Vec3d{1, 1, 7});
                const auto twice = vc3d::line_annotation::normalizedOrZero(once);
                vc::objects::Anchor stored, written;
                stored.points = written.points = {{"control_point_ids", {7}}};
                stored.position = written.position = std::array<double, 3>{1, 2, 3};
                stored.direction = std::array<double, 3>{once[0], once[1], once[2]};
                written.direction = std::array<double, 3>{twice[0], twice[1], twice[2]};
                require(!(stored == written) && !anchorMoved(stored, written, std::nullopt), "a renormalized direction is the same anchor");
                require(anchoredControlId(stored) == 7, "an anchor names its control");
                written.points = {{"control_point_ids", {8}}};
                require(anchorMoved(stored, written, std::nullopt), "an anchor on another control has moved");
                written.points = stored.points; written.position = std::array<double, 3>{1, 2, 3.5};
                require(anchorMoved(stored, written, std::nullopt), "an anchor at another position has moved");
                // Positions are compared where panes hold them: in a volume the
                // fiber is registered into at twice its stored scale, a stored
                // displacement under the tolerance is a move past it.
                written.position = std::array<double, 3>{1, 2, 3 + 7.5e-7};
                vc3d::line_annotation::FiberVolumeFrame doubled;
                doubled.storedToVolume = vc::objects::VolumeAffine::scale(2.0);
                require(!anchorMoved(stored, written, std::nullopt) && anchorMoved(stored, written, doubled),
                        "an anchor's move is measured in the frame its fiber is shown in");
            }
            // A save of g in flight; its linked control is then replaced at a new
            // position, and a width change saves g's metadata (from the stored
            // copy, links as stored). The pane is brought to that commit, and
            // the autosave stores the link on the replacement.
            {
                saving._panes.clear();
                auto g = newFiber(110), h = newFiber(120);
                Controller::FiberBranchRef gh = link;
                gh.controlPointIndex = 1; gh.branchControlPointIndex = 1;
                gh.branchFiberId = h.id; gh.branchFileName = h.fileName;
                gh.controlPointPosition = g.controlPoints[1]; gh.branchControlPointPosition = h.controlPoints[1];
                g.branches.push_back(gh);
                h.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(gh, g.id, g.fileName));
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(g), saving.makeFiberSaveSnapshot(h)};
                saving.commitFiberStoreAction(created);
                auto gPane = openPane(read(g.uuid), "panes-g");
                {
                    std::vector<Controller::FiberSaveSnapshot> inFlight{
                        saving.makeFiberSaveSnapshot(saving.makeStoredFiberSessionSnapshot(*gPane).fiber)};
                    auto job = saving.prepareFiberCommit(inFlight, {});
                    auto replacement = gPane->controlPoints[1];
                    replacement.storeId = -1; replacement.identity = 0; replacement.volumePoint = cv::Vec3d{12, 110, 1};
                    gPane->controlPoints[1] = replacement;
                    gPane->optimizedLine.points[1].position = replacement.volumePoint;
                    gPane->branches.front().controlPointPosition = replacement.volumePoint;
                    auto result = Controller::executeFiberCommit(job, *panesState.objectStoreSession());
                    saving.installFiberCommit(job, result);
                    saving.applyCommittedFibers(job.edits, false);
                }
                gPane->fiberWidth += 0.5;
                saving.saveFiberDisplayAnnotations(*gPane); // and the autosave it schedules
                const auto stored = read(g.uuid);
                const auto linkedAt = size_t(stored.branches.front().controlPointIndex);
                require(gPane->graphCommit == position() && stored.branches.size() == 1 &&
                        vc3d::line_annotation::pointsApproximatelyEqual(stored.controlPoints.at(linkedAt), cv::Vec3d{12, 110, 1}) &&
                        gPane->controlPoints[1].storeId >= 0 && stored.controlPoints.at(linkedAt).storeId == gPane->controlPoints[1].storeId &&
                        read(h.uuid).branches.size() == 1,
                        "after a width save, the pane's geometry save stores the link on the replaced control");
                saving._panes.clear();
            }
            // Linked fibers s and t in two panes, as an annotator alternating
            // between them: while a save of s is in flight, t's pane gets a
            // control of its own and replaces its linked control. t is brought
            // to s's commit with its own edits kept, and saves them.
            {
                saving._panes.clear();
                auto s = newFiber(210), t = newFiber(220);
                Controller::FiberBranchRef st = link;
                st.controlPointIndex = 1; st.branchControlPointIndex = 1;
                st.branchFiberId = t.id; st.branchFileName = t.fileName;
                st.controlPointPosition = s.controlPoints[1]; st.branchControlPointPosition = t.controlPoints[1];
                s.branches.push_back(st);
                t.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(st, s.id, s.fileName));
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(s), saving.makeFiberSaveSnapshot(t)};
                saving.commitFiberStoreAction(created);
                auto sPane = openPane(read(s.uuid), "panes-s");
                auto tPane = openPane(read(t.uuid), "panes-t");
                const auto saveInFlight = [&](const std::shared_ptr<Controller::LineAnnotationSession>& pane,
                                              const std::function<void()>& meanwhile) {
                    pane->fiberTags.push_back("in-flight-" + std::to_string(pane->fiberTags.size()));
                    std::vector<Controller::FiberSaveSnapshot> inFlight{
                        saving.makeFiberSaveSnapshot(saving.makeStoredFiberSessionSnapshot(*pane).fiber)};
                    auto job = saving.prepareFiberCommit(inFlight, {});
                    meanwhile();
                    auto result = Controller::executeFiberCommit(job, *panesState.objectStoreSession());
                    saving.installFiberCommit(job, result);
                    saving.applyCommittedFibers(job.edits, false);
                };
                const auto savePane = [&](const std::shared_ptr<Controller::LineAnnotationSession>& pane, const std::string& tag) {
                    pane->fiberTags.push_back(tag);
                    auto snapshot = saving.makeStoredFiberSessionSnapshot(*pane);
                    saving.commitFibers({&snapshot.fiber});
                };
                const auto replaceLinked = [&](const std::shared_ptr<Controller::LineAnnotationSession>& pane, const cv::Vec3d& at) {
                    const auto index = size_t(pane->branches.front().controlPointIndex);
                    auto replacement = pane->controlPoints[index];
                    replacement.storeId = -1; replacement.identity = 0; replacement.volumePoint = at;
                    pane->controlPoints[index] = replacement;
                    pane->optimizedLine.points[index].position = at;
                    pane->branches.front().controlPointPosition = at;
                };
                // s's save leaves the link as stored.
                saveInFlight(sPane, [&] {
                    vc3d::line_annotation::LineControlPoint added(3.0, cv::Vec3d{30, 220, 1}, false, 3);
                    tPane->controlPoints.push_back(added);
                    vc::lasagna::LinePoint end; end.position = cv::Vec3d{30, 220, 1}; tPane->optimizedLine.points.push_back(end);
                    replaceLinked(tPane, cv::Vec3d{12, 220, 1});
                });
                require(tPane->graphCommit == position() && tPane->controlPoints[1].storeId == -1 &&
                        tPane->controlPoints.size() == 4 && tPane->branches.size() == 1,
                        "a pane with unsaved controls of its own follows the linked fiber's save");
                savePane(tPane, "t-own-edits");
                {
                    const auto stored = read(t.uuid);
                    const auto linkedAt = size_t(stored.branches.front().controlPointIndex);
                    require(tagged(t.uuid, "t-own-edits") && stored.controlPoints.size() == 4 && stored.branches.size() == 1 &&
                            vc3d::line_annotation::pointsApproximatelyEqual(stored.controlPoints.at(linkedAt), cv::Vec3d{12, 220, 1}) &&
                            tPane->controlPoints[1].storeId >= 0 &&
                            stored.controlPoints.at(linkedAt).storeId == tPane->controlPoints[1].storeId &&
                            read(s.uuid).branches.size() == 1 && paneMatchesStore(tPane),
                            "the pane saves its own controls, the link on its replaced control");
                    require(sPane->graphCommit == position() && paneMatchesStore(sPane),
                            "the linked fiber's pane follows the link to the replaced control");
                }
                // s's save moves its end of the link (its linked control is
                // replaced), while t again has a control it has not saved.
                replaceLinked(sPane, cv::Vec3d{12, 210, 1});
                saveInFlight(sPane, [&] {
                    vc3d::line_annotation::LineControlPoint added(4.0, cv::Vec3d{40, 220, 1}, false, 4);
                    tPane->controlPoints.push_back(added);
                    vc::lasagna::LinePoint end; end.position = cv::Vec3d{40, 220, 1}; tPane->optimizedLine.points.push_back(end);
                });
                require(tPane->graphCommit == position() && vc3d::line_annotation::pointsApproximatelyEqual(
                            tPane->branches.front().branchControlPointPosition, cv::Vec3d{12, 210, 1}),
                        "a pane with an unsaved control takes the linked fiber's moved end");
                savePane(tPane, "t-second-control");
                require(tagged(t.uuid, "t-second-control") && read(t.uuid).controlPoints.size() == 5 &&
                        paneMatchesStore(tPane) && paneMatchesStore(sPane) && sPane->graphCommit == position(),
                        "both panes are current after alternating saves");
                // t unlinks (not saved yet); then s's save moves its end of the
                // link once more. The link stays out of t's pane, and t's save
                // removes it.
                tPane->branches.clear();
                replaceLinked(sPane, cv::Vec3d{13, 210, 1});
                saveInFlight(sPane, [] {});
                require(tPane->graphCommit == position() && tPane->branches.empty() && read(t.uuid).branches.size() == 1,
                        "a link unlinked in a pane before the linked fiber's save stays unlinked there");
                savePane(tPane, "t-unlinked");
                require(tagged(t.uuid, "t-unlinked") && read(t.uuid).branches.empty() && read(s.uuid).branches.empty() &&
                        sPane->branches.empty() && sPane->graphCommit == position(),
                        "the pane's unlink is saved, and leaves the linked fiber's pane");
                saving._panes.clear();
                // An approved link whose control the pane itself replaces: its
                // own save commits the link pending, and the pane shows it.
                const auto linkedPair = [&](double y, bool pending, bool automatic) {
                    auto c = newFiber(y), d = newFiber(y + 5);
                    Controller::FiberBranchRef cd = link;
                    cd.controlPointIndex = 1; cd.branchControlPointIndex = 1;
                    cd.branchFiberId = d.id; cd.branchFileName = d.fileName;
                    cd.controlPointPosition = c.controlPoints[1]; cd.branchControlPointPosition = d.controlPoints[1];
                    cd.pending = pending; cd.autoLink = automatic;
                    c.branches.push_back(cd);
                    d.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(cd, c.id, c.fileName));
                    std::vector<Controller::FiberSaveSnapshot> made{saving.makeFiberSaveSnapshot(c), saving.makeFiberSaveSnapshot(d)};
                    saving.commitFiberStoreAction(made);
                    return std::pair{c, d};
                };
                {
                    const auto [c, d] = linkedPair(300, false, false);
                    auto cPane = openPane(read(c.uuid), "panes-c");
                    auto dPane = openPane(read(d.uuid), "panes-d");
                    require(!cPane->branches.front().pending && !dPane->branches.front().pending, "the link is stored approved");
                    replaceLinked(cPane, cv::Vec3d{12, 300, 1});
                    savePane(cPane, "moved-approved");
                    require(read(c.uuid).branches.front().pending && cPane->branches.front().pending &&
                            dPane->branches.front().pending && dPane->graphCommit == position() && paneMatchesStore(cPane) &&
                            paneMatchesStore(dPane), "a pane that moves an approved link's control saves and shows it pending");
                    saving._panes.clear();
                }
                // A pane whose ref of a link lost its edge id approves it: the
                // other fiber's pane, which holds the link with its id, takes
                // the approval.
                {
                    const auto [c, d] = linkedPair(320, true, false);
                    auto cPane = openPane(read(c.uuid), "panes-c");
                    auto dPane = openPane(read(d.uuid), "panes-d");
                    cPane->branches.front().edgeUuid.clear();
                    cPane->branches.front().pending = false;
                    savePane(cPane, "approved-without-id");
                    require(!read(d.uuid).branches.front().pending && !dPane->branches.front().pending &&
                            dPane->graphCommit == position() && paneMatchesStore(dPane) && paneMatchesStore(cPane),
                            "an approval saved from a ref without its edge id reaches the linked fiber's pane");
                    saving._panes.clear();
                }
                // A pane approves a link (not saved yet); the linked fiber's save
                // then moves its end to another control. The pane follows, and
                // its approval, which was of the link as it was, does not carry
                // over: its save leaves the link pending.
                {
                    const auto [c, d] = linkedPair(380, true, false);
                    auto dPane = openPane(read(d.uuid), "panes-d");
                    dPane->branches.front().pending = false;
                    auto moving = read(c.uuid);
                    moving.branches.front().controlPointIndex = 2;
                    moving.branches.front().controlPointPosition = moving.controlPoints[2];
                    saving.commitFibers({&moving});
                    require(dPane->graphCommit == position() && dPane->branches.front().pending && paneMatchesStore(dPane),
                            "an unsaved approval does not carry over to a link whose end the linked fiber's save moved");
                    savePane(dPane, "after-moved-end");
                    require(read(d.uuid).branches.front().pending && read(c.uuid).branches.front().controlPointIndex == 2,
                            "the pane's save leaves the moved link pending");
                    saving._panes.clear();
                }
                // A stored link that one pane has unlinked (not saved yet) while
                // the other fiber's pane holds it without its edge id and saves:
                // whether the first pane was never shown the link cannot be
                // told, so it keeps its loaded position and the link stays out.
                {
                    const auto [c, d] = linkedPair(400, true, false);
                    auto cPane = openPane(read(c.uuid), "panes-c");
                    auto dPane = openPane(read(d.uuid), "panes-d");
                    dPane->branches.clear();
                    cPane->branches.front().edgeUuid.clear();
                    const auto before = dPane->graphCommit;
                    savePane(cPane, "id-less-over-unlink");
                    require(dPane->branches.empty() && dPane->graphCommit == before,
                            "a save from a ref without its edge id does not bring an unlinked link back into the other pane");
                    saving._panes.clear();
                }
                // A link's end moved by less than the tolerance links are
                // resolved with is not a move: the link stays approved and the
                // linked fiber's pane stays saveable.
                {
                    const auto [c, d] = linkedPair(420, false, false);
                    auto dPane = openPane(read(d.uuid), "panes-d");
                    auto nudged = read(c.uuid);
                    nudged.controlPoints[1] += cv::Vec3d{4e-7, 4e-7, 0}; nudged.linePoints[1] += cv::Vec3d{4e-7, 4e-7, 0};
                    nudged.branches.front().controlPointPosition = nudged.controlPoints[1];
                    saving.commitFibers({&nudged});
                    require(!read(d.uuid).branches.front().pending && dPane->graphCommit == position(), "a nudge within tolerance keeps approval");
                    savePane(dPane, "after-nudge");
                    require(tagged(d.uuid, "after-nudge") && !read(d.uuid).branches.front().pending, "the pane saves past the nudge");
                    // Past the tolerance it is a move: pending, and the pane follows.
                    auto moved = read(c.uuid);
                    moved.controlPoints[1] += cv::Vec3d{8e-7, 8e-7, 0}; moved.linePoints[1] += cv::Vec3d{8e-7, 8e-7, 0};
                    moved.branches.front().controlPointPosition = moved.controlPoints[1];
                    saving.commitFibers({&moved});
                    require(read(d.uuid).branches.front().pending && dPane->graphCommit == position() && paneMatchesStore(dPane),
                            "a move past the tolerance makes the link pending and the pane follows it");
                    savePane(dPane, "after-move");
                    require(tagged(d.uuid, "after-move"), "the pane saves past the move");
                    saving._panes.clear();
                }
                // A link just made between two panes (each holds its ref, without
                // an edge id yet) and unlinked in the other pane while the save
                // that stores it is in flight: installing that save does not
                // bring it back there.
                {
                    auto c = newFiber(440), d = newFiber(445);
                    std::vector<Controller::FiberSaveSnapshot> made{saving.makeFiberSaveSnapshot(c), saving.makeFiberSaveSnapshot(d)};
                    saving.commitFiberStoreAction(made);
                    auto cPane = openPane(read(c.uuid), "panes-c");
                    auto dPane = openPane(read(d.uuid), "panes-d");
                    Controller::FiberBranchRef cd = link;
                    cd.controlPointIndex = 1; cd.branchControlPointIndex = 1;
                    cd.branchFiberId = d.id; cd.branchFileName = d.fileName;
                    cd.controlPointPosition = cPane->controlPoints[1].volumePoint;
                    cd.branchControlPointPosition = dPane->controlPoints[1].volumePoint;
                    cPane->branches.push_back(cd);
                    dPane->branches.push_back(vc3d::line_annotation::reciprocalBranchRef(cd, c.id, c.fileName));
                    saveInFlight(cPane, [&] { dPane->branches.clear(); });
                    require(read(d.uuid).branches.size() == 1 && dPane->branches.empty() && dPane->graphCommit == position() &&
                            !cPane->branches.front().edgeUuid.empty(),
                            "a new link unlinked in the other pane while its save was in flight stays unlinked there");
                    savePane(dPane, "new-link-unlinked");
                    require(read(d.uuid).branches.empty() && read(c.uuid).branches.empty() && cPane->branches.empty() &&
                            cPane->graphCommit == position(), "the unlink's save removes the new link from both");
                    saving._panes.clear();
                }
                // Two different changes of one link's review state (the pane
                // drops the automatic mark, not saved; the linked fiber's save
                // approves the link): the pane keeps its loaded position.
                {
                    const auto [c, d] = linkedPair(340, true, true);
                    auto dPane = openPane(read(d.uuid), "panes-d");
                    require(dPane->branches.front().autoLink && dPane->branches.front().pending, "the link is stored automatic");
                    dPane->branches.front().autoLink = false;
                    const auto before = dPane->graphCommit;
                    auto approving = read(c.uuid);
                    approving.branches.front().pending = false; approving.branches.front().autoLink = false;
                    saving.commitFibers({&approving});
                    require(dPane->graphCommit == before && dPane->branches.front().pending,
                            "a pane with another unsaved change of a link's review state is not brought past its approval");
                    saving._panes.clear();
                }
                // Two different moves of one end of a link (the pane has it on
                // another control of the linked fiber, not saved; that fiber's
                // save moves it to a third): the pane keeps its loaded
                // position. A pane that still has it as stored follows.
                {
                    const auto [c, d] = linkedPair(360, true, false);
                    auto dMoved = openPane(read(d.uuid), "panes-d-moved");
                    auto dClean = openPane(read(d.uuid), "panes-d-clean");
                    dMoved->branches.front().branchControlPointIndex = 0;
                    dMoved->branches.front().branchControlPointPosition = read(c.uuid).controlPoints[0];
                    const auto before = dMoved->graphCommit;
                    auto moving = read(c.uuid);
                    moving.branches.front().controlPointIndex = 2;
                    moving.branches.front().controlPointPosition = moving.controlPoints[2];
                    saving.commitFibers({&moving});
                    require(dMoved->graphCommit == before && dMoved->branches.front().branchControlPointIndex == 0,
                            "a pane that moved a link's linked end itself is not brought past another move of it");
                    require(dClean->graphCommit == position() && paneMatchesStore(dClean) &&
                            dClean->branches.front().branchControlPointIndex == 2, "a pane that had it as stored follows the move");
                    saving._panes.clear();
                }
            }
            // A save of one fiber alone that moves the other end of a link to
            // another control of the linked fiber: that fiber's pane follows
            // it. A pane that re-anchored the link itself meanwhile does not
            // (two different moves): it keeps its loaded position.
            {
                saving._panes.clear();
                auto a = newFiber(230), b = newFiber(240);
                Controller::FiberBranchRef ab = link;
                ab.controlPointIndex = 1; ab.branchControlPointIndex = 1;
                ab.branchFiberId = b.id; ab.branchFileName = b.fileName;
                ab.controlPointPosition = a.controlPoints[1]; ab.branchControlPointPosition = b.controlPoints[1];
                a.branches.push_back(ab);
                b.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(ab, a.id, a.fileName));
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(a), saving.makeFiberSaveSnapshot(b)};
                saving.commitFiberStoreAction(created);
                auto bPane = openPane(read(b.uuid), "panes-b", true);
                const auto retarget = [&](int index) {
                    auto moving = read(a.uuid);
                    moving.branches.front().branchControlPointIndex = index;
                    moving.branches.front().branchControlPointPosition = read(b.uuid).controlPoints.at(size_t(index));
                    saving.commitFibers({&moving});
                };
                retarget(2);
                require(bPane->graphCommit == position() && paneMatchesStore(bPane) &&
                        bPane->controlPoints[size_t(bPane->branches.front().controlPointIndex)].storeId ==
                            read(b.uuid).controlPoints[2].storeId,
                        "a pane follows its end of a link moved to another of its controls by the linked fiber's save");
                const auto own = size_t(bPane->branches.front().controlPointIndex);
                auto replacement = bPane->controlPoints[own];
                replacement.storeId = -1; replacement.identity = 0; replacement.volumePoint = cv::Vec3d{21, 240, 1};
                bPane->controlPoints[own] = replacement;
                bPane->branches.front().controlPointPosition = replacement.volumePoint;
                const auto before = bPane->graphCommit;
                retarget(1);
                require(bPane->graphCommit == before && bPane->controlPoints[own].storeId == -1 &&
                        size_t(bPane->branches.front().controlPointIndex) == own,
                        "a pane that re-anchored a link itself is not brought past another move of that end");
                saving._panes.clear();
            }
            // An adjacent link and a pending automatic link: a save of the other
            // end mirrors their kind and review state into the open pane, and
            // approving the automatic one in the pane (its handler) clears its
            // mark on both ends.
            {
                saving._panes.clear();
                auto m = newFiber(130), n = newFiber(140);
                Controller::FiberBranchRef adjacentLink = link, automaticLink = link;
                adjacentLink.controlPointIndex = 0; adjacentLink.branchControlPointIndex = 0;
                adjacentLink.branchFiberId = n.id; adjacentLink.branchFileName = n.fileName; adjacentLink.adjacent = true;
                adjacentLink.pending = false;
                adjacentLink.controlPointPosition = m.controlPoints[0]; adjacentLink.branchControlPointPosition = n.controlPoints[0];
                automaticLink.controlPointIndex = 2; automaticLink.branchControlPointIndex = 2;
                automaticLink.branchFiberId = n.id; automaticLink.branchFileName = n.fileName;
                automaticLink.pending = true; automaticLink.autoLink = true;
                automaticLink.controlPointPosition = m.controlPoints[2]; automaticLink.branchControlPointPosition = n.controlPoints[2];
                m.branches = {adjacentLink, automaticLink};
                n.branches = {vc3d::line_annotation::reciprocalBranchRef(adjacentLink, m.id, m.fileName),
                              vc3d::line_annotation::reciprocalBranchRef(automaticLink, m.id, m.fileName)};
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(m), saving.makeFiberSaveSnapshot(n)};
                saving.commitFiberStoreAction(created);
                auto nPane = openPane(read(n.uuid), "panes-n");
                const auto flags = [&](bool adjacent, bool pending, bool automatic) {
                    return std::any_of(nPane->branches.begin(), nPane->branches.end(), [&](const auto& branch) {
                        return branch.adjacent == adjacent && branch.pending == pending && branch.autoLink == automatic;
                    });
                };
                require(flags(true, false, false) && flags(false, true, true), "the pane loads both kinds of link");
                { auto tagged = read(m.uuid); tagged.tags.push_back("flags"); saving.commitFibers({&tagged}); }
                require(nPane->graphCommit == position() && paneMatchesStore(nPane) && flags(true, false, false) &&
                        flags(false, true, true), "a save of the other end keeps both links' kind and review state in the pane");
                std::string automaticEdge;
                {
                    const auto automatic = std::find_if(nPane->branches.begin(), nPane->branches.end(),
                                                        [](const auto& branch) { return branch.autoLink; });
                    require(automatic != nPane->branches.end() && !automatic->edgeUuid.empty(), "n's pane holds the automatic link");
                    automaticEdge = automatic->edgeUuid;
                    saving.handleGeneratedControlPointSetLinkPending("panes-n", size_t(automatic->controlPointIndex), m.id,
                                                                     automatic->branchControlPointIndex, false);
                }
                const auto storedM = read(m.uuid);
                const auto approvedOnM = std::any_of(storedM.branches.begin(), storedM.branches.end(), [&](const auto& branch) {
                    return branch.edgeUuid == automaticEdge && !branch.adjacent && !branch.pending && !branch.autoLink;
                });
                require(nPane->graphCommit == position() && paneMatchesStore(nPane) && flags(true, false, false) &&
                        flags(false, false, false) && !flags(false, true, true) && approvedOnM,
                        "approving the automatic link in the open pane clears its mark on both ends");
                saving._panes.clear();
            }
            // A pane holding a link without its edge id (as a link just made in
            // it does) takes the id with its save, so retiring the linked fiber
            // later removes the link from the pane.
            {
                saving._panes.clear();
                auto k = newFiber(150), l = newFiber(160);
                Controller::FiberBranchRef kl = link;
                kl.controlPointIndex = 1; kl.branchControlPointIndex = 1;
                kl.branchFiberId = l.id; kl.branchFileName = l.fileName;
                kl.controlPointPosition = k.controlPoints[1]; kl.branchControlPointPosition = l.controlPoints[1];
                k.branches.push_back(kl);
                l.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(kl, k.id, k.fileName));
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(k), saving.makeFiberSaveSnapshot(l)};
                saving.commitFiberStoreAction(created);
                auto kPane = openPane(read(k.uuid), "panes-k");
                kPane->branches.front().edgeUuid.clear();
                kPane->fiberTags.push_back("edge-less");
                auto fromK = saving.makeStoredFiberSessionSnapshot(*kPane).fiber;
                {
                    // The save written as the existing relationship is recorded as
                    // a link its fibers' panes may not have been shown (the
                    // editor held it without its edge id).
                    std::vector<Controller::FiberSaveSnapshot> save{saving.makeFiberSaveSnapshot(fromK)};
                    auto job = saving.prepareFiberCommit(save, {});
                    auto result = Controller::executeFiberCommit(job, *panesState.objectStoreSession());
                    saving.installFiberCommit(job, result);
                    saving.applyCommittedFibers(job.edits);
                    const auto change = job.links->changes.find(read(k.uuid).branches.front().edgeUuid);
                    require(change != job.links->changes.end() && change->second.unseen && change->second.before &&
                            change->second.after && read(k.uuid).branches.size() == 1,
                            "a link written from a ref without its edge id is recorded as one the panes may not have been shown");
                }
                require(!kPane->branches.front().edgeUuid.empty() && kPane->branches.front().edgeUuid == read(k.uuid).branches.front().edgeUuid,
                        "the pane's link takes its edge id with the save");
                std::vector<Controller::FiberSaveSnapshot> none;
                saving.commitFiberStoreAction(none, {read(l.uuid)});
                require(kPane->branches.empty() && kPane->graphCommit == position() && read(k.uuid).branches.empty(),
                        "retiring the linked fiber removes the link from the pane");
                saving._panes.clear();
            }
            // Every control of a pane replaced while its save is in flight: the
            // pane is still the save's source (recorded with the snapshot), is
            // brought to its commit, and its next save stores the replacements.
            {
                saving._panes.clear();
                auto e = newFiber(170), f = newFiber(180);
                Controller::FiberBranchRef ef = link;
                ef.controlPointIndex = 1; ef.branchControlPointIndex = 1;
                ef.branchFiberId = f.id; ef.branchFileName = f.fileName;
                ef.controlPointPosition = e.controlPoints[1]; ef.branchControlPointPosition = f.controlPoints[1];
                e.branches.push_back(ef);
                f.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(ef, e.id, e.fileName));
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(e), saving.makeFiberSaveSnapshot(f)};
                saving.commitFiberStoreAction(created);
                auto ePane = openPane(read(e.uuid), "panes-e");
                ePane->fiberTags.push_back("in-flight");
                // The in-flight save itself replaces the linked control (a new
                // id, so it changes the link's anchor).
                ePane->controlPoints[1].storeId = -1; ePane->controlPoints[1].identity = 0;
                std::vector<Controller::FiberSaveSnapshot> inFlight{
                    saving.makeFiberSaveSnapshot(saving.makeStoredFiberSessionSnapshot(*ePane).fiber)};
                auto job = saving.prepareFiberCommit(inFlight, {});
                for (auto& control : ePane->controlPoints) { control.storeId = -1; control.identity = 0; }
                auto result = Controller::executeFiberCommit(job, *panesState.objectStoreSession());
                saving.installFiberCommit(job, result);
                saving.applyCommittedFibers(job.edits, false);
                require(ePane->graphCommit == position() &&
                        std::all_of(ePane->controlPoints.begin(), ePane->controlPoints.end(), [](const auto& control) { return control.storeId < 0; }),
                        "a source pane whose controls were all replaced is brought to its commit, its replacements without ids");
                ePane->fiberTags.push_back("replaced");
                auto again = saving.makeStoredFiberSessionSnapshot(*ePane).fiber;
                saving.commitFibers({&again});
                const auto stored = read(e.uuid);
                require(tagged(e.uuid, "replaced") && stored.branches.size() == 1 && read(f.uuid).branches.size() == 1 &&
                        std::all_of(ePane->controlPoints.begin(), ePane->controlPoints.end(), [](const auto& control) { return control.storeId >= 0; }),
                        "its next save stores the replacements and keeps the link");
                saving._panes.clear();
            }
            // A successful edit with its collapse rollback still armed (a
            // collapse saves before it disarms): the pane is brought to its own
            // commit, and its next geometry save goes through.
            {
                saving._panes.clear();
                auto c = newFiber(190), d = newFiber(200);
                Controller::FiberBranchRef cd = link;
                cd.controlPointIndex = 1; cd.branchControlPointIndex = 1;
                cd.branchFiberId = d.id; cd.branchFileName = d.fileName;
                cd.controlPointPosition = c.controlPoints[1]; cd.branchControlPointPosition = d.controlPoints[1];
                c.branches.push_back(cd);
                d.branches.push_back(vc3d::line_annotation::reciprocalBranchRef(cd, c.id, c.fileName));
                std::vector<Controller::FiberSaveSnapshot> created{saving.makeFiberSaveSnapshot(c), saving.makeFiberSaveSnapshot(d)};
                saving.commitFiberStoreAction(created);
                auto cPane = openPane(read(c.uuid), "panes-c");
                Controller::LineAnnotationSession::ControlPointCollapseRollback rollback;
                rollback.controlPoints = cPane->controlPoints; rollback.branches = cPane->branches;
                cPane->controlPointCollapseRollback = rollback;
                // The collapse: the linked control is replaced (a new id).
                cPane->controlPoints[1].storeId = -1; cPane->controlPoints[1].identity = 0;
                auto collapsed = saving.makeStoredFiberSessionSnapshot(*cPane).fiber;
                saving.commitFibers({&collapsed});
                require(cPane->graphCommit == position(), "the pane is brought to its own commit with its rollback armed");
                cPane->controlPointCollapseRollback.reset(); // the disarm
                cPane->controlPoints[2].volumePoint[1] += 1; cPane->optimizedLine.points[2].position[1] += 1;
                auto moved = saving.makeStoredFiberSessionSnapshot(*cPane).fiber;
                saving.commitFibers({&moved});
                require(cPane->graphCommit == position() && read(c.uuid).controlPoints[2][1] == cPane->controlPoints[2].volumePoint[1] &&
                        read(d.uuid).branches.size() == 1, "its next geometry save goes through");
                saving._panes.clear();
            }
            panesState.closeProject();
        }
        {
            QTemporaryDir directory;
            const auto target = fs::path(directory.path().toStdString()) / "service.vcobj";
            { auto db = ObjectStore::create(target, options); }
            CState destination;
            Controller imported(&destination, nullptr, nullptr, nullptr);
            imported._errorDialogsSuppressed = true;
            openStoreAsProject(destination, QString::fromStdString(target.string()), "service-import-test");
            QString error;
            if (!imported.importFibersFromPath(transferPaths.at(a.uuid), &error))
                throw std::runtime_error(error.toStdString());
            require(imported.readStoredFibers(std::nullopt).size() == 2,
                    "a transported component is imported into the attached store");
            auto document = aWire;
            const auto uuid = vc::objects::util::newUuid(); document["uuid"] = uuid;
            document["branches"] = nlohmann::json::array(); document["adjacent_branches"] = nlohmann::json::array();
            document.erase("optimization_mode"); document["custom_annotation"] = {{"note", "retain"}};
            const auto volumeUuid = vc::objects::util::newUuid();
            document["coordinates"] = {{"volume_uuid", volumeUuid}, {"scale_factor", 4}};
            document["coordinate_binding"] = {{"active_volume_id", "display"}};
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
        a = resolve(a.uuid); b = resolve(b.uuid);
        require(controller.generatedBranchLinkMarkers(a.branches).empty(), "unbound peers have no branch markers");
        {
            auto changedSession = std::make_shared<Controller::LineAnnotationSession>();
            changedSession->fiberId = a.id; changedSession->fiberUuid = a.uuid; changedSession->storePin = a.storePin; changedSession->graphCommit = a.graphCommit;
            auto untouched = std::make_shared<Controller::LineAnnotationSession>();
            untouched->fiberId = b.id; untouched->fiberUuid = b.uuid; untouched->storePin = b.storePin; untouched->graphCommit = b.graphCommit;
            untouched->fiberTags = {"unsaved-work"};
            untouched->runningSolveCancel = std::make_shared<std::atomic_bool>(false);
            Controller::PaneRecord changedPane, keptPane;
            changedPane.surfaceName = "changed-test-pane"; changedPane.session = changedSession;
            keptPane.surfaceName = "kept-test-pane"; keptPane.session = untouched;
            controller._panes.push_back(changedPane); controller._panes.push_back(keptPane);
            { auto db = ObjectStore::open(path, options); auto txn = db.write({{a.uuid, a.storePin->reference()}});
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
            require(failed.front().fiber.storePin->reference().version == invalid.storePin->reference().version, "failed transaction leaves working pins unchanged");
            auto db = ObjectStore::open(path, options); auto txn = db.read();
            require(db.currentObjectRevision(txn, a.uuid)->version == a.storePin->reference().version,
                    "failed relationship rolls back geometry and metadata");
        }
        const auto edge=a.branches.front().edgeUuid;
        const auto graphBaseline = a.graphCommit;
        controller.commitFibers({&a});
        controller.commitFibers({&a});
        require(a.graphCommit == graphBaseline, "repeated no-op saves preserve the loaded graph baseline");
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
        { auto db=ObjectStore::open(path,options);auto txn=db.write({{a.uuid,a.storePin->reference()}});db.setMeta(txn,a.uuid,"external","true");txn.commit(); }
        bool conflict=false;try { auto old=stale;controller.commitFiberStoreAction(old); } catch(...) { conflict=true; }
        require(conflict,"stale controller edits reject");
        a=resolve(a.uuid);
        auto outcome=controller.deleteFibers({a.id});require(outcome.deletedIds.size()==1,"retire through controller");
        require(retiredNotifications == std::vector<std::string>{a.uuid}, "retirement publishes object UUID");
        {
        auto db=ObjectStore::open(path,options);auto txn=db.read();
        require(db.objectByUuid(txn,a.uuid)->retired,"retirement persisted");
        require(db.objectAtVersion(txn,first->reference()).has_value(),"historical revision retained");
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
            live->storePin=merged.storePin;
            live->graphCommit=merged.graphCommit;
            live->coordinateIdentity=merged.coordinateIdentity;live->frame=merged.frame;
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
            require(live->storePin->reference().version!=merged.storePin->reference().version,"interaction advances pin immediately");
            const auto savedPin=live->storePin;
            (void)controller.saveOpenFibers();
            require(live->storePin==savedPin,"Save barrier does not create another revision");
            {
                // Split and merge plan from session snapshots; new fibers must keep the stored frame.
                const auto snapshot=controller.makeStoredFiberSessionSnapshot(*live).fiber;
                require(snapshot.coordinateIdentity.value("coordinates",nlohmann::json())==frame,"session snapshot carries stored frame");
                const auto halves=la::planFiberSplit(snapshot,1,identity("live-prefix.json"),identity("live-suffix.json"),false,&error);
                if(!halves) throw std::runtime_error(error);
                require(halves->prefix.coordinateIdentity.value("coordinates",nlohmann::json())==frame &&
                        halves->suffix.coordinateIdentity.value("coordinates",nlohmann::json())==frame,"split of an open fiber keeps its frame");
            }
            controller._panes.clear();
            auto db=ObjectStore::open(path,options);auto txn=db.read();
            const auto saved=std::get<vc::geometry::Fiber>(vc::geometry::decode(db.readGeometryData(txn,*savedPin->geometryRevision)));
            require(saved.controlPoints[0][0]==live->controlPoints[0].volumePoint[0],"manual edit survives independent read");
        }
        const auto artifact=controller.fiberTransportPaths({merged.id}).at(merged.uuid);
        require(std::filesystem::exists(artifact) && controller.fiberIdForFilePath(artifact)==merged.id,"pinned Spiral transport retains UUID");
        {
            CState child(nullptr,&state);
            require(child.objectStoreSession()==state.objectStoreSession(),"secondary workspaces share session");
        }
        require(state.storeFor("fiber").has_value(),"secondary destruction keeps main attachment");
        state.closeProject();
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
            openStoreAsProject(state, QString::fromStdString(path.string()), {}, true);
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
            openStoreAsProject(destination, QString::fromStdString(target.string()), "control-only-roundtrip");
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
            state.closeProject();
        }
        if(const char* smoke=std::getenv("VC3D_OBJECT_STORE_SMOKE_DB")) {
            openStoreAsProject(state, QString::fromUtf8(smoke),"vc3d-real-smoke");
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
                std::cerr << "CT smoke: " << ct << " scale 2, voxel size " << state.currentVolume()->voxelSize() << " um, fiber XYZ " << point << "\n";
            }
            const auto originalPin=fiber.storePin;
            fiber.tags.push_back("vc3d_smoke");
            controller.commitFibers({&fiber});
            state.closeProject();openStoreAsProject(state, QString::fromUtf8(smoke),"vc3d-real-smoke");
            require(controller.readStoredFibers(std::nullopt).size()==count,"real database reopens");
            const auto restored=resolve(fiber.uuid);
            require(std::find(restored.tags.begin(),restored.tags.end(),"vc3d_smoke")!=restored.tags.end(),"real database edit persists");
            require(restored.storePin->reference().version!=originalPin->reference().version,"real database revision advances");
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
            state.closeProject();openStoreAsProject(state, QString::fromUtf8(smoke),"vc3d-real-smoke");
            const auto roundtrip=resolve(realMerge->merged.uuid);
            require(roundtrip.controlPoints.size()==restored.controlPoints.size(),"real split/merge control count");
            require(roundtrip.linePoints==realMerge->merged.linePoints,"real split/merge preserves planner numerical geometry");
            require(roundtrip.branches.size()==restored.branches.size(),"real split/merge preserves external branches");
            std::cerr << "real-data smoke: " << count << " fibers; edit, shared link, split/merge and reopen; "
                      << roundtrip.controlPoints.size() << " controls and " << roundtrip.linePoints.size() << " planned line points preserved\n";
            state.closeProject();
        }
        std::cerr << "vc3d object store controller tests passed\n";
    } catch(const std::exception& error) { std::cerr << "vc3d object store test: " << error.what() << '\n'; return 1; }
    std::filesystem::remove(path);
    return 0;
}
