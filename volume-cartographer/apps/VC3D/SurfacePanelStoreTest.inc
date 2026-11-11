#include <QApplication>
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SurfaceStoreEdit.hpp"
#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <iostream>
#include <vc_object_store/Uuid.hpp>
#include "viewer_controls/panels/VolumeRegistrationsPanel.hpp"
#include <vc_reader/registrations.hpp>
#include <mutex>
#include <QTreeWidget>
#include <QDoubleSpinBox>
#include "vc/core/types/Volume.hpp"
#include <nlohmann/json.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <filesystem>
#include <fstream>
#include "ViewerManager.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "overlays/StoreSurfaceOverlayController.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "test/support/TestVolume.hpp"
// Built only with VC_TESTING. The surface panel over a real object store:
// listing, filters, tags, activation and deletion through commit events; an
// Open Data sample's project (catalog surfaces of a volume the open one shows
// at pyramid level 1), fetched on activation, edited in place and reset.
namespace {
// Open a store as the state's project, as VC3D opens one.
void openStoreAsProject(CState& state, const QString& path, const QString& author)
{
    auto projectPath = std::filesystem::path(path.toStdString());
    projectPath.replace_extension(".vcproj");
    if (!std::filesystem::exists(projectPath))
        vc::project::ProjectDocument::create(projectPath, "test", path.toStdString(), false);
    state.setVpkg(VolumePkg::open(projectPath));
}
}
int runSurfacePanelStoreTests()
{
    using namespace vc::objects;
    namespace fs = std::filesystem;
    auto require = [](bool value, const char* message) { if (!value) throw std::runtime_error(message); };
    try {
        QTemporaryDir dir;
        const std::filesystem::path volumeRoot = fs::path(dir.path().toStdString()) / "ct.zarr";
        const vcr::zarr::CreateOptions levelZero{.shape = {4, 4, 4}, .chunks = {4, 4, 4}, .compressor = "", .overwrite = true};
        vcr::zarr::Writer::create(volumeRoot / "0", levelZero);
        std::ofstream(volumeRoot / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "panel-volume"}, {"name", "panel-volume"}, {"format", "zarr"}, {"width", 4}, {"height", 4}, {"slices", 4}, {"voxelsize", 1.0}}.dump();
        const auto volume = Volume::New(volumeRoot);
        const auto path = fs::path(dir.path().toStdString()) / "objects.vcobj";
        OpenOptions options; options.author = "surface-panel-test";
        { auto created = ObjectStore::create(path, options); }

        CState state;
        openStoreAsProject(state, QString::fromStdString(path.string()), "surface-panel-test");
        auto attachment = *state.storeFor("surface");
        cv::Mat_<cv::Vec3f> points(3, 3);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) points(r, c) = {float(c), float(r), 2.f};
        QuadSurface grid(points, {1.f, 1.f});
        auto add = [&](const std::string& name, std::vector<std::string> tags, bool bound) {
            const auto uuid = util::newUuid();
            auto request = state.objectStoreSession()->newRequest(attachment);
            request.expectedObjects = {{uuid, std::nullopt}};
            request.operations = vc::objects::createSurfaceOperations(uuid, name, tags,
                bound ? vc::objects::volumeCoordinates(volume->id(), 1) : nlohmann::json::object(),
                {"surface-panel-test"}, grid.meta, quad_surface_geometry(grid));
            state.objectStoreSession()->submit(attachment, request);
            return uuid;
        };
        const auto approved = add("approved", {"approved"}, true);
        const auto plain = add("plain", {}, true);
        const auto unbound = add("unbound", {}, false);

        SurfaceCatalog catalog(state.objectStoreSession());
        QObject::connect(&state, &CState::objectStoreChanged, &catalog,
                         [&](const StoreSessionEvent& event) { catalog.handle(event); });
        catalog.setSource(attachment, volume, {volume->id()});
        SurfaceTreeWidget tree;
        tree.setColumnCount(TIMESTAMP_COLUMN + 1);
        SurfacePanelController panel({&tree}, &state, &catalog, nullptr, {});
        QCheckBox approvedTag, defectiveTag, reviewedTag, inspectTag, hideUnapproved;
        QDoubleSpinBox zLower, zUpper;
        panel.configureTags({nullptr, &approvedTag, &defectiveTag, &reviewedTag, &inspectTag});
        SurfacePanelController::FilterUiRefs filters;
        filters.hideUnapproved = &hideUnapproved;
        filters.zLowerBound = &zLower;
        filters.zUpperBound = &zUpper;
        panel.configureFilters(filters, state.pointCollection());
        std::vector<std::shared_ptr<QuadSurface>> activated;
        QObject::connect(&panel, &SurfacePanelController::surfaceActivated,
                         [&](const QString&, const std::shared_ptr<QuadSurface>& surface) { activated.push_back(surface); });

        auto pump = [&](const std::function<bool()>& done, const char* message) {
            QElapsedTimer timer; timer.start();
            while (!done() && timer.elapsed() < 10000) {
                QApplication::processEvents();
                QThread::msleep(1);
            }
            require(done(), message);
        };
        auto visible = [&] { return panel.visibleSurfaceIds(); };
        pump([&] { return tree.topLevelItemCount() == 3; }, "the panel lists the store's surfaces");
        auto sorted = [](std::vector<std::string> ids) { std::sort(ids.begin(), ids.end()); return ids; };
        require(visible() == sorted({approved, plain}), "unbound surfaces are listed but not drawn");

        hideUnapproved.setChecked(true);
        pump([&] { return visible() == std::vector<std::string>{approved}; }, "tag filters select record tags in the store");
        hideUnapproved.setChecked(false);
        pump([&] { return visible() == sorted({approved, plain}); }, "clearing a tag filter shows the surfaces again");

        // The Z-range filter asks the store for the surfaces whose bounds reach
        // the range in the open volume (the grids lie at z = 2).
        zUpper.setValue(10);
        zLower.setValue(5);
        pump([&] { return visible().empty(); }, "the Z range excludes surfaces below it");
        zLower.setValue(1);
        zUpper.setValue(3);
        pump([&] { return visible() == sorted({approved, plain}); }, "the Z range keeps surfaces that reach it");
        zLower.setValue(zLower.minimum());
        zUpper.setValue(zUpper.maximum());

        QString error;
        require(panel.activateSurfaceById(plain, &error), "activating a bound surface opens it");
        require(activated.size() == 1 && activated.back() && (*activated.back()->rawPointsPtr())(1, 2) == cv::Vec3f(2, 1, 2),
                "activation hands out the stored geometry");
        require(panel.opened() && panel.opened()->editable && panel.opened()->pin.reference().objectUuid == plain,
                "the opened surface is pinned and editable");
        panel.setSelectionChangeGuard([] { return false; });
        require(!panel.activateSurfaceById(approved, &error) && panel.opened()->pin.reference().objectUuid == plain && activated.size() == 1,
                "failed pending saves veto selection before replacing the displayed surface");
        panel.setSelectionChangeGuard({});
        require(!panel.activateSurfaceById(unbound, &error), "unbound surfaces do not open");

        require(panel.setTagChecked(SurfacePanelController::Tag::Approved, true), "tag checkboxes are enabled for the selection");
        pump([&] { const auto* row = catalog.row(plain); return row && row->record.tags == std::vector<std::string>{"approved"}; },
             "tag checkboxes commit record tags");
        hideUnapproved.setChecked(true);
        pump([&] { return visible() == sorted({approved, plain}); }, "filters follow committed tags");

        require(panel.retireSurfaces({QString::fromStdString(approved)}, &error), "surfaces retire");
        pump([&] { return tree.topLevelItemCount() == 2; }, "retired surfaces leave the panel");
        require(visible() == std::vector<std::string>{plain}, "retired surfaces are not drawn");

        // An Open Data sample's project of the catalog volume the open volume
        // shows at level 1: one surface with geometry, one still remote, fetched
        // by the session.
        const auto catalogPath = fs::path(dir.path().toStdString()) / "catalog.vcobj";
        std::string shown, remote;
        {
            auto db = ObjectStore::create(catalogPath, options);
            const auto binding = nlohmann::json{{"coordinates", {{"volume_uuid", "catalog-volume"}, {"scale_factor", 1}}},
                                                {"catalog", {{"sample", "sample"}, {"segment", "segment"}}}}.dump();
            shown = util::newUuid();
            remote = util::newUuid();
            auto txn = db.write({{shown, std::nullopt}, {remote, std::nullopt}});
            QuadSurface published(points, {1.f, 1.f});
            published.meta = nlohmann::json{{"catalog", nlohmann::json::parse(binding).at("catalog")}};
            vc::objects::createSurface(db, txn, shown, "shown", {}, nlohmann::json::parse(binding).at("coordinates"),
                                       {"catalog-test"}, published.meta, quad_surface_geometry(published));
            // As a catalog fetch records it.
            db.setMeta(txn, shown, "catalog_geometry_revision", nlohmann::json(db.currentGeometry(txn, shown)->uuid).dump());
            db.setMeta(txn, shown, "catalog_write", "\"fetched\"");
            ObjectRecord record;
            record.uuid = remote; record.kind = "surface"; record.name = "remote"; record.meta = binding;
            db.upsertObject(txn, record);
            txn.commit();
        }
        std::mutex fetchedMutex;
        std::vector<std::string> fetched;
        state.objectStoreSession()->setGeometryFetcher(vc::objects::kCatalogMetaKey,
            [&](ObjectStore& store, const std::string& uuid, const ObjectStoreSession::FetchWrite& write) {
                { std::lock_guard lock(fetchedMutex); fetched.push_back(uuid); }
                const auto selected = store.currentObjectRevision(store.read(), uuid)->reference();
                write({{uuid, selected}}, [&](ObjectStore& db, WriteTxn& txn) {
                    const auto geometry = db.putGeometry(txn, uuid, quad_surface_geometry(grid));
                    db.setMeta(txn, uuid, "catalog_geometry_revision", nlohmann::json(geometry).dump());
                    db.setMeta(txn, uuid, "catalog_write", "\"fetched\"");
                });
            });
        auto fetchedNow = [&] { std::lock_guard lock(fetchedMutex); return fetched; };

        // The sample's store is a project like any other: opening it closes the other.
        openStoreAsProject(state, QString::fromStdString(catalogPath.string()), "surface-panel-test");
        const auto catalogAttachment = *state.storeFor("surface");
        require(!state.objectStoreSession()->current(attachment), "the previous project's store is detached");
        // The level the open volume shows, as the catalog registers it.
        vc::objects::registerVolumeTransform(*state.objectStoreSession(), catalogAttachment, "catalog-volume", "panel-volume",
                                             vc::objects::VolumeAffine::scale(0.5), "pyramid level 1");
        catalog.setSource(state.storeFor("surface"), volume, {volume->id()});
        pump([&] { return tree.topLevelItemCount() == 2; }, "the sample's surfaces are listed");
        require(catalog.row(shown)->fromCatalog() && catalog.row(shown)->source == catalogAttachment, "rows know their store");
        require(catalog.row(shown)->mapping == vc::objects::VolumeAffine::scale(0.5) && !catalog.row(shown)->toolInput(),
                "catalog rows are scaled to the open volume's level");

        require(panel.activateSurfaceById(shown, &error), "catalog surfaces open");
        require((*activated.back()->rawPointsPtr())(1, 2) == cv::Vec3f(1, 0.5f, 1), "in the open volume's voxels");
        require(panel.opened()->editable, "catalog surfaces are directly editable");
        require(panel.activateSurfaceById(remote, &error) && fetchedNow() == std::vector<std::string>{remote},
                "a remote catalog surface is fetched when activated headlessly");

        // Segmentation editing saves to the sample's project and preserves the view's scale.
        require(panel.activateSurfaceById(shown, &error), "the catalog surface reopens");
        const auto onScreen = panel.opened()->surface;
        const auto publishedVersion = panel.opened()->pin.reference();
        require(panel.prepareSurfaceEditing(shown, &error) && panel.opened()->editable &&
                    panel.opened()->pin.attachment == catalogAttachment && panel.opened()->surface == onScreen,
                "editing keeps the sample's store and the surface on screen");
        {
            SegmentationEditManager edits;
            SegmentationModule module(nullptr, &edits, nullptr, nullptr, &state, nullptr, false);
            module.setEditingDestinationResolver([&](const auto& surface) {
                return panel.prepareSurfaceEditing(surface->id, &error);
            });
            module.setStorePinResolver([&](const QuadSurface* surface) -> std::optional<SegmentationModule::StoreOrigin> {
                if (panel.opened() && panel.opened()->editable && panel.opened()->surface->id == surface->id)
                    return SegmentationModule::StoreOrigin{panel.opened()->pin, panel.opened()->view};
                return std::nullopt;
            });
            state.setSurface("segmentation", onScreen, false, false);
            module.setEditingEnabled(true);
            require(module.beginEditingSession(onScreen) && module.editingEnabled() && module.hasActiveSession(),
                    "segmentation editing starts directly on a sample surface");
            (*onScreen->rawPointsPtr())(1, 2) = {3, 4, 5};
            require(module.saveSurface(onScreen.get()), "segmentation edits save to the sample store");
            module.setEditingEnabled(false);
            module.endEditingSession();
        }
        require((*catalog.open(shown).surface->rawPointsPtr())(1, 2) == cv::Vec3f(3, 4, 5),
                "reopening the sample surface retains the edit in volume coordinates");
        require(panel.restoreRevision(shown, publishedVersion, &error), "sample surfaces can restore their history");
        require((*catalog.open(shown).surface->rawPointsPtr())(1, 2) == cv::Vec3f(1, 0.5f, 1),
                "restoring history recovers the published geometry");
        state.objectStoreSession()->editRecords(catalogAttachment, {shown}, [](ObjectRecord& record) { record.name = "edited"; });
        pump([&] { return catalog.row(shown)->record.name == "edited"; }, "the catalog surface is edited in place");
        require(panel.resetToCatalogVersion(shown, &error), "a catalog surface resets to its catalog version");
        pump([&] { return catalog.row(shown)->record.name == "shown"; }, "the reset restores the catalog version");

        // The Transforms dock lists the volumes the project maps into the open one,
        // including a registration committed while it is shown.
        attachment = catalogAttachment;
        state.setCurrentVolume(volume);
        VolumeRegistrationsPanel registrations(&state);
        auto* related = registrations.findChild<QTreeWidget*>();
        require(related && related->topLevelItemCount() == 1 && related->topLevelItem(0)->text(0) == "catalog-volume",
                "the dock lists the catalog volume the open volume shows");
        vc::objects::registerVolumeTransform(*state.objectStoreSession(), attachment, "scan-b", "panel-volume",
                                             vc::objects::VolumeAffine::scale(2), "test registration");
        state.pollObjectStoreChanges();
        require(related->topLevelItemCount() == 2 && !related->findItems("scan-b", Qt::MatchExactly).isEmpty(),
                "a committed registration appears in the Transforms dock");
        state.closeProject();
        std::cerr << "surface panel store tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "surface panel store test: " << error.what() << '\n';
        return 1;
    }
}

// Viewer regressions run independently of the surface panel's filter suite.
int runViewerRegressionTests()
{
    using namespace vc::objects;
    namespace fs = std::filesystem;
    auto require = [](bool value, const char* message) { if (!value) throw std::runtime_error(message); };
    auto pump = [&](const std::function<bool()>& done, const char* message) {
        QElapsedTimer timer; timer.start();
        while (!done() && timer.elapsed() < 10000) {
            QApplication::processEvents();
            QThread::msleep(1);
        }
        require(done(), message);
    };
    try {
        QTemporaryDir dir;
        const auto volume = vc3d::test::createVolume(fs::path(dir.path().toStdString()) / "panel.zarr", {});
        const auto path = fs::path(dir.path().toStdString()) / "objects.vcobj";
        OpenOptions options; options.author = "viewer-test";
        { auto created = ObjectStore::create(path, options); }
        CState state;
        openStoreAsProject(state, QString::fromStdString(path.string()), "viewer-test");
        auto attachment = *state.storeFor("surface");
        cv::Mat_<cv::Vec3f> points(3, 3);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) points(r, c) = {float(c), float(r), 2};
        QuadSurface grid(points, {1, 1});
        const auto plain = util::newUuid();
        {
            auto request = state.objectStoreSession()->newRequest(attachment);
            request.expectedObjects = {{plain, std::nullopt}};
            request.operations = vc::objects::createSurfaceOperations(plain, "surface", {}, volumeCoordinates(volume->id(), 1),
                                                                      {"viewer-test"}, grid.meta, quad_surface_geometry(grid));
            state.objectStoreSession()->submit(attachment, request);
        }
        SurfaceCatalog catalog(state.objectStoreSession());
        catalog.setSource(attachment, volume, {volume->id()});
        SurfaceTreeWidget tree;
        tree.setColumnCount(TIMESTAMP_COLUMN + 1);
        SurfacePanelController panel({&tree}, &state, &catalog, nullptr, {});
        pump([&] { return tree.topLevelItemCount() == 1; }, "the viewer fixture surface is listed");
        require(panel.activateSurfaceById(plain), "the viewer fixture surface opens");
        const auto originalFrame = *panel.opened();
        int cleared = 0;
        QObject::connect(&panel, &SurfacePanelController::surfaceSelectionCleared, [&] { ++cleared; });
        catalog.setSource(attachment, volume, {volume->id(), 0.5});
        require(cleared == 0 && panel.opened(), "a working-level reload retains the active selection");
        panel.replaceOpenedSurface(catalog.open(plain));
        pump([&] { return !catalog.loading(); }, "the working-level listing completes");
        require(cleared == 0 && panel.opened()->surface->id == plain && !tree.selectedItems().empty(),
                "the surface stays selected after the new mappings arrive");
        require((*panel.opened()->surface->rawPointsPtr())(1, 2) == cv::Vec3f(1, 0.5f, 1),
                "the selected geometry is reopened in the new working voxels");
        {
            auto editFrame = originalFrame;
            SegmentationModule edits(nullptr, nullptr, nullptr, nullptr, &state, nullptr, false);
            edits.setStorePinResolver([&](const QuadSurface*) -> std::optional<SegmentationModule::StoreOrigin> {
                return SegmentationModule::StoreOrigin{editFrame.pin, editFrame.view};
            });
            require(edits.storePin(editFrame.surface.get()).has_value(), "an edit remembers the original frame");
            editFrame = *panel.opened();
            require(edits.saveSurface(editFrame.surface.get()), "the same revision can be edited at another level");
        }
        catalog.setSource(attachment, volume, {volume->id()});
        panel.replaceOpenedSurface(catalog.open(plain));
        pump([&] { return !catalog.loading(); }, "returning to the original level completes");
        require((*panel.opened()->surface->rawPointsPtr())(1, 2) == cv::Vec3f(2, 1, 2),
                "saving after a level switch maps back through the new frame");
        state.closeProject();
        {
            // Exercise the real asynchronous viewer with a stationary camera.
            const auto root = fs::path(dir.path().toStdString()) / "viewer.zarr";
            auto ct = vc3d::test::createVolume(root, {.chunkShapeZYX = {4, 4, 4}, .numLevels = 2});
            vc3d::test::Grid<uint8_t> data({32, 32, 32});
            for (int z = 0; z < 32; ++z)
                for (int y = 0; y < 32; ++y)
                    for (int x = 0; x < 32; ++x) data(z, y, x) = uint8_t(8 * z);
            vc3d::test::writeZYX(*ct, data, {0, 0, 0});
            vc3d::test::Grid<uint8_t> coarse({16, 16, 16});
            for (int z = 0; z < 16; ++z)
                for (int y = 0; y < 16; ++y)
                    for (int x = 0; x < 16; ++x) coarse(z, y, x) = uint8_t(16 * z + 4);
            vc3d::test::writeZYX(*ct, coarse, {0, 0, 0}, 1);
            state.setCurrentVolume(ct);
            cv::Mat_<cv::Vec3f> vertices(9, 9);
            for (int y = 0; y < 9; ++y)
                for (int x = 0; x < 9; ++x) vertices(y, x) = {float(4 * x), float(4 * y), 8};
            auto surface = std::make_shared<QuadSurface>(vertices, cv::Vec2f(0.25f, 0.25f));
            surface->id = util::newUuid();
            state.setSurface("segmentation", surface, false, true);
            ViewerManager manager(&state, state.pointCollection());
            QWidget host;
            auto* viewer = static_cast<CChunkedVolumeViewer*>(manager.createViewerInWidget("segmentation", &host));
            viewer->resize(128, 128);
            host.resize(128, 128);
            host.show(); viewer->show();
            viewer->applyCameraState({0, 0, 8, 0});
            auto centerPixel = [&] {
                const auto& image = viewer->_framebuffer;
                return image.isNull() ? -1 : qRed(image.pixel(image.width() / 2, image.height() / 2));
            };
            pump([&] { return centerPixel() == 64 && !viewer->_renderWorkerBusy && !viewer->_pendingRenderJob; },
                 "the surface reaches native resolution without camera motion");
            // Two edits before the first render completes must retain the latest.
            for (int z : {9, 10}) {
                for (auto& p : *surface->rawPointsPtr()) p[2] = float(z);
                viewer->invalidateVis();
                viewer->requestRender("segmentation edit refresh");
            }
            pump([&] { return centerPixel() == 80 && !viewer->_renderWorkerBusy && !viewer->_pendingRenderJob; },
                 "successive in-place edits repaint without pointer motion");
            for (auto& p : *surface->rawPointsPtr()) p[2] = 11;
            state.setSurface("segmentation", surface, false, true);
            pump([&] { return centerPixel() == 88; }, "surface edit signals also repaint immediately");
            bool flashed = false;
            const auto frameConnection = QObject::connect(viewer, &CChunkedVolumeViewer::renderFrameCompleted,
                [&](auto, auto) { if (centerPixel() == 0) flashed = true; });
            for (auto& p : *surface->rawPointsPtr()) p[2] = 16;
            viewer->invalidateVis();
            viewer->requestRender("edit into cold chunks");
            pump([&] { return centerPixel() == 128 && !viewer->_renderWorkerBusy && !viewer->_pendingRenderJob && !viewer->_loadRunning; },
                 "an edit into uncached chunks finishes without further input");
            require(!flashed, "an edit retains pixels while its new chunks load");
            QObject::disconnect(frameConnection);

            state.setSurface("test plane", std::make_shared<PlaneSurface>(cv::Vec3f(16, 16, 16), cv::Vec3f(0, 0, 1)));
            auto* idle = static_cast<CChunkedVolumeViewer*>(manager.createViewerInWidget("test plane", &host));
            QStringList status;
            QObject::connect(&manager, &ViewerManager::sharedCacheStatsChanged, [&](const QStringList& items) { status = items; });
            viewer->_loadRunning = true;
            idle->refreshStatus();
            require(status.join(' ').contains("segmentation: loading L0"),
                    "an idle viewer does not hide another viewer's loading status");
            viewer->_loadRunning = false;
            idle->refreshStatus();
            require(!status.join(' ').contains("loading"), "loading clears when the reads finish");

            viewer->applyCameraState({2, -3, 8, 2});
            const auto camera = viewer->cameraState();
            const auto snapshot = manager.captureNavigation();
            cv::Mat_<cv::Vec3f> scaledPoints = surface->rawPoints() * 0.5f;
            auto scaled = std::make_shared<QuadSurface>(scaledPoints, surface->scale() * 2.f);
            scaled->id = surface->id;
            state.setSurface("segmentation", scaled, false, true);
            manager.restoreNavigation(snapshot, vcr::catalog::Affine::scale(0.5));
            const auto restored = viewer->cameraState();
            const auto beforeGrid = surface->surfaceToGrid({camera.surfacePtrX, camera.surfacePtrY});
            const auto afterGrid = scaled->surfaceToGrid({restored.surfacePtrX, restored.surfacePtrY});
            require(cv::norm(beforeGrid - afterGrid) < 1e-5 && restored.scale == camera.scale * 2 &&
                    restored.zOffset == camera.zOffset * 0.5f,
                    "reloaded surfaces retain the physical camera center and zoom");
            // Draft intersections honor the declaration's scale and volume ID.
            struct WorkingVolume : Volume {
                explicit WorkingVolume(const fs::path& path) : Volume(path) { baseScaleLevel_ = 1; }
            };
            state.setCurrentVolume(std::make_shared<WorkingVolume>(root));
            StoreSurfaceOverlayController overlay(&state, &manager);
            const auto voxels = *state.annotationVolume();
            state.objectStoreSession()->setDraft(scaled->id, quad_surface_grid(*scaled), volumeCoordinates(voxels));
            PlaneSurface plane({8, 8, 8}, {1, 0, 0});
            const auto slab = StoreSurfaceOverlayController::planeSlab(plane);
            require(slab.has_value(), "the plane slab is valid");
            const auto crossings = overlay.draftRuns({scaled->id}, *slab);
            require(!crossings.empty() && std::abs(crossings.front().points.front()[2] - 8) < 1e-5,
                    "the selected surface has plane intersections");
            state.objectStoreSession()->clearDraft(scaled->id);
            state.setSurface("test plane", std::make_shared<PlaneSurface>(cv::Vec3f(8, 8, 8), cv::Vec3f(1, 0, 0)));
            overlay.setSurfaces({scaled->id});
            StoreSurfaceOverlayController::OverlayBuilder selectedLines(idle);
            overlay.collectPrimitives(idle, selectedLines);
            const auto& selectedView = *overlay._views.at(idle);
            pump([&] { return !selectedView.cutRequests.busy && !selectedView.residentRuns.empty(); },
                 "displayed plane intersections complete asynchronously");
            require(selectedView.targets.empty() && !selectedView.residentRuns.empty() &&
                    std::abs(selectedView.residentRuns.front().points.front()[2] - 8) < 1e-5,
                    "plane overlays use the displayed grid rather than querying the UUID's current head");
            auto anotherSelection = std::make_shared<QuadSurface>(scaled->rawPoints().clone(), scaled->scale());
            anotherSelection->id = scaled->id;
            for (auto& p : *anotherSelection->rawPointsPtr()) p[2] = 6;
            state.setSurface("segmentation", anotherSelection, false, true);
            StoreSurfaceOverlayController::OverlayBuilder switchedLines(idle);
            overlay.collectPrimitives(idle, switchedLines);
            pump([&] { return !selectedView.cutRequests.busy && !selectedView.residentRuns.empty(); },
                 "changed displayed plane intersections complete asynchronously");
            require(!selectedView.residentRuns.empty() &&
                    std::abs(selectedView.residentRuns.front().points.front()[2] - 6) < 1e-5,
                    "same-UUID selection changes replace orthogonal-plane intersections");
            viewer->quiesceForClose();
        }
        std::cerr << "viewer regression tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "viewer regression test: " << error.what() << '\n';
        return 1;
    }
}
