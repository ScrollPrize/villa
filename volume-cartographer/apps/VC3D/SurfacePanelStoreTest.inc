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
#include "support/TestZarrVolume.hpp"
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
        vc::test::ZarrVolumeSpec create;
        create.shapeZYX = {4, 4, 4}; create.chunkShapeZYX = {4, 4, 4}; create.uuid = "panel-volume";
        const auto volume = vc::test::createZarrVolume(fs::path(dir.path().toStdString()) / "ct.zarr", create);
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
            state.objectStoreSession()->act(attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
                create_quad_in_store(db, txn, uuid, grid, name, tags,
                                     bound ? vc::objects::volumeCoordinates(volume->id(), 1) : nlohmann::json::object(),
                                     {"surface-panel-test"});
            });
            return uuid;
        };
        const auto approved = add("approved", {"approved"}, true);
        const auto plain = add("plain", {}, true);
        const auto unbound = add("unbound", {}, false);

        SurfaceCatalog catalog(state.objectStoreSession());
        QObject::connect(&state, &CState::objectStoreChanged, &catalog,
                         [&](const StoreSessionEvent& event) { catalog.handle(event); });
        catalog.setSource(attachment, volume);
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
        require(panel.opened() && panel.opened()->editable && panel.opened()->pin.object.objectUuid == plain,
                "the opened surface is pinned and editable");
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
            published.meta = utils::Json::parse(nlohmann::json{{"catalog", nlohmann::json::parse(binding).at("catalog")}}.dump());
            create_quad_in_store(db, txn, shown, published, "shown", {}, nlohmann::json::parse(binding).at("coordinates"),
                                 {"catalog-test"});
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
                const auto revision = store.currentObjectRevision(store.read(), uuid)->uuid;
                write({{uuid, revision}}, [&](ObjectStore& db, WriteTxn& txn) {
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
        catalog.setSource(state.storeFor("surface"), volume);
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
        const auto publishedRevision = panel.opened()->pin.objectRevision;
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
        require(panel.restoreRevision(shown, publishedRevision, &error), "sample surfaces can restore their history");
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
