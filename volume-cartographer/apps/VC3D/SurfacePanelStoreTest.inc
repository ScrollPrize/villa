#include <QApplication>
#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <iostream>
#include <vc_object_store/Uuid.hpp>
#include "segmentation/growth/SegmentationGrower.hpp"
#include "viewer_controls/panels/VolumeRegistrationsPanel.hpp"
#include <vc_reader/registrations.hpp>
#include <mutex>
#include <QTreeWidget>
// Built only with VC_TESTING. The surface panel over a real object store:
// listing, filters, tags, activation and deletion through commit events; a
// fetched catalog store (as an Open Data sample's) of a volume the open one
// shows at pyramid level 1, listed beside it, fetched on activation and edited
// in the user's store as the same surface.
int runSurfacePanelStoreTests()
{
    using namespace vc::objects;
    namespace fs = std::filesystem;
    auto require = [](bool value, const char* message) { if (!value) throw std::runtime_error(message); };
    try {
        QTemporaryDir dir;
        Volume::ZarrCreateOptions create;
        create.shapeZYX = {4, 4, 4}; create.chunkShapeZYX = {4, 4, 4}; create.compressor = "none"; create.uuid = "panel-volume";
        const auto volume = Volume::New(fs::path(dir.path().toStdString()) / "ct.zarr", create);
        const auto path = fs::path(dir.path().toStdString()) / "objects.vcobj";
        OpenOptions options; options.author = "surface-panel-test";
        { auto created = ObjectStore::create(path, options); }

        CState state;
        state.attachObjectStore(QString::fromStdString(path.string()), "surface-panel-test");
        const auto attachment = *state.objectStoreAttachment();
        cv::Mat_<cv::Vec3f> points(3, 3);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) points(r, c) = {float(c), float(r), 2.f};
        QuadSurface grid(points, {1.f, 1.f});
        auto add = [&](const std::string& name, std::vector<std::string> tags, bool bound) {
            const auto uuid = util::newUuid();
            state.objectStoreSession()->act(attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
                create_quad_in_store(db, txn, uuid, grid, name, tags,
                                     bound ? nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(volume->id(), 1)}}.dump() : std::string("{}"));
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
        panel.configureTags({nullptr, &approvedTag, &defectiveTag, &reviewedTag, &inspectTag});
        SurfacePanelController::FilterUiRefs filters;
        filters.hideUnapproved = &hideUnapproved;
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
        require(visible() == std::vector<std::string>{approved}, "tag filters read record tags");
        hideUnapproved.setChecked(false);

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
        require(visible() == sorted({approved, plain}), "filters follow committed tags");

        require(panel.retireSurfaces({QString::fromStdString(approved)}, &error), "surfaces retire");
        pump([&] { return tree.topLevelItemCount() == 2; }, "retired surfaces leave the panel");
        require(visible() == std::vector<std::string>{plain}, "retired surfaces are not drawn");

        // A catalog store of the catalog volume the open volume shows at level 1
        // (the catalog's level registration): one surface with geometry, one
        // still remote, fetched by the session.
        const auto catalogPath = fs::path(dir.path().toStdString()) / "catalog.vcobj";
        std::string shown, remote;
        {
            auto db = ObjectStore::create(catalogPath, options);
            const auto binding = nlohmann::json{{"coordinates", {{"volume_uuid", "catalog-volume"}, {"scale_factor", 1}}},
                                                {"catalog", {{"sample", "sample"}, {"segment", "segment"}}}}.dump();
            shown = util::newUuid();
            remote = util::newUuid();
            auto txn = db.write({{shown, std::nullopt}, {remote, std::nullopt}});
            create_quad_in_store(db, txn, shown, grid, "shown", {}, binding);
            ObjectRecord record;
            record.uuid = remote; record.kind = "surface"; record.name = "remote"; record.meta = binding;
            db.upsertObject(txn, record);
            txn.commit();
        }
        auto levels = std::make_shared<vcr::catalog::RegistrationGraph>();
        levels->add({"level:sample:catalog-volume:1", "1", "catalog-volume", "panel-volume",
                     vcr::catalog::Affine::scale(0.5), "pyramid level 1"});
        auto catalogOptions = options;
        catalogOptions.catalogTransforms = levels;
        std::mutex fetchedMutex;
        std::vector<std::string> fetched;
        const auto catalogAttachment = state.objectStoreSession()->attachFetched(catalogPath, catalogOptions,
            [&](ObjectStore& store, const std::string& uuid, const ObjectStoreSession::FetchWrite& write) {
                { std::lock_guard lock(fetchedMutex); fetched.push_back(uuid); }
                const auto revision = store.currentObjectRevision(store.read(), uuid)->uuid;
                write({{uuid, revision}}, [&](ObjectStore& db, WriteTxn& txn) {
                    db.putGeometry(txn, uuid, quad_surface_geometry(grid));
                });
            });
        auto fetchedNow = [&] { std::lock_guard lock(fetchedMutex); return fetched; };
        catalog.setCatalogSources({catalogAttachment});
        pump([&] { return tree.topLevelItemCount() == 4; }, "catalog surfaces are listed beside the user's");
        require(catalog.row(shown)->catalog && !catalog.row(plain)->catalog, "rows know their store");
        require(catalog.row(shown)->mapping == vc::objects::VolumeAffine::scale(0.5) && !catalog.row(shown)->toolInput(), "catalog rows are scaled to the open volume's level");

        require(panel.activateSurfaceById(shown, &error), "catalog surfaces open");
        require((*activated.back()->rawPointsPtr())(1, 2) == cv::Vec3f(1, 0.5f, 1), "in the open volume's voxels");
        require(!panel.opened()->editable, "catalog surfaces are read-only");
        require(panel.activateSurfaceById(remote, &error) && fetchedNow() == std::vector<std::string>{remote},
                "a remote catalog surface is fetched when activated headlessly");

        // The Patch Tracer reads each source surface from the store it is in.
        PatchTracerSource tracerSource{state.objectStoreSession(), {}, {}};
        for (const auto& [uuid, row] : catalog.rows())
            if (row.mapping) tracerSource.surfaces.push_back({row.source, uuid, volume->id()});
        const auto tracerSurfaces = loadPatchTracerSurfaces(tracerSource);
        require(tracerSource.surfaces.size() == 3 && tracerSurfaces.size() == 3 &&
                std::any_of(tracerSurfaces.begin(), tracerSurfaces.end(), [](const auto& surface) {
                    return (*surface->rawPointsPtr())(1, 2) == cv::Vec3f(1, 0.5f, 1);
                }),
                "Patch Tracer sources load from their own stores, catalog ones scaled to the volume");

        // Editing a catalog surface edits it in the user's store as the same
        // surface. The user's store relates the two volumes as the catalog does
        // (VC3D gives every attachment the catalog's registrations).
        vc::objects::registerVolumeTransform(*state.objectStoreSession(), attachment, "catalog-volume", "panel-volume",
                                             vc::objects::VolumeAffine::scale(0.5), "pyramid level 1");
        require(panel.activateSurfaceById(shown, &error), "the catalog surface opens");
        const auto onScreen = panel.opened()->surface;
        require(panel.editInUserStore(shown, &error) && panel.opened()->editable &&
                    panel.opened()->pin.attachment == attachment && panel.opened()->surface == onScreen,
                "editing a catalog surface continues it, on screen, in the user's store");
        require(!catalog.row(shown)->catalog && catalog.row(shown)->imported() &&
                    catalog.row(shown)->mapping == vc::objects::VolumeAffine::scale(0.5),
                "its row is the user's, as stored, with the catalog version as its original");
        require(tree.topLevelItemCount() == 4, "it is listed once");
        std::string catalogVersion;
        state.objectStoreSession()->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            const auto history = db.objectHistory(txn, shown);
            require(history.size() == 1 && nlohmann::json::parse(history.front().record.meta)["coordinates"]["volume_uuid"] ==
                                               "catalog-volume", "its first revision is the catalog version as stored");
            catalogVersion = history.front().uuid;
        });
        catalog.rename(shown, "edited");
        pump([&] { return catalog.row(shown)->record.name == "edited"; }, "the imported surface is edited in the user's store");
        const auto past = catalog.open(shown, catalogVersion);
        require(!past.current && !past.editable && past.surface->id == shown &&
                    (*past.surface->rawPointsPtr())(1, 2) == cv::Vec3f(1, 0.5f, 1), "a past revision opens read-only");
        require(panel.resetToCatalogVersion(shown, &error), "an imported surface resets to its catalog version");
        pump([&] { return catalog.row(shown)->record.name == "shown"; }, "the reset restores the catalog version");

        // The Transforms dock lists the volumes the store maps into the open one,
        // including a registration committed while it is shown.
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
        state.detachObjectStore();
        std::cerr << "surface panel store tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "surface panel store test: " << error.what() << '\n';
        return 1;
    }
}
