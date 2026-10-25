#include <QApplication>
#include <QPushButton>
#include <QSpinBox>
#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <iostream>
#include <vc_object_store/Uuid.hpp>
#include "SurfaceAffineTransformController.hpp"
#include "segmentation/growth/SegmentationGrower.hpp"
#include "viewer_controls/panels/ViewerTransformsPanel.hpp"
// Built only with VC_TESTING. The surface panel over a real object store:
// listing, filters, tags, activation and deletion through commit events; a
// read-only catalog store listed beside it at a pyramid level, fetched on
// activation and copied into the user's store.
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
                                     bound ? vc3d::coordinateBindingForVolume(*volume, 1).dump() : std::string("{}"));
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

        // A catalog store of another (catalog) volume that the open volume shows at
        // level 1: one surface with geometry, one still remote.
        const auto catalogPath = fs::path(dir.path().toStdString()) / "catalog.vcobj";
        std::string shown, remote;
        {
            auto db = ObjectStore::create(catalogPath, options);
            const auto binding = nlohmann::json{{"coordinates", {{"volume_uuid", "catalog-volume"}, {"scale_factor", 1}}}}.dump();
            shown = util::newUuid();
            remote = util::newUuid();
            auto txn = db.write({{shown, std::nullopt}, {remote, std::nullopt}});
            create_quad_in_store(db, txn, shown, grid, "shown", {}, binding);
            ObjectRecord record;
            record.uuid = remote; record.kind = "surface"; record.name = "remote"; record.meta = binding;
            db.upsertObject(txn, record);
            txn.commit();
        }
        std::vector<std::string> fetched;
        catalog.setFetcher([&](const fs::path& store, const std::string& uuid) {
            fetched.push_back(uuid);
            auto db = ObjectStore::open(store, options);
            auto txn = db.write({{uuid, db.currentObjectRevision(db.read(), uuid)->uuid}});
            db.putGeometry(txn, uuid, quad_surface_geometry(grid));
            txn.commit();
        });
        catalog.setCatalogSources({{state.catalogAttachment(catalogPath), "catalog-volume", 0.5}});
        pump([&] { return tree.topLevelItemCount() == 4; }, "catalog surfaces are listed beside the user's");
        require(catalog.row(shown)->catalog && !catalog.row(plain)->catalog, "rows know their store");
        require(catalog.row(shown)->toVolume == 0.5, "catalog rows are scaled to the open volume's level");

        require(panel.activateSurfaceById(shown, &error), "catalog surfaces open");
        require((*activated.back()->rawPointsPtr())(1, 2) == cv::Vec3f(1, 0.5f, 1), "in the open volume's voxels");
        require(!panel.opened()->editable, "catalog surfaces are read-only");
        require(panel.activateSurfaceById(remote, &error) && fetched == std::vector<std::string>{remote},
                "a remote catalog surface is fetched when activated headlessly");

        // The Patch Tracer reads each source surface from the store it is in.
        PatchTracerSource tracerSource{state.objectStoreSession(), {}, {}};
        for (const auto& [uuid, row] : catalog.rows())
            if (row.toVolume) tracerSource.surfaces.push_back({row.source.attachment, uuid, *row.toVolume});
        const auto tracerSurfaces = loadPatchTracerSurfaces(tracerSource);
        require(tracerSource.surfaces.size() == 3 && tracerSurfaces.size() == 3 &&
                std::any_of(tracerSurfaces.begin(), tracerSurfaces.end(), [](const auto& surface) {
                    return (*surface->rawPointsPtr())(1, 2) == cv::Vec3f(1, 0.5f, 1);
                }),
                "Patch Tracer sources load from their own stores, catalog ones scaled to the volume");

        const auto copy = catalog.copy(shown, "editable");
        catalog.readNow({copy});
        require(catalog.row(copy) && !catalog.row(copy)->catalog && catalog.row(copy)->toVolume == 1.0,
                "a copy of a catalog surface is the user's, in the open volume's voxels");
        require(panel.activateSurfaceById(copy, &error) && panel.opened()->editable &&
                (*activated.back()->rawPointsPtr())(1, 2) == cv::Vec3f(1, 0.5f, 1), "the copy opens editable");

        // A transform of a store surface can be saved (as a new surface).
        state.setSurface("segmentation", activated.back());
        ViewerTransformsPanel transforms;
        SurfaceAffineTransformController transformController({&state, &transforms, nullptr, nullptr, &panel, nullptr, nullptr, {}});
        transforms.findChild<QSpinBox*>()->setValue(2);
        QPushButton* save = nullptr;
        for (auto* button : transforms.findChildren<QPushButton*>())
            if (button->text().contains("Save")) save = button;
        transformController.refresh();
        require(save && save->isEnabled(), "a transformed store surface can be saved");
        state.detachObjectStore();
        std::cerr << "surface panel store tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "surface panel store test: " << error.what() << '\n';
        return 1;
    }
}
