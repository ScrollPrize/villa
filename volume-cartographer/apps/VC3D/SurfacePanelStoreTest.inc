#include <QApplication>
#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <iostream>
#include <vc_object_store/Uuid.hpp>
// Built only with VC_TESTING. The surface panel over a real object store:
// listing, filters, tags, activation and deletion through commit events.
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
        state.detachObjectStore();
        std::cerr << "surface panel store tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "surface panel store test: " << error.what() << '\n';
        return 1;
    }
}
