#include <utility>
// The VC3D surface catalog over a real object store: rows of the current
// volume's surfaces kept current from commit events, opening a surface in the
// volume's voxels, and record edits (tags, name, retire).
#include "SurfaceCatalog.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "segmentation/SurfaceCorrections.hpp"
#include "segmentation/SurfaceStoreEdit.hpp"
#include "vc/core/PointCollections.hpp"
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_reader/catalog.hpp>
#include <vc_reader/registrations.hpp>
#include "VolumeRegistrations.hpp"
#include "vc/core/types/Volume.hpp"
#include <nlohmann/json.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <filesystem>
#include <fstream>
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <iostream>
#include <set>

namespace fs = std::filesystem;
using namespace vc::objects;

namespace {
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
}

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    try {
        QTemporaryDir dir;
        const std::filesystem::path volumeRoot1 = fs::path(dir.path().toStdString()) / "ct.zarr";
        const vcr::zarr::CreateOptions levelZero1{.shape = {4, 4, 4}, .chunks = {4, 4, 4}, .compressor = "", .overwrite = true};
        vcr::zarr::Writer::create(volumeRoot1 / "0", levelZero1);
        std::ofstream(volumeRoot1 / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "vol1"}, {"name", "vol1"}, {"format", "zarr"}, {"width", 4}, {"height", 4}, {"slices", 4}, {"voxelsize", 1.0}}.dump();
        const auto volume = Volume::New(volumeRoot1);
        const auto path = fs::path(dir.path().toStdString()) / "objects.vcobj";
        OpenOptions options; options.author = "surface-catalog-test";
        { auto created = ObjectStore::create(path, options); }
        auto session = std::make_shared<ObjectStoreSession>();
        const auto attachment = session->attach(path, options);

        cv::Mat_<cv::Vec3f> points(3, 3);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) points(r, c) = {float(c), float(r), 1.f};
        QuadSurface grid(points, {1.f, 1.f});
        auto add = [&](const std::string& name, const std::optional<nlohmann::json>& coordinates) {
            const auto uuid = util::newUuid();
            session->act(attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
                ObjectRecord record; record.uuid = uuid; record.kind = "surface"; record.name = name; record.tags = {"approved"};
                auto meta = nlohmann::json::object();
                if (coordinates) meta.update(*coordinates);
                record.meta = meta.dump();
                db.upsertObject(txn, record);
                db.putGeometry(txn, uuid, quad_surface_geometry(grid));
            });
            return uuid;
        };
        const auto native = add("native", nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(volume->id(), 1)}});
        const auto half = add("half", nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(volume->id(), 0.5)}});
        const auto unbound = add("unbound", std::nullopt);
        const auto other = add("other", nlohmann::json{{"coordinates", geometryCoordinates({{"volume_uuid", "vol2"}, {"scale_factor", 1}})}});

        SurfaceCatalog catalog(session);
        std::vector<std::string> committed;
        QObject::connect(&catalog, &SurfaceCatalog::surfacesCommitted,
                         [&](const std::vector<std::string>& uuids) { committed.insert(committed.end(), uuids.begin(), uuids.end()); });
        auto pump = [&](const std::function<bool()>& done, const char* message) {
            QElapsedTimer timer; timer.start();
            while (!done() && timer.elapsed() < 10000) {
                for (const auto& event : session->takeEvents()) catalog.handle(event);
                QCoreApplication::processEvents();
                QThread::msleep(1);
            }
            if (!done()) std::cerr << "rows=" << catalog.rows().size() << " loading=" << catalog.loading()
                                   << " error=" << catalog.error().toStdString() << '\n';
            require(done(), message);
        };
        session->takeEvents();
        catalog.setSource(attachment, volume, {volume->id()});
        pump([&] { return !catalog.loading() && catalog.rows().size() == 4; }, "lists the volume's surfaces and those not bound to it");
        require(catalog.row(native)->mapping == VolumeAffine::identity() && catalog.row(half)->mapping == VolumeAffine::scale(2.0) &&
                !catalog.row(unbound)->bound() && !catalog.row(other)->bound(),
                "rows carry the mapping to the volume; other volumes' surfaces are unbound here");
        require(catalog.row(native)->toolInput() && !catalog.row(half)->toolInput(),
                "tools take surfaces stored in the volume's voxels");
        require(catalog.row(native)->record.name == "native" && catalog.row(native)->record.tags == std::vector<std::string>{"approved"},
                "rows carry name and tags");

        const auto opened = catalog.open(native);
        require(opened.editable && opened.surface->id == native && opened.pin.geometryRevision, "native surfaces open editable and pinned");
        require((*opened.surface->rawPointsPtr())(2, 1) == cv::Vec3f(1, 2, 1), "native points are stored points");
        const auto scaled = catalog.open(half);
        require(scaled.editable && scaled.view.mapping == VolumeAffine::scale(2.0), "store views of scaled surfaces support inverse edits");
        require((*scaled.surface->rawPointsPtr())(2, 1) == cv::Vec3f(2, 4, 2), "scaled points are in volume voxels");
        {
            SurfaceStoreEdit edit(session,scaled.pin,scaled.view);
            cv::Mat_<cv::Vec3f> grown(4,3,cv::Vec3f(-1,-1,-1));
            scaled.surface->rawPointsPtr()->copyTo(grown(cv::Rect(0,0,3,3)));
            grown(3,1)={2,6,2};
            QuadSurface result(grown,{0.5f,0.5f});
            edit.save(result,cv::Rect(0,0,3,4));
            session->read(attachment,[&](const auto& db,const auto& txn,const auto&) {
                require(GeometrySnapshot(db,txn).surface(half).vertex(3,1)==vc::geometry::SurfacePoint{1,3,1},
                        "growth keeps the opened view binding even when its output has no metadata");
            });
        }
        bool refused = false;
        try { catalog.open(unbound); } catch (const std::exception&) { refused = true; }
        require(refused, "unbound surfaces do not open");

        // Editing an opened surface commits to its store object and re-pins.
        {
            auto edited = catalog.open(native);
            SurfaceStoreEdit edit(session, edited.pin);
            auto lease = session->holdForEditing(edited.pin);
            auto& grid = *edited.surface->rawPointsPtr();
            grid(1, 1) = {5.f, 5.f, 5.f};
            edit.save(*edited.surface, cv::Rect(1, 1, 1, 1));
            require(edit.pin().objectRevision != edited.pin.objectRevision, "saves re-pin from the receipt");
            require((*catalog.open(native).surface->rawPointsPtr())(1, 1) == cv::Vec3f(5, 5, 5), "vertex edits reach the store");
            // The mask brush: validity 0 and the sentinel in memory; stored coordinates survive.
            cv::Mat_<uint8_t> mask(3, 3, uint8_t(255));
            mask(0, 0) = 0;
            grid(0, 0) = {-1.f, -1.f, -1.f};
            edited.surface->setChannel("mask", mask);
            edit.save(*edited.surface, cv::Rect(0, 0, 3, 3), {"mask"});
            require((*catalog.open(native).surface->rawPointsPtr())(0, 0) == cv::Vec3f(-1, -1, -1), "masked vertices load masked");
            session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                const auto unmasked = load_quad_from_store(db, txn, native, std::nullopt, false);
                require((*unmasked->rawPointsPtr())(0, 0) == cv::Vec3f(0, 0, 1), "masking keeps stored coordinates");
            });
            // Growth: another grid size replaces the geometry.
            cv::Mat_<cv::Vec3f> grown(4, 3, cv::Vec3f(-1.f, -1.f, -1.f));
            grid.copyTo(grown(cv::Rect(0, 0, 3, 3)));
            grown(3, 1) = {1.f, 3.f, 1.f};
            QuadSurface larger(grown, {1.f, 1.f});
            edit.save(larger, cv::Rect(0, 0, 3, 4));
            require(catalog.open(native).surface->rawPointsPtr()->size() == cv::Size(3, 4), "growth replaces the stored grid");
            lease = {};
            bool stale = false;
            try { session->holdForEditing(edited.pin); } catch (const std::exception&) { stale = true; }
            require(stale, "an edit from a superseded pin is refused");
        }

        // Correction points live with their surface as linked point collections.
        {
            PointCollections points;
            points.addPoint("fix", {1, 2, 3});
            const auto fix = points.getCollectionId("fix").value();
            points.setCollectionAnchor2d(fix, cv::Vec2f(1.5f, 0.5f));
            const auto saved = points.anchoredCollections();
            vc3d::corrections::write(*session, attachment, half, saved, {volume->id()});
            auto stored = vc3d::corrections::read(*session, attachment, half);
            require(stored.size() == 1 && stored.begin()->first == fix, "corrections round-trip by collection id");
            session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                const auto uuid = db.neighbours(txn, half, std::string(vc::objects::kAttachmentEdgeKind)).at(0);
                require(nlohmann::json::parse(db.objectByUuid(txn, uuid)->meta)["role"] == "corrections" &&
                            db.neighbours(txn, half).empty(),
                        "corrections are attached data, not graph links");
            });
            PointCollections reloaded;
            require(reloaded.replaceAnchoredCollections(stored), "stored corrections load");
            const auto& collection = reloaded.getAllCollections().at(fix);
            require(collection.uuid == points.getAllCollections().at(fix).uuid && collection.anchor2d == cv::Vec2f(1.5f, 0.5f) &&
                    collection.points.size() == 1 && collection.points.begin()->second.p == cv::Vec3f(1, 2, 3),
                    "corrections keep identity, anchor and points");
            points.addPoint("fix", {4, 5, 6});
            vc3d::corrections::write(*session, attachment, half, points.anchoredCollections(), {volume->id()});
            require(vc3d::corrections::read(*session, attachment, half).at(fix).at("points").size() == 2, "corrections update in place");
            {
                const auto firstPoint = points.getAllCollections().at(fix).points.begin()->second;
                auto movedPoint = firstPoint;
                movedPoint.p = {7, 8, 9};
                points.updatePoint(movedPoint);
                vc3d::corrections::write(*session, attachment, half, points.anchoredCollections(), {volume->id()});
                const auto stored = vc3d::corrections::read(*session, attachment, half).at(fix);
                const auto& moved = stored.at("points").at(std::to_string(firstPoint.id)).at("p");
                require(moved[0].get<double>() == 7 && moved[2].get<double>() == 9, "moved correction points keep their id");
            }
            {
                // Through a view at half scale: points read halved, unchanged ones
                // keep their stored values on save, moved ones map back.
                const vc::objects::ViewFrame coarse{"vol1", 0.5, VolumeAffine::scale(0.5)};
                auto shown = vc3d::corrections::read(*session, attachment, half, coarse);
                const auto& ordered = points.getAllCollections().at(fix).points;
                const auto firstId = std::to_string(ordered.begin()->first), secondId = std::to_string(std::next(ordered.begin())->first);
                require(shown.at(fix).at("points").at(firstId).at("p")[0].get<double>() == 3.5, "corrections read through the view");
                shown.at(fix)["points"][secondId]["p"] = nlohmann::json::parse("[1.0, 1.0, 1.0]");
                vc3d::corrections::write(*session, attachment, half, shown, {volume->id()}, coarse);
                const auto stored = vc3d::corrections::read(*session, attachment, half).at(fix).at("points");
                require(stored.at(firstId).at("p")[0].get<double>() == 7 && stored.at(secondId).at("p")[0].get<double>() == 2,
                        "corrections save back to their stored frame");
            }
            vc3d::corrections::write(*session, attachment, half, {}, {volume->id()});
            require(vc3d::corrections::read(*session, attachment, half).empty(), "dropped corrections are retired");
            require(vc3d::corrections::read(*session, attachment, native).empty(), "corrections belong to one surface");
            vc3d::corrections::GrowthAnnotation annotation;
            annotation.surfaceUuid = half;
            annotation.beforeGeometry = "before";
            annotation.afterGeometry = "after";
            annotation.points = {{fix, 2, "fix", {1, 2, 3}, 20}, {fix, 1, "fix", {4, 5, 6}, 10}};
            const auto annotationUuid = vc3d::corrections::writeGrowthAnnotation(*session, attachment, annotation, {volume->id()});
            session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                const auto object = db.objectByUuid(txn, annotationUuid);
                require(object && nlohmann::json::parse(object->meta)["source"]["growth_annotation"]["after_geometry"] == "after",
                        "growth annotations record the revisions");
                const auto points = std::get<vc::geometry::PointCollection>(
                    vc::geometry::decode(db.readGeometryData(txn, db.currentGeometry(txn, annotationUuid)->uuid)));
                require(points.ids == std::vector<int64_t>{1, 2}, "annotation points keep placement order");
            });
            require(vc3d::corrections::read(*session, attachment, half).empty(), "annotations are not corrections");
        }

        session->tagObjects(attachment, {native}, {"reviewed"}, {"approved"});
        session->editRecords(attachment, {native}, [](ObjectRecord& record) { record.name = "renamed"; });
        pump([&] {
            const auto* row = catalog.row(native);
            return row && row->record.name == "renamed" && row->record.tags == std::vector<std::string>{"reviewed"};
        }, "record edits reach the rows through commit events");
        require(std::count(committed.begin(), committed.end(), native) >= 1, "commits name the edited surface");
        session->editCurrent(attachment, {half}, [](auto& db, auto& txn, const auto& uuid) { db.retireObject(txn, uuid); });
        pump([&] { return !catalog.row(half) && catalog.rows().size() == 3; }, "retired surfaces leave the listing");

        // The store supplies registered views; the application only adapts their tiles.
        VolumeAffine affine;
        affine(0,0)=2; affine(0,1)=0.5; affine(0,3)=10; affine(2,3)=20;
        const auto registration=util::newUuid();
        session->act(attachment,{{registration,std::nullopt}},[&](auto& db,auto& txn) {
            db.registerVolumeTransform(txn,"vol2","vol1",affine,"registered scan",registration);
        });
        pump([&]{return catalog.row(other)&&catalog.row(other)->mapping.has_value();},"registration commits refresh compatible surfaces");
        {
            // Area: the store's, in stored units (vol2: the 2x2 unit grid); the
            // open volume, where the shear doubles x, sees twice it.
            require(catalog.row(other)->record.surface&&catalog.row(other)->record.surface->area==4.0,
                    "the store keeps the area in stored units");
            require(std::abs(([&] { const auto opened = catalog.open(other); const auto& points = *std::as_const(*opened.surface).rawPointsPtr(); return vc::geometry::surfaceGridArea(points.rows, points.cols,
            [&](int64_t r, int64_t c) -> const cv::Vec3f& { return points(int(r), int(c)); },
            [&](int64_t r, int64_t c) { return vc::geometry::validSurfacePoint(points(int(r), int(c))); }); }())-8.0)<1e-9,
                    "the open volume sees the sheared area");
        }
        auto registered=catalog.open(other);
        require((*registered.surface->rawPointsPtr())(2,1)==cv::Vec3f(13,2,21),"surface uses the store's affine view");
        SurfaceStoreEdit registeredEdit(session,registered.pin,registered.view);
        (*registered.surface->rawPointsPtr())(2,1)={15,2,21};
        registeredEdit.save(*registered.surface,cv::Rect(1,2,1,1));
        session->read(attachment,[&](const auto& db,const auto& txn,const auto&) {
            require(GeometrySnapshot(db,txn).surface(other).vertex(2,1)==vc::geometry::SurfacePoint{2,2,1},
                    "registered view edits remain in canonical coordinates");
            require(nlohmann::json::parse(db.objectByUuid(txn,other)->meta)["coordinates"]["volume_uuid"]=="vol2",
                    "view edits preserve the paired volume");
        });
        const auto conflicting=util::newUuid();
        session->act(attachment,{{conflicting,std::nullopt}},[&](auto& db,auto& txn) {
            db.registerVolumeTransform(txn,"vol2","vol1",VolumeAffine::scale(3),"alternative",conflicting);
        });
        bool staleMapping=false;
        try {registeredEdit.save(*registered.surface,cv::Rect(1,2,1,1));}catch(const std::exception&){staleMapping=true;}
        require(staleMapping,"registration conflicts invalidate writable views");

        // What VC3D does with registrations, through the store: list the related
        // volumes, choose a route for an ambiguous pair, register a new affine.
        {
            namespace reg = vc3d::registrations;
            auto relatedTo = [&](const std::string& volumeId) {
                std::vector<VolumeTransforms::Related> result;
                session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                    result = db.volumeTransforms(txn).related(volumeId);
                });
                return result;
            };
            auto related = relatedTo("vol1");
            require(related.size() == 1 && related[0].volume == "vol2" &&
                    related[0].status == vcr::catalog::TransformStatus::Ambiguous && related[0].routes.size() == 2 &&
                    !related[0].chosen, "two registrations of vol2 make an ambiguous pair");
            const auto chosenPath = related[0].routes[1].path;
            chooseVolumeTransform(*session, attachment, "vol2", "vol1", chosenPath);
            related = relatedTo("vol1");
            require(related[0].status == vcr::catalog::TransformStatus::Resolved && related[0].chosen == chosenPath,
                    "a chosen route resolves the pair");
            chooseVolumeTransform(*session, attachment, "vol2", "vol1", related[0].routes[0].path);
            require(relatedTo("vol1")[0].chosen == related[0].routes[0].path, "choosing again replaces the choice");
            const auto opened = catalog.open(other);
            require(opened.view.mapping == related[0].routes[0].forward, "views follow the chosen route");

            require(reg::parseAffine("[[2, 0, 0, 1], [0, 2, 0, 0], [0, 0, 2, 0]]") ==
                        VolumeAffine::fromJson(nlohmann::json::parse("[[2,0,0,1],[0,2,0,0],[0,0,2,0]]")),
                    "a pasted JSON matrix parses");
            require(reg::parseAffine("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1") == VolumeAffine::identity(), "4x4 rows parse");
            bool rejected = false;
            try { reg::parseAffine("1 0 0\n0 1 0"); } catch (const std::exception&) { rejected = true; }
            require(rejected, "a matrix of the wrong size is refused");
            rejected = false;
            try { reg::parseAffine("0 0 0 0 0 0 0 0 0 0 0 0"); } catch (const std::exception&) { rejected = true; }
            require(rejected, "a singular matrix is refused");

            registerVolumeTransform(*session, attachment, "vol3", "vol1", VolumeAffine::scale(4), "test scan");
            const auto withNew = relatedTo("vol1");
            const auto vol3 = std::find_if(withNew.begin(), withNew.end(), [](const auto& r) { return r.volume == "vol3"; });
            require(vol3 != withNew.end() && vol3->status == vcr::catalog::TransformStatus::Resolved &&
                    vol3->routes.front().forward == VolumeAffine::scale(4), "a registered affine relates its volume");
            session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                require(db.volumeTransforms(txn).graph.describe(vol3->routes.front().path).find("test scan") != std::string::npos,
                        "routes are described by registration name");
            });
        }

        // The volume at working level 1 is the same volume in half its voxels:
        // surfaces are listed at that scale, and those made there are kept in
        // the volume's own voxels at it, seen at full scale in the full volume.
        const std::filesystem::path volumeRoot2 = fs::path(dir.path().toStdString()) / "level.zarr";
        const vcr::zarr::CreateOptions levelZero2{.shape = {2, 2, 2}, .chunks = {2, 2, 2}, .compressor = "", .overwrite = true};
        vcr::zarr::Writer::create(volumeRoot2 / "0", levelZero2);
        std::ofstream(volumeRoot2 / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "vol1"}, {"name", "vol1"}, {"format", "zarr"}, {"width", 2}, {"height", 2}, {"slices", 2}, {"voxelsize", 1.0}}.dump();
        const auto level = Volume::New(volumeRoot2);
        const VolumeVoxels atHalf{volume->id(), 0.5};
        const auto atLevel = util::newUuid();
        session->act(attachment, {{atLevel, std::nullopt}}, [&](auto& db, auto& txn) {
            ObjectRecord record; record.uuid = atLevel; record.kind = "surface"; record.name = "at-level";
            record.meta = nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(atHalf)}}.dump();
            db.upsertObject(txn, record);
            db.putGeometry(txn, atLevel, quad_surface_geometry(grid));
        });
        catalog.setSource(attachment, level, atHalf);
        pump([&] { const auto* row = catalog.row(native); return !catalog.loading() && row && row->mapping; },
             "the level lists the surfaces of its volume");
        require(*catalog.row(native)->mapping == VolumeAffine::scale(0.5), "seen at the level's scale");
        require(catalog.row(atLevel)->toolInput(), "tools at the level take surfaces made there");
        catalog.setSource(attachment, volume, {volume->id()});
        pump([&] { const auto* row = catalog.row(atLevel); return !catalog.loading() && row && row->mapping; },
             "the full volume lists surfaces made at the level");
        require(*catalog.row(atLevel)->mapping == VolumeAffine::scale(2), "seen at the full volume's scale");

        catalog.setSource(std::nullopt, volume, {volume->id()});
        require(catalog.rows().empty(), "no attachment lists nothing");

        // A cold intersection query waits for every selected placeholder and
        // returns their crossings together, without fetching unrelated surfaces.
        {
            const auto remotePath = fs::path(dir.path().toStdString()) / "remote.vcobj";
            const std::vector<std::string> ids{util::newUuid(), util::newUuid(), util::newUuid(), util::newUuid()};
            {
                auto db = ObjectStore::create(remotePath, options);
                auto txn = db.write();
                for (const auto& uuid : ids) {
                    ObjectRecord record;
                    record.uuid = uuid;
                    record.kind = "surface";
                    record.storageKind = "file";
                    record.storageRef = uuid;
                    record.meta = nlohmann::json{{"coordinates", volumeCoordinates(volume->id(), 1)},
                                                 {"published", true}}.dump();
                    record.surface.emplace();
                    record.surface->bbox = std::array<double, 6>{0, 0, 1, 2, 2, 1};
                    db.upsertObject(txn, record);
                }
                txn.commit();
            }
            // The session owns the fetch thread; destroy it before its captures.
            std::atomic_int fetchCount{0};
            ObjectStoreSession remoteSession;
            remoteSession.setGeometryFetcher("published",
                [&](ObjectStore& store, const std::string& uuid, const ObjectStoreSession::FetchWrite& write) {
                    if (uuid == ids[2]) throw std::runtime_error("test fetch failure");
                    const auto revision = store.currentObjectRevision(store.read(), uuid)->uuid;
                    ++fetchCount;
                    write({{uuid, revision}}, [&](auto& db, auto& txn) {
                        db.putGeometry(txn, uuid, quad_surface_geometry(grid));
                    });
                });
            const auto remote = remoteSession.attach(remotePath, options);
            RegionSlice slice;
            slice.origin = {0, 1, 0};
            slice.u = {1, 0, 0};
            slice.v = {0, 0, 1};
            slice.normal = {0, 1, 0};
            slice.localBounds = {-1, -1, -1, 3, 3, 1};
            RegionOptions query;
            query.uuids = std::vector<std::string>{ids[0], ids[1]};
            const auto cold = SurfaceCatalog::region(remoteSession, {remote}, {volume->id()}, slice, query);
            std::set<std::string> reached;
            for (const auto& [attachment, uuid] : cold.remote) reached.insert(uuid);
            require(cold.runs.empty() && reached == std::set<std::string>{ids[0], ids[1]},
                    "a cold query lists the selected placeholders it reached");
            for (const auto& [attachment, uuid] : cold.remote) remoteSession.ensureGeometry(attachment, uuid);
            const auto crossings = SurfaceCatalog::region(remoteSession, {remote}, {volume->id()}, slice, query);
            std::set<std::string> crossed;
            for (const auto& run : crossings.runs) crossed.insert(run.objectUuid);
            require(crossings.complete && crossings.remote.empty() && crossed == std::set<std::string>{ids[0], ids[1]} &&
                        fetchCount == 2,
                    "fetched placeholders are read, and only the selected surfaces were fetched");
            const auto warm = SurfaceCatalog::region(remoteSession, {remote}, {volume->id()}, slice, query);
            require(warm.complete && warm.runs.size() == crossings.runs.size() && fetchCount == 2,
                    "warm queries reuse fetched geometry");
            query.uuids = std::vector<std::string>{ids[2]};
            const auto failing = SurfaceCatalog::region(remoteSession, {remote}, {volume->id()}, slice, query);
            bool failed = false;
            try { for (const auto& [attachment, uuid] : failing.remote) remoteSession.ensureGeometry(attachment, uuid); }
            catch (const GeometryUnavailable&) { failed = true; }
            require(failing.remote.size() == 1 && failed, "fetch errors are reported to the fetcher");
            query.uuids = std::vector<std::string>{ids[3]};
            const auto cancelled = SurfaceCatalog::region(remoteSession, {remote}, {volume->id()}, slice, query,
                                                          [] { return true; });
            require(!cancelled.complete && cancelled.runs.empty() && fetchCount == 2,
                    "cancelled queries neither publish intersections nor fetch more surfaces");
        }
        session->detach(attachment);
        std::cout << "surface catalog tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "surface catalog test: " << error.what() << '\n';
        return 1;
    }
}
