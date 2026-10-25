// The VC3D surface catalog over a real object store: rows of the current
// volume's surfaces kept current from commit events, opening a surface in the
// volume's voxels, and record edits (tags, name, retire).
#include "SurfaceCatalog.hpp"
#include "SurfaceAreaCalculator.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "segmentation/SurfaceCorrections.hpp"
#include "segmentation/SurfaceStoreEdit.hpp"
#include "vc/core/PointCollections.hpp"
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include "ObjectStoreCoordinates.hpp"
#include "OpenDataSurfaceStore.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <filesystem>
#include <iostream>

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
        Volume::ZarrCreateOptions create;
        create.shapeZYX = {4, 4, 4}; create.chunkShapeZYX = {4, 4, 4}; create.compressor = "none"; create.uuid = "vol1";
        const auto volume = Volume::New(fs::path(dir.path().toStdString()) / "ct.zarr", create);
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
        const auto native = add("native", vc3d::coordinateBindingForVolume(*volume, 1));
        const auto half = add("half", vc3d::coordinateBindingForVolume(*volume, 0.5));
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
        catalog.setSource(attachment, volume);
        pump([&] { return !catalog.loading() && catalog.rows().size() == 4; }, "lists the volume's surfaces and those not bound to it");
        require(catalog.row(native)->toVolume == 1.0 && catalog.row(half)->toVolume == 2.0 && !catalog.row(unbound)->toVolume &&
                !catalog.row(other)->toVolume, "rows carry the volume scale; other volumes' surfaces are unbound here");
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
            edit.hold();
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
            edit.release();
            bool stale = false;
            try { SurfaceStoreEdit(session, edited.pin).hold(); } catch (const std::exception&) { stale = true; }
            require(stale, "an edit from a superseded pin is refused");
        }

        // Correction points live with their surface as linked point collections.
        {
            PointCollections points;
            points.addPoint("fix", {1, 2, 3});
            const auto fix = points.getCollectionId("fix").value();
            points.setCollectionAnchor2d(fix, cv::Vec2f(1.5f, 0.5f));
            const auto saved = points.anchoredCollections();
            vc3d::corrections::write(*session, attachment, half, saved, *volume);
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
            vc3d::corrections::write(*session, attachment, half, points.anchoredCollections(), *volume);
            require(vc3d::corrections::read(*session, attachment, half).at(fix).at("points").size() == 2, "corrections update in place");
            {
                const auto firstPoint = points.getAllCollections().at(fix).points.begin()->second;
                auto movedPoint = firstPoint;
                movedPoint.p = {7, 8, 9};
                points.updatePoint(movedPoint);
                vc3d::corrections::write(*session, attachment, half, points.anchoredCollections(), *volume);
                const auto stored = vc3d::corrections::read(*session, attachment, half).at(fix);
                const auto& moved = stored.at("points").at(std::to_string(firstPoint.id)).at("p");
                require(moved[0].get_double() == 7 && moved[2].get_double() == 9, "moved correction points keep their id");
            }
            vc3d::corrections::write(*session, attachment, half, {}, *volume);
            require(vc3d::corrections::read(*session, attachment, half).empty(), "dropped corrections are retired");
            require(vc3d::corrections::read(*session, attachment, native).empty(), "corrections belong to one surface");
            vc3d::corrections::GrowthAnnotation annotation;
            annotation.surfaceUuid = half;
            annotation.beforeGeometry = "before";
            annotation.afterGeometry = "after";
            annotation.points = {{fix, 2, "fix", {1, 2, 3}, 20}, {fix, 1, "fix", {4, 5, 6}, 10}};
            const auto annotationUuid = vc3d::corrections::writeGrowthAnnotation(*session, attachment, annotation, *volume);
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

        catalog.setTags(native, {"reviewed"});
        catalog.rename(native, "renamed");
        pump([&] {
            const auto* row = catalog.row(native);
            return row && row->record.name == "renamed" && row->record.tags == std::vector<std::string>{"reviewed"};
        }, "record edits reach the rows through commit events");
        require(std::count(committed.begin(), committed.end(), native) >= 1, "commits name the edited surface");
        catalog.retire({half});
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
            // Area: area_vx2 in the bound volume's voxels (vol2: the 2x2 unit grid),
            // area_cm2 physical from the open volume, where the shear doubles x.
            const auto areas=SurfaceAreaCalculator::calculateAreas(catalog,{other});
            require(areas.size()==1&&areas.front().success&&areas.front().areaVx2==4.0,
                    "area_vx2 is measured in the bound volume's voxels");
            require(std::abs(vc::surface::computeSurfaceAreaVox2(*catalog.open(other).surface)-8.0)<1e-9,
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

        // A volume opened at pyramid level 1 has its own id: registering the level
        // relates it to the volume it shows, both ways, once.
        Volume::ZarrCreateOptions coarse = create;
        coarse.shapeZYX = {2, 2, 2}; coarse.chunkShapeZYX = {2, 2, 2}; coarse.uuid = "vol1-level1";
        const auto level = Volume::New(fs::path(dir.path().toStdString()) / "level.zarr", coarse);
        const auto atLevel = add("at-level", vc3d::coordinateBindingForVolume(*level, 1));
        require(vc3d::opendata::registerVolumeLevel(*session, attachment, "vol1-level1", "vol1", 1), "the level is registered");
        require(!vc3d::opendata::registerVolumeLevel(*session, attachment, "vol1-level1", "vol1", 1), "an existing relation is kept");
        catalog.setSource(attachment, level);
        pump([&] { const auto* row = catalog.row(native); return !catalog.loading() && row && row->mapping; },
             "the level lists the surfaces of the volume it shows");
        require(*catalog.row(native)->mapping == VolumeAffine::scale(0.5), "seen at the level's scale");
        catalog.setSource(attachment, volume);
        pump([&] { const auto* row = catalog.row(atLevel); return !catalog.loading() && row && row->mapping; },
             "the full volume lists surfaces made at the level");
        require(*catalog.row(atLevel)->mapping == VolumeAffine::scale(2), "seen at the full volume's scale");

        catalog.setSource(std::nullopt, volume);
        require(catalog.rows().empty(), "no attachment lists nothing");
        session->detach(attachment);
        std::cout << "surface catalog tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "surface catalog test: " << error.what() << '\n';
        return 1;
    }
}
