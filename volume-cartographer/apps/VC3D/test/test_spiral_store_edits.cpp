// Spiral drawings saved into the object store (SpiralStoreEdits): point
// collections keep their identity, their points' ids and their metadata role;
// brush patches are created, replaced and retired as surface objects in the
// preview's declared frame, with leases on a hosted replica.
#include "SpiralStoreEdits.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/Hub.hpp>

#include <QJsonArray>
#include <QTemporaryDir>
#include <QTest>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
constexpr const char* kVolume = "fit-volume";

QJsonDocument fitDocument(const QString& objectUuid, bool absolute, const std::string& revision = {})
{
    // A fit point-collection document as the workspace builds it from the
    // store: stable ids as keys, reference-volume positions.
    QJsonObject points;
    for (const auto& [id, x, wind] : {std::tuple{10, 1.0, 0.0}, {20, 2.0, 1.0}, {30, 3.0, 2.0}})
        points[QString::number(id)] = QJsonObject{{QStringLiteral("p"), QJsonArray{x, 5.0, 7.0}},
                                                  {QStringLiteral("wind_a"), wind},
                                                  {QStringLiteral("creation_time"), 0}};
    QJsonObject metadata{{QStringLiteral("winding_is_absolute"), absolute}};
    if (!objectUuid.isEmpty()) metadata[QStringLiteral("object_uuid")] = objectUuid;
    if (!revision.empty())
        metadata[QLatin1String(vc3d::spiral::kLocalObjectRevisionKey)] = QString::fromStdString(revision);
    return QJsonDocument(QJsonObject{
        {QStringLiteral("vc_pointcollections_json_version"), QStringLiteral("1")},
        {QStringLiteral("collections"), QJsonObject{{QStringLiteral("1"), QJsonObject{
            {QStringLiteral("name"), QStringLiteral("walk")}, {QStringLiteral("points"), points},
            {QStringLiteral("metadata"), metadata}}}}}});
}

std::shared_ptr<QuadSurface> grid(float z, int rows = 3, int cols = 4)
{
    auto points = std::make_unique<cv::Mat_<cv::Vec3f>>(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) (*points)(r, c) = cv::Vec3f(float(c) * 2, float(r) * 2, z);
    auto surface = std::make_shared<QuadSurface>(points.release(), cv::Vec2f(0.5f, 0.5f));
    surface->meta["source"] = {{"spiral_patch_erode_cells", 0}};
    return surface;
}
}

class SpiralStoreEditsTest : public QObject
{
    Q_OBJECT

private slots:
    void savesPointCollectionsWithStableIds()
    {
        QTemporaryDir directory;
        const auto path = std::filesystem::path(directory.filePath(QStringLiteral("store.vcobj")).toStdString());
        OpenOptions options; options.author = "spiral-test";
        { auto created = ObjectStore::create(path, options); }
        ObjectStoreSession session;
        const auto attachment = session.attach(path, options);

        // A new collection gets an object, its points and their annotations.
        auto fresh = vc3d::spiral::importEditablePcls(fitDocument({}, true), 1.0, {}, true,
                                                     vc3d::spiral::PclRole::Relative);
        QCOMPARE(fresh.size(), std::size_t{1});
        const auto created = vc3d::spiral::savePointCollections(session, attachment, kVolume, fresh);
        QCOMPARE(created.size(), std::size_t{1});
        const auto uuid = created.front().uuid;
        QVERIFY(!created.front().revision.empty());
        auto read = [&](const std::string& object) {
            Json result;
            session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                const auto record = db.objectByUuid(txn, object);
                result["meta"] = Json::parse(record->meta);
                result["retired"] = record->retired;
                const auto head = db.currentGeometry(txn, object);
                const auto points = std::get<vc::geometry::PointCollection>(
                    vc::geometry::decode(db.readGeometryData(txn, head->uuid)));
                result["ids"] = points.ids;
                result["points"] = points.points;
                result["annotations"] = Json::parse(db.geometryPointAnnotations(txn, head->uuid, "points"));
            });
            return result;
        };
        auto stored = read(uuid);
        QVERIFY(stored["ids"] == Json({10, 20, 30}));
        QVERIFY(stored["meta"]["source"]["winding_is_absolute"] == Json(true));
        QVERIFY(stored["annotations"][1]["wind_a"] == Json(1.0));
        QVERIFY(stored["meta"]["coordinates"] == volumeCoordinates(kVolume, 1));
        // Saved again before an update brings it into the fit, it replaces
        // the same object: the draft names it and the revision saved.
        vc3d::spiral::setPointCollectionObject(fresh.front(), uuid, created.front().revision);
        fresh.front().appendPreviewPoint(cv::Vec3f(4.0f, 5.0f, 7.0f));
        const auto again = vc3d::spiral::savePointCollections(session, attachment, kVolume, fresh);
        QVERIFY(again.front().uuid == uuid && again.front().revision != created.front().revision);
        QVERIFY(read(uuid)["ids"] == Json({10, 20, 30, 31}));
        // A save whose revision could not be read back (the store detached
        // after its commit) is refused with the current one, not duplicated.
        vc3d::spiral::setPointCollectionObject(fresh.front(), uuid, {});
        try {
            vc3d::spiral::savePointCollections(session, attachment, kVolume, fresh);
            QFAIL("a save from an unknown revision was not refused");
        } catch (const vc3d::spiral::StoreObjectChanged& changed) {
            QVERIFY(changed.uuid == uuid && changed.current == again.front().revision);
        }
        std::size_t collections = 0;
        session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            collections = db.objectsByKind(txn, "point_collection").size();
        });
        QCOMPARE(collections, std::size_t{1});

        // The fit shows it with its object UUID and the revision it read: an
        // edit replaces that object, keeping the ids of kept points and the
        // rest of its source.
        auto head = [&] {
            std::string revision;
            session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                revision = db.currentObjectRevision(txn, uuid)->uuid;
            });
            return revision;
        };
        auto draft = [&] {
            return vc3d::spiral::importEditablePcls(fitDocument(QString::fromStdString(uuid), false, head()), 1.0,
                                                   {}, true, vc3d::spiral::PclRole::Relative);
        };
        auto review = [&](const std::string& note) {
            session.act(attachment, {{uuid, head()}}, [&](auto& db, auto& txn) {
                auto meta = Json::parse(db.objectByUuid(txn, uuid)->meta);
                meta["source"]["reviewer"] = note;
                db.setMeta(txn, uuid, "source", meta["source"].dump());
            });
        };
        review("kept");
        auto edited = draft();
        QVERIFY(edited.front().erase(1));
        edited.front().appendPreviewPoint(cv::Vec3f(9.0f, 5.0f, 7.0f));
        // Another editor's change after the draft was read refuses the save
        // instead of being overwritten, deletion included.
        const auto before = head();
        review("changed meanwhile");
        auto refused = [&](const std::vector<vc3d::spiral::EditablePclDraft>& drafts) {
            try {
                vc3d::spiral::savePointCollections(session, attachment, kVolume, drafts);
            } catch (const std::runtime_error& error) {
                return std::string(error.what()).find("after this edit began") != std::string::npos;
            }
            return false;
        };
        QVERIFY(refused(edited));
        auto stale = edited;
        stale.front().setDeleted(true);
        QVERIFY(refused(stale));
        QVERIFY(read(uuid)["ids"] == Json({10, 20, 30, 31}) && read(uuid)["retired"] == Json(false));
        QVERIFY(head() != before);
        review("kept");
        edited = draft();
        QVERIFY(edited.front().erase(1));
        edited.front().appendPreviewPoint(cv::Vec3f(9.0f, 5.0f, 7.0f));
        QVERIFY(vc3d::spiral::savePointCollections(session, attachment, kVolume, edited).front().uuid == uuid);
        stored = read(uuid);
        QVERIFY(stored["ids"] == Json({10, 30, 31}));
        QVERIFY(stored["points"][2] == Json({9.0, 5.0, 7.0}));
        QVERIFY(stored["meta"]["source"]["reviewer"] == Json("kept"));
        QVERIFY(stored["meta"]["source"]["winding_is_absolute"] == Json(false));
        QVERIFY(stored["annotations"][2]["wind_a"] == Json(3.0));

        // Deleting it (as read now) retires the object.
        edited = draft();
        edited.front().setDeleted(true);
        vc3d::spiral::savePointCollections(session, attachment, kVolume, edited);
        QVERIFY(read(uuid)["retired"] == Json(true));
    }

    void savesBrushPatchesOnAHostedReplica()
    {
        QTemporaryDir directory;
        const auto hub = std::filesystem::path(directory.filePath(QStringLiteral("hub.vcobj")).toStdString());
        const auto replica = std::filesystem::path(directory.filePath(QStringLiteral("replica.vcobj")).toStdString());
        OpenOptions options; options.author = "spiral-test";
        { auto created = ObjectStore::create(hub, options); }
        initHub(hub, "curator");
        cloneHub(hub, replica, options);
        ObjectStoreSession session;
        const auto attachment = session.attach(replica, options);

        const auto created = vc3d::spiral::saveBrushPatches(
            session, attachment, kVolume, 2.0, {{{}, {}, "brush_1", grid(5.0f), false}}).front();
        const auto uuid = created.uuid;
        Json first;
        session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            const auto record = db.objectByUuid(txn, uuid);
            first = {{"meta", Json::parse(record->meta)}, {"tags", record->tags}, {"name", record->name},
                     {"geometry", db.currentGeometry(txn, uuid)->uuid}};
        });
        QVERIFY(first["name"] == Json("brush_1"));
        QVERIFY(first["tags"] == Json({vc3d::spiral::kBrushPatchTag}));
        QVERIFY(first["meta"]["coordinates"] == volumeCoordinates(kVolume, 2.0));
        QVERIFY(first["meta"]["source"]["spiral_patch_erode_cells"] == Json(0));
        QVERIFY(session.syncHubAndWait(attachment, std::chrono::seconds(30))->pending.empty());

        // Pushed, the patch lives on the hub, so saves take its lease. Another
        // editor saves it meanwhile: replacing or retiring from the revision
        // this edit started from is refused, and theirs is kept.
        const auto other = std::filesystem::path(directory.filePath(QStringLiteral("other.vcobj")).toStdString());
        OpenOptions otherOptions; otherOptions.author = "other";
        cloneHub(hub, other, otherOptions);
        {
            ObjectStoreSession theirs;
            const auto mine = theirs.attach(other, otherOptions);
            std::string head;
            theirs.read(mine, [&](const auto& db, const auto& txn, const auto&) {
                head = db.currentObjectRevision(txn, uuid)->uuid;
            });
            vc3d::spiral::saveBrushPatches(theirs, mine, kVolume, 2.0, {{uuid, head, "brush_1", grid(11.0f), false}});
            QVERIFY(theirs.syncHubAndWait(mine, std::chrono::seconds(30))->pending.empty());
            theirs.detach(mine);
        }
        std::string theirs;
        for (const bool remove : {false, true}) {
            try {
                vc3d::spiral::saveBrushPatches(session, attachment, kVolume, 2.0,
                                               {{uuid, created.revision, "brush_1", grid(13.0f), remove}});
                QFAIL("a save over another editor's was not refused");
            } catch (const vc3d::spiral::StoreObjectChanged& changed) {
                QVERIFY(changed.uuid == uuid && changed.current != created.revision);
                theirs = changed.current;
            }
        }
        session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            QVERIFY(!db.objectByUuid(txn, uuid)->retired);
            QVERIFY(db.currentGeometry(txn, uuid)->uuid != first["geometry"].get<std::string>());  // theirs
        });
        // From the revision the refusal named (theirs), saving again replaces it.
        const auto mine = vc3d::spiral::saveBrushPatches(session, attachment, kVolume, 2.0,
                                                         {{uuid, theirs, "brush_1", grid(13.0f), false}}).front();
        session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            QVERIFY(db.currentObjectRevision(txn, uuid)->uuid == mine.revision);
        });
        vc3d::spiral::saveBrushPatches(session, attachment, kVolume, 2.0, {{uuid, mine.revision, "brush_1", nullptr, true}});
        session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            QVERIFY(db.objectByUuid(txn, uuid)->retired);
        });
        session.detach(attachment);
    }
};

QTEST_MAIN(SpiralStoreEditsTest)

#include "test_spiral_store_edits.moc"
