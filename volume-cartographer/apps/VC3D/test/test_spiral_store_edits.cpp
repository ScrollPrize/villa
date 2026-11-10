// Spiral drawings saved into the object store (SpiralStoreEdits): point
// collections keep their identity, their points' ids and their metadata role;
// brush patches are created, replaced and retired as surface objects in the
// preview's declared frame; a save over another editor's is refused.
#include "SpiralStoreEdits.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/TransactionJson.hpp>
#include <vc_object_store/objects/ObjectCollection.hpp>

#include <QJsonArray>
#include <QTemporaryDir>
#include <QTest>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
constexpr const char* kVolume = "fit-volume";

QJsonDocument fitDocument(const QString& objectUuid, bool absolute, const ObjectVersion& reference = {})
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
    if (reference.version != 0)
        metadata[QLatin1String(vc3d::spiral::kLocalObjectReferenceKey)] = QJsonDocument::fromJson(
            QByteArray::fromStdString(objectVersionsRequestToJson({reference}))).object()["versions"].toArray().first();
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

// A read of the attached store's current state.
template <typename Fn> void readStore(ObjectStoreSession& session, const AttachmentIdentity& attachment, Fn&& fn)
{
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) { fn(reader); });
}
ObjectVersion headVersion(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& uuid)
{
    ObjectVersion version;
    readStore(session, attachment, [&](const StoreReader& reader) { version = reader.currentObject(uuid)->reference(); });
    return version;
}
}

class SpiralStoreEditsTest : public QObject
{
    Q_OBJECT

private slots:
    void inheritedBrushEditsGuardOwnerAndWorkWithoutOverlayHistory()
    {
        QTemporaryDir directory;
        const auto base = std::filesystem::path(directory.filePath("base.vcobj").toStdString());
        OpenOptions options; options.author = "spiral-test";
        { auto db = ObjectStore::create(base, options); }
        ObjectStoreSession session;
        const auto attachment = session.attach(base, options);
        const auto created = vc3d::spiral::saveBrushPatches(session, attachment, kVolume, 1,
            {{{}, {}, "brush", grid(5), false}}).front();
        session.detach(attachment);
        { auto db = ObjectStore::open(base, options); auto txn = db.write(); db.seal(txn); txn.commit(); }
        for (const auto mode : {HistoryMode::On, HistoryMode::Off}) {
            const auto path = std::filesystem::path(directory.filePath(mode == HistoryMode::On ? "on.vcobj" : "off.vcobj").toStdString());
            options.history = mode;
            { auto collection = ObjectCollection::create(path, {base}, options); }
            const auto overlay = session.attach(path, options);
            auto forged = created.reference;
            session.readStore(overlay, [&](const StoreReader& reader, const auto&) { forged.storeUuid = reader.status().storeUuid; });
            QVERIFY_EXCEPTION_THROWN(vc3d::spiral::saveBrushPatches(session, overlay, kVolume, 1,
                {{created.uuid, forged, "brush", grid(7), false}}), vc3d::spiral::StoreObjectChanged);
            const auto saved = vc3d::spiral::saveBrushPatches(session, overlay, kVolume, 1,
                {{created.uuid, created.reference, "brush", grid(7), false}}).front();
            QVERIFY(saved.reference.storeUuid != created.reference.storeUuid);
            QVERIFY(headVersion(session, overlay, created.uuid) == saved.reference);
            session.detach(overlay);
        }
    }
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
        QVERIFY(created.front().reference.version > 0 && !created.front().requestUuid.empty());
        auto read = [&](const std::string& object) {
            Json result;
            readStore(session, attachment, [&](const StoreReader& reader) {
                const auto revision = reader.currentObject(object);
                const auto& record = revision->record;
                result["meta"] = Json::parse(record.meta);
                result["retired"] = record.retired;
                const auto head = reader.currentGeometry(object);
                const auto points = std::get<vc::geometry::PointCollection>(
                    vc::geometry::decode(reader.readGeometryData(head->uuid)));
                result["ids"] = points.ids;
                result["points"] = points.points;
                result["annotations"] = reader.geometryPointAnnotations(head->uuid, "points");
            });
            return result;
        };
        auto stored = read(uuid);
        QVERIFY(stored["ids"] == Json({10, 20, 30}));
        QVERIFY(stored["meta"]["source"]["winding_is_absolute"] == Json(true));
        QVERIFY(stored["annotations"][1]["wind_a"] == Json(1.0));
        QVERIFY(stored["meta"]["coordinates"] == volumeCoordinates(kVolume, 1));
        // Saved again before an update brings it into the fit, it replaces
        // the same object: the draft names it and the version saved.
        vc3d::spiral::setPointCollectionObject(fresh.front(), created.front().reference);
        fresh.front().appendPreviewPoint(cv::Vec3f(4.0f, 5.0f, 7.0f));
        const auto again = vc3d::spiral::savePointCollections(session, attachment, kVolume, fresh);
        QVERIFY(again.front().uuid == uuid && again.front().reference.version > created.front().reference.version);
        QVERIFY(read(uuid)["ids"] == Json({10, 20, 30, 31}));
        // A save whose version could not be read back (the store detached
        // after its commit) is refused with the current one, not duplicated.
        fresh.front().sourceCollection["metadata"] = QJsonObject{{"object_uuid", QString::fromStdString(uuid)}};
        try {
            vc3d::spiral::savePointCollections(session, attachment, kVolume, fresh);
            QFAIL("a save from an unknown version was not refused");
        } catch (const vc3d::spiral::StoreObjectChanged& changed) {
            QVERIFY(changed.uuid == uuid && changed.current == again.front().reference);
        }
        std::size_t collections = 0;
        readStore(session, attachment, [&](const StoreReader& reader) {
            ObjectSelection selection; selection.kind = "point_collection";
            collections = reader.selectObjects(selection).size();
        });
        QCOMPARE(collections, std::size_t{1});

        // The fit shows it with its object UUID and the version it read: an
        // edit replaces that object, keeping the ids of kept points and the
        // rest of its source.
        auto head = [&] { return headVersion(session, attachment, uuid); };
        auto draft = [&] {
            return vc3d::spiral::importEditablePcls(fitDocument(QString::fromStdString(uuid), false, head()), 1.0,
                                                   {}, true, vc3d::spiral::PclRole::Relative);
        };
        auto review = [&](const std::string& note) {
            session.editRecords(attachment, {uuid}, [&](ObjectRecord& record) {
                auto meta = Json::parse(record.meta);
                meta["source"]["reviewer"] = note;
                record.meta = meta.dump();
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
        // ID 31 existed before this draft: dropping it must not make it reusable.
        QVERIFY(stored["ids"] == Json({10, 30, 32}));
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

    void savesBrushPatchesBesideAnotherEditor()
    {
        QTemporaryDir directory;
        const auto path = std::filesystem::path(directory.filePath(QStringLiteral("shared.vcobj")).toStdString());
        OpenOptions options; options.author = "spiral-test";
        { auto created = ObjectStore::create(path, options); }
        ObjectStoreSession session;
        const auto attachment = session.attach(path, options);

        const auto created = vc3d::spiral::saveBrushPatches(
            session, attachment, kVolume, 2.0, {{{}, {}, "brush_1", grid(5.0f), false}}).front();
        const auto uuid = created.uuid;
        QVERIFY(created.reference.version > 0 && !created.requestUuid.empty());
        Json first;
        readStore(session, attachment, [&](const StoreReader& reader) {
            const auto revision = reader.currentObject(uuid);
            const auto& record = revision->record;
            first = {{"meta", Json::parse(record.meta)}, {"tags", record.tags}, {"name", record.name},
                     {"geometry", reader.currentGeometry(uuid)->uuid}};
        });
        QVERIFY(first["name"] == Json("brush_1"));
        QVERIFY(first["tags"] == Json({vc3d::spiral::kBrushPatchTag}));
        QVERIFY(first["meta"]["coordinates"] == volumeCoordinates(kVolume, 2.0));
        QVERIFY(first["meta"]["source"]["spiral_patch_erode_cells"] == Json(0));

        // Another editor of the same store file saves it meanwhile: replacing
        // or retiring from the version this edit started from is refused, and
        // theirs is kept.
        {
            ObjectStoreSession theirs;
            OpenOptions otherOptions; otherOptions.author = "other";
            const auto mine = theirs.attach(path, otherOptions);
            const auto head = headVersion(theirs, mine, uuid);
            vc3d::spiral::saveBrushPatches(theirs, mine, kVolume, 2.0, {{uuid, head, "brush_1", grid(11.0f), false}});
            theirs.detach(mine);
        }
        ObjectVersion theirs;
        for (const bool remove : {false, true}) {
            try {
                vc3d::spiral::saveBrushPatches(session, attachment, kVolume, 2.0,
                                               {{uuid, created.reference, "brush_1", grid(13.0f), remove}});
                QFAIL("a save over another editor's was not refused");
            } catch (const vc3d::spiral::StoreObjectChanged& changed) {
                QVERIFY(changed.uuid == uuid && changed.current != created.reference);
                theirs = changed.current;
            }
        }
        readStore(session, attachment, [&](const StoreReader& reader) {
            QVERIFY(!reader.currentObject(uuid)->record.retired);
            QVERIFY(reader.currentGeometry(uuid)->uuid != first["geometry"].get<std::string>());  // theirs
        });
        // From the version the refusal named (theirs), saving again replaces it.
        const auto mine = vc3d::spiral::saveBrushPatches(session, attachment, kVolume, 2.0,
                                                         {{uuid, theirs, "brush_1", grid(13.0f), false}}).front();
        QVERIFY(headVersion(session, attachment, uuid) == mine.reference);
        vc3d::spiral::saveBrushPatches(session, attachment, kVolume, 2.0, {{uuid, mine.reference, "brush_1", nullptr, true}});
        readStore(session, attachment, [&](const StoreReader& reader) {
            QVERIFY(reader.currentObject(uuid)->record.retired);
        });
        session.detach(attachment);
    }
};

QTEST_MAIN(SpiralStoreEditsTest)

#include "test_spiral_store_edits.moc"
