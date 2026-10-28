// The Spiral workspace's pinned reads of a fit's object-store inputs: pins
// resolve by geometry revision, coordinates convert to reference-volume
// voxels, roles come from the fitted metadata, consumed relationships are
// compared with the store's, and the hashes match the Python adapter's.
//
// With SPIRAL_STORE_INPUTS_DUMP=STORE|MANIFEST|OUT it instead reads MANIFEST's
// pins from STORE and writes OUT in display_geometry_dump.py's format, for
//   display_geometry_dump.py compare <baseline> OUT
#include "SpiralStoreInputs.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QTemporaryDir>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
using namespace vc::objects;
using Json = nlohmann::json;
using vc3d::spiral::PinState;

namespace {
void require(bool value, const std::string& message)
{
    if (!value) throw std::runtime_error(message);
}

template <typename T>
GeometryArray array(const std::string& name, const std::string& dtype,
                    std::vector<int64_t> shape, const std::vector<T>& values)
{
    GeometryArray result{name, dtype, std::move(shape), {}};
    result.bytes.resize(values.size() * sizeof(T));
    std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
    return result;
}

const std::string kVolume = "20260411134726";

// Expected hashes from object_store_inputs.adapter_metadata/content_sha256.
void metadataHashesMatchThePythonAdapter()
{
    struct Case { const char* kind; const char* meta; std::optional<FiberDetail> fiber; const char* sha; };
    const FiberDetail h{"H", false};
    const FiberDetail manual{"", true};
    const Case cases[] = {
        {"surface", R"({"coordinates":{"volume_uuid":"s1-reference","scale_factor":1.0},"source":{"spiral_patch_erode_cells":2,"other":"x"}})",
         std::nullopt, "3d881c171e61199c7fd1001b13de093557c403396d761d2ef00c1f779b1bb976"},
        {"surface", "{}", std::nullopt, "ccc5d42f1d2a63966abbcf7f1f9e73d444233a3ffc0262f875d5cff3cfcd9af1"},
        {"point_collection", R"({"coordinates":{"volume_uuid":"20260411134726","scale_factor":4},"sampling_group":"reviewed","source":{"winding_is_absolute":true}})",
         std::nullopt, "bbfec3e74559083a07f321f90ce5ddc50ebe055a6b1dd5df5ddb2c2bbf3361c7"},
        {"point_collection", "{\"coordinates\":{\"volume_uuid\":\"v\",\"scale_factor\":0.25},\"source\":{\"note\":\"\xc3\xbcn\xc3\xaf" "code\"}}",
         std::nullopt, "63657d11b3a7df71fb4ac710bf0f409c96b25eb28108aeff342794b995fb7654"},
        {"fiber", "{\"coordinates\":{\"volume_uuid\":\"s1-reference\",\"scale_factor\":4.0},\"source\":{\"hv_classification\":{\"family\":\"H\",\"certainty\":0.873,\"votes\":[1,2],\"label\":\"\xc3\xa9\"}}}",
         h, "0f981bea61ce3539b69db570493eb762e0a0e3842486120a5491ace58761896c"},
        {"fiber", R"({"coordinates":{"volume_uuid":"s1-reference","scale_factor":1e-05}})",
         manual, "77bc6f6477a68e798c3c9200faf00722658b7a9d78a0f23b4dc769588bf283b3"},
    };
    for (const auto& item : cases)
        require(vc3d::spiral::adapterMetadataSha256(item.kind, Json::parse(item.meta), item.fiber) == item.sha,
                std::string("metadata hash differs from the Python adapter for ") + item.kind + " " + item.meta);
}

// Expected text from json.dumps(sort_keys=True, separators=(",", ":")).
void canonicalJsonMatchesPython()
{
    const Json value = {
        {"f", {0.1, 1e-05, 0.0001, 1e16, 1e15, 123456.789, -0.0, 5e-324, 1.7976931348623157e308, 2.5,
               100.0, 1.0 / 3, 1e22, 0.30000000000000004, -1.5e-07, 12345678901234567.0}},
        {"s", "\xc3\xbcn\xc3\xaf\n\""},
        {"i", {1, -2, int64_t(1000000000000000000)}},
        {"n", nullptr},
        {"b", {true, false}},
        {"k", {{"b", 1}, {"a", Json::object()}}}};
    const std::string expected =
        R"({"b":[true,false],"f":[0.1,1e-05,0.0001,1e+16,1000000000000000.0,123456.789,-0.0,5e-324,)"
        R"(1.7976931348623157e+308,2.5,100.0,0.3333333333333333,1e+22,0.30000000000000004,-1.5e-07,)"
        R"(1.2345678901234568e+16],"i":[1,-2,1000000000000000000],"k":{"a":{},"b":1},"n":null,)"
        R"("s":"\u00fcn\u00ef\n\""})";
    const auto text = vc3d::spiral::pythonCanonicalJson(value);
    require(text == expected, "canonical JSON differs from Python's: " + text);
}

// Expected hashes from object_store_inputs.content_sha256 of
// {"a_anchor", "b_anchor", "meta"} with the binding's anchor dictionaries.
void relationshipHashesMatchThePythonAdapter()
{
    Anchor surface;
    surface.points = {{"points", {0, 1}}};
    surface.position = std::array<double, 3>{10.5, 20.25, 30.1};
    Anchor point;
    point.points = {{"point_ids", {4}}};
    require(vc3d::spiral::relationshipContentSha256(surface, point, {{"distance", 0.125}, {"source", "linker"}})
                == "b4bd93221c171d46fd7210e76e1195362b9a97d396edd6e2ab3b27a01abe006e",
            "point relationship hash differs from the Python adapter");
    Anchor whole;
    whole.wholeGeometry = true;
    whole.channels = {"mask"};
    whole.metadataKeys = {"coordinates"};
    whole.direction = std::array<double, 3>{0.0, -1e-05, 1.0};
    require(vc3d::spiral::relationshipContentSha256(whole, Anchor{}, Json::object())
                == "621acf914b0b769564a430b6f9e9aa84b9e2c9d70fe47d87246e9ab0c3df3cd8",
            "whole-geometry relationship hash differs from the Python adapter");
}

struct Fixture {
    std::shared_ptr<ObjectStoreSession> session = std::make_shared<ObjectStoreSession>();
    AttachmentIdentity attachment;
    std::string surface, pcl, fiber, link;

    explicit Fixture(const fs::path& path)
    {
        OpenOptions options; options.author = "spiral-store-inputs-test";
        { auto created = ObjectStore::create(path, options); }
        attachment = session->attach(path, options);
        surface = util::newUuid();
        pcl = util::newUuid();
        fiber = util::newUuid();
        session->act(attachment, {{surface, std::nullopt}, {pcl, std::nullopt}, {fiber, std::nullopt}},
                     [&](auto& db, auto& txn) {
            // A 2x3 patch at scale 1 whose last vertex is invalid.
            ObjectRecord record; record.uuid = surface; record.kind = "surface"; record.name = "near";
            record.meta = Json{{"coordinates", volumeCoordinates(kVolume, 1)},
                               {"source", {{"spiral_patch_erode_cells", 1}}}}.dump();
            db.upsertObject(txn, record);
            std::vector<float> points;
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 3; ++col) {
                    const bool invalid = row == 1 && col == 2;
                    points.insert(points.end(), {invalid ? -1.f : 10.f + col, invalid ? -1.f : 20.f + row,
                                                 invalid ? -1.f : 30.f});
                }
            GeometryData surfaceData{"surface", 1, "source_xyz", R"({"scale":[1.0,1.0]})",
                                     {array("points", "<f4", {2, 3, 3}, points)}};
            db.putGeometry(txn, surface, surfaceData);
            // A collection at scale 2 with noncontiguous IDs and one winding.
            ObjectRecord collection; collection.uuid = pcl; collection.kind = "point_collection";
            collection.name = "walk";
            collection.meta = Json{{"coordinates", volumeCoordinates(kVolume, 2)},
                                   {"source", {{"winding_is_absolute", true},
                                               {"point_attributes", Json::array({Json{{"wind_a", 3.0}},
                                                                                 Json{{"wind_a", nullptr}}})}}}}.dump();
            db.upsertObject(txn, collection);
            GeometryData pclData{"point_collection", 1, "source_xyz", "{}",
                                 {array("points", "<f8", {2, 3}, std::vector<double>{40, 42, 44, 2, 4, 6}),
                                  array("point_ids", "<i8", {2}, std::vector<int64_t>{9, 4})}};
            db.putGeometry(txn, pcl, pclData);
            // A fiber at scale 4.
            ObjectRecord line; line.uuid = fiber; line.kind = "fiber"; line.name = "fiber";
            line.meta = Json{{"coordinates", volumeCoordinates(kVolume, 4)}}.dump();
            line.fiber = FiberDetail{"H", true};
            db.upsertObject(txn, line);
            GeometryData fiberData{"fiber", 1, "source_xyz", "{}",
                                   {array("control_points", "<f8", {2, 3}, std::vector<double>{4, 8, 12, 12, 16, 20}),
                                    array("line_points", "<f8", {3, 3}, std::vector<double>{4, 8, 12, 8, 12, 16, 12, 16, 20}),
                                    array("control_point_ids", "<i8", {2}, std::vector<int64_t>{0, 1})}};
            db.putGeometry(txn, fiber, fiberData);
        });
        // The collection's point 4 lies on the patch.
        session->act(attachment, pins({surface, pcl}), [&](auto& db, auto& txn) {
            EdgeRecord edge; edge.kind = "point-surface"; edge.aUuid = surface; edge.bUuid = pcl;
            edge.aAnchor.points = {{"points", {0, 1}}};
            edge.aAnchor.position = std::array<double, 3>{11, 20, 30};
            edge.bAnchor.points = {{"point_ids", {4}}};
            edge.meta = R"({"distance":0.125})";
            link = db.addEdge(txn, edge);
        });
    }

    // The input manifest a fit of every object would record.
    Json manifest()
    {
        Json objects = Json::array();
        std::string storeUuid;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            storeUuid = db.uuid(txn);
            for (const auto& [uuid, kind, scale] : {std::tuple{surface, "surface", 1.0},
                                                    std::tuple{pcl, "point_collection", 2.0},
                                                    std::tuple{fiber, "fiber", 4.0}}) {
                const auto current = db.currentObjectRevision(txn, uuid);
                objects.push_back({{"uuid", uuid}, {"kind", kind}, {"name", current->record.name},
                                   {"object_revision", "another-replica"},
                                   {"geometry_revision", *current->geometryRevision},
                                   {"metadata_sha256", vc3d::spiral::adapterMetadataSha256(
                                        kind, Json::parse(current->record.meta), current->record.fiber)},
                                   {"scale_factor", scale}});
            }
        });
        // The link as the adapter records it: surface end first.
        Json relationships = Json::array();
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            for (const auto& selected : db.relationshipsAt(txn, {{surface, std::nullopt}, {pcl, std::nullopt}})) {
                const auto& version = selected.relationship;
                const bool stored = version.record.aUuid == surface;
                const auto& a = stored ? version.record.aAnchor : version.record.bAnchor;
                const auto& b = stored ? version.record.bAnchor : version.record.aAnchor;
                relationships.push_back(
                    {{"uuid", version.record.uuid}, {"revision", "another-replica"}, {"kind", version.record.kind},
                     {"family", "pcl_patch"}, {"approval_state", version.record.approvalState},
                     {"generation", version.record.generation}, {"a", surface},
                     {"a_geometry_revision", *(stored ? version.aGeometryRevision : version.bGeometryRevision)},
                     {"b", pcl},
                     {"b_geometry_revision", *(stored ? version.bGeometryRevision : version.aGeometryRevision)},
                     {"content_sha256", vc3d::spiral::relationshipContentSha256(a, b, Json::parse(version.record.meta))}});
            }
        });
        return {{"digest", "fixture"},
                {"manifest", {{"store", {{"uuid", storeUuid}, {"path", "/service/host/path"}}},
                              {"volume_uuid", kVolume}, {"objects", objects}, {"relationships", relationships}}}};
    }

    // Checked writes expect the current revisions of what they change.
    RevisionSelection pins(std::initializer_list<std::string> uuids)
    {
        RevisionSelection selection;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            for (const auto& uuid : uuids) selection[uuid] = db.currentObjectRevision(txn, uuid)->uuid;
        });
        return selection;
    }

    vc3d::spiral::PinnedStoreInputs read(const Json& document)
    {
        vc3d::spiral::PinnedStoreInputs result;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            result = vc3d::spiral::readPinnedStoreInputs(db, txn, document);
        });
        return result;
    }
};

void pinnedReadsConvertToReferenceVoxels(const fs::path& directory)
{
    Fixture fixture(directory / "inputs.vcobj");
    const auto document = fixture.manifest();
    auto inputs = fixture.read(document);
    require(inputs.resolved(), "fixture pins resolve");
    require(inputs.count(PinState::Current) == 3, "every pin is current");

    const auto& surface = *inputs.surfaces.at(0).surface;
    const auto& points = *surface.rawPointsPtr();
    require(points.rows == 2 && points.cols == 3, "surface grid shape");
    require(points(0, 1) == cv::Vec3f(11, 20, 30), "scale-1 surface vertices are stored XYZ");
    require(points(1, 2) == cv::Vec3f(-1, -1, -1), "invalid vertices keep the sentinel");

    const auto& pcl = inputs.pointCollections.at(0);
    require(pcl.absolute, "absolute role from metadata");
    require((pcl.ids == std::vector<int64_t>{4, 9}), "points ordered by stable ID");
    require(pcl.points[0] == cv::Vec3d(1, 2, 3) && pcl.points[1] == cv::Vec3d(20, 21, 22),
            "scale-2 points divided by their scale");
    require(!pcl.windings[0] && pcl.windings[1] == 3.0, "windings follow their point IDs");

    const auto& fiber = inputs.fibers.at(0);
    require(fiber.line.size() == 3 && fiber.line[1] == cv::Vec3d(2, 3, 4), "scale-4 fiber line");
    require(fiber.controls.back() == cv::Vec3d(3, 4, 5), "scale-4 fiber controls");
    require(fiber.family == "H", "fitted fiber family");

    using vc3d::spiral::LinkState;
    require(inputs.relationships.size() == 1 && inputs.relationships[0].state == LinkState::Current
                && inputs.addedRelationships == 0,
            "the fitted link is current");

    // A metadata edit the adapter reads, then a geometry edit: the pinned
    // geometry is still what is read, and its state says what changed.
    fixture.session->act(fixture.attachment, fixture.pins({fixture.pcl, fixture.surface}),
                         [&](auto& db, auto& txn) {
        db.setMeta(txn, fixture.pcl, "sampling_group", "\"reviewed\"");
        db.setMeta(txn, fixture.surface, "review_note", "\"unrelated\"");
    });
    inputs = fixture.read(document);
    require(inputs.pointCollections.at(0).state == PinState::MetadataChanged, "metadata change detected");
    require(inputs.surfaces.at(0).state == PinState::Current, "unread metadata is no change");
    // The role is the fitted one, not the current head's.
    fixture.session->act(fixture.attachment, fixture.pins({fixture.pcl}), [&](auto& db, auto& txn) {
        db.setMeta(txn, fixture.pcl, "source", R"({"winding_is_absolute":false})");
    });
    inputs = fixture.read(document);
    require(inputs.pointCollections.at(0).fittedMetadata && inputs.pointCollections.at(0).absolute,
            "the role comes from the fitted metadata");
    auto unknown = document;
    unknown["manifest"]["objects"][1]["metadata_sha256"] = std::string(64, '0');
    inputs = fixture.read(unknown);
    require(!inputs.pointCollections.at(0).fittedMetadata && inputs.pointCollections.at(0).points.empty(),
            "a collection whose fitted metadata is not held is withheld");
    require(inputs.problems.size() == 1, "the withheld collection is reported");

    fixture.session->act(fixture.attachment, fixture.pins({fixture.fiber}), [&](auto& db, auto& txn) {
        const auto current = db.currentGeometry(txn, fixture.fiber);
        const double moved[3] = {0, 0, 0};
        GeometryArrayEdit edit{"line_points", 3, {}};
        edit.bytes.resize(sizeof(moved));
        std::memcpy(edit.bytes.data(), moved, sizeof(moved));
        db.editGeometry(txn, fixture.fiber, {edit}, current->uuid,
                        [](const std::vector<EdgeRecord>&) { return std::vector<GeometryEdgeEdit>{}; });
    });
    inputs = fixture.read(document);
    require(inputs.fibers.at(0).state == PinState::Edited, "geometry edit detected");
    require(inputs.fibers.at(0).line.size() == 3, "the pinned geometry is still read");

    // Links: an edit changes the fitted one, a new one among the fitted
    // inputs is reported, and a rejected one is no longer consumed.
    fixture.session->act(fixture.attachment, fixture.pins({fixture.surface, fixture.pcl}),
                         [&](auto& db, auto& txn) {
        db.setMeta(txn, fixture.link, "distance", "0.25");
    });
    inputs = fixture.read(document);
    require(inputs.relationships.at(0).state == LinkState::Changed, "link edit detected");
    fixture.session->act(fixture.attachment, fixture.pins({fixture.surface, fixture.fiber}),
                         [&](auto& db, auto& txn) {
        EdgeRecord edge; edge.kind = "same-surface"; edge.aUuid = fixture.surface; edge.bUuid = fixture.fiber;
        edge.aAnchor.wholeGeometry = true; edge.bAnchor.wholeGeometry = true;
        db.addEdge(txn, edge);
    });
    inputs = fixture.read(document);
    require(inputs.addedRelationships == 1, "a new consumable link is reported");
    fixture.session->act(fixture.attachment, fixture.pins({fixture.surface, fixture.pcl}),
                         [&](auto& db, auto& txn) { db.setApproval(txn, fixture.link, "rejected"); });
    inputs = fixture.read(document);
    require(inputs.relationships.at(0).state == LinkState::Dropped, "a rejected link is dropped");

    // A pin this store lacks is missing, never replaced by a current head.
    auto missing = document;
    missing["manifest"]["objects"][0]["geometry_revision"] = util::newUuid();
    inputs = fixture.read(missing);
    require(inputs.surfaces.at(0).state == PinState::Missing && !inputs.surfaces.at(0).surface,
            "missing pin has no geometry");
    require(!inputs.resolved() && inputs.problems.size() == 1, "missing pin reported");

    auto other = document;
    other["manifest"]["store"]["uuid"] = util::newUuid();
    inputs = fixture.read(other);
    require(inputs.surfaces.empty() && inputs.problems.size() == 1, "another store is refused");
}

// display_geometry_dump.py's entry for each pinned input.
std::string sha256(const std::vector<float>& values)
{
    return QCryptographicHash::hash(
        QByteArray(reinterpret_cast<const char*>(values.data()), qsizetype(values.size() * sizeof(float))),
        QCryptographicHash::Sha256).toHex().toStdString();
}

Json summary(const std::vector<cv::Vec3f>& xyz)
{
    std::vector<float> flat;
    Json first = Json::array();
    std::optional<cv::Vec3f> low, high;
    std::size_t invalid = 0;
    for (const auto& point : xyz) {
        flat.insert(flat.end(), {point[0], point[1], point[2]});
        if (point == cv::Vec3f(-1, -1, -1)) { ++invalid; continue; }
        if (first.size() < 2) first.push_back({point[0], point[1], point[2]});
        if (!low) { low = point; high = point; }
        for (int axis = 0; axis < 3; ++axis) {
            (*low)[axis] = std::min((*low)[axis], point[axis]);
            (*high)[axis] = std::max((*high)[axis], point[axis]);
        }
    }
    return {{"count", xyz.size()}, {"invalid", invalid}, {"first", first},
            {"min", low ? Json{(*low)[0], (*low)[1], (*low)[2]} : Json(nullptr)},
            {"max", high ? Json{(*high)[0], (*high)[1], (*high)[2]} : Json(nullptr)},
            {"sha256", sha256(flat)}};
}

std::vector<cv::Vec3f> asFloat(const std::vector<cv::Vec3d>& points)
{
    std::vector<cv::Vec3f> result;
    for (const auto& point : points) result.emplace_back(float(point[0]), float(point[1]), float(point[2]));
    return result;
}

int dump(const std::string& spec)
{
    const auto first = spec.find('|'), second = spec.find('|', first + 1);
    const fs::path store = spec.substr(0, first);
    const fs::path manifestPath = spec.substr(first + 1, second - first - 1);
    const fs::path out = spec.substr(second + 1);
    std::ifstream manifestFile(manifestPath);
    const Json document = Json::parse(manifestFile);
    auto session = std::make_shared<ObjectStoreSession>();
    OpenOptions options; options.readOnly = true;
    const auto attachment = session->attach(store, options);
    vc3d::spiral::PinnedStoreInputs inputs;
    std::map<std::string, std::string> derived;
    session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        inputs = vc3d::spiral::readPinnedStoreInputs(db, txn, document);
        for (const auto& pin : vc3d::spiral::manifestPins(document)) {
            const auto record = db.objectByUuid(txn, pin.uuid);
            const auto meta = Json::parse(record->meta);
            auto name = meta.value("source", Json::object()).value("uuid_derivation", Json::object())
                            .value("name", pin.kind + ":" + pin.name);
            if (name.rfind("patch:", 0) == 0 && name.size() > 7 && name.ends_with(".tifxyz"))
                name.resize(name.size() - 7);
            derived[pin.uuid] = name;
        }
    });
    Json objects = Json::object();
    for (const auto& entry : inputs.surfaces) {
        if (!entry.surface) continue;
        const auto& points = *entry.surface->rawPointsPtr();
        std::vector<cv::Vec3f> xyz(points.begin(), points.end());
        auto item = summary(xyz);
        item["kind"] = "surface";
        item["shape"] = {points.rows, points.cols};
        const cv::Mat mask = entry.surface->channel("mask");
        item["masked"] = mask.empty() ? Json(nullptr) : Json(cv::countNonZero(mask == 0));
        objects[derived[entry.pin.uuid]] = item;
    }
    for (const auto& entry : inputs.pointCollections) {
        auto item = summary(asFloat(entry.points));
        item["kind"] = "point_collection";
        Json points = Json::object();
        for (std::size_t i = 0; i < entry.ids.size(); ++i)
            points[std::to_string(entry.ids[i])] = {float(entry.points[i][0]), float(entry.points[i][1]),
                                                    float(entry.points[i][2])};
        item["points"] = points;
        objects[derived[entry.pin.uuid]] = item;
    }
    for (const auto& entry : inputs.fibers)
        objects[derived[entry.pin.uuid]] = {{"kind", "fiber"}, {"controls", summary(asFloat(entry.controls))},
                                            {"line", summary(asFloat(entry.line))}};
    std::ofstream(out) << Json{{"frame", "reference-volume voxels, XYZ"}, {"objects", objects}}.dump(1);
    for (const auto& problem : inputs.problems) std::cerr << problem << "\n";
    using vc3d::spiral::LinkState;
    std::cout << "links: " << inputs.count(LinkState::Current) << " current, "
              << inputs.count(LinkState::Changed) << " changed, " << inputs.count(LinkState::Dropped)
              << " dropped, " << inputs.addedRelationships << " added\n";
    return inputs.resolved() ? 0 : 1;
}
} // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    try {
        if (const char* spec = std::getenv("SPIRAL_STORE_INPUTS_DUMP"); spec && *spec)
            return dump(spec);
        metadataHashesMatchThePythonAdapter();
        canonicalJsonMatchesPython();
        relationshipHashesMatchThePythonAdapter();
        QTemporaryDir directory;
        pinnedReadsConvertToReferenceVoxels(directory.path().toStdString());
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "spiral store inputs: ok\n";
    return 0;
}
