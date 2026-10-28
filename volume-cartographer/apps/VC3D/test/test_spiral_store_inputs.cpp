// The Spiral workspace's reads of a fit's object-store inputs: they are read
// as the store holds them now and converted to reference-volume voxels, and
// each input and consumed relationship says whether it is what the fit read,
// by geometry revision, the adapter's metadata values and link generations.
//
// With SPIRAL_STORE_INPUTS_DUMP=STORE|MANIFEST|OUT it instead reads MANIFEST's
// inputs from STORE and writes OUT in display_geometry_dump.py's format, for
//   display_geometry_dump.py compare <baseline> OUT
#include "SpiralStoreInputs.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QTemporaryDir>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
using namespace vc::objects;
using Json = nlohmann::json;
using vc3d::spiral::InputState;

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

// Expected values from object_store_inputs.adapter_metadata.
void adapterMetadataMatchesThePythonAdapter()
{
    struct Case { const char* kind; const char* meta; std::optional<FiberDetail> fiber; const char* expected; };
    const FiberDetail h{"H", false};
    const FiberDetail manual{"", true};
    const Case cases[] = {
        {"surface", R"({"coordinates":{"volume_uuid":"s1-reference","scale_factor":1.0},"source":{"spiral_patch_erode_cells":2,"other":"x"}})",
         std::nullopt, R"({"coordinates": {"scale_factor": 1.0, "volume_uuid": "s1-reference"}, "spiral_patch_erode_cells": 2})"},
        {"surface", "{}", std::nullopt, R"({"coordinates": null, "spiral_patch_erode_cells": null})"},
        {"point_collection", R"({"coordinates":{"volume_uuid":"20260411134726","scale_factor":4},"sampling_group":"reviewed","source":{"winding_is_absolute":true}})",
         std::nullopt, R"({"coordinates": {"scale_factor": 4, "volume_uuid": "20260411134726"}, "sampling_group": "reviewed", "winding_is_absolute": true})"},
        {"fiber", R"({"coordinates":{"volume_uuid":"s1-reference","scale_factor":4.0},"source":{"hv_classification":{"family":"H","certainty":0.873}}})",
         h, R"({"coordinates": {"scale_factor": 4.0, "volume_uuid": "s1-reference"}, "family": "H", "hv_classification": {"certainty": 0.873, "family": "H"}, "manual_family": false})"},
        {"fiber", R"({"coordinates":{"volume_uuid":"s1-reference","scale_factor":1e-05}})",
         manual, R"({"coordinates": {"scale_factor": 1e-05, "volume_uuid": "s1-reference"}, "family": "", "hv_classification": null, "manual_family": true})"},
    };
    for (const auto& item : cases)
        require(vc3d::spiral::adapterMetadata(item.kind, Json::parse(item.meta), item.fiber) == Json::parse(item.expected),
                std::string("adapter metadata differs from the Python adapter's for ") + item.kind + " " + item.meta);
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
                                   {"metadata", vc3d::spiral::adapterMetadata(
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
                relationships.push_back(
                    {{"uuid", version.record.uuid}, {"revision", "another-replica"}, {"kind", version.record.kind},
                     {"family", "pcl_patch"}, {"approval_state", version.record.approvalState},
                     {"generation", version.record.generation}, {"a", surface},
                     {"a_geometry_revision", *(stored ? version.aGeometryRevision : version.bGeometryRevision)},
                     {"b", pcl},
                     {"b_geometry_revision", *(stored ? version.bGeometryRevision : version.aGeometryRevision)}});
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

    vc3d::spiral::FitInputChanges changes(const Json& document, const std::set<std::string>& objects,
                                          const std::set<std::string>& links)
    {
        vc3d::spiral::FitInputChanges result;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            result = vc3d::spiral::readFitInputChanges(db, txn, document, objects, links);
        });
        return result;
    }

    vc3d::spiral::FitInputs read(const Json& document)
    {
        vc3d::spiral::FitInputs result;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            result = vc3d::spiral::readFitInputs(db, txn, document);
        });
        return result;
    }
};

// Everything the workspace displays or reports of a read.
Json describe(const vc3d::spiral::FitInputs& inputs)
{
    Json result = {{"problems", inputs.problems}, {"added", Json::array()}};
    for (const auto& link : inputs.addedRelationships) result["added"].push_back({link.uuid, link.a, link.b});
    const auto points = [](const std::vector<cv::Vec3d>& values) {
        Json list = Json::array();
        for (const auto& value : values) list.push_back({value[0], value[1], value[2]});
        return list;
    };
    for (const auto& entry : inputs.surfaces) {
        Json grid = Json::array();
        if (entry.surface)
            for (const auto& point : *entry.surface->rawPointsPtr()) grid.push_back({point[0], point[1], point[2]});
        result["objects"][entry.pin.uuid] = {{"state", vc3d::spiral::inputStateName(entry.state)},
                                             {"problem", entry.problem}, {"grid", grid}};
    }
    for (const auto& entry : inputs.pointCollections)
        result["objects"][entry.pin.uuid] = {{"state", vc3d::spiral::inputStateName(entry.state)},
                                             {"problem", entry.problem}, {"absolute", entry.absolute},
                                             {"ids", entry.ids}, {"points", points(entry.points)}};
    for (const auto& entry : inputs.fibers)
        result["objects"][entry.pin.uuid] = {{"state", vc3d::spiral::inputStateName(entry.state)},
                                             {"problem", entry.problem}, {"family", entry.family},
                                             {"line", points(entry.line)}};
    for (const auto& link : inputs.relationships)
        result["links"][link.uuid] = vc3d::spiral::linkStateName(link.state);
    return result;
}

// The incremental read of what a commit changed, applied to the previous
// read, must give what a full read gives.
vc3d::spiral::FitInputs incremental(Fixture& fixture, const Json& document, vc3d::spiral::FitInputs previous,
                                    const std::set<std::string>& objects, const std::set<std::string>& links,
                                    const std::string& what)
{
    vc3d::spiral::applyFitInputChanges(previous, fixture.changes(document, objects, links));
    auto full = fixture.read(document);
    require(describe(previous) == describe(full), "incremental read differs from a full read after " + what
                + ":\n" + describe(previous).dump() + "\n" + describe(full).dump());
    return full;
}

void readsCurrentInputsAndTheirDifferences(const fs::path& directory)
{
    Fixture fixture(directory / "inputs.vcobj");
    const auto document = fixture.manifest();
    auto inputs = fixture.read(document);
    require(inputs.unchanged(), "the fixture holds its inputs as fitted");
    require(inputs.count(InputState::Fitted) == 3, "every input is as fitted");

    const auto& surface = *inputs.surfaces.at(0).surface;
    require(surface.id == fixture.surface, "input geometry retains its store UUID for viewer queries");
    SurfaceProximityQuery query;
    query.uuids = std::vector<std::string>{surface.id};
    query.volume = kVolume;
    query.point = {10.5, 20.25, 30};
    query.tolerance = 0.1;
    const auto hits = fixture.session->nearestSurfaces({fixture.attachment}, query);
    require(hits.size() == 1 && hits.front().hit.objectUuid == fixture.surface,
            "a displayed fit input can be located directly without a session draft");
    require(!fixture.session->hasDraft(fixture.surface), "reading fit inputs does not create editing drafts");
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
    require(inputs.relationships.size() == 1 && inputs.relationships[0].state == LinkState::Fitted
                && inputs.addedRelationships.empty(),
            "the fitted link is as fitted");

    // A metadata edit the adapter reads, then a geometry edit: what is read
    // is the store now, and the state says what changed since the fit.
    fixture.session->act(fixture.attachment, fixture.pins({fixture.pcl, fixture.surface}),
                         [&](auto& db, auto& txn) {
        db.setMeta(txn, fixture.pcl, "sampling_group", "\"reviewed\"");
        db.setMeta(txn, fixture.surface, "review_note", "\"unrelated\"");
    });
    inputs = incremental(fixture, document, inputs, {fixture.pcl, fixture.surface}, {}, "metadata edits");
    require(inputs.pointCollections.at(0).state == InputState::MetadataChanged, "metadata change detected");
    require(inputs.surfaces.at(0).state == InputState::Fitted, "unread metadata is no change");
    auto unrecorded = document;
    unrecorded["manifest"]["objects"][1].erase("metadata");
    require(fixture.read(unrecorded).pointCollections.at(0).state == InputState::Fitted,
            "a manifest without metadata values compares geometry only");
    fixture.session->act(fixture.attachment, fixture.pins({fixture.pcl}), [&](auto& db, auto& txn) {
        db.setMeta(txn, fixture.pcl, "source", R"({"winding_is_absolute":false})");
    });
    inputs = incremental(fixture, document, inputs, {fixture.pcl}, {}, "a role change");
    require(!inputs.pointCollections.at(0).absolute, "the role is the store's now");

    fixture.session->act(fixture.attachment, fixture.pins({fixture.fiber}), [&](auto& db, auto& txn) {
        const auto current = db.currentGeometry(txn, fixture.fiber);
        const double moved[3] = {0, 0, 0};
        GeometryArrayEdit edit{"line_points", 3, {}};
        edit.bytes.resize(sizeof(moved));
        std::memcpy(edit.bytes.data(), moved, sizeof(moved));
        db.editGeometry(txn, fixture.fiber, {edit}, current->uuid,
                        [](const std::vector<EdgeRecord>&) { return std::vector<GeometryEdgeEdit>{}; });
    });
    inputs = incremental(fixture, document, inputs, {fixture.fiber}, {}, "a geometry edit");
    require(inputs.fibers.at(0).state == InputState::Edited, "geometry edit detected");
    require(inputs.fibers.at(0).line.size() == 3 && inputs.fibers.at(0).line[1] == cv::Vec3d(0, 0, 0),
            "the edited geometry is read");

    // Links: an edit changes the fitted one, a new one among the fitted
    // inputs is reported, and a rejected one is no longer consumed.
    fixture.session->act(fixture.attachment, fixture.pins({fixture.surface, fixture.pcl}),
                         [&](auto& db, auto& txn) {
        db.setMeta(txn, fixture.link, "distance", "0.25");
    });
    inputs = incremental(fixture, document, inputs, {}, {fixture.link}, "a link edit");
    require(inputs.relationships.at(0).state == LinkState::Changed, "link edit detected");
    std::string added;
    fixture.session->act(fixture.attachment, fixture.pins({fixture.surface, fixture.fiber}),
                         [&](auto& db, auto& txn) {
        EdgeRecord edge; edge.kind = "same-surface"; edge.aUuid = fixture.surface; edge.bUuid = fixture.fiber;
        edge.aAnchor.wholeGeometry = true; edge.bAnchor.wholeGeometry = true;
        added = db.addEdge(txn, edge);
    });
    inputs = incremental(fixture, document, inputs, {}, {added}, "a new link");
    require(inputs.addedRelationships.size() == 1 && inputs.addedRelationships[0].uuid == added
                && inputs.addedRelationships[0].a == fixture.surface,
            "a new consumable link is reported");
    fixture.session->act(fixture.attachment, fixture.pins({fixture.surface, fixture.pcl}),
                         [&](auto& db, auto& txn) { db.setApproval(txn, fixture.link, "rejected"); });
    inputs = incremental(fixture, document, inputs, {}, {fixture.link}, "a rejection");
    require(inputs.relationships.at(0).state == LinkState::Dropped, "a rejected link is dropped");
    // Removing the new link: reported by the link alone, or by an endpoint.
    fixture.session->act(fixture.attachment, fixture.pins({fixture.surface, fixture.fiber}),
                         [&](auto& db, auto& txn) { db.removeEdge(txn, added); });
    require(incremental(fixture, document, inputs, {}, {added}, "a removed link").addedRelationships.empty(),
            "a removed link is no longer reported");
    inputs = incremental(fixture, document, inputs, {fixture.fiber}, {}, "a removed link, by its endpoint");

    // An object this store does not hold is missing and shows nothing.
    auto missing = document;
    missing["manifest"]["objects"][0]["uuid"] = util::newUuid();
    inputs = fixture.read(missing);
    require(inputs.surfaces.at(0).state == InputState::Missing && !inputs.surfaces.at(0).surface,
            "a missing object has no geometry");
    require(!inputs.unchanged() && !inputs.surfaces.at(0).problem.empty() && inputs.problems.empty(),
            "a missing object is reported");

    auto other = document;
    other["manifest"]["store"]["uuid"] = util::newUuid();
    inputs = fixture.read(other);
    require(inputs.surfaces.empty() && inputs.problems.size() == 1, "another store is refused");
}

// display_geometry_dump.py's entry for each input.
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
    vc3d::spiral::FitInputs inputs;
    std::map<std::string, std::string> derived;
    double fullSeconds = 0, changeSeconds = 0;
    session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto start = std::chrono::steady_clock::now();
        inputs = vc3d::spiral::readFitInputs(db, txn, document);
        const auto middle = std::chrono::steady_clock::now();
        // What one commit to the first surface and the first link costs.
        if (!inputs.surfaces.empty() && !inputs.relationships.empty())
            (void)vc3d::spiral::readFitInputChanges(db, txn, document, {inputs.surfaces[0].pin.uuid},
                                                    {inputs.relationships[0].uuid});
        fullSeconds = std::chrono::duration<double>(middle - start).count();
        changeSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - middle).count();
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
    for (const auto& entry : inputs.surfaces) if (!entry.problem.empty()) std::cerr << entry.problem << "\n";
    for (const auto& entry : inputs.pointCollections) if (!entry.problem.empty()) std::cerr << entry.problem << "\n";
    for (const auto& entry : inputs.fibers) if (!entry.problem.empty()) std::cerr << entry.problem << "\n";
    using vc3d::spiral::LinkState;
    std::cout << "read: full " << fullSeconds << " s, one surface and one link " << changeSeconds << " s\n";
    std::cout << "inputs: " << inputs.count(vc3d::spiral::InputState::Fitted) << " as fitted, "
              << inputs.count(vc3d::spiral::InputState::Edited) << " edited, "
              << inputs.count(vc3d::spiral::InputState::MetadataChanged) << " metadata changed, "
              << inputs.count(vc3d::spiral::InputState::Missing) << " missing\n";
    std::cout << "links: " << inputs.count(LinkState::Fitted) << " as fitted, "
              << inputs.count(LinkState::Changed) << " changed, " << inputs.count(LinkState::Dropped)
              << " dropped, " << inputs.addedRelationships.size() << " added\n";
    return inputs.unchanged() ? 0 : 1;
}
} // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    try {
        if (const char* spec = std::getenv("SPIRAL_STORE_INPUTS_DUMP"); spec && *spec)
            return dump(spec);
        adapterMetadataMatchesThePythonAdapter();
        QTemporaryDir directory;
        readsCurrentInputsAndTheirDifferences(directory.path().toStdString());
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "spiral store inputs: ok\n";
    return 0;
}
