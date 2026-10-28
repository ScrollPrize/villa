#include "SpiralStoreInputs.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/objects/GeometryCoordinates.hpp>

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>

namespace vc3d::spiral {
namespace {

using vc::objects::ObjectStore;
using vc::objects::Txn;
using Json = nlohmann::json;

Json valueOrNull(const Json& object, const char* key)
{
    return object.is_object() && object.contains(key) ? object.at(key) : Json(nullptr);
}

std::vector<cv::Vec3d> readXyz(const ObjectStore& db, const Txn& txn,
                               const vc::objects::GeometryRevision& revision,
                               const std::string& name, double scale)
{
    const auto info = std::find_if(revision.arrays.begin(), revision.arrays.end(),
                                   [&](const auto& array) { return array.name == name; });
    if (info == revision.arrays.end())
        throw std::runtime_error("geometry " + revision.uuid + " has no " + name + " array");
    const auto bytes = db.readGeometryArray(txn, revision.uuid, name);
    const auto count = info->shape.empty() ? 0 : std::size_t(info->shape[0]);
    std::vector<cv::Vec3d> points(count);
    for (std::size_t i = 0; i < count; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            double value = 0;
            if (info->dtype == "<f8") {
                std::memcpy(&value, bytes.data() + (i * 3 + axis) * 8, 8);
            } else if (info->dtype == "<f4") {
                float single = 0;
                std::memcpy(&single, bytes.data() + (i * 3 + axis) * 4, 4);
                value = single;
            } else {
                throw std::runtime_error(name + " has unsupported dtype " + info->dtype);
            }
            points[i][axis] = value * scale;
        }
    }
    return points;
}

std::vector<int64_t> readIds(const ObjectStore& db, const Txn& txn,
                             const vc::objects::GeometryRevision& revision)
{
    const auto info = std::find_if(revision.arrays.begin(), revision.arrays.end(),
                                   [](const auto& array) { return array.name == "point_ids"; });
    if (info == revision.arrays.end() || info->dtype != "<i8")
        throw std::runtime_error("geometry " + revision.uuid + " has no int64 point_ids");
    const auto bytes = db.readGeometryArray(txn, revision.uuid, "point_ids");
    std::vector<int64_t> ids(bytes.size() / 8);
    std::memcpy(ids.data(), bytes.data(), ids.size() * 8);
    return ids;
}

bool truthy(const Json& value)
{
    if (value.is_null()) return false;
    if (value.is_boolean()) return value.get<bool>();
    if (value.is_number()) return value.get<double>() != 0;
    if (value.is_string() || value.is_array() || value.is_object()) return !value.empty();
    return true;
}

// object_store_inputs.KINDS and FAMILIES: the relationship families the
// fitter consumes, by edge kind and endpoint kinds in KINDS order.
int kindIndex(const std::string& kind)
{
    constexpr const char* kinds[] = {"surface", "point_collection", "fiber"};
    for (int i = 0; i < 3; ++i)
        if (kind == kinds[i]) return i;
    return 3;
}

std::string relationshipFamily(const std::string& edgeKind, const std::string& first,
                               const std::string& second)
{
    if (edgeKind == "point-surface" && first == "surface" && second == "point_collection") return "pcl_patch";
    if (edgeKind == "same-surface" && first == "surface" && second == "fiber") return "fiber_patch";
    if (edgeKind == "same-surface" && first == "surface" && second == "surface") return "patch_overlap";
    if (edgeKind == "fiber_branch" && first == "fiber" && second == "fiber") return "fiber_branch";
    return {};
}

// The identity fields of StoredRelationship.provenance() for a relationship
// version the adapter would consume (filter_relationships); nullopt when it
// would not be consumed. A content change bumps the generation, so the
// generation stands for the anchors and metadata.
std::optional<Json> consumedIdentity(const vc::objects::SelectedRelationship& selected,
                                     const std::map<std::string, std::string>& kinds,
                                     bool approvedLinksOnly)
{
    const auto& version = selected.relationship;
    const auto& record = version.record;
    const auto aKind = kinds.find(record.aUuid), bKind = kinds.find(record.bUuid);
    if (aKind == kinds.end() || bKind == kinds.end() || !selected.applicable) return std::nullopt;
    struct End {
        std::string uuid, kind;
        std::optional<std::string> geometry;
    };
    End ends[2] = {{record.aUuid, aKind->second, version.aGeometryRevision},
                   {record.bUuid, bKind->second, version.bGeometryRevision}};
    // Ordered by kind, then UUID; a stable sort keeps stored order otherwise.
    if (std::pair(kindIndex(ends[1].kind), ends[1].uuid) < std::pair(kindIndex(ends[0].kind), ends[0].uuid))
        std::swap(ends[0], ends[1]);
    const auto family = relationshipFamily(record.kind, ends[0].kind, ends[1].kind);
    if (family.empty()) return std::nullopt;
    const Json meta = record.meta.empty() ? Json::object() : Json::parse(record.meta);
    if (family == "fiber_branch" && truthy(valueOrNull(meta, "adjacent"))) return std::nullopt;
    if (record.approvalState == "rejected") return std::nullopt;
    if (approvedLinksOnly && record.approvalState != "approved") return std::nullopt;
    const auto geometry = [](const auto& revision) { return revision ? Json(*revision) : Json(nullptr); };
    return Json{{"uuid", record.uuid}, {"kind", record.kind}, {"family", family},
                {"approval_state", record.approvalState}, {"generation", record.generation},
                {"a", ends[0].uuid}, {"a_geometry_revision", geometry(ends[0].geometry)},
                {"b", ends[1].uuid}, {"b_geometry_revision", geometry(ends[1].geometry)}};
}

// The manifest's consumed relationships against the ones the current heads
// of the fitted objects give, by record UUID.
void compareRelationships(const ObjectStore& db, const Txn& txn, const Json& manifest,
                          const std::vector<StoreInputPin>& pins, FitInputs& result)
{
    std::map<std::string, std::string> kinds;
    vc::objects::RevisionSelection heads;
    for (const auto& pin : pins) {
        kinds[pin.uuid] = pin.kind;
        if (const auto current = db.currentObjectRevision(txn, pin.uuid); current && !current->deleted)
            heads[pin.uuid] = std::nullopt;
    }
    std::map<std::string, Json> consumed;
    try {
        const bool approvedOnly = manifest.value("approved_links_only", false);
        for (const auto& selected : db.relationshipsAt(txn, heads))
            if (auto identity = consumedIdentity(selected, kinds, approvedOnly))
                consumed[selected.relationship.record.uuid] = std::move(*identity);
    } catch (const std::exception& error) {
        result.problems.push_back(std::string("relationships: ") + error.what());
        return;
    }
    for (const auto& fitted : manifest.value("relationships", Json::array())) {
        FitRelationship entry{fitted.at("uuid").get<std::string>(), fitted.value("family", std::string()),
                              LinkState::Dropped};
        if (const auto local = consumed.find(entry.uuid); local != consumed.end()) {
            entry.state = LinkState::Fitted;
            for (const auto& [key, value] : local->second.items())
                if (fitted.contains(key) && fitted.at(key) != value) entry.state = LinkState::Changed;
            consumed.erase(local);
        }
        result.relationships.push_back(std::move(entry));
    }
    result.addedRelationships = consumed.size();
}

} // namespace

const char* inputStateName(InputState state)
{
    switch (state) {
    case InputState::Fitted: return "as fitted";
    case InputState::Edited: return "edited";
    case InputState::MetadataChanged: return "metadata changed";
    case InputState::Missing: return "missing";
    }
    return "unknown";
}

const char* linkStateName(LinkState state)
{
    switch (state) {
    case LinkState::Fitted: return "as fitted";
    case LinkState::Changed: return "changed";
    case LinkState::Dropped: return "dropped";
    }
    return "unknown";
}

std::size_t FitInputs::count(InputState state) const
{
    std::size_t total = 0;
    for (const auto& entry : surfaces) total += entry.state == state;
    for (const auto& entry : pointCollections) total += entry.state == state;
    for (const auto& entry : fibers) total += entry.state == state;
    return total;
}

std::size_t FitInputs::count(LinkState state) const
{
    return std::size_t(std::count_if(relationships.begin(), relationships.end(),
                                     [state](const auto& entry) { return entry.state == state; }));
}

std::vector<StoreInputPin> manifestPins(const Json& document)
{
    std::vector<StoreInputPin> pins;
    for (const auto& object : document.at("manifest").value("objects", Json::array())) {
        StoreInputPin pin;
        pin.uuid = object.at("uuid").get<std::string>();
        pin.kind = object.at("kind").get<std::string>();
        pin.name = object.value("name", std::string());
        pin.geometryRevision = object.at("geometry_revision").get<std::string>();
        pin.metadata = valueOrNull(object, "metadata");
        pins.push_back(std::move(pin));
    }
    return pins;
}

Json adapterMetadata(const std::string& kind, const Json& meta,
                     const std::optional<vc::objects::FiberDetail>& fiber)
{
    const Json source = valueOrNull(meta, "source");
    Json read = Json::object();
    read["coordinates"] = valueOrNull(meta, "coordinates");
    if (kind == "surface") {
        read["spiral_patch_erode_cells"] = valueOrNull(source, "spiral_patch_erode_cells");
    } else if (kind == "point_collection") {
        read["winding_is_absolute"] = valueOrNull(source, "winding_is_absolute");
        read["sampling_group"] = valueOrNull(meta, "sampling_group");
    } else if (kind == "fiber") {
        read["hv_classification"] = valueOrNull(source, "hv_classification");
        read["family"] = fiber ? Json(fiber->family) : Json(nullptr);
        read["manual_family"] = fiber ? Json(fiber->manualFamily) : Json(nullptr);
    }
    return read;
}

FitInputs readFitInputs(const ObjectStore& db, const Txn& txn, const Json& document)
{
    FitInputs result;
    const auto& manifest = document.at("manifest");
    result.digest = document.value("digest", std::string());
    result.storeUuid = manifest.at("store").at("uuid").get<std::string>();
    result.volumeUuid = manifest.at("volume_uuid").get<std::string>();
    if (const auto local = db.uuid(txn); local != result.storeUuid) {
        result.problems.push_back("the fit read store " + result.storeUuid + ", not this store "
                                  + local);
        return result;
    }
    const auto pins = manifestPins(document);
    for (const auto& pin : pins) {
        const auto current = db.currentObjectRevision(txn, pin.uuid);
        const bool present = current && !current->deleted && current->geometryRevision;
        auto state = InputState::Missing;
        Json meta = Json::object();
        if (present) {
            meta = Json::parse(current->record.meta);
            // Compared as values: JSON numbers compare numerically.
            state = *current->geometryRevision != pin.geometryRevision ? InputState::Edited
                : !pin.metadata.is_null()
                        && adapterMetadata(pin.kind, meta, current->record.fiber) != pin.metadata
                    ? InputState::MetadataChanged
                    : InputState::Fitted;
        } else {
            result.problems.push_back(pin.kind + " " + pin.uuid + " (" + pin.name
                                      + ") is not in this store");
        }
        std::optional<vc::objects::GeometryRevision> geometry;
        std::optional<double> scale;
        try {
            if (present) {
                geometry = db.geometryRevision(txn, *current->geometryRevision);
                // Stored XYZ -> reference-volume voxels, in the frame the
                // object declares now.
                scale = vc::objects::geometryScaleToVolume(meta, result.volumeUuid);
                if (!scale)
                    throw std::runtime_error("it declares no frame in volume " + result.volumeUuid);
            }
            if (pin.kind == "surface") {
                FitSurface entry{pin, state, nullptr};
                if (geometry) {
                    auto stored = load_quad_from_store(db, txn, pin.uuid, geometry->uuid);
                    if (*scale != 1.0) {
                        // Invalid vertices keep the sentinel; grid steps follow the frame.
                        auto* points = new cv::Mat_<cv::Vec3f>(stored->rawPointsPtr()->clone());
                        for (auto& point : *points)
                            if (point[0] != -1.f) point *= float(*scale);
                        auto converted = std::make_unique<QuadSurface>(
                            points, stored->scale() / float(*scale));
                        converted->id = stored->id;
                        converted->meta = stored->meta;
                        converted->tags = stored->tags;
                        const cv::Mat mask = stored->channel("mask");
                        if (!mask.empty()) converted->setChannel("mask", mask);
                        stored = std::move(converted);
                    }
                    entry.surface = std::shared_ptr<QuadSurface>(std::move(stored));
                }
                result.surfaces.push_back(std::move(entry));
            } else if (pin.kind == "point_collection") {
                FitPointCollection entry;
                entry.pin = pin;
                entry.state = state;
                if (geometry) {
                    entry.absolute = valueOrNull(valueOrNull(meta, "source"), "winding_is_absolute") == Json(true);
                    const auto ids = readIds(db, txn, *geometry);
                    const auto points = readXyz(db, txn, *geometry, "points", *scale);
                    if (ids.size() != points.size())
                        throw std::runtime_error("point IDs and points differ in length");
                    const auto annotations = Json::parse(
                        db.geometryPointAnnotations(txn, geometry->uuid, "points"));
                    std::vector<std::size_t> order(ids.size());
                    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
                    std::sort(order.begin(), order.end(),
                              [&](std::size_t a, std::size_t b) { return ids[a] < ids[b]; });
                    for (const auto i : order) {
                        entry.ids.push_back(ids[i]);
                        entry.points.push_back(points[i]);
                        const Json annotation = annotations.is_array() && i < annotations.size()
                            ? annotations[i] : Json(nullptr);
                        const Json wind = valueOrNull(annotation, "wind_a");
                        entry.windings.push_back(wind.is_number() ? std::optional(wind.get<double>())
                                                                  : std::nullopt);
                    }
                }
                result.pointCollections.push_back(std::move(entry));
            } else if (pin.kind == "fiber") {
                FitFiber entry{pin, state, {}, {}, {}};
                if (geometry) {
                    if (current->record.fiber) entry.family = current->record.fiber->family;
                    entry.controls = readXyz(db, txn, *geometry, "control_points", *scale);
                    entry.line = readXyz(db, txn, *geometry, "line_points", *scale);
                }
                result.fibers.push_back(std::move(entry));
            }
        } catch (const std::exception& error) {
            result.problems.push_back(pin.kind + " " + pin.uuid + ": " + error.what());
        }
    }
    compareRelationships(db, txn, manifest, pins, result);
    return result;
}

} // namespace vc3d::spiral
