#include "SpiralStoreInputs.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/objects/GeometryCoordinates.hpp>

#include <QCryptographicHash>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

std::string sha256Hex(const std::string& text)
{
    return QCryptographicHash::hash(QByteArray::fromStdString(text), QCryptographicHash::Sha256)
        .toHex().toStdString();
}

// repr(float): the shortest digits that round-trip, fixed notation for
// decimal exponents -4..15 and d.ddde+XX otherwise.
std::string pythonFloat(double value)
{
    if (std::isnan(value)) return "NaN";
    if (std::isinf(value)) return value < 0 ? "-Infinity" : "Infinity";
    char text[40];
    for (int precision = 0; precision < 17; ++precision) {
        std::snprintf(text, sizeof(text), "%.*e", precision, value);
        if (std::strtod(text, nullptr) == value) break;
    }
    std::string mantissa(text, std::strchr(text, 'e'));
    const int exponent = std::atoi(std::strchr(text, 'e') + 1);
    std::string sign;
    if (mantissa.front() == '-') {
        sign = "-";
        mantissa.erase(0, 1);
    }
    std::string digits;
    for (const char c : mantissa)
        if (c != '.') digits.push_back(c);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    if (exponent < -4 || exponent >= 16) {
        std::string result = sign + digits.substr(0, 1);
        if (digits.size() > 1) result += "." + digits.substr(1);
        char suffix[8];
        std::snprintf(suffix, sizeof(suffix), "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
        return result + suffix;
    }
    if (exponent < 0) return sign + "0." + std::string(std::size_t(-exponent - 1), '0') + digits;
    if (digits.size() <= std::size_t(exponent) + 1)
        return sign + digits + std::string(std::size_t(exponent) + 1 - digits.size(), '0') + ".0";
    return sign + digits.substr(0, std::size_t(exponent) + 1) + "." + digits.substr(std::size_t(exponent) + 1);
}

void appendCanonical(std::string& out, const Json& value)
{
    switch (value.type()) {
    case Json::value_t::object: {
        // nlohmann orders keys bytewise, as sort_keys orders ASCII and UTF-8.
        out.push_back('{');
        bool first = true;
        for (const auto& [key, item] : value.items()) {
            if (!first) out.push_back(',');
            first = false;
            out += Json(key).dump(-1, ' ', true);
            out.push_back(':');
            appendCanonical(out, item);
        }
        out.push_back('}');
        return;
    }
    case Json::value_t::array: {
        out.push_back('[');
        for (std::size_t i = 0; i < value.size(); ++i) {
            if (i) out.push_back(',');
            appendCanonical(out, value[i]);
        }
        out.push_back(']');
        return;
    }
    case Json::value_t::number_float:
        out += pythonFloat(value.get<double>());
        return;
    default:
        out += value.dump(-1, ' ', true);
    }
}

// The anchor dictionary the Python binding returns (anchorToPy).
Json anchorJson(const vc::objects::Anchor& anchor)
{
    Json value = Json::object();
    if (anchor.wholeGeometry) value["whole_geometry"] = true;
    if (!anchor.channels.empty()) value["channels"] = anchor.channels;
    if (!anchor.metadataKeys.empty()) value["metadata_keys"] = anchor.metadataKeys;
    if (!anchor.points.empty()) {
        value["points"] = Json::array();
        for (const auto& point : anchor.points)
            value["points"].push_back({{"array", point.array}, {"index", point.index}});
    }
    if (anchor.position) value["position"] = *anchor.position;
    if (anchor.direction) value["direction"] = *anchor.direction;
    return value;
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
constexpr const char* kKinds[] = {"surface", "point_collection", "fiber"};

int kindIndex(const std::string& kind)
{
    for (int i = 0; i < 3; ++i)
        if (kind == kKinds[i]) return i;
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

// StoredRelationship.provenance() of a relationship version the adapter
// would consume (filter_relationships), without the replica-local
// "revision"; nullopt when it would not be consumed.
std::optional<Json> consumedProvenance(const vc::objects::SelectedRelationship& selected,
                                       const std::map<std::string, std::string>& kinds,
                                       bool approvedLinksOnly)
{
    const auto& version = selected.relationship;
    const auto& record = version.record;
    const auto aKind = kinds.find(record.aUuid), bKind = kinds.find(record.bUuid);
    if (aKind == kinds.end() || bKind == kinds.end() || !selected.applicable) return std::nullopt;
    struct End {
        std::string uuid, kind;
        const vc::objects::Anchor* anchor;
        std::optional<std::string> geometry;
    };
    End ends[2] = {{record.aUuid, aKind->second, &record.aAnchor, version.aGeometryRevision},
                   {record.bUuid, bKind->second, &record.bAnchor, version.bGeometryRevision}};
    // Ordered by kind, then UUID; a stable sort keeps stored order otherwise.
    if (std::pair(kindIndex(ends[1].kind), ends[1].uuid) < std::pair(kindIndex(ends[0].kind), ends[0].uuid))
        std::swap(ends[0], ends[1]);
    const Json meta = record.meta.empty() ? Json::object() : Json::parse(record.meta);
    const auto family = relationshipFamily(record.kind, ends[0].kind, ends[1].kind);
    if (family.empty()) return std::nullopt;
    if (family == "fiber_branch" && truthy(valueOrNull(meta, "adjacent"))) return std::nullopt;
    if (record.approvalState == "rejected") return std::nullopt;
    if (approvedLinksOnly && record.approvalState != "approved") return std::nullopt;
    const auto geometry = [](const auto& revision) { return revision ? Json(*revision) : Json(nullptr); };
    return Json{{"uuid", record.uuid}, {"kind", record.kind}, {"family", family},
                {"approval_state", record.approvalState}, {"generation", record.generation},
                {"a", ends[0].uuid}, {"a_geometry_revision", geometry(ends[0].geometry)},
                {"b", ends[1].uuid}, {"b_geometry_revision", geometry(ends[1].geometry)},
                {"content_sha256", relationshipContentSha256(*ends[0].anchor, *ends[1].anchor,
                                                             meta.is_null() ? Json::object() : meta)}};
}

// Stored XYZ -> reference-volume voxels, in the frame the fit was read in
// (the manifest's scale factor), through the store's coordinate helpers.
double toVolume(const StoreInputPin& pin, const std::string& volume)
{
    const auto frame = Json{{"coordinates", vc::objects::volumeCoordinates(volume, pin.scaleFactor)}};
    return *vc::objects::geometryScaleToVolume(frame, volume);
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

// Where the local store stands against one pin; nullopt when the pinned
// geometry revision is not this object's here.
std::optional<PinState> pinState(const ObjectStore& db, const Txn& txn, const StoreInputPin& pin,
                                 std::optional<vc::objects::GeometryRevision>& geometry)
{
    geometry = db.geometryRevision(txn, pin.geometryRevision);
    if (!geometry || geometry->objectUuid != pin.uuid) return std::nullopt;
    const auto current = db.currentObjectRevision(txn, pin.uuid);
    if (!current || current->deleted || current->geometryRevision != pin.geometryRevision)
        return PinState::Edited;
    if (!pin.metadataSha256.empty()
        && adapterMetadataSha256(pin.kind, Json::parse(current->record.meta), current->record.fiber)
               != pin.metadataSha256)
        return PinState::MetadataChanged;
    return PinState::Current;
}

// The revision of the pinned object that holds the pinned geometry with the
// metadata the fit read: the current one when it still does, else the latest
// such revision in its history. nullopt when this store holds none.
std::optional<vc::objects::ObjectRevision> fittedRevision(const ObjectStore& db, const Txn& txn,
                                                          const StoreInputPin& pin)
{
    const auto holds = [&](const vc::objects::ObjectRevision& revision) {
        if (revision.deleted || revision.geometryRevision != pin.geometryRevision) return false;
        // Without a pinned hash only revisions with their own metadata qualify.
        if (pin.metadataSha256.empty()) return revision.metadataComplete;
        return adapterMetadataSha256(pin.kind, Json::parse(revision.record.meta), revision.record.fiber)
            == pin.metadataSha256;
    };
    if (auto current = db.currentObjectRevision(txn, pin.uuid); current && holds(*current)) return current;
    auto history = db.objectHistory(txn, pin.uuid);
    for (auto revision = history.rbegin(); revision != history.rend(); ++revision)
        if (holds(*revision)) return std::move(*revision);
    return std::nullopt;
}

} // namespace

const char* pinStateName(PinState state)
{
    switch (state) {
    case PinState::Current: return "current";
    case PinState::Edited: return "edited";
    case PinState::MetadataChanged: return "metadata changed";
    case PinState::Missing: return "missing";
    }
    return "unknown";
}

const char* linkStateName(LinkState state)
{
    switch (state) {
    case LinkState::Current: return "current";
    case LinkState::Changed: return "changed";
    case LinkState::Dropped: return "dropped";
    }
    return "unknown";
}

std::size_t PinnedStoreInputs::count(LinkState state) const
{
    return std::size_t(std::count_if(relationships.begin(), relationships.end(),
                                     [state](const auto& entry) { return entry.state == state; }));
}

std::size_t PinnedStoreInputs::count(PinState state) const
{
    std::size_t total = 0;
    for (const auto& entry : surfaces) total += entry.state == state;
    for (const auto& entry : pointCollections) total += entry.state == state;
    for (const auto& entry : fibers) total += entry.state == state;
    return total;
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
        pin.metadataSha256 = object.value("metadata_sha256", std::string());
        pin.scaleFactor = object.at("scale_factor").get<double>();
        pins.push_back(std::move(pin));
    }
    return pins;
}

std::string adapterMetadataSha256(const std::string& kind, const Json& meta,
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
    return sha256Hex(pythonCanonicalJson(read));
}

std::string pythonCanonicalJson(const Json& value)
{
    std::string text;
    appendCanonical(text, value);
    return text;
}

std::string relationshipContentSha256(const vc::objects::Anchor& a, const vc::objects::Anchor& b,
                                      const Json& meta)
{
    return sha256Hex(pythonCanonicalJson(
        Json{{"a_anchor", anchorJson(a)}, {"b_anchor", anchorJson(b)}, {"meta", meta}}));
}

namespace {

// The manifest's consumed relationships against the ones the current heads
// of the fitted objects would give, by record UUID (shared by replicas).
void compareRelationships(const ObjectStore& db, const Txn& txn, const Json& manifest,
                          PinnedStoreInputs& result)
{
    std::map<std::string, std::string> kinds;
    vc::objects::RevisionSelection heads;
    const auto select = [&](const StoreInputPin& pin) {
        kinds[pin.uuid] = pin.kind;
        if (const auto current = db.currentObjectRevision(txn, pin.uuid); current && !current->deleted)
            heads[pin.uuid] = std::nullopt;
    };
    for (const auto& entry : result.surfaces) select(entry.pin);
    for (const auto& entry : result.pointCollections) select(entry.pin);
    for (const auto& entry : result.fibers) select(entry.pin);
    std::map<std::string, Json> consumed;
    try {
        const bool approvedOnly = manifest.value("approved_links_only", false);
        for (const auto& selected : db.relationshipsAt(txn, heads))
            if (auto provenance = consumedProvenance(selected, kinds, approvedOnly))
                consumed[selected.relationship.record.uuid] = std::move(*provenance);
    } catch (const std::exception& error) {
        result.problems.push_back(std::string("relationships: ") + error.what());
        return;
    }
    for (const auto& fitted : manifest.value("relationships", Json::array())) {
        PinnedRelationship entry{fitted.at("uuid").get<std::string>(),
                                 fitted.value("family", std::string()), LinkState::Dropped};
        if (const auto local = consumed.find(entry.uuid); local != consumed.end()) {
            entry.state = LinkState::Current;
            // Fields an older manifest lacks are not compared; "revision"
            // names the service replica's version and never matches here.
            for (const auto& [key, value] : local->second.items())
                if (fitted.contains(key) && fitted.at(key) != value) entry.state = LinkState::Changed;
            consumed.erase(local);
        }
        result.relationships.push_back(std::move(entry));
    }
    result.addedRelationships = consumed.size();
}

} // namespace

PinnedStoreInputs readPinnedStoreInputs(const ObjectStore& db, const Txn& txn, const Json& document)
{
    PinnedStoreInputs result;
    const auto& manifest = document.at("manifest");
    result.digest = document.value("digest", std::string());
    result.storeUuid = manifest.at("store").at("uuid").get<std::string>();
    result.volumeUuid = manifest.at("volume_uuid").get<std::string>();
    if (const auto local = db.uuid(txn); local != result.storeUuid) {
        result.problems.push_back("the fit read store " + result.storeUuid + ", not this store "
                                  + local);
        return result;
    }
    for (const auto& pin : manifestPins(document)) {
        std::optional<vc::objects::GeometryRevision> geometry;
        const auto state = pinState(db, txn, pin, geometry);
        const double scale = toVolume(pin, result.volumeUuid);
        try {
            if (pin.kind == "surface") {
                PinnedSurface entry{pin, state.value_or(PinState::Missing), nullptr};
                if (state) {
                    auto stored = load_quad_from_store(db, txn, pin.uuid, pin.geometryRevision);
                    if (scale != 1.0) {
                        // Invalid vertices keep the sentinel; grid steps follow the frame.
                        auto* points = new cv::Mat_<cv::Vec3f>(stored->rawPointsPtr()->clone());
                        for (auto& point : *points)
                            if (point[0] != -1.f) point *= float(scale);
                        auto converted = std::make_unique<QuadSurface>(
                            points, stored->scale() / float(scale));
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
                PinnedPointCollection entry;
                entry.pin = pin;
                entry.state = state.value_or(PinState::Missing);
                // The role comes from the metadata the fit read, never from a
                // current head with other metadata.
                const auto fitted = state ? fittedRevision(db, txn, pin) : std::nullopt;
                entry.fittedMetadata = fitted.has_value();
                if (state && !fitted)
                    result.problems.push_back("point collection " + pin.uuid + " (" + pin.name
                                              + "): this store lacks the metadata it was fitted with;"
                                                " not shown");
                if (fitted) {
                    const auto meta = Json::parse(fitted->record.meta);
                    entry.absolute = valueOrNull(valueOrNull(meta, "source"), "winding_is_absolute") == Json(true);
                    const auto ids = readIds(db, txn, *geometry);
                    const auto points = readXyz(db, txn, *geometry, "points", scale);
                    if (ids.size() != points.size())
                        throw std::runtime_error("point IDs and points differ in length");
                    const auto annotations = Json::parse(
                        db.geometryPointAnnotations(txn, pin.geometryRevision, "points"));
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
                PinnedFiber entry{pin, state.value_or(PinState::Missing), {}, {}, {}};
                if (state) {
                    if (const auto fitted = fittedRevision(db, txn, pin); fitted && fitted->record.fiber)
                        entry.family = fitted->record.fiber->family;
                    entry.controls = readXyz(db, txn, *geometry, "control_points", scale);
                    entry.line = readXyz(db, txn, *geometry, "line_points", scale);
                }
                result.fibers.push_back(std::move(entry));
            }
        } catch (const std::exception& error) {
            result.problems.push_back(pin.kind + " " + pin.uuid + ": " + error.what());
        }
        if (!state)
            result.problems.push_back(pin.kind + " " + pin.uuid + " (" + pin.name
                                      + "): this store lacks geometry revision "
                                      + pin.geometryRevision);
    }
    compareRelationships(db, txn, manifest, result);
    return result;
}

} // namespace vc3d::spiral
