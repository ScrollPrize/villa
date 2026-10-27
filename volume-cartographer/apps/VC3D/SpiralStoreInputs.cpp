#include "SpiralStoreInputs.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/objects/GeometryCoordinates.hpp>

#include <QCryptographicHash>

#include <algorithm>
#include <cstring>
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
    // json.dumps(sort_keys=True, separators=(",", ":")): ASCII escapes.
    const std::string text = read.dump(-1, ' ', true);
    return QCryptographicHash::hash(QByteArray::fromStdString(text), QCryptographicHash::Sha256)
        .toHex().toStdString();
}

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
                if (state) {
                    const auto record = db.objectByUuid(txn, pin.uuid);
                    const auto meta = record ? Json::parse(record->meta) : Json::object();
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
                PinnedFiber entry{pin, state.value_or(PinState::Missing), {}, {}};
                if (state) {
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
    return result;
}

} // namespace vc3d::spiral
