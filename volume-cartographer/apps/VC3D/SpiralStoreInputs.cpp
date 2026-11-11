#include "SpiralStoreInputs.hpp"
#include <vc_object_store/store/GeometryViews.hpp>

#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/TransactionJson.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/RelationshipInputs.hpp>
#include <vc_object_store/objects/ManifestReferences.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace vc3d::spiral {
namespace {

using vc::objects::StoreReader;
using Json = nlohmann::json;

Json valueOrNull(const Json& object, const char* key)
{
    return object.is_object() && object.contains(key) ? object.at(key) : Json(nullptr);
}

std::vector<cv::Vec3d> scaledPoints(const std::vector<vc::geometry::Point>& points, double scale)
{
    std::vector<cv::Vec3d> result;
    result.reserve(points.size());
    for (const auto& p : points) result.emplace_back(p[0] * scale, p[1] * scale, p[2] * scale);
    return result;
}

struct ConsumedRelationshipIdentity {
    vc::objects::RelationshipVersion reference;
    std::string family, a, b;
};

// The manifest's consumed relationships against the ones the current heads
// of the fitted objects give, by record UUID. With `scope`, only those
// relationships are compared, from the heads of their fitted endpoints.
void compareRelationships(const StoreReader& reader, const Json& manifest,
                          const std::vector<StoreInputPin>& pins, const std::set<std::string>* scope,
                          const std::map<std::string, vc::objects::RelationshipVersion>& references, FitInputs& result)
{
    std::map<std::string, std::string> kinds;
    for (const auto& pin : pins) kinds[pin.uuid] = pin.kind;
    const auto inScope = [&](const std::string& uuid) { return !scope || scope->contains(uuid); };
    const auto fitted = manifest.value("relationships", Json::array());
    std::set<std::string> endpoints;
    if (scope) {
        for (const auto& entry : fitted)
            if (inScope(entry.at("uuid").get<std::string>()))
                endpoints.insert({entry.value("a", std::string()), entry.value("b", std::string())});
        vc::objects::EdgeSelection scoped;
        scoped.uuids = std::vector<std::string>(scope->begin(), scope->end());
        scoped.includeRetired = true;
        for (const auto& edge : reader.selectEdges(scoped)) endpoints.insert({edge.aUuid, edge.bUuid});
    } else {
        for (const auto& [uuid, kind] : kinds) endpoints.insert(uuid);
    }
    // The fitted endpoints at their current versions.
    std::vector<vc::objects::ObjectVersion> heads;
    for (const auto& uuid : endpoints) {
        if (!kinds.contains(uuid)) continue;
        if (const auto current = reader.currentObject(uuid); current && !current->deleted)
            heads.push_back(current->reference());
    }
    std::map<std::string, ConsumedRelationshipIdentity> consumed;
    try {
        const bool approvedOnly = manifest.value("approved_links_only", false);
        if (!heads.empty())
            for (const auto& selected : reader.relationshipsAt(heads))
                if (inScope(selected.relationship.record.uuid)) {
                    const auto& edge = selected.relationship.record;
                    const auto decision = vc::objects::fittingRelationshipInput(edge, selected.applicable, kinds, approvedOnly);
                    if (decision.exclusion.empty())
                        consumed[edge.uuid] = {selected.relationship.reference(), decision.family,
                            decision.reversed ? edge.bUuid : edge.aUuid, decision.reversed ? edge.aUuid : edge.bUuid};
                }
    } catch (const std::exception& error) {
        result.problems.push_back(std::string("relationships: ") + error.what());
        return;
    }
    for (const auto& entry : fitted) {
        FitRelationship link{entry.at("uuid").get<std::string>(), entry.value("family", std::string()),
                             LinkState::Dropped, entry.value("a", std::string()), entry.value("b", std::string())};
        if (!inScope(link.uuid)) continue;
        if (const auto local = consumed.find(link.uuid); local != consumed.end()) {
            link.state = LinkState::Fitted;
            if (references.at(link.uuid) != local->second.reference)
                link.state = LinkState::Changed;
            consumed.erase(local);
        }
        result.relationships.push_back(std::move(link));
    }
    for (const auto& [uuid, identity] : consumed)
        result.addedRelationships.push_back({uuid, identity.family, LinkState::Added, identity.a, identity.b});
}

// Read one manifest object as the store holds it now into `result`.
void readObject(const StoreReader& reader, const StoreInputPin& pin, const std::string& volume,
                FitInputs& result)
{
    const auto current = reader.currentObject(pin.uuid);
    const bool present = current && !current->deleted && current->geometryRevision;
    auto state = InputState::Missing;
    Json meta = Json::object();
    std::string problem;
    if (present) {
        meta = Json::parse(current->record.meta);
        // Compared as values: JSON numbers compare numerically.
        state = *current->geometryRevision != pin.geometryRevision ? InputState::Edited
            : !pin.metadata.is_null()
                    && adapterMetadata(pin.kind, meta, current->record.fiber, current->record.track) != pin.metadata
                ? InputState::MetadataChanged
                : InputState::Fitted;
    } else {
        problem = pin.kind + " " + pin.uuid + " (" + pin.name + ") is not in this store";
    }
    FitSurface surface{pin, state, nullptr, {}};
    FitPointCollection collection;
    collection.pin = pin;
    collection.state = state;
    if (present) collection.reference = current->reference();
    FitFiber fiber{pin, state, {}, {}, {}, {}};
    if (pin.kind == "track") {
        // Its state is all a track contributes; its geometry is not shown.
        result.tracks.push_back({pin, state, problem});
        return;
    }
    try {
        std::optional<vc::objects::GeometryRevision> geometry;
        std::optional<double> scale;
        if (present) {
            geometry = reader.geometryRevision(*current->geometryRevision);
            // Stored XYZ -> reference-volume voxels, in the frame the object
            // declares now.
            scale = vc::objects::geometryScaleToVolume(meta, volume);
            if (!scale) throw std::runtime_error("it declares no frame in volume " + volume);
        }
        if (geometry && pin.kind == "surface") {
            const auto view=vc::objects::GeometrySnapshot(reader).surface(*current).viewTransformedAt(volume);
            surface.surface = std::shared_ptr<QuadSurface>(load_quad_from_store(reader,*current,true,&view));
        } else if (geometry && pin.kind == "point_collection") {
            collection.absolute = valueOrNull(valueOrNull(meta, "source"), "winding_is_absolute") == Json(true);
            const auto view = vc::objects::GeometrySnapshot(reader).pointCollection(pin.uuid, geometry->uuid);
            const auto ids = view.ids(0, view.size());
            const auto points = scaledPoints(view.points(0, view.size()), *scale);
            if (ids.size() != points.size()) throw std::runtime_error("point IDs and points differ in length");
            const auto annotations = reader.geometryPointAnnotations(geometry->uuid, "points");
            std::vector<std::size_t> order(ids.size());
            for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ids[a] < ids[b]; });
            for (const auto i : order) {
                collection.ids.push_back(ids[i]);
                collection.points.push_back(points[i]);
                const Json annotation = annotations.is_array() && i < annotations.size() ? annotations[i]
                                                                                         : Json(nullptr);
                const Json wind = valueOrNull(annotation, "wind_a");
                collection.windings.push_back(wind.is_number() ? std::optional(wind.get<double>()) : std::nullopt);
            }
        } else if (geometry && pin.kind == "fiber") {
            if (current->record.fiber) fiber.family = current->record.fiber->family;
            const auto view = vc::objects::GeometrySnapshot(reader).fiber(pin.uuid, geometry->uuid);
            fiber.controls = scaledPoints(view.controlPoints(0, view.controlPointCount()), *scale);
            fiber.line = scaledPoints(view.linePoints(0, view.linePointCount()), *scale);
        }
    } catch (const std::exception& error) {
        // Nothing of an unreadable input is shown.
        problem = pin.kind + " " + pin.uuid + ": " + error.what();
        surface.surface.reset();
        collection = FitPointCollection{pin, state, false, {}, {}, {}, {}, {}};
        fiber = FitFiber{pin, state, {}, {}, {}, {}};
    }
    if (pin.kind == "surface") {
        surface.problem = problem;
        result.surfaces.push_back(std::move(surface));
    } else if (pin.kind == "point_collection") {
        collection.problem = problem;
        result.pointCollections.push_back(std::move(collection));
    } else if (pin.kind == "fiber") {
        fiber.problem = problem;
        result.fibers.push_back(std::move(fiber));
    }
}

// The manifest's store and volume; false (with the problem) for another store.
bool sameStore(const StoreReader& reader, const Json& document, FitInputs& result,
               vc::objects::ManifestReferences& references)
{
    const auto& manifest = document.at("manifest");
    try { references=vc::objects::manifestReferences(manifest); }
    catch(const std::exception& error){result.problems.push_back(error.what());return false;}
    result.digest = document.value("digest", std::string());
    result.storeUuid = manifest.at("store").at("uuid").get<std::string>();
    result.volumeUuid = manifest.at("volume_uuid").get<std::string>();
    if (const auto local = reader.status().storeUuid; local != result.storeUuid) {
        result.problems.push_back("the fit read store " + result.storeUuid + ", not this store " + local);
        return false;
    }
    return true;
}

template <typename Entry>
void replaceByUuid(std::vector<Entry>& entries, std::vector<Entry>& changed)
{
    for (auto& entry : changed) {
        const auto found = std::find_if(entries.begin(), entries.end(),
                                        [&](const Entry& old) { return old.pin.uuid == entry.pin.uuid; });
        if (found != entries.end()) *found = std::move(entry);
    }
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
    case LinkState::Added: return "added";
    }
    return "unknown";
}

std::size_t FitInputs::count(InputState state) const
{
    std::size_t total = 0;
    for (const auto& entry : surfaces) total += entry.state == state;
    for (const auto& entry : pointCollections) total += entry.state == state;
    for (const auto& entry : fibers) total += entry.state == state;
    for (const auto& entry : tracks) total += entry.state == state;
    return total;
}

std::size_t FitInputs::count(LinkState state) const
{
    return std::size_t(std::count_if(relationships.begin(), relationships.end(),
                                     [state](const auto& entry) { return entry.state == state; }));
}

namespace {
StoreInputPin objectPin(const Json& object,const vc::objects::ManifestReferences& references)
{
    StoreInputPin pin;
    pin.uuid = object.at("uuid").get<std::string>();
    pin.kind = object.at("kind").get<std::string>();
    pin.name = object.value("name", std::string());
    pin.geometryRevision = object.at("geometry_revision").get<std::string>();
    pin.metadata = valueOrNull(object, "metadata");
    pin.reference=references.objects.at(pin.uuid);
    return pin;
}
std::vector<StoreInputPin> objectPins(const Json& document,const vc::objects::ManifestReferences& references)
{
    std::vector<StoreInputPin> pins;
    for (const auto& object : document.at("manifest").value("objects", Json::array()))
        pins.push_back(objectPin(object,references));
    return pins;
}
}

std::vector<StoreInputPin> manifestPins(const Json& document)
{
    return objectPins(document,vc::objects::manifestReferences(document.at("manifest")));
}

// Spiral input-protocol projection, mirrored by object_store_inputs.adapter_metadata.
// test_spiral_store_inputs pins this domain contract; geometry/placement decoding
// belongs exclusively to GeometrySnapshot, not to either adapter.
Json adapterMetadata(const std::string& kind, const Json& meta,
                     const std::optional<vc::objects::FiberDetail>& fiber,
                     const std::optional<vc::objects::TrackDetail>& track)
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
    } else if (kind == "track") {
        read["family"] = track ? Json(track->family) : Json(nullptr);
        read["dataset_ref"] = track ? Json(track->datasetRef) : Json(nullptr);
        read["external_key"] = track ? Json(track->externalKey) : Json(nullptr);
    }
    return read;
}

FitInputs readFitInputs(const StoreReader& reader, const Json& document)
{
    FitInputs result;
    vc::objects::ManifestReferences references;
    if (!sameStore(reader, document, result, references)) return result;
    const auto pins = objectPins(document,references);
    for (const auto& pin : pins) readObject(reader, pin, result.volumeUuid, result);
    // The outer shell: read as a surface, held apart from the patches.
    if (const Json shell = valueOrNull(document.at("manifest"), "outer_shell"); shell.is_object()) {
        FitInputs read;
        const auto pin = objectPin(shell,references);
        readObject(reader, pin, result.volumeUuid, read);
        // Only a surface pin lands among the surfaces: anything else the
        // manifest names as its shell is a problem, not a shell.
        if (read.surfaces.empty())
            result.problems.push_back("the fit's outer shell " + pin.uuid + " is a " + pin.kind
                                      + ", not a surface");
        else
            result.outerShell = std::move(read.surfaces.front());
    }
    compareRelationships(reader, document.at("manifest"), pins, nullptr, references.relationships, result);
    return result;
}

FitInputChanges readFitInputChanges(const StoreReader& reader, const Json& document,
                                    const std::set<std::string>& objectUuids,
                                    const std::set<std::string>& relationshipUuids)
{
    FitInputChanges changes;
    vc::objects::ManifestReferences references;
    if (!sameStore(reader, document, changes.inputs, references)) return changes;
    const auto pins = objectPins(document,references);
    for (const auto& pin : pins)
        if (objectUuids.contains(pin.uuid)) readObject(reader, pin, changes.inputs.volumeUuid, changes.inputs);
    // A changed object's links: the fitted ones, and the ones it has now.
    changes.checkedRelationships = relationshipUuids;
    for (const auto& entry : document.at("manifest").value("relationships", Json::array()))
        if (objectUuids.contains(entry.value("a", std::string()))
            || objectUuids.contains(entry.value("b", std::string())))
            changes.checkedRelationships.insert(entry.at("uuid").get<std::string>());
    vc::objects::EdgeSelection incident;
    incident.endpointUuids = std::vector<std::string>();
    for (const auto& pin : pins)
        if (objectUuids.contains(pin.uuid)) incident.endpointUuids->push_back(pin.uuid);
    incident.includeRetired = true;
    if (!incident.endpointUuids->empty())
        for (const auto& edge : reader.selectEdges(incident, std::vector<std::string>{"uuid"}))
            changes.checkedRelationships.insert(edge.uuid);
    compareRelationships(reader, document.at("manifest"), pins, &changes.checkedRelationships, references.relationships, changes.inputs);
    return changes;
}

void applyFitInputChanges(FitInputs& inputs, FitInputChanges changes)
{
    // Every relationship a re-read object has now was compared again, so an
    // added one at a re-read endpoint that was not found again is gone.
    std::set<std::string> reread;
    for (const auto& entry : changes.inputs.surfaces) reread.insert(entry.pin.uuid);
    for (const auto& entry : changes.inputs.pointCollections) reread.insert(entry.pin.uuid);
    for (const auto& entry : changes.inputs.fibers) reread.insert(entry.pin.uuid);
    for (const auto& entry : changes.inputs.tracks) reread.insert(entry.pin.uuid);
    replaceByUuid(inputs.surfaces, changes.inputs.surfaces);
    replaceByUuid(inputs.pointCollections, changes.inputs.pointCollections);
    replaceByUuid(inputs.fibers, changes.inputs.fibers);
    replaceByUuid(inputs.tracks, changes.inputs.tracks);
    for (auto& link : changes.inputs.relationships) {
        const auto found = std::find_if(inputs.relationships.begin(), inputs.relationships.end(),
                                        [&](const auto& old) { return old.uuid == link.uuid; });
        if (found != inputs.relationships.end()) *found = std::move(link);
    }
    std::erase_if(inputs.addedRelationships, [&](const FitRelationship& link) {
        return changes.checkedRelationships.contains(link.uuid) || reread.contains(link.a)
            || reread.contains(link.b);
    });
    for (auto& link : changes.inputs.addedRelationships) inputs.addedRelationships.push_back(std::move(link));
    std::sort(inputs.addedRelationships.begin(), inputs.addedRelationships.end(),
              [](const auto& left, const auto& right) { return left.uuid < right.uuid; });
    // Store-level problems describe the latest read.
    inputs.problems = std::move(changes.inputs.problems);
}

} // namespace vc3d::spiral
