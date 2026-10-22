#include "vc_object_store/objects/FiberEdit.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <numeric>
#include <set>

namespace vc::objects {
RevisionSelection publishFiberEdits(ObjectStoreSession& connection, const AttachmentIdentity& attachment,
    std::vector<FiberEdit>& requestedEdits, const std::vector<FiberEdit>& retired,
    const std::vector<FiberEdit>& endpoints)
{
    using Json = nlohmann::json;
    auto edits = requestedEdits;
    RevisionSelection expected;
    std::map<std::string, EdgeRecord> incident;
    std::map<std::string, FiberEdit*> changed;
    for (auto& snapshot : edits) changed[snapshot.uuid] = &snapshot;
    auto expect = [&](const FiberEdit& fiber) {
        if (fiber.storePin && fiber.storePin->attachment != attachment)
            throw std::runtime_error("The fiber belongs to a previous object store attachment");
        expected[fiber.uuid] = fiber.storePin && !fiber.storePin->objectRevision.empty() ? std::optional(fiber.storePin->objectRevision) : std::nullopt;
    };
    for (const auto& snapshot : edits) expect(snapshot);
    for (const auto& fiber : retired) expect(fiber);
    auto findFiber = [&](const std::string& id) -> const FiberEdit& {
        if (changed.contains(id)) return *changed.at(id);
        const auto found = std::find_if(endpoints.begin(), endpoints.end(), [&](const auto& fiber) { return fiber.uuid == id; });
        if (found == endpoints.end()) throw std::runtime_error("Branch endpoint no longer exists");
        return *found;
    };
    for (const auto& snapshot : edits)
        for (const auto& branch : snapshot.branches) expect(findFiber(branch.targetUuid));
    connection.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto requested = expected;
        for (const auto& [uuid, pin] : requested) {
            const auto head = db.currentObjectRevision(txn, uuid);
            if ((head ? std::optional(head->uuid) : std::nullopt) != pin)
                throw std::runtime_error("Fiber revision changed; refresh before editing");
            if (!head) continue;
            for (const auto& edge : db.edgesOf(txn, uuid, true)) {
                incident[edge.uuid] = edge;
                for (const auto& endpoint : {edge.aUuid, edge.bUuid}) {
                    if (expected.contains(endpoint)) continue;
                    const auto cached = std::find_if(endpoints.begin(), endpoints.end(), [&](const auto& f) { return f.uuid == endpoint; });
                    if (cached != endpoints.end()) expect(*cached);
                    else expected[endpoint] = db.currentObjectRevision(txn, endpoint)->uuid;
                }
            }
        }
    });
    std::map<std::string, std::string> geometryHeads;
    const auto committed = connection.act(attachment, expected, [&](auto& db, auto& txn) {
        auto decisions = [&](const auto& affected) {
            std::vector<GeometryEdgeEdit> result;
            for (const auto& edge : affected) {
                GeometryEdgeEdit edit; edit.edgeUuid = edge.uuid; edit.expectedGeneration = edge.generation;
                edit.action = "keep_recorded";
                if (auto r = db.currentGeometry(txn, edge.aUuid)) edit.aRevision = r->uuid;
                if (auto r = db.currentGeometry(txn, edge.bUuid)) edit.bRevision = r->uuid;
                result.push_back(std::move(edit));
            }
            return result;
        };
        for (auto& snapshot : edits) {
            auto& fiber = snapshot;
            FiberDocument document;
            document.object = fiber.record; document.source = fiber.sourceAnnotations;
            vc::geometry::Fiber geometry;
            for (const auto& cp : fiber.controlPoints) geometry.controlPoints.push_back(cp.position);
            geometry.linePoints = fiber.linePoints;
            const auto current = db.objectByUuid(txn, fiber.uuid);
            auto record = current.value_or(document.object);
            record.uuid = fiber.uuid; record.kind = "fiber";
            record.name = fiber.name; record.tags = fiber.tags;
            record.formatVersion = document.object.formatVersion; record.fiber = document.object.fiber;
            auto meta = Json::parse(record.meta);
            if (!current) {
                if (!retired.empty()) {
                    std::optional<Json> coordinates;
                    for (const auto& original : retired) {
                        const auto parent = db.objectRevision(txn, original.storePin->objectRevision);
                        if (!parent) throw std::runtime_error("Structural source revision missing");
                        const auto parentMeta = Json::parse(parent->record.meta);
                        const auto frame = parentMeta.value("coordinates",Json::object());
                        if (coordinates && *coordinates != frame)
                            throw std::runtime_error("Cannot combine fibers from different coordinate frames");
                        coordinates = frame;
                        meta["coordinates"] = frame;
                        if (parentMeta.contains("coordinate_binding")) meta["coordinate_binding"] = parentMeta["coordinate_binding"];
                    }
                } else {
                    for (const char* key : {"coordinates","coordinate_binding"})
                        if (const auto identity = Json::parse(snapshot.coordinateIdentity); identity.contains(key)) meta[key]=identity[key];
                }
            }
            Json identity = Json::object();
            for (const auto* key : {"coordinates", "coordinate_binding"})
                if (meta.contains(key)) identity[key] = meta.at(key);
            fiber.coordinateIdentity = identity.dump();
            auto source = meta.value("source", Json::object());
            auto incoming = Json::parse(document.source);
            incoming.erase("branches"); incoming.erase("adjacent_branches");
            incoming.erase("coordinates"); incoming.erase("coordinate_binding");
            source.erase("control_points_attributes"); source.erase("line_points_attributes");
            source.update(incoming);
            source.erase("branches"); source.erase("adjacent_branches");
            int64_t nextId = source.value("next_control_point_id", int64_t(0));
            for (const auto& cp : fiber.controlPoints) nextId = std::max(nextId, cp.storeId + 1);
            std::vector<int64_t> sourcePointIds;
            for(const auto& cp:fiber.controlPoints) sourcePointIds.push_back(cp.storeId);
            std::set<int64_t> ids;
            for (auto& cp : fiber.controlPoints) {
                if (cp.storeId < 0 || ids.contains(cp.storeId)) cp.storeId = nextId++;
                ids.insert(cp.storeId); geometry.controlPointIds.push_back(cp.storeId);
            }
            source["next_control_point_id"] = nextId;
            document.geometry = vc::geometry::encode(geometry);
            auto head = current ? db.currentGeometry(txn, fiber.uuid) : std::nullopt;
            if (!head) {
                if (retired.empty()) {
                    meta["source"] = source; record.meta = meta.dump();
                    db.upsertObject(txn, record);
                    geometryHeads[fiber.uuid] = db.putGeometry(txn, fiber.uuid, document.geometry);
                } else {
                    // Split/merge reuses immutable source slices, including stable
                    // control identities. Only genuinely new rows carry payloads.
                    auto baseSource = source;
                    baseSource.erase("control_points_attributes"); baseSource.erase("line_points_attributes");
                    meta["source"] = baseSource; record.meta = meta.dump(); db.upsertObject(txn,record);
                    auto revision = db.putGeometry(txn,fiber.uuid,vc::geometry::encode(vc::geometry::Fiber{}));
                    Json origins = Json::array();
                    for (const auto* name : {"control_points", "line_points"}) {
                        const bool controls = std::string_view(name)=="control_points";
                        const auto& desired = *std::find_if(document.geometry.arrays.begin(),document.geometry.arrays.end(),[&](const auto& a) { return a.name==name; });
                        GeometrySequenceEdit edit; edit.array=name;
                        edit.transformedAnnotations = source.value(std::string(name)+"_attributes",Json(std::vector<Json>(desired.shape[0],Json::object()))).dump();
                        for (int64_t i=0;i<desired.shape[0];++i) {
                            GeometrySequencePart part; part.count=1;
                            bool matched = false;
                            for (const auto& original : retired) {
                                std::vector<std::array<double,3>> points;
                                if (controls) for (const auto& cp : original.controlPoints) points.push_back(cp.position);
                                else points = original.linePoints;
                                const auto& point = controls ? geometry.controlPoints[i] : geometry.linePoints[i];
                                for (size_t j=0;j<points.size();++j) {
                                    if (points[j] != point) continue;
                                    if (controls && original.controlPoints[j].storeId != sourcePointIds[i]) continue;
                                    matched = true;
                                    if(controls) origins.push_back({{"id",geometry.controlPointIds[i]},{"object_uuid",original.uuid},{"source_id",original.controlPoints[j].storeId}});
                                    if(!controls || original.controlPoints[j].storeId==geometry.controlPointIds[i]) {
                                        part.revision=original.storePin->geometryRevision;part.start=j;
                                    }
                                    break;
                                }
                                if (matched) break;
                            }
                            if (!part.revision) {
                                part.bytes.assign(desired.bytes.begin()+i*24,desired.bytes.begin()+(i+1)*24);
                                if (controls) part.ids={geometry.controlPointIds[i]};
                            }
                            edit.parts.push_back(std::move(part));
                        }
                        revision=db.editGeometrySequence(txn,fiber.uuid,edit,revision);
                    }
                    if (!origins.empty()) db.setMeta(txn,fiber.uuid,"control_point_origins",origins.dump());
                    geometryHeads[fiber.uuid]=revision;
                }
            } else {
                auto data = db.readGeometryData(txn, head->uuid);
                auto original = std::get<vc::geometry::Fiber>(vc::geometry::decode(data));
                if (original.controlPointIds.empty()) {
                    original.controlPointIds.resize(original.controlPoints.size());
                    std::iota(original.controlPointIds.begin(), original.controlPointIds.end(), int64_t(0));
                }
                for (const auto* name : {"control_points", "line_points"}) {
                    const bool controls = std::string_view(name) == "control_points";
                    auto desired = *std::find_if(document.geometry.arrays.begin(), document.geometry.arrays.end(), [&](const auto& a) { return a.name == name; });
                    auto old = *std::find_if(data.arrays.begin(), data.arrays.end(), [&](const auto& a) { return a.name == name; });
                    auto attrs = source.value(std::string(name)+"_attributes", Json(std::vector<Json>(desired.shape[0],Json::object())));
                    const bool structure = desired.shape != old.shape || (controls && (geometry.controlPointIds != original.controlPointIds || Json::parse(db.geometryPointAnnotations(txn, head->uuid, name)) != attrs));
                    if (structure) {
                        GeometrySequenceEdit edit; edit.array = name; edit.eraseCount = old.shape[0]; edit.transformedAnnotations = attrs.dump();
                        vc::geometry::Fiber proposed = std::get<vc::geometry::Fiber>(vc::geometry::decode(data));
                        if (controls) {
                            proposed.controlPoints = geometry.controlPoints; proposed.controlPointIds = geometry.controlPointIds;
                            for (size_t i=0;i<geometry.controlPoints.size();++i) {
                                GeometrySequencePart part; part.count = 1;
                                const auto found = std::find(original.controlPointIds.begin(),original.controlPointIds.end(),geometry.controlPointIds[i]);
                                if (found != original.controlPointIds.end()) {
                                    part.revision = head->uuid; part.start = found-original.controlPointIds.begin();
                                    proposed.controlPoints[i] = original.controlPoints[part.start];
                                } else {
                                    part.bytes.assign(desired.bytes.begin()+i*24,desired.bytes.begin()+(i+1)*24);
                                    part.ids = {geometry.controlPointIds[i]};
                                }
                                edit.parts.push_back(std::move(part));
                            }
                        } else {
                            proposed.linePoints = geometry.linePoints;
                            GeometrySequencePart part; part.count = desired.shape[0]; part.bytes = desired.bytes;
                            edit.parts.push_back(std::move(part));
                        }
                        const auto affected = db.affectedGeometryEdges(txn, fiber.uuid, vc::geometry::encode(proposed));
                        const auto revision = db.editGeometrySequence(txn, fiber.uuid, edit, head->uuid, decisions(affected));
                        head = db.geometryRevision(txn, revision); data = db.readGeometryData(txn, head->uuid);
                        old = *std::find_if(data.arrays.begin(), data.arrays.end(), [&](const auto& a) { return a.name == name; });
                    }
                    std::vector<GeometryArrayEdit> edits;
                    for (size_t offset=0; offset<desired.bytes.size(); offset+=24) {
                        if (std::equal(desired.bytes.begin()+offset,desired.bytes.begin()+offset+24,old.bytes.begin()+offset)) continue;
                        edits.push_back({name,int64_t(offset/8), {desired.bytes.begin()+offset,desired.bytes.begin()+offset+24}});
                    }
                    if (!edits.empty()) {
                        const auto affected = db.affectedGeometryEditEdges(txn, fiber.uuid, edits, head->uuid);
                        const auto revision = db.editGeometry(txn, fiber.uuid, edits, head->uuid, decisions(affected));
                        head = db.geometryRevision(txn, revision); data = db.readGeometryData(txn, head->uuid);
                    }
                }
                geometryHeads[fiber.uuid] = head->uuid;
                // Sequence annotations are immutable and live with geometry.
                source.erase("control_points_attributes"); source.erase("line_points_attributes");
                // Plain (pre-sequence) arrays still use source annotations.
                for (const auto* name : {"control_points", "line_points"})
                    source[std::string(name)+"_attributes"] = Json::parse(db.geometryPointAnnotations(txn, head->uuid, name));
                meta["source"] = source; record.meta = meta.dump(); db.upsertObject(txn, record);
            }
        }
        std::set<std::string> retiredUuids;
        for (const auto& fiber : retired) retiredUuids.insert(fiber.uuid);
        std::set<std::string> retainedBranches;
        for (auto& snapshot : edits) {
            auto& local = snapshot;
            for (auto& branch : local.branches) {
                const auto& target = findFiber(branch.targetUuid);
                if (retiredUuids.contains(target.uuid)) throw std::runtime_error("Branch still references a retired fiber");
                auto anchor = [&](const auto& fiber, int index, const std::array<double,3>& direction) {
                    Anchor a;
                    const auto& cp = fiber.controlPoints.at(index);
                    a.position = {cp[0],cp[1],cp[2]}; a.direction = {direction[0],direction[1],direction[2]};
                    const auto geometry = db.currentGeometry(txn, fiber.uuid);
                    const bool hasIds = geometry && std::any_of(geometry->arrays.begin(),geometry->arrays.end(),[](const auto& array) { return array.name == "control_point_ids"; });
                    a.points = {{hasIds ? "control_point_ids" : "control_points", {hasIds ? cp.storeId : int64_t(index)}}}; return a;
                };
                EdgeRecord edge; edge.uuid = branch.edgeUuid; edge.kind = "fiber_branch";
                edge.aUuid = local.uuid; edge.bUuid = target.uuid;
                edge.aAnchor = anchor(local,branch.controlPointIndex,branch.controlPointDirection);
                edge.bAnchor = anchor(target,branch.branchControlPointIndex,branch.branchControlPointDirection);
                edge.meta = Json{{"adjacent",branch.adjacent}}.dump();
                // Reciprocal UI references describe one shared edge.
                for (const auto& existing : db.edgesOf(txn,local.uuid)) {
                    const bool forward = existing.aUuid == edge.aUuid && existing.bUuid == edge.bUuid;
                    const bool reverse = existing.bUuid == edge.aUuid && existing.aUuid == edge.bUuid;
                    if (existing.kind != "fiber_branch" || (!forward && !reverse)) continue;
                    if ((!edge.uuid.empty() && edge.uuid == existing.uuid) ||
                        ((forward ? existing.aAnchor.points == edge.aAnchor.points && existing.bAnchor.points == edge.bAnchor.points : existing.bAnchor.points == edge.aAnchor.points && existing.aAnchor.points == edge.bAnchor.points) &&
                         Json::parse(existing.meta).value("adjacent",false) == branch.adjacent)) {
                        edge.uuid = existing.uuid; edge.tags = existing.tags;
                        auto metadata = Json::parse(existing.meta); metadata["adjacent"] = branch.adjacent; edge.meta = metadata.dump(); break;
                    }
                }
                if (!edge.uuid.empty()) {
                    const auto old = db.edgeByUuid(txn, edge.uuid);
                    if (old && !((old->aUuid == edge.aUuid && old->bUuid == edge.bUuid) ||
                                 (old->bUuid == edge.aUuid && old->aUuid == edge.bUuid))) edge.uuid.clear();
                }
                if (!edge.uuid.empty() && db.edgeByUuid(txn,edge.uuid)) db.updateEdge(txn,edge);
                else edge.uuid = db.addEdge(txn,edge);
                db.retainRelationship(txn,edge.uuid);
                const auto approval = branch.pending ? "pending" : "approved";
                if (db.edgeByUuid(txn,edge.uuid)->approvalState != approval) db.setApproval(txn,edge.uuid,approval);
                retainedBranches.insert(edge.uuid); branch.edgeUuid = edge.uuid;
            }
        }
        std::set<std::string> edited;
        for (const auto& snapshot : edits) edited.insert(snapshot.uuid);
        for (const auto& [uuid,edge] : incident) {
            if (edge.kind != "fiber_branch") { db.keepRecordedRelationship(txn,uuid); continue; }
            if ((edited.contains(edge.aUuid) || edited.contains(edge.bUuid) || retiredUuids.contains(edge.aUuid) || retiredUuids.contains(edge.bUuid)) && !retainedBranches.contains(uuid)) db.removeEdge(txn,uuid);
        }
        for (const auto& fiber : retired) db.retireObject(txn,fiber.uuid);
    });
    for (auto& snapshot : edits) {
        auto& fiber = snapshot;
        fiber.storePin = PinnedObject{attachment,{attachment.storeUuid,fiber.uuid},*committed.at(fiber.uuid),geometryHeads.at(fiber.uuid)};
    }
    requestedEdits = std::move(edits);
    return committed;
}
}
