#include "vc_object_store/objects/GeometryEdits.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/SequenceIdentity.hpp"
#include "geometry/ArrayLayout.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include "geometry/SharedGeometry.hpp"
#include "internal/History.hpp"
#include <nlohmann/json.hpp>
#include <limits>
#include <optional>
#include <set>

namespace vc::objects {
namespace {
using Json = nlohmann::json;
const GeometryArray* findArray(const GeometryData& data, const std::string& name)
{
    const auto it = std::find_if(data.arrays.begin(), data.arrays.end(), [&](const auto& a) { return a.name == name; });
    return it == data.arrays.end() ? nullptr : &*it;
}
std::vector<int64_t> pointIds(const GeometryData& data, const std::string& name, size_t count)
{
    std::vector<int64_t> ids(count);
    if (const auto* array = findArray(data, name)) detail::copyScalars<int64_t>(array->bytes, ids);
    else throw std::invalid_argument("sequence requires stable IDs: " + name);
    std::set<int64_t> unique;
    for (auto id : ids) if (id < 0 || !unique.insert(id).second)
        throw std::invalid_argument("sequence IDs must be nonnegative and unique");
    return ids;
}
}
std::string updateGeometryArrays(ObjectStore& db, WriteTxn& txn, const std::string& uuid,
    const GeometryData& desired, const std::string& expected,
    const std::map<std::string, std::string>& annotations,
    const std::function<std::vector<GeometryEdgeEdit>(const std::vector<EdgeRecord>&)>& decide)
{
    auto* connection = ObjectStoreImplAccess::conn(txn);
    detail::Savepoint savepoint(connection);
    try {
        // Even a no-op must check the caller's pin.
        auto head = db.editGeometry(txn, uuid, {}, expected);
        const auto revision = *db.geometryRevision(txn, head);
        (void)detail::geometrySchema(desired);
        if (revision.kind != desired.kind || revision.formatVersion != desired.formatVersion ||
            revision.coordinateSystem != desired.coordinateSystem || Json::parse(revision.attributes) != Json::parse(desired.attributes))
            throw std::invalid_argument("array updates require unchanged geometry schema and attributes");
        // Sequence edits (fibers, point collections, tracks) work on whole arrays.
        // Other arrays, including every surface array, are diffed in bounded
        // windows so surfaces of any size update without a full read.
        std::optional<GeometryData> whole;
        auto sequenceData = [&]() -> GeometryData& {
            if (!whole) whole = db.readGeometryData(txn, head, std::numeric_limits<uint64_t>::max());
            return *whole;
        };
        std::set<std::string> identityArrays;
        for (const auto& array : desired.arrays) {
            const auto name = detail::sequenceIdArray(desired.kind, array.name);
            if (!name.empty()) identityArrays.insert(name);
        }
        for (const auto& array : revision.arrays)
            if (!findArray(desired, array.name))
                throw std::invalid_argument("array updates cannot remove arrays");
        for (const auto& [name, value] : annotations)
            if (!findArray(desired, name) || (desired.kind == "fiber" ? name != "control_points" && name != "line_points"
                : desired.kind == "surface" || name != "points"))
                throw std::invalid_argument("invalid sequence annotations: " + name);
        auto decisions = [&](const std::vector<EdgeRecord>& edges) {
            return decide ? decide(edges) : std::vector<GeometryEdgeEdit>{};
        };
        for (const auto& target : desired.arrays) {
            if (identityArrays.contains(target.name)) continue;
            const auto previous = std::find_if(revision.arrays.begin(), revision.arrays.end(),
                [&](const auto& a) { return a.name == target.name; });
            if (previous == revision.arrays.end() || previous->dtype != target.dtype)
                throw std::invalid_argument("array updates cannot change array schema");
            const auto scalarBytes = detail::geometryByteLength(target.dtype, {1});
            const auto idsName = detail::sequenceIdArray(desired.kind, target.name);
            const bool sequence = desired.kind == "fiber" ? target.name == "control_points" || target.name == "line_points"
                : desired.kind != "surface" && target.name == "points";
            if (sequence) {
                auto& data = sequenceData();
                const auto old = *findArray(data, target.name);
                const size_t count = target.shape.at(0), oldCount = old.shape.at(0);
                const auto oldIds = idsName.empty() ? std::vector<int64_t>{} : pointIds(data, idsName, oldCount);
                const auto newIds = idsName.empty() ? std::vector<int64_t>{} : pointIds(desired, idsName, count);
                const auto oldAttrs = Json::parse(db.geometryPointAnnotations(txn, head, target.name));
                auto attrs = Json(std::vector<Json>(count, Json::object()));
                std::map<int64_t, size_t> rows;
                for (size_t i = 0; i < oldIds.size(); ++i) rows.emplace(oldIds[i], i);
                if (annotations.contains(target.name)) attrs = Json::parse(annotations.at(target.name));
                else if (!idsName.empty()) {
                    for (size_t i = 0; i < count; ++i)
                        if (rows.contains(newIds[i])) attrs[i] = oldAttrs.at(rows.at(newIds[i]));
                } else if (count == oldCount) attrs = oldAttrs;
                else if (std::any_of(oldAttrs.begin(), oldAttrs.end(), [](const auto& a) { return !a.is_null() && !a.empty(); }))
                    throw std::invalid_argument("resizing an annotated sequence requires transformed annotations");
                const bool structure = target.shape != old.shape || oldIds != newIds || oldAttrs != attrs;
                if (structure) {
                    GeometrySequenceEdit edit; edit.array = target.name; edit.eraseCount = oldCount;
                    if (annotations.contains(target.name)) edit.transformedAnnotations = attrs.dump();
                    auto proposed = data;
                    auto& proposedArray = *std::find_if(proposed.arrays.begin(), proposed.arrays.end(), [&](const auto& a) { return a.name == target.name; });
                    proposedArray = target;
                    const size_t stride = scalarBytes * 3;
                    if (!idsName.empty()) {
                        // Retained points reference the head revision; a run of
                        // consecutive retained rows is one reference part, as the
                        // structural edit reads every reference part's source
                        // arrays and annotations in full (one part per point made
                        // an insert into an n-point sequence cost n such reads).
                        for (size_t i = 0; i < count;) {
                            GeometrySequencePart part;
                            if (const auto it = rows.find(newIds[i]); it != rows.end()) {
                                part.revision = head; part.start = it->second;
                                size_t run = 0;
                                for (auto row = it->second; i + run < count; ++run, ++row) {
                                    const auto next = rows.find(newIds[i + run]);
                                    if (next == rows.end() || next->second != row) break;
                                    std::copy_n(old.bytes.begin() + row * stride, stride, proposedArray.bytes.begin() + (i + run) * stride);
                                }
                                part.count = run; i += run;
                            } else {
                                part.count = 1;
                                part.bytes.assign(target.bytes.begin() + i * stride, target.bytes.begin() + (i + 1) * stride);
                                part.ids = {newIds[i]};
                                ++i;
                            }
                            edit.parts.push_back(std::move(part));
                        }
                        std::erase_if(proposed.arrays, [&](const auto& a) { return a.name == idsName; });
                        proposed.arrays.push_back(detail::scalarArray<int64_t>(idsName, "<i8", {int64_t(count)}, newIds));
                    } else {
                        GeometrySequencePart part; part.count = count; part.bytes = target.bytes;
                        edit.parts.push_back(std::move(part));
                    }
                    std::optional<std::pair<std::string, std::vector<int64_t>>> changedIds;
                    if (!idsName.empty()) {
                        // Every erased and every resulting ID, as the structural edit records them.
                        changedIds.emplace(idsName, oldIds);
                        changedIds->second.insert(changedIds->second.end(), newIds.begin(), newIds.end());
                    }
                    const auto affected = detail::structuralSnapshotEdges(db, txn, uuid, proposed, changedIds);
                    head = db.editGeometrySequence(txn, uuid, edit, head, decisions(affected));
                    whole.reset();
                }
            } else if (target.shape != previous->shape) {
                throw std::invalid_argument("array updates cannot resize a grid or auxiliary array");
            }
            std::vector<GeometryArrayEdit> edits;
            // Coalesce consecutive changed scalars, preserving exact encoded bytes.
            // Runs continue across read windows.
            std::optional<size_t> run;
            auto close = [&](size_t end) {
                edits.push_back({target.name, int64_t(*run / scalarBytes), {target.bytes.begin() + *run, target.bytes.begin() + end}});
                run.reset();
            };
            for (size_t window = 0; window < target.bytes.size(); window += detail::geometryChunkSize) {
                const auto count = std::min(detail::geometryChunkSize, target.bytes.size() - window);
                const auto old = db.readGeometryArray(txn, head, target.name, int64_t(window), int64_t(count));
                if (old.size() != count) throw std::runtime_error("geometry: short array read");
                for (size_t offset = 0; offset < count; offset += scalarBytes) {
                    const bool same = std::equal(old.begin() + offset, old.begin() + offset + scalarBytes,
                                                 target.bytes.begin() + window + offset);
                    if (!same && !run) run = window + offset;
                    else if (same && run) close(window + offset);
                }
            }
            if (run) close(target.bytes.size());
            if (!edits.empty()) {
                head = db.editGeometry(txn, uuid, edits, head, ObjectStore::EdgeDecider(decisions));
                whole.reset();
            }
        }
        savepoint.release();
        return head;
    } catch (const std::exception& error) {
        detail::historyConflict(connection, error.what());
    }
}
}
