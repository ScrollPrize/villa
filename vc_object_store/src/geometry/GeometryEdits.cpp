#include "vc_object_store/objects/GeometryEdits.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "geometry/SequenceIdentity.hpp"
#include "geometry/ArrayLayout.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include <nlohmann/json.hpp>
#include <numeric>
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
    else std::iota(ids.begin(), ids.end(), int64_t(0));
    return ids;
}
}
std::string updateGeometryArrays(ObjectStore& db, WriteTxn& txn, const std::string& uuid,
    const GeometryData& desired, const std::string& expected,
    const std::map<std::string, std::string>& annotations,
    const std::function<std::vector<GeometryEdgeEdit>(const std::vector<EdgeRecord>&)>& decide)
{
    // Even a no-op must check the caller's pin.
    auto head = db.editGeometry(txn, uuid, {}, expected);
    auto data = db.readGeometryData(txn, head);
    (void)vc::geometry::decode(desired);
    if (data.kind != desired.kind || data.formatVersion != desired.formatVersion ||
        data.coordinateSystem != desired.coordinateSystem || Json::parse(data.attributes) != Json::parse(desired.attributes))
        throw std::invalid_argument("array updates require unchanged geometry schema and attributes");
    std::set<std::string> identityArrays;
    for (const auto& array : desired.arrays) {
        const auto name = detail::sequenceIdArray(desired.kind, array.name);
        if (!name.empty()) identityArrays.insert(name);
    }
    for (const auto& array : data.arrays)
        if (!findArray(desired, array.name))
            throw std::invalid_argument("array updates cannot remove arrays");
    for (const auto& [name, value] : annotations)
        if (!findArray(desired, name) || desired.kind == "patch" || identityArrays.contains(name))
            throw std::invalid_argument("invalid sequence annotations: " + name);
    auto decisions = [&](const std::vector<EdgeRecord>& edges) {
        return decide ? decide(edges) : std::vector<GeometryEdgeEdit>{};
    };
    for (const auto& target : desired.arrays) {
        if (identityArrays.contains(target.name)) continue;
        const auto* previous = findArray(data, target.name);
        if (!previous || previous->dtype != target.dtype)
            throw std::invalid_argument("array updates cannot change array schema");
        auto old = *previous;
        const auto scalarBytes = detail::geometryByteLength(target.dtype, {1});
        const auto idsName = detail::sequenceIdArray(desired.kind, target.name);
        if (desired.kind != "patch") {
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
            const bool structure = target.shape != old.shape || oldIds != newIds || oldAttrs != attrs ||
                (!idsName.empty() && !findArray(data, idsName) && findArray(desired, idsName));
            if (structure) {
                GeometrySequenceEdit edit; edit.array = target.name; edit.eraseCount = oldCount;
                edit.transformedAnnotations = attrs.dump();
                auto proposed = data;
                auto& proposedArray = *std::find_if(proposed.arrays.begin(), proposed.arrays.end(), [&](const auto& a) { return a.name == target.name; });
                proposedArray = target;
                const size_t stride = scalarBytes * 3;
                if (!idsName.empty()) {
                    for (size_t i = 0; i < count; ++i) {
                        GeometrySequencePart part; part.count = 1;
                        if (const auto it = rows.find(newIds[i]); it != rows.end()) {
                            part.revision = head; part.start = it->second;
                            std::copy_n(old.bytes.begin() + it->second * stride, stride, proposedArray.bytes.begin() + i * stride);
                        } else {
                            part.bytes.assign(target.bytes.begin() + i * stride, target.bytes.begin() + (i + 1) * stride);
                            part.ids = {newIds[i]};
                        }
                        edit.parts.push_back(std::move(part));
                    }
                    std::erase_if(proposed.arrays, [&](const auto& a) { return a.name == idsName; });
                    proposed.arrays.push_back(detail::scalarArray<int64_t>(idsName, "<i8", {int64_t(count)}, newIds));
                } else {
                    GeometrySequencePart part; part.count = count; part.bytes = target.bytes;
                    edit.parts.push_back(std::move(part));
                }
                const auto affected = db.affectedGeometryEdges(txn, uuid, proposed);
                head = db.editGeometrySequence(txn, uuid, edit, head, decisions(affected));
                data = db.readGeometryData(txn, head);
                old = *findArray(data, target.name);
            }
        } else if (target.shape != old.shape) {
            throw std::invalid_argument("use surface resize operations to change a patch grid");
        }
        std::vector<GeometryArrayEdit> edits;
        // Coalesce consecutive changed scalars, preserving exact encoded bytes.
        for (size_t offset = 0; offset < target.bytes.size();) {
            if (std::equal(target.bytes.begin() + offset, target.bytes.begin() + offset + scalarBytes, old.bytes.begin() + offset)) {
                offset += scalarBytes; continue;
            }
            const auto start = offset;
            do { offset += scalarBytes; }
            while (offset < target.bytes.size() && !std::equal(target.bytes.begin() + offset,
                   target.bytes.begin() + offset + scalarBytes, old.bytes.begin() + offset));
            edits.push_back({target.name, int64_t(start / scalarBytes), {target.bytes.begin() + start, target.bytes.begin() + offset}});
        }
        if (!edits.empty()) {
            const auto affected = db.affectedGeometryEditEdges(txn, uuid, edits, head);
            head = db.editGeometry(txn, uuid, edits, head, decisions(affected));
            data = db.readGeometryData(txn, head);
        }
    }
    return head;
}
}
