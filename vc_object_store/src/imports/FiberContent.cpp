#include "vc_object_store/objects/FiberContent.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>

namespace vc::objects {
namespace {
using Json = nlohmann::json;

constexpr uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);

struct Hasher {
    uint64_t state = kFnvOffset;
    void byte(uint8_t b) { state ^= b; state *= kFnvPrime; }
    void u64(uint64_t v) { for (int i = 0; i < 8; ++i) byte(uint8_t(v >> (8 * i))); }
    void f64(double v) { u64(std::bit_cast<uint64_t>(v == 0.0 ? 0.0 : v)); } // -0 == +0
    void point(const geometry::Point& p) { for (double c : p) f64(c); }
    void text(const std::string& s) { u64(s.size()); for (unsigned char c : s) byte(c); }
};

const Json& annotationAt(const FiberPieceContent& content, size_t index)
{
    static const Json empty = Json::object();
    if (index < content.annotations.size() && content.annotations[index].is_object()) return content.annotations[index];
    return empty;
}
} // namespace

std::optional<size_t> FiberPieceContent::indexOf(int64_t id) const
{
    const auto it = std::find(controlIds.begin(), controlIds.end(), id);
    if (it == controlIds.end()) return std::nullopt;
    return size_t(it - controlIds.begin());
}

void FiberPieceContent::resolveLineIndices()
{
    controlLineIndex.assign(controlPoints.size(), -1);
    size_t cursor = 0;
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        for (size_t j = cursor; j < linePoints.size(); ++j) {
            if (linePoints[j] == controlPoints[i]) { controlLineIndex[i] = int64_t(j); cursor = j; break; }
        }
    }
}

FiberPieceContent fiberPieceContent(const geometry::Fiber& fiber, const Json& controlAnnotations)
{
    FiberPieceContent content;
    content.controlIds = fiber.controlPointIds;
    content.controlPoints = fiber.controlPoints;
    content.linePoints = fiber.linePoints;
    if (content.controlIds.size() != content.controlPoints.size())
        throw std::invalid_argument("fiber content: control points without stable ids");
    content.annotations.assign(content.controlPoints.size(), Json::object());
    if (controlAnnotations.is_array())
        for (size_t i = 0; i < std::min(controlAnnotations.size(), content.annotations.size()); ++i)
            if (controlAnnotations[i].is_object()) content.annotations[i] = controlAnnotations[i];
    content.resolveLineIndices();
    return content;
}

FiberPieceContent fiberPieceContent(const StoreReader& reader, const ObjectRevision& revision)
{
    GeometrySnapshot snapshot(reader);
    const auto fiber = snapshot.fiber(revision).load();
    Json annotations = Json::array();
    if (revision.geometryRevision) annotations = reader.geometryPointAnnotations(*revision.geometryRevision, "control_points");
    return fiberPieceContent(fiber, annotations);
}

uint64_t fiberSpanHash(const FiberPieceContent& content, size_t lower)
{
    if (lower + 1 >= content.controlPoints.size()) throw std::out_of_range("fiber content: span index");
    const bool forward = content.controlIds[lower] < content.controlIds[lower + 1];
    Hasher hasher;
    const int64_t from = content.controlLineIndex.at(lower), to = content.controlLineIndex.at(lower + 1);
    if (from >= 0 && to >= from) {
        hasher.u64(uint64_t(to - from + 1));
        if (forward) for (int64_t i = from; i <= to; ++i) hasher.point(content.linePoints[size_t(i)]);
        else for (int64_t i = to; i >= from; --i) hasher.point(content.linePoints[size_t(i)]);
    } else {
        hasher.u64(0);
        hasher.point(content.controlPoints[forward ? lower : lower + 1]);
        hasher.point(content.controlPoints[forward ? lower + 1 : lower]);
    }
    const auto& annotation = annotationAt(content, lower);
    const auto segment = annotation.find("segment_to_next");
    hasher.text(segment != annotation.end() && segment->is_object() ? segment->dump() : std::string());
    return hasher.state;
}
} // namespace vc::objects
