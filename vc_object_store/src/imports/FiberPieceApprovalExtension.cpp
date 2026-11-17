#include "vc_object_store/objects/FiberPieceApprovalExtension.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include "vc_object_store/Uuid.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace vc::objects {
namespace {
using Type = ExtensionType;
using Json = nlohmann::json;
ExtensionColumn required(const char* name, Type type) { return {name, type, false}; }
ExtensionColumn optional(const char* name, Type type) { return {name, type, true}; }

constexpr uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);

struct Hasher {
    uint64_t state = kFnvOffset;
    void byte(uint8_t b) { state ^= b; state *= kFnvPrime; }
    void u64(uint64_t v) { for (int i = 0; i < 8; ++i) byte(uint8_t(v >> (8 * i))); }
    void i64(int64_t v) { u64(uint64_t(v)); }
    void f64(double v) { u64(std::bit_cast<uint64_t>(v == 0.0 ? 0.0 : v)); } // -0 == +0
    void point(const geometry::Point& p) { for (double c : p) f64(c); }
    void text(const std::string& s) { u64(s.size()); for (unsigned char c : s) byte(c); }
};

void putU64(std::vector<uint8_t>& out, uint64_t v) { for (int i = 0; i < 8; ++i) out.push_back(uint8_t(v >> (8 * i))); }
uint64_t getU64(std::span<const uint8_t> in, size_t at)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= uint64_t(in[at + size_t(i)]) << (8 * i);
    return v;
}

std::optional<std::array<double, 3>> vector3(const Json& annotation, const char* key)
{
    const auto it = annotation.find(key);
    if (it == annotation.end() || !it->is_array() || it->size() != 3) return std::nullopt;
    std::array<double, 3> v{};
    for (size_t i = 0; i < 3; ++i) {
        if (!(*it)[i].is_number()) return std::nullopt;
        v[i] = (*it)[i].get<double>();
        if (!std::isfinite(v[i])) return std::nullopt;
    }
    return v;
}

std::vector<std::string> controlTags(const Json& annotation)
{
    std::vector<std::string> tags;
    const auto it = annotation.find("tags");
    if (it != annotation.end() && it->is_array())
        for (const auto& tag : *it) if (tag.is_string()) tags.push_back(tag.get<std::string>());
    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    return tags;
}

const Json& annotationAt(const FiberPieceContent& content, size_t index)
{
    static const Json empty = Json::object();
    if (index < content.annotations.size() && content.annotations[index].is_object()) return content.annotations[index];
    return empty;
}

std::string blobText(const ExtensionValue& value, const char* what)
{
    const auto* blob = std::get_if<std::vector<uint8_t>>(&value);
    if (!blob || blob->size() != 16) throw std::invalid_argument(std::string("fiber piece approval: invalid ") + what);
    return extensionUuid(value);
}
std::optional<std::string> optionalBlobText(const ExtensionValue& value, const char* what)
{
    if (std::holds_alternative<std::monostate>(value)) return std::nullopt;
    return blobText(value, what);
}
int64_t integer(const ExtensionValue& value, const char* what)
{
    const auto* v = std::get_if<int64_t>(&value);
    if (!v) throw std::invalid_argument(std::string("fiber piece approval: invalid ") + what);
    return *v;
}
std::optional<int64_t> optionalInteger(const ExtensionValue& value, const char* what)
{
    if (std::holds_alternative<std::monostate>(value)) return std::nullopt;
    return integer(value, what);
}
std::string text(const ExtensionValue& value, const char* what)
{
    const auto* v = std::get_if<std::string>(&value);
    if (!v) throw std::invalid_argument(std::string("fiber piece approval: invalid ") + what);
    return *v;
}
std::optional<std::string> optionalText(const ExtensionValue& value, const char* what)
{
    if (std::holds_alternative<std::monostate>(value)) return std::nullopt;
    return text(value, what);
}
const std::vector<uint8_t>& blob(const ExtensionValue& value, const char* what)
{
    const auto* v = std::get_if<std::vector<uint8_t>>(&value);
    if (!v) throw std::invalid_argument(std::string("fiber piece approval: invalid ") + what);
    return *v;
}
ExtensionValue optionalValue(const std::optional<std::string>& v, bool uuid)
{
    if (!v) return std::monostate{};
    return uuid ? extensionUuid(*v) : ExtensionValue(*v);
}

void validateApproval(const FiberPieceApproval& row)
{
    if (row.state != FiberPieceState::Approved && row.state != FiberPieceState::Revoked)
        throw std::invalid_argument("fiber piece approval: invalid state");
    if (row.aspect != FiberPieceAspect::Geometry && row.aspect != FiberPieceAspect::Normals)
        throw std::invalid_argument("fiber piece approval: invalid aspect");
    if (row.lockCpB && *row.lockCpB == row.lockCpA)
        throw std::invalid_argument("fiber piece approval: lock control points coincide");
    if (row.insideCp == row.lockCpA) throw std::invalid_argument("fiber piece approval: inside control is the lock");
    if (row.fiberVersion < 1) throw std::invalid_argument("fiber piece approval: invalid fiber version");
    if (row.author.empty() || row.approvedAt.empty()) throw std::invalid_argument("fiber piece approval: author and time required");
    if (row.state == FiberPieceState::Revoked) {
        if (!row.supersedes) throw std::invalid_argument("fiber piece approval: revocation must supersede a row");
        if (!row.spans.empty() || !row.controls.empty())
            throw std::invalid_argument("fiber piece approval: revocation carries no fingerprints");
    } else {
        if (row.controls.size() < 2) throw std::invalid_argument("fiber piece approval: a piece covers at least one span");
        if (row.aspect == FiberPieceAspect::Geometry && row.spans.size() + 1 != row.controls.size())
            throw std::invalid_argument("fiber piece approval: span and control fingerprints disagree");
        if (row.aspect == FiberPieceAspect::Normals && !row.spans.empty())
            throw std::invalid_argument("fiber piece approval: normals carry no span fingerprints");
        if (row.controls.front().id != row.lockCpA || row.controls[1].id != row.insideCp)
            throw std::invalid_argument("fiber piece approval: fingerprints must walk from the lock through the inside control");
        if (row.lockCpB && row.controls.back().id != *row.lockCpB)
            throw std::invalid_argument("fiber piece approval: fingerprints must end at the second lock");
    }
    if (row.supersedes && *row.supersedes == row.rowUuid) throw std::invalid_argument("fiber piece approval: row supersedes itself");
}
} // namespace

ExtensionTable fiberPieceApprovalTable()
{
    ExtensionTable table;
    table.name = kFiberPieceApprovalTable;
    table.extension = kFiberPieceApprovalExtension;
    table.formatVersion = kFiberPieceApprovalFormat;
    table.provisioning = R"({"fingerprint":"fnv1a64-v1","aspects":{"geometry":0,"normals":1},"states":{"approved":1,"revoked":2}})";
    table.keySpace = "objects";
    table.versioned = false;
    table.columns = {
        required("fiber_uuid", Type::Blob), required("row_uuid", Type::Blob), required("approval_uuid", Type::Blob),
        required("aspect", Type::Integer), required("state", Type::Integer), optional("supersedes", Type::Blob),
        required("lock_cp_a", Type::Integer), optional("lock_cp_b", Type::Integer), required("inside_cp", Type::Integer),
        required("fiber_version", Type::Integer), required("author", Type::Text), required("approved_at", Type::Text),
        required("host", Type::Text), optional("changed_by", Type::Text), optional("changed_at", Type::Text),
        required("span_fingerprints", Type::Blob), required("cp_fingerprints", Type::Blob)};
    table.primaryKey = {"fiber_uuid", "row_uuid"};
    table.indexes = {{{"fiber_uuid", "aspect"}, false}};
    return table;
}

std::vector<uint8_t> encodeFiberSpanFingerprints(std::span<const FiberSpanFingerprint> spans)
{
    std::vector<uint8_t> out;
    out.reserve(spans.size() * 24);
    for (const auto& span : spans) { putU64(out, uint64_t(span.idLow)); putU64(out, uint64_t(span.idHigh)); putU64(out, span.hash); }
    return out;
}
std::vector<FiberSpanFingerprint> decodeFiberSpanFingerprints(std::span<const uint8_t> bytes)
{
    if (bytes.size() % 24) throw std::invalid_argument("fiber piece approval: invalid span fingerprints");
    std::vector<FiberSpanFingerprint> spans(bytes.size() / 24);
    for (size_t i = 0; i < spans.size(); ++i)
        spans[i] = {int64_t(getU64(bytes, 24 * i)), int64_t(getU64(bytes, 24 * i + 8)), getU64(bytes, 24 * i + 16)};
    return spans;
}
std::vector<uint8_t> encodeFiberControlFingerprints(std::span<const FiberControlFingerprint> controls)
{
    std::vector<uint8_t> out;
    out.reserve(controls.size() * 16);
    for (const auto& control : controls) { putU64(out, uint64_t(control.id)); putU64(out, control.hash); }
    return out;
}
std::vector<FiberControlFingerprint> decodeFiberControlFingerprints(std::span<const uint8_t> bytes)
{
    if (bytes.size() % 16) throw std::invalid_argument("fiber piece approval: invalid control fingerprints");
    std::vector<FiberControlFingerprint> controls(bytes.size() / 16);
    for (size_t i = 0; i < controls.size(); ++i) controls[i] = {int64_t(getU64(bytes, 16 * i)), getU64(bytes, 16 * i + 8)};
    return controls;
}

ExtensionRow fiberPieceApprovalRow(const FiberPieceApproval& row)
{
    validateApproval(row);
    return {extensionUuid(row.fiberUuid), extensionUuid(row.rowUuid), extensionUuid(row.approvalUuid),
            int64_t(row.aspect), int64_t(row.state), optionalValue(row.supersedes, true),
            row.lockCpA, row.lockCpB ? ExtensionValue(*row.lockCpB) : ExtensionValue(std::monostate{}), row.insideCp,
            row.fiberVersion, row.author, row.approvedAt, row.host,
            optionalValue(row.changedBy, false), optionalValue(row.changedAt, false),
            encodeFiberSpanFingerprints(row.spans), encodeFiberControlFingerprints(row.controls)};
}

FiberPieceApproval decodeFiberPieceApprovalRow(const ExtensionRow& values)
{
    if (values.size() != fiberPieceApprovalTable().columns.size())
        throw std::invalid_argument("fiber piece approval: unexpected column count");
    FiberPieceApproval row;
    row.fiberUuid = blobText(values[0], "fiber uuid");
    row.rowUuid = blobText(values[1], "row uuid");
    row.approvalUuid = blobText(values[2], "approval uuid");
    row.aspect = FiberPieceAspect(integer(values[3], "aspect"));
    row.state = FiberPieceState(integer(values[4], "state"));
    row.supersedes = optionalBlobText(values[5], "supersedes");
    row.lockCpA = integer(values[6], "lock_cp_a");
    row.lockCpB = optionalInteger(values[7], "lock_cp_b");
    row.insideCp = integer(values[8], "inside_cp");
    row.fiberVersion = integer(values[9], "fiber_version");
    row.author = text(values[10], "author");
    row.approvedAt = text(values[11], "approved_at");
    row.host = text(values[12], "host");
    row.changedBy = optionalText(values[13], "changed_by");
    row.changedAt = optionalText(values[14], "changed_at");
    row.spans = decodeFiberSpanFingerprints(blob(values[15], "span_fingerprints"));
    row.controls = decodeFiberControlFingerprints(blob(values[16], "cp_fingerprints"));
    validateApproval(row);
    return row;
}

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

bool FiberPieceContent::isLockControl(size_t index) const
{
    const auto tags = controlTags(annotationAt(*this, index));
    return std::find(tags.begin(), tags.end(), kFiberLockControlPointTag) != tags.end();
}

FiberPieceContent fiberPieceContent(const geometry::Fiber& fiber, const Json& controlAnnotations)
{
    FiberPieceContent content;
    content.controlIds = fiber.controlPointIds;
    content.controlPoints = fiber.controlPoints;
    content.linePoints = fiber.linePoints;
    if (content.controlIds.size() != content.controlPoints.size())
        throw std::invalid_argument("fiber piece approval: control points without stable ids");
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

std::optional<FiberPieceRange> fiberPieceRange(const FiberPieceContent& content, int64_t lockCpA,
                                               std::optional<int64_t> lockCpB, int64_t insideCp)
{
    const auto a = content.indexOf(lockCpA), inside = content.indexOf(insideCp);
    if (!a || !inside || *a == *inside) return std::nullopt;
    if (*inside != *a + 1 && (*a == 0 || *inside != *a - 1)) return std::nullopt;
    const bool forward = *inside > *a;
    FiberPieceRange range;
    range.forward = forward;
    if (lockCpB) {
        const auto b = content.indexOf(*lockCpB);
        if (!b || *b == *a) return std::nullopt;
        if (forward != (*b > *a)) return std::nullopt;
        range.first = std::min(*a, *b);
        range.last = std::max(*a, *b);
    } else {
        range.first = forward ? *a : 0;
        range.last = forward ? content.controlIds.size() - 1 : *a;
        range.openEnd = true;
    }
    if (range.last <= range.first) return std::nullopt;
    return range;
}

std::optional<FiberPieceRange> fiberPieceRange(const FiberPieceContent& content, const FiberPieceApproval& row)
{
    return fiberPieceRange(content, row.lockCpA, row.lockCpB, row.insideCp);
}

uint64_t fiberControlHash(const FiberPieceContent& content, size_t index, FiberPieceAspect aspect)
{
    Hasher hasher;
    const auto& annotation = annotationAt(content, index);
    if (aspect == FiberPieceAspect::Geometry) {
        hasher.point(content.controlPoints.at(index));
        if (auto direction = vector3(annotation, "direction")) {
            for (double c : *direction) {
                if (c == 0.0) continue;
                if (c < 0.0) for (double& d : *direction) d = -d;
                break;
            }
            hasher.byte(1);
            for (double c : *direction) hasher.f64(c);
        } else {
            hasher.byte(0);
        }
        for (const auto& tag : controlTags(annotation)) if (tag != kFiberLockControlPointTag) hasher.text(tag);
    } else {
        if (auto normal = vector3(annotation, "display_normal")) {
            hasher.byte(1);
            for (double c : *normal) hasher.f64(c);
        } else {
            hasher.byte(0);
        }
        const auto source = annotation.find("display_normal_source");
        hasher.text(source != annotation.end() && source->is_string() ? source->get<std::string>() : std::string());
    }
    return hasher.state;
}

uint64_t fiberSpanHash(const FiberPieceContent& content, size_t lower)
{
    if (lower + 1 >= content.controlPoints.size()) throw std::out_of_range("fiber piece approval: span index");
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

FiberPieceFingerprints fingerprintFiberPiece(const FiberPieceContent& content, const FiberPieceRange& range, FiberPieceAspect aspect)
{
    FiberPieceFingerprints result;
    for (size_t k = 0; k <= range.spanCount(); ++k) {
        const size_t index = range.at(k);
        result.controls.push_back({content.controlIds.at(index), fiberControlHash(content, index, aspect)});
    }
    if (aspect == FiberPieceAspect::Geometry) {
        for (size_t k = 0; k < range.spanCount(); ++k) {
            const size_t a = range.at(k), b = range.at(k + 1);
            const size_t lower = std::min(a, b);
            const int64_t idA = content.controlIds[a], idB = content.controlIds[b];
            result.spans.push_back({std::min(idA, idB), std::max(idA, idB), fiberSpanHash(content, lower)});
        }
    }
    return result;
}

std::vector<FiberPieceApproval> liveFiberPieceApprovals(std::span<const FiberPieceApproval> rows)
{
    std::set<std::string> superseded;
    for (const auto& row : rows) if (row.supersedes) superseded.insert(*row.supersedes);
    std::vector<FiberPieceApproval> live;
    for (const auto& row : rows)
        if (row.state == FiberPieceState::Approved && !superseded.count(row.rowUuid)) live.push_back(row);
    std::sort(live.begin(), live.end(), [](const auto& a, const auto& b) { return a.rowUuid < b.rowUuid; });
    return live;
}

FiberPieceApprovalStates fiberPieceApprovalStates(const FiberPieceContent& content, std::span<const FiberPieceApproval> rows)
{
    FiberPieceApprovalStates states;
    states.live = liveFiberPieceApprovals(rows);
    const size_t spanCount = content.controlPoints.empty() ? 0 : content.controlPoints.size() - 1;
    for (auto& aspect : states.spans) aspect.assign(spanCount, {});
    states.rowCurrent.assign(states.live.size(), false);
    std::set<int64_t> superseded;
    for (const auto& row : rows) {
        const bool isLive = std::any_of(states.live.begin(), states.live.end(), [&](const auto& l) { return l.rowUuid == row.rowUuid; });
        auto& target = isLive ? states.lockCpIds : superseded;
        target.insert(row.lockCpA);
        if (row.lockCpB) target.insert(*row.lockCpB);
    }
    for (int64_t id : superseded) if (!states.lockCpIds.count(id)) states.releasableLockCpIds.insert(id);

    for (size_t r = 0; r < states.live.size(); ++r) {
        const auto& row = states.live[r];
        const size_t aspect = size_t(row.aspect);
        auto& spans = states.spans[aspect];
        bool allCurrent = true;
        // Walk the row's control list; consecutive controls that are adjacent
        // in the content form a span whose currency is checked, controls that
        // are no longer adjacent cover every content span between them as stale.
        std::optional<size_t> previous, previousK;
        bool previousCurrent = false;
        for (size_t k = 0; k < row.controls.size(); ++k) {
            const auto index = content.indexOf(row.controls[k].id);
            if (!index) { allCurrent = false; continue; }
            const bool controlCurrent = fiberControlHash(content, *index, row.aspect) == row.controls[k].hash;
            if (!controlCurrent) allCurrent = false;
            if (previous) {
                const size_t lower = std::min(*previous, *index), upper = std::max(*previous, *index);
                bool spanCurrent = controlCurrent && previousCurrent && upper == lower + 1 && *previousK + 1 == k;
                if (spanCurrent && row.aspect == FiberPieceAspect::Geometry) {
                    const auto& stored = row.spans[k - 1];
                    const int64_t idA = content.controlIds[*previous], idB = content.controlIds[*index];
                    spanCurrent = stored.idLow == std::min(idA, idB) && stored.idHigh == std::max(idA, idB) &&
                                  stored.hash == fiberSpanHash(content, lower);
                }
                if (!spanCurrent) allCurrent = false;
                for (size_t s = lower; s < upper; ++s) {
                    if (spans[s].row && *spans[s].row != r) { spanCurrent = false; allCurrent = false; } // overlapping live rows: neither trusted
                    spans[s].row = r;
                    spans[s].current = spanCurrent && upper == lower + 1;
                }
            }
            previous = index;
            previousK = k;
            previousCurrent = controlCurrent;
        }
        states.rowCurrent[r] = allCurrent;
    }
    for (size_t aspect = 0; aspect < 2; ++aspect)
        for (size_t s = 0; s < spanCount; ++s)
            if (states.spans[aspect][s].row && states.spans[aspect][s].current) states.approvedSpans[aspect].push_back(s);
    return states;
}

namespace {
FiberPieceApproval rowForRange(const FiberPieceContent& content, const FiberPieceApproval& base,
                               const FiberPieceRange& range, std::optional<int64_t> lockCpB)
{
    FiberPieceApproval row = base;
    row.rowUuid = util::newUuid();
    row.lockCpA = content.controlIds[range.lockIndexA()];
    row.insideCp = content.controlIds[range.at(1)];
    row.lockCpB = lockCpB;
    auto fingerprints = fingerprintFiberPiece(content, range, row.aspect);
    row.spans = std::move(fingerprints.spans);
    row.controls = std::move(fingerprints.controls);
    return row;
}
} // namespace

std::vector<FiberPieceApproval> cropFiberPieceApprovals(const FiberPieceContent& content,
    std::span<const FiberPieceApproval> live, size_t openedFirst, size_t openedLast,
    const std::string& changedBy, const std::string& now)
{
    if (openedLast <= openedFirst || openedLast >= content.controlPoints.size())
        throw std::invalid_argument("fiber piece approval: invalid opened range");
    std::vector<FiberPieceApproval> result;
    for (const auto& row : live) {
        const auto range = fiberPieceRange(content, row);
        if (!range) continue; // unresolvable rows are stale already; cropping cannot place them
        // Shared spans: span owners in [first, last-1] of both ranges.
        const size_t sharedFirst = std::max(range->first, openedFirst), sharedLast = std::min(range->last, openedLast);
        if (sharedLast <= sharedFirst) continue;
        // The piece's own rows (the same range) are not cropped: an edit
        // inside leaves them stale by fingerprint, a re-approval supersedes.
        if (range->first == openedFirst && range->last == openedLast) continue;
        const bool insideLeft = range->first >= openedFirst, insideRight = range->last <= openedLast;
        if (insideLeft && insideRight) { result.push_back(revokeFiberPieceApproval(row, changedBy, now)); continue; }
        FiberPieceApproval stamp = row;
        stamp.supersedes = row.rowUuid;
        stamp.changedBy = changedBy;
        stamp.changedAt = now;
        const bool openEnd = !row.lockCpB;
        if (!insideLeft) { // part [range.first, openedFirst]
            FiberPieceRange part{range->first, openedFirst, true};
            const bool keepsLockA = range->forward; // lock A at range.first
            part.forward = keepsLockA;
            const std::optional<int64_t> lockB = keepsLockA ? std::optional<int64_t>(content.controlIds[openedFirst])
                : (openEnd ? std::nullopt : std::optional<int64_t>(content.controlIds[range->first]));
            result.push_back(rowForRange(content, stamp, part, lockB));
        }
        if (!insideRight) { // part [openedLast, range.last]
            FiberPieceRange part{openedLast, range->last, false};
            const bool keepsLockA = !range->forward; // lock A at range.last
            part.forward = !keepsLockA;
            const std::optional<int64_t> lockB = keepsLockA ? std::optional<int64_t>(content.controlIds[openedLast])
                : (openEnd ? std::nullopt : std::optional<int64_t>(content.controlIds[range->last]));
            result.push_back(rowForRange(content, stamp, part, lockB));
        }
    }
    return result;
}

FiberPieceApproval approveFiberPiece(const FiberPieceContent& content, const std::string& fiberUuid,
    const FiberPieceRange& range, FiberPieceAspect aspect, int64_t fiberVersion,
    const std::string& author, const std::string& approvedAt, const std::string& host)
{
    FiberPieceApproval row;
    row.fiberUuid = fiberUuid;
    row.rowUuid = util::newUuid();
    row.approvalUuid = row.rowUuid;
    row.aspect = aspect;
    row.state = FiberPieceState::Approved;
    row.fiberVersion = fiberVersion;
    row.author = author;
    row.approvedAt = approvedAt;
    row.host = host;
    row.lockCpA = content.controlIds.at(range.lockIndexA());
    row.insideCp = content.controlIds.at(range.at(1));
    const size_t other = range.forward ? range.last : range.first;
    row.lockCpB = range.openEnd ? std::nullopt : std::optional<int64_t>(content.controlIds.at(other));
    auto fingerprints = fingerprintFiberPiece(content, range, aspect);
    row.spans = std::move(fingerprints.spans);
    row.controls = std::move(fingerprints.controls);
    validateApproval(row);
    return row;
}

FiberPieceApproval revokeFiberPieceApproval(const FiberPieceApproval& live, const std::string& changedBy, const std::string& now)
{
    FiberPieceApproval row = live;
    row.rowUuid = util::newUuid();
    row.state = FiberPieceState::Revoked;
    row.supersedes = live.rowUuid;
    row.changedBy = changedBy;
    row.changedAt = now;
    row.spans.clear();
    row.controls.clear();
    return row;
}

std::vector<FiberPieceApproval> readFiberPieceApprovals(const ObjectStore& store, const Txn& txn, const std::string& fiberUuid)
{
    std::vector<FiberPieceApproval> rows;
    const auto tables = store.extensionTables(txn);
    if (std::none_of(tables.begin(), tables.end(), [](const auto& t) { return t.name == kFiberPieceApprovalTable; })) return rows;
    const std::array<ExtensionValue, 1> prefix{extensionUuid(fiberUuid)};
    store.scanExtensionRows(txn, kFiberPieceApprovalTable, [&](const ExtensionRow& row) {
        rows.push_back(decodeFiberPieceApprovalRow(row));
        return true;
    }, prefix);
    return rows;
}

std::vector<FiberPieceApproval> readFiberPieceApprovals(const StoreReader& reader, const std::string& fiberUuid)
{
    std::vector<FiberPieceApproval> rows;
    if (!reader.status().extensionsPresent) return rows;
    const std::array<ExtensionValue, 1> prefix{extensionUuid(fiberUuid)};
    try {
        reader.scanExtensionRows(kFiberPieceApprovalTable, [&](const ExtensionRow& row) {
            rows.push_back(decodeFiberPieceApprovalRow(row));
            return true;
        }, prefix);
    } catch (const std::invalid_argument&) {
        // No such table in this store: nothing approved.
        rows.clear();
    }
    return rows;
}
} // namespace vc::objects
