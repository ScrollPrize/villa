#include "vc_object_store/objects/FiberConstraintApprovalExtension.hpp"

#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/StoreReader.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <set>
#include <stdexcept>

namespace vc::objects {
namespace {
using Type = ExtensionType;
ExtensionColumn required(const char* name, Type type) { return {name, type, false}; }
ExtensionColumn optional(const char* name, Type type) { return {name, type, true}; }

int64_t integer(const ExtensionValue& value, const char* what)
{
    const auto* number = std::get_if<int64_t>(&value);
    if (!number) throw std::invalid_argument(std::string("fiber constraint approval: invalid ") + what);
    return *number;
}
std::string text(const ExtensionValue& value, const char* what)
{
    const auto* string = std::get_if<std::string>(&value);
    if (!string) throw std::invalid_argument(std::string("fiber constraint approval: invalid ") + what);
    return *string;
}
std::optional<std::string> optionalText(const ExtensionValue& value, const char* what)
{
    if (std::holds_alternative<std::monostate>(value)) return std::nullopt;
    return text(value, what);
}
std::string blobText(const ExtensionValue& value, const char* what)
{
    if (std::holds_alternative<std::monostate>(value))
        throw std::invalid_argument(std::string("fiber constraint approval: missing ") + what);
    return extensionUuid(value);
}
std::optional<std::string> optionalBlobText(const ExtensionValue& value, const char* what)
{
    if (std::holds_alternative<std::monostate>(value)) return std::nullopt;
    return blobText(value, what);
}
ExtensionValue optionalUuid(const std::optional<std::string>& value)
{
    if (!value) return std::monostate{};
    return extensionUuid(*value);
}
ExtensionValue optionalString(const std::optional<std::string>& value)
{
    if (!value) return std::monostate{};
    return *value;
}
// A fingerprint's three fields, or three NULLs for a revocation.
ExtensionValue fingerprintField(const std::optional<FiberSpanFingerprint>& span, int field)
{
    if (!span) return std::monostate{};
    if (field == 0) return span->idLow;
    if (field == 1) return span->idHigh;
    return int64_t(std::bit_cast<int64_t>(span->hash));
}
std::optional<FiberSpanFingerprint> fingerprintOf(const ExtensionRow& values, size_t first, const char* what)
{
    if (std::holds_alternative<std::monostate>(values[first])) {
        for (size_t i = first; i < first + 3; ++i)
            if (!std::holds_alternative<std::monostate>(values[i]))
                throw std::invalid_argument(std::string("fiber constraint approval: partial ") + what);
        return std::nullopt;
    }
    FiberSpanFingerprint span;
    span.idLow = integer(values[first], what);
    span.idHigh = integer(values[first + 1], what);
    span.hash = std::bit_cast<uint64_t>(integer(values[first + 2], what));
    return span;
}

void validateApproval(const FiberConstraintApproval& row)
{
    if (row.state != FiberConstraintApprovalState::Approved && row.state != FiberConstraintApprovalState::Revoked)
        throw std::invalid_argument("fiber constraint approval: invalid state");
    if (row.distance != FiberConstraintDistance::Direct && row.distance != FiberConstraintDistance::Distant)
        throw std::invalid_argument("fiber constraint approval: invalid distance");
    const auto kind = row.identity.kind;
    if (kind != FiberConstraintKind::Perpendicular && kind != FiberConstraintKind::Parallel &&
        kind != FiberConstraintKind::Enclosure)
        throw std::invalid_argument("fiber constraint approval: invalid constraint kind");
    if (row.identity.fiberA.empty() || row.identity.fiberB.empty())
        throw std::invalid_argument("fiber constraint approval: both fibers are required");
    if (row.identity.fiberA >= row.identity.fiberB)
        throw std::invalid_argument("fiber constraint approval: the fibers must be in ascending UUID order");
    if (row.fiberVersionA < 1 || row.fiberVersionB < 1)
        throw std::invalid_argument("fiber constraint approval: invalid fiber version");
    if (row.author.empty() || row.approvedAt.empty())
        throw std::invalid_argument("fiber constraint approval: author and time required");
    if (row.state == FiberConstraintApprovalState::Revoked) {
        if (!row.supersedes) throw std::invalid_argument("fiber constraint approval: revocation must supersede a row");
        if (row.spanA || row.spanB)
            throw std::invalid_argument("fiber constraint approval: revocation carries no fingerprints");
    } else if (!row.spanA || !row.spanB) {
        throw std::invalid_argument("fiber constraint approval: an approval fingerprints both covering spans");
    }
    if (row.supersedes && *row.supersedes == row.rowUuid)
        throw std::invalid_argument("fiber constraint approval: row supersedes itself");
}

} // namespace

FiberSpanFingerprint fiberSpanFingerprint(const FiberPieceContent& content, size_t span)
{
    if (content.controlPoints.size() < 2 || span + 1 >= content.controlPoints.size())
        throw std::invalid_argument("fiber constraint approval: span index outside the fiber");
    const auto low = content.controlIds[span], high = content.controlIds[span + 1];
    FiberSpanFingerprint fingerprint;
    fingerprint.idLow = std::min(low, high);
    fingerprint.idHigh = std::max(low, high);
    fingerprint.hash = fiberSpanHash(content, span);
    return fingerprint;
}

FiberConstraintIdentity fiberConstraintIdentity(const std::string& fiberA, double arcA, const std::string& fiberB,
                                                double arcB, FiberConstraintKind kind, double quantum)
{
    if (!(quantum > 0) || !std::isfinite(quantum))
        throw std::invalid_argument("fiber constraint approval: the arc quantum must be finite and positive");
    if (!std::isfinite(arcA) || !std::isfinite(arcB) || arcA < 0 || arcB < 0)
        throw std::invalid_argument("fiber constraint approval: anchor arcs must be finite and non-negative");
    if (fiberA == fiberB) throw std::invalid_argument("fiber constraint approval: a constraint joins two fibers");
    const bool ordered = fiberA < fiberB;
    FiberConstraintIdentity identity;
    identity.fiberA = ordered ? fiberA : fiberB;
    identity.fiberB = ordered ? fiberB : fiberA;
    identity.kind = kind;
    identity.arcA = int64_t(std::llround(std::floor((ordered ? arcA : arcB) / quantum)));
    identity.arcB = int64_t(std::llround(std::floor((ordered ? arcB : arcA) / quantum)));
    return identity;
}

ExtensionTable fiberConstraintApprovalTable()
{
    ExtensionTable table;
    table.name = kFiberConstraintApprovalTable;
    table.extension = kFiberConstraintApprovalExtension;
    table.formatVersion = kFiberConstraintApprovalFormat;
    table.provisioning = R"({"fingerprint":"fnv1a64-v1","arc_quantum_base_voxels":64.0,)"
                         R"("kinds":{"perpendicular":1,"parallel":2,"enclosure":3},)"
                         R"("states":{"approved":1,"revoked":2},"distances":{"direct":1,"distant":2}})";
    table.keySpace = "objects";
    table.versioned = false;
    table.columns = {required("fiber_a", Type::Blob),     required("fiber_b", Type::Blob),
                     required("kind", Type::Integer),     required("arc_a", Type::Integer),
                     required("arc_b", Type::Integer),    required("row_uuid", Type::Blob),
                     required("approval_uuid", Type::Blob), required("state", Type::Integer),
                     required("distance", Type::Integer), optional("supersedes", Type::Blob),  required("fiber_version_a", Type::Integer),
                     required("fiber_version_b", Type::Integer), optional("span_a_id_low", Type::Integer),
                     optional("span_a_id_high", Type::Integer), optional("span_a_hash", Type::Integer),
                     optional("span_b_id_low", Type::Integer), optional("span_b_id_high", Type::Integer),
                     optional("span_b_hash", Type::Integer), required("author", Type::Text),
                     required("approved_at", Type::Text), required("host", Type::Text),
                     optional("changed_by", Type::Text),  optional("changed_at", Type::Text)};
    table.primaryKey = {"fiber_a", "fiber_b", "kind", "arc_a", "arc_b", "row_uuid"};
    table.indexes = {{{"fiber_b", "fiber_a"}, false}, {{"approval_uuid"}, false}};
    return table;
}

ExtensionRow fiberConstraintApprovalRow(const FiberConstraintApproval& row)
{
    validateApproval(row);
    return {extensionUuid(row.identity.fiberA),
            extensionUuid(row.identity.fiberB),
            int64_t(row.identity.kind),
            row.identity.arcA,
            row.identity.arcB,
            extensionUuid(row.rowUuid),
            extensionUuid(row.approvalUuid),
            int64_t(row.state),
            int64_t(row.distance),
            optionalUuid(row.supersedes),
            row.fiberVersionA,
            row.fiberVersionB,
            fingerprintField(row.spanA, 0),
            fingerprintField(row.spanA, 1),
            fingerprintField(row.spanA, 2),
            fingerprintField(row.spanB, 0),
            fingerprintField(row.spanB, 1),
            fingerprintField(row.spanB, 2),
            row.author,
            row.approvedAt,
            row.host,
            optionalString(row.changedBy),
            optionalString(row.changedAt)};
}

FiberConstraintApproval decodeFiberConstraintApprovalRow(const ExtensionRow& values)
{
    if (values.size() != fiberConstraintApprovalTable().columns.size())
        throw std::invalid_argument("fiber constraint approval: unexpected column count");
    FiberConstraintApproval row;
    row.identity.fiberA = blobText(values[0], "fiber a");
    row.identity.fiberB = blobText(values[1], "fiber b");
    row.identity.kind = FiberConstraintKind(integer(values[2], "kind"));
    row.identity.arcA = integer(values[3], "arc a");
    row.identity.arcB = integer(values[4], "arc b");
    row.rowUuid = blobText(values[5], "row uuid");
    row.approvalUuid = blobText(values[6], "approval uuid");
    row.state = FiberConstraintApprovalState(integer(values[7], "state"));
    row.distance = FiberConstraintDistance(integer(values[8], "distance"));
    row.supersedes = optionalBlobText(values[9], "supersedes");
    row.fiberVersionA = integer(values[10], "fiber version a");
    row.fiberVersionB = integer(values[11], "fiber version b");
    row.spanA = fingerprintOf(values, 12, "span a fingerprint");
    row.spanB = fingerprintOf(values, 15, "span b fingerprint");
    row.author = text(values[18], "author");
    row.approvedAt = text(values[19], "approved_at");
    row.host = text(values[20], "host");
    row.changedBy = optionalText(values[21], "changed_by");
    row.changedAt = optionalText(values[22], "changed_at");
    validateApproval(row);
    return row;
}

std::optional<size_t> fiberSpanCoveringLineIndex(const FiberPieceContent& content, int64_t lineIndex)
{
    if (content.controlLineIndex.size() < 2 || lineIndex < 0) return std::nullopt;
    // A line point past the first or last control point (a tail) belongs to
    // the nearest span: the tail is drawn from it.
    std::optional<size_t> nearest;
    int64_t nearestDistance = 0;
    for (size_t span = 0; span + 1 < content.controlLineIndex.size(); ++span) {
        const auto from = content.controlLineIndex[span], to = content.controlLineIndex[span + 1];
        if (from < 0 || to < 0) continue;
        const auto low = std::min(from, to), high = std::max(from, to);
        if (lineIndex >= low && lineIndex <= high) return span;
        const auto distance = lineIndex < low ? low - lineIndex : lineIndex - high;
        if (!nearest || distance < nearestDistance) {
            nearest = span;
            nearestDistance = distance;
        }
    }
    return nearest;
}

FiberConstraintApproval approveFiberConstraint(const FiberConstraintIdentity& identity, const FiberPieceContent& a,
                                               size_t spanA, int64_t fiberVersionA, const FiberPieceContent& b,
                                               size_t spanB, int64_t fiberVersionB,
                                               FiberConstraintDistance distance, const std::string& author,
                                               const std::string& approvedAt, const std::string& host)
{
    FiberConstraintApproval row;
    row.identity = identity;
    row.rowUuid = util::newUuid();
    row.approvalUuid = row.rowUuid;
    row.state = FiberConstraintApprovalState::Approved;
    row.distance = distance;
    row.fiberVersionA = fiberVersionA;
    row.fiberVersionB = fiberVersionB;
    row.spanA = fiberSpanFingerprint(a, spanA);
    row.spanB = fiberSpanFingerprint(b, spanB);
    row.author = author;
    row.approvedAt = approvedAt;
    row.host = host;
    validateApproval(row);
    return row;
}

FiberConstraintApproval revokeFiberConstraintApproval(const FiberConstraintApproval& live,
                                                      const std::string& changedBy, const std::string& now)
{
    FiberConstraintApproval row = live;
    row.rowUuid = util::newUuid();
    row.state = FiberConstraintApprovalState::Revoked;
    row.supersedes = live.rowUuid;
    row.spanA.reset();
    row.spanB.reset();
    row.changedBy = changedBy;
    row.changedAt = now;
    validateApproval(row);
    return row;
}

std::vector<FiberConstraintApproval> liveFiberConstraintApprovals(std::span<const FiberConstraintApproval> rows)
{
    std::set<std::string> superseded;
    for (const auto& row : rows)
        if (row.supersedes) superseded.insert(*row.supersedes);
    std::vector<FiberConstraintApproval> live;
    for (const auto& row : rows)
        if (row.state == FiberConstraintApprovalState::Approved && !superseded.count(row.rowUuid))
            live.push_back(row);
    std::sort(live.begin(), live.end(), [](const auto& a, const auto& b) { return a.rowUuid < b.rowUuid; });
    return live;
}

bool fiberConstraintApprovalCurrent(const FiberConstraintApproval& row, const FiberPieceContent& a, size_t spanA,
                                    const FiberPieceContent& b, size_t spanB)
{
    if (!row.spanA || !row.spanB) return false;
    if (spanA + 1 >= a.controlPoints.size() || spanB + 1 >= b.controlPoints.size()) return false;
    return fiberSpanFingerprint(a, spanA) == *row.spanA && fiberSpanFingerprint(b, spanB) == *row.spanB;
}

std::vector<FiberConstraintApproval> readFiberConstraintApprovals(const StoreReader& reader)
{
    std::vector<FiberConstraintApproval> rows;
    if (!reader.status().extensionsPresent) return rows;
    try {
        reader.scanExtensionRows(kFiberConstraintApprovalTable, [&](const ExtensionRow& row) {
            rows.push_back(decodeFiberConstraintApprovalRow(row));
            return true;
        });
    } catch (const std::invalid_argument&) {
        // No such table in this store: nothing approved.
        rows.clear();
    }
    return rows;
}

std::vector<FiberConstraintApproval> readFiberConstraintApprovals(const StoreReader& reader,
                                                                 const std::string& fiberUuid)
{
    std::vector<FiberConstraintApproval> rows;
    for (auto& row : readFiberConstraintApprovals(reader))
        if (row.identity.fiberA == fiberUuid || row.identity.fiberB == fiberUuid) rows.push_back(std::move(row));
    return rows;
}
} // namespace vc::objects
