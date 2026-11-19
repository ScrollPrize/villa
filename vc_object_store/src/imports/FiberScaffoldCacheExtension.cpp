#include "vc_object_store/objects/FiberScaffoldCacheExtension.hpp"

#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/StoreReader.hpp"

#include <cmath>
#include <stdexcept>

namespace vc::objects {
namespace {
using Type = ExtensionType;
ExtensionColumn required(const char* name, Type type) { return {name, type, false}; }
ExtensionColumn optional(const char* name, Type type) { return {name, type, true}; }

int64_t integer(const ExtensionValue& value, const char* what)
{
    const auto* number = std::get_if<int64_t>(&value);
    if (!number) throw std::invalid_argument(std::string("fiber scaffold cache: invalid ") + what);
    return *number;
}
double real(const ExtensionValue& value, const char* what)
{
    if (const auto* number = std::get_if<double>(&value)) return *number;
    if (std::holds_alternative<std::monostate>(value)) return std::numeric_limits<double>::quiet_NaN();
    throw std::invalid_argument(std::string("fiber scaffold cache: invalid ") + what);
}
std::string text(const ExtensionValue& value, const char* what)
{
    const auto* string = std::get_if<std::string>(&value);
    if (!string) throw std::invalid_argument(std::string("fiber scaffold cache: invalid ") + what);
    return *string;
}
std::vector<uint8_t> blob(const ExtensionValue& value, const char* what)
{
    const auto* bytes = std::get_if<std::vector<uint8_t>>(&value);
    if (!bytes) throw std::invalid_argument(std::string("fiber scaffold cache: invalid ") + what);
    return *bytes;
}
// A real column that holds NaN for "not measured": stored as NULL so the
// column stays numeric and a reader cannot mistake it for zero.
ExtensionValue realOrNull(double value)
{
    if (std::isnan(value)) return ExtensionValue{std::monostate{}};
    return ExtensionValue{value};
}

void validateNormals(const FiberScaffoldNormalsRow& row)
{
    if (row.fiberUuid.empty()) throw std::invalid_argument("fiber scaffold cache: a normals row needs a fiber");
    if (row.fiberVersion <= 0)
        throw std::invalid_argument("fiber scaffold cache: a normals row needs the fiber version it was solved from");
    if (row.normalsUuid.empty() || row.inputs.empty())
        throw std::invalid_argument("fiber scaffold cache: a normals row needs its identity and its inputs");
    if (row.normals.size() % 24 != 0)
        throw std::invalid_argument("fiber scaffold cache: normals are three float64 per line point");
}

void validateCrossing(const FiberScaffoldCrossingRow& row)
{
    if (row.fiberA.empty() || row.fiberB.empty())
        throw std::invalid_argument("fiber scaffold cache: a crossing row needs both fibers");
    if (!(row.fiberA < row.fiberB))
        throw std::invalid_argument("fiber scaffold cache: a crossing row states its fibers in UUID order");
    if (row.versionA <= 0 || row.versionB <= 0)
        throw std::invalid_argument("fiber scaffold cache: a crossing row needs both fiber versions");
    if (row.inputs.empty() || row.normalsA.empty() || row.normalsB.empty())
        throw std::invalid_argument("fiber scaffold cache: a crossing row needs its inputs and both normals rows");
    if (row.eventIndex < kFiberScaffoldNoCrossings)
        throw std::invalid_argument("fiber scaffold cache: invalid crossing index");
}
} // namespace

ExtensionTable fiberScaffoldNormalsTable()
{
    ExtensionTable table;
    table.name = kFiberScaffoldNormalsTable;
    table.extension = kFiberScaffoldCacheExtension;
    table.formatVersion = kFiberScaffoldCacheFormat;
    table.provisioning = R"({"normals":"float64x3 per line point, little endian",)"
                         R"("key":["fiber_uuid","fiber_version","inputs"]})";
    table.keySpace = "objects";
    table.versioned = false;
    table.columns = {required("fiber_uuid", Type::Blob),   required("fiber_version", Type::Integer),
                     required("normals_uuid", Type::Blob), required("inputs", Type::Text),
                     required("sign_source", Type::Integer),
                     required("sign", Type::Integer),      required("converged", Type::Integer),
                     required("normals", Type::Blob),
                     required("author", Type::Text),       required("computed_at", Type::Text)};
    table.primaryKey = {"fiber_uuid", "fiber_version", "normals_uuid"};
    table.indexes = {};
    return table;
}

ExtensionTable fiberScaffoldCrossingsTable()
{
    ExtensionTable table;
    table.name = kFiberScaffoldCrossingsTable;
    table.extension = kFiberScaffoldCacheExtension;
    table.formatVersion = kFiberScaffoldCacheFormat;
    table.provisioning = R"({"no_crossings_index":-1,)"
                         R"("key":["fiber_a","fiber_b","version_a","version_b","normals_a","normals_b","inputs",)"
                         R"("event_index"]})";
    table.keySpace = "objects";
    table.versioned = false;
    table.columns = {required("fiber_a", Type::Blob),      required("fiber_b", Type::Blob),
                     required("version_a", Type::Integer), required("version_b", Type::Integer),
                     required("normals_a", Type::Blob),    required("normals_b", Type::Blob),
                     required("inputs", Type::Text),       required("event_index", Type::Integer),
                     required("arc_a", Type::Real),        required("arc_b", Type::Real),
                     required("point_a_x", Type::Real),    required("point_a_y", Type::Real),
                     required("point_a_z", Type::Real),    required("point_b_x", Type::Real),
                     required("point_b_y", Type::Real),    required("point_b_z", Type::Real),
                     required("parallel_score", Type::Real), required("perpendicular_score", Type::Real),
                     optional("winding_distance", Type::Real),
                     optional("parallel_winding_distance", Type::Real),
                     required("parallel", Type::Integer),  required("author", Type::Text),
                     required("computed_at", Type::Text)};
    table.primaryKey = {"fiber_a", "fiber_b", "version_a", "version_b", "normals_a", "normals_b", "inputs",
                        "event_index"};
    table.indexes = {{{"fiber_b", "fiber_a"}, false}};
    return table;
}

ExtensionRow fiberScaffoldNormalsRow(const FiberScaffoldNormalsRow& row)
{
    validateNormals(row);
    return {extensionUuid(row.fiberUuid), row.fiberVersion, extensionUuid(row.normalsUuid),
            row.inputs,                   row.signSource,   row.sign,
            int64_t(row.converged ? 1 : 0), row.normals,    row.author,
            row.computedAt};
}

ExtensionRow fiberScaffoldCrossingsRow(const FiberScaffoldCrossingRow& row)
{
    validateCrossing(row);
    return {extensionUuid(row.fiberA),
            extensionUuid(row.fiberB),
            row.versionA,
            row.versionB,
            extensionUuid(row.normalsA),
            extensionUuid(row.normalsB),
            row.inputs,
            row.eventIndex,
            row.arcA,
            row.arcB,
            row.pointA[0],
            row.pointA[1],
            row.pointA[2],
            row.pointB[0],
            row.pointB[1],
            row.pointB[2],
            row.parallelScore,
            row.perpendicularScore,
            realOrNull(row.windingDistance),
            realOrNull(row.parallelWindingDistance),
            int64_t(row.parallel ? 1 : 0),
            row.author,
            row.computedAt};
}

std::vector<FiberScaffoldNormalsRow> readFiberScaffoldNormals(const StoreReader& reader)
{
    std::vector<FiberScaffoldNormalsRow> out;
    try {
        reader.scanExtensionRows(kFiberScaffoldNormalsTable, [&](const ExtensionRow& row) {
            if (row.size() < 10) throw std::invalid_argument("fiber scaffold cache: short normals row");
            FiberScaffoldNormalsRow entry;
            entry.fiberUuid = extensionUuid(row[0]);
            entry.fiberVersion = integer(row[1], "fiber version");
            entry.normalsUuid = extensionUuid(row[2]);
            entry.inputs = text(row[3], "inputs");
            entry.signSource = integer(row[4], "sign source");
            entry.sign = integer(row[5], "sign");
            entry.converged = integer(row[6], "converged") != 0;
            entry.normals = blob(row[7], "normals");
            entry.author = text(row[8], "author");
            entry.computedAt = text(row[9], "computed at");
            out.push_back(std::move(entry));
            return true;
        });
    } catch (const std::invalid_argument&) {
        out.clear();  // no such table in this store: nothing cached
    }
    return out;
}

std::vector<FiberScaffoldCrossingRow> readFiberScaffoldCrossings(const StoreReader& reader)
{
    std::vector<FiberScaffoldCrossingRow> out;
    try {
        reader.scanExtensionRows(kFiberScaffoldCrossingsTable, [&](const ExtensionRow& row) {
            if (row.size() < 23) throw std::invalid_argument("fiber scaffold cache: short crossing row");
            FiberScaffoldCrossingRow entry;
            entry.fiberA = extensionUuid(row[0]);
            entry.fiberB = extensionUuid(row[1]);
            entry.versionA = integer(row[2], "version a");
            entry.versionB = integer(row[3], "version b");
            entry.normalsA = extensionUuid(row[4]);
            entry.normalsB = extensionUuid(row[5]);
            entry.inputs = text(row[6], "inputs");
            entry.eventIndex = integer(row[7], "event index");
            entry.arcA = real(row[8], "arc a");
            entry.arcB = real(row[9], "arc b");
            entry.pointA = {real(row[10], "point a"), real(row[11], "point a"), real(row[12], "point a")};
            entry.pointB = {real(row[13], "point b"), real(row[14], "point b"), real(row[15], "point b")};
            entry.parallelScore = real(row[16], "parallel score");
            entry.perpendicularScore = real(row[17], "perpendicular score");
            entry.windingDistance = real(row[18], "winding distance");
            entry.parallelWindingDistance = real(row[19], "parallel winding distance");
            entry.parallel = integer(row[20], "parallel") != 0;
            entry.author = text(row[21], "author");
            entry.computedAt = text(row[22], "computed at");
            out.push_back(std::move(entry));
            return true;
        });
    } catch (const std::invalid_argument&) {
        out.clear();
    }
    return out;
}
} // namespace vc::objects
