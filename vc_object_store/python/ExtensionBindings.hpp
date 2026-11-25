#pragma once
#include "vc_object_store/objects/FiberGraphSolveExtension.hpp"
#include "vc_object_store/objects/FiberConstraintApprovalExtension.hpp"
#include "vc_object_store/objects/FiberApprovalExtension.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include <nanobind/stl/tuple.h>
#include "vc_object_store/objects/TransactionJson.hpp"

// Included after StorageBindings.hpp: use the same transaction wrappers and
// JSON metadata codec, but never round-trip typed row data through JSON.
namespace {
std::map<std::string,vo::FiberGraphSetting> fiberGraphSettingsFromPy(const nb::dict& input)
{
    std::map<std::string,vo::FiberGraphSetting> settings;
    for(const auto [name,value]:input){
        const auto key=nb::cast<std::string>(name);
        if(nb::isinstance<nb::bool_>(value))settings.emplace(key,nb::cast<bool>(value));
        else if(nb::isinstance<nb::int_>(value))settings.emplace(key,nb::cast<int64_t>(value));
        else if(nb::isinstance<nb::float_>(value))settings.emplace(key,nb::cast<double>(value));
        else throw std::invalid_argument("fiber graph settings must be explicit boolean/integer/real scalars");
    }
    return settings;
}

vo::FiberGraphTable fiberGraphTableFromPy(const std::string& name)
{
    using Table = vo::FiberGraphTable;
    const std::map<std::string, Table> names = {{"runs", Table::Runs}, {"inputs", Table::Inputs},
        {"fibers", Table::FiberResults}, {"samples", Table::Samples}, {"hypotheses", Table::Hypotheses},
        {"widths", Table::Widths}, {"positions", Table::Positions}, {"relations", Table::ConstraintResults},
        {"metrics", Table::Metrics}, {"residuals", Table::Residuals},
        {"residual_pieces", Table::ResidualPieces}};
    const auto found = names.find(name);
    if (found == names.end()) throw std::invalid_argument("unknown fiber graph table role: " + name);
    return found->second;
}

vo::ExtensionType extensionTypeFromPy(const std::string& name)
{
    if (name == "integer") return vo::ExtensionType::Integer;
    if (name == "real") return vo::ExtensionType::Real;
    if (name == "text") return vo::ExtensionType::Text;
    if (name == "blob") return vo::ExtensionType::Blob;
    throw std::invalid_argument("extension column type must be integer, real, text or blob");
}

const char* extensionTypeToPy(vo::ExtensionType type)
{
    switch (type) {
    case vo::ExtensionType::Integer: return "integer";
    case vo::ExtensionType::Real: return "real";
    case vo::ExtensionType::Text: return "text";
    case vo::ExtensionType::Blob: return "blob";
    }
    throw std::invalid_argument("invalid extension column type");
}

vo::ExtensionValue extensionValueFromPy(const nb::handle& value)
{
    if (value.is_none()) return std::monostate{};
    if (nb::isinstance<nb::bytes>(value)) {
        const auto data = nb::borrow<nb::bytes>(value);
        const auto* begin = reinterpret_cast<const uint8_t*>(data.c_str());
        return std::vector<uint8_t>(begin, begin + data.size());
    }
    if (nb::isinstance<nb::int_>(value)) return nb::cast<int64_t>(value);
    if (nb::isinstance<nb::float_>(value)) return nb::cast<double>(value);
    if (nb::isinstance<nb::str>(value)) return nb::cast<std::string>(value);
    throw std::invalid_argument("extension values must be None, int64, float, str or bytes");
}

nb::object extensionValueToPy(const vo::ExtensionValue& value)
{
    return std::visit([](const auto& item) -> nb::object {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, std::monostate>) return nb::none();
        else if constexpr (std::is_same_v<T, std::vector<uint8_t>>)
            return nb::bytes(item.empty() ? "" : reinterpret_cast<const char*>(item.data()), item.size());
        else return nb::cast(item);
    }, value);
}

vo::ExtensionRow extensionRowFromPy(const nb::handle& input)
{
    vo::ExtensionRow result;
    for (const auto value : nb::cast<nb::sequence>(input)) result.push_back(extensionValueFromPy(value));
    return result;
}

vo::ExtensionKeyFilter extensionKeyFilterFromPy(const nb::dict& input)
{
    vo::ExtensionKeyFilter result;
    for(const auto [name,value]:input)result.emplace(nb::cast<std::string>(name),extensionValueFromPy(value));
    return result;
}

std::vector<vo::ExtensionRow> extensionRowsFromPy(const nb::sequence& input)
{
    std::vector<vo::ExtensionRow> result;
    result.reserve(nb::len(input));
    for (const auto row : input) result.push_back(extensionRowFromPy(row));
    return result;
}

nb::list extensionRowToPy(const vo::ExtensionRow& row)
{
    nb::list result;
    for (const auto& value : row) result.append(extensionValueToPy(value));
    return result;
}

vo::ExtensionTable extensionTableFromPy(const nb::dict& input)
{
    static const std::set<std::string> keys{"name", "extension", "format_version", "provisioning", "based_on",
        "columns", "primary_key", "indexes", "key_space", "versioned"};
    for (const auto [key, value] : input)
        if (!keys.contains(nb::cast<std::string>(key))) throw std::invalid_argument("unknown extension table field");
    vo::ExtensionTable table;
    table.name = getStr(input, "name"); table.extension = getStr(input, "extension");
    table.formatVersion = getInt(input, "format_version", 1);
    table.keySpace = getStr(input, "key_space");
    if (has(input, "versioned")) table.versioned = nb::cast<bool>(input["versioned"]);
    if (has(input, "provisioning")) table.provisioning = jsonDumps(input["provisioning"]);
    if (has(input, "based_on")) table.basedOn = vo::extensionSourcesFromJson(jsonDumps(input["based_on"]));
    for (const auto item : nb::cast<nb::sequence>(input["columns"])) {
        const auto column = nb::cast<nb::dict>(item);
        table.columns.push_back({getStr(column, "name"), extensionTypeFromPy(getStr(column, "type")),
                                 has(column, "nullable") && nb::cast<bool>(column["nullable"])});
    }
    table.primaryKey = nb::cast<std::vector<std::string>>(input["primary_key"]);
    if (has(input, "indexes")) for (const auto item : nb::cast<nb::sequence>(input["indexes"])) {
        const auto index = nb::cast<nb::dict>(item);
        table.indexes.push_back({nb::cast<std::vector<std::string>>(index["columns"]),
                                 has(index, "unique") && nb::cast<bool>(index["unique"])});
    }
    return table;
}

template<class Table>
nb::dict extensionTableToPy(const Table& table)
{
    nb::dict result;
    result["name"] = table.name; result["extension"] = table.extension;
    result["format_version"] = table.formatVersion; result["key_space"] = table.keySpace;
    result["versioned"] = table.versioned; result["provisioning"] = jsonLoads(table.provisioning);
    result["based_on"] = jsonLoads(vo::extensionSourcesJson(table.basedOn));
    result["primary_key"] = table.primaryKey;
    nb::list columns;
    for (const auto& column : table.columns) {
        nb::dict item; item["name"] = column.name; item["type"] = extensionTypeToPy(column.type);
        item["nullable"] = column.nullable; columns.append(item);
    }
    result["columns"] = columns;
    if constexpr (requires { table.indexes; }) {
        nb::list indexes;
        for (const auto& index : table.indexes) {
            nb::dict item; item["columns"] = index.columns; item["unique"] = index.unique; indexes.append(item);
        }
        result["indexes"] = indexes;
    }
    if constexpr (requires { table.tableUuid; }) {
        result["table_uuid"] = table.tableUuid; result["last_modified_commit"] = table.lastModifiedCommit;
    }
    return result;
}

inline nb::dict fiberApprovalLayerToPy(const vo::FiberApprovalLayer& layer);

template<class T>
void bindExtensionApi(nb::class_<T>& cls)
{
    cls.def("create_extension_table", [](T& self, const nb::dict& definition) {
        const auto table = extensionTableFromPy(definition);
        self.template withWrite<true>([&](auto& db, auto& txn) { db.createExtensionTable(txn, table); });
    }, nb::arg("definition"))
    .def("append_extension_rows", [](T& self, const std::string& table, const nb::sequence& input) {
        const auto rows = extensionRowsFromPy(input);
        self.template withWrite<true>([&](auto& db, auto& txn) { db.appendExtensionRows(txn, table, rows); });
    }, nb::arg("table"), nb::arg("rows"))
    .def("replace_extension_rows", [](T& self, const std::string& table, const std::string& owner, const nb::sequence& input) {
        const auto rows = extensionRowsFromPy(input);
        self.template withWrite<true>([&](auto& db, auto& txn) { db.replaceExtensionRows(txn, table, owner, rows); });
    }, nb::arg("table"), nb::arg("owner_uuid"), nb::arg("rows"))
    .def("extension_tables", [](T& self) {
        nb::list result;
        const auto tables = self.withRead([](auto& db, auto& txn) { return db.extensionTables(txn); });
        for (const auto& table : tables) result.append(extensionTableToPy(table));
        return result;
    })
    .def("extension_rows", [](T& self, const std::string& table, const nb::sequence& prefix,const nb::dict& filter) {
        const auto key = extensionRowFromPy(prefix);
        const auto selection = extensionKeyFilterFromPy(filter);
        nb::list result;
        self.withRead([&](auto& db, auto& txn) {
            db.scanExtensionRows(txn, table, [&](const auto& row) { result.append(extensionRowToPy(row)); return true; }, key,selection);
        });
        return result;
    }, nb::arg("table"), nb::arg("key_prefix") = nb::tuple(),nb::arg("key_filter")=nb::dict())
    .def("scan_extension_rows", [](T& self, const std::string& table, const nb::callable& visitor, const nb::sequence& prefix,const nb::dict& filter) {
        const auto key = extensionRowFromPy(prefix);
        const auto selection = extensionKeyFilterFromPy(filter);
        size_t visited = 0;
        self.withRead([&](auto& db, auto& txn) {
            db.scanExtensionRows(txn, table, [&](const auto& row) {
                ++visited;
                const auto keepGoing = visitor(extensionRowToPy(row));
                return keepGoing.is_none() || nb::cast<bool>(keepGoing);
            }, key,selection);
        });
        return visited;
    }, nb::arg("table"), nb::arg("visitor"), nb::arg("key_prefix") = nb::tuple(),nb::arg("key_filter")=nb::dict())
    .def("extension_rows_at_version", [](T& self, const std::string& table, const nb::dict& ref) {
        const bool object = has(ref, "object"), relationship = has(ref, "relationship");
        if (object == relationship) throw std::invalid_argument("extension reference needs exactly one object or relationship UUID");
        nb::list result;
        const auto visitor = [&](const vo::ExtensionRow& row) { result.append(extensionRowToPy(row)); return true; };
        self.withRead([&](auto& db, auto& txn) {
            if (object) db.scanExtensionRowsAtVersion(txn, table,
                vo::objectVersionFromJson(jsonDumps(ref)), visitor);
            else db.scanExtensionRowsAtVersion(txn, table,
                vo::relationshipVersionFromJson(jsonDumps(ref)), visitor);
        });
        return result;
    }, nb::arg("table"), nb::arg("reference"))
    .def("read_fiber_approval", [](T& self, const std::string& fiberUuid) -> nb::object {
        const auto layer = self.withRead([&](auto& db, auto& txn) {
            return vo::readFiberApproval(vo::LocalStoreReader(db, txn), fiberUuid);
        });
        return layer ? nb::object(fiberApprovalLayerToPy(*layer)) : nb::none();
    }, nb::arg("fiber_uuid"), "The newest generation of the fiber's approval layer as a dict, or None.");
}

// ---- Fiber approval layer (docs/fiber-approval.md) -------------------------
inline vo::FiberApprovalAspect fiberApprovalAspectFromPy(const std::string& name)
{
    if (name == "geometry") return vo::FiberApprovalAspect::Geometry;
    if (name == "normals") return vo::FiberApprovalAspect::Normals;
    throw std::invalid_argument("fiber approval aspect is geometry or normals");
}
inline vo::FiberStretchState fiberStretchStateFromPy(const std::string& name)
{
    if (name == "undefined") return vo::FiberStretchState::Undefined;
    if (name == "approved") return vo::FiberStretchState::Approved;
    if (name == "broken") return vo::FiberStretchState::Broken;
    throw std::invalid_argument("fiber stretch state is undefined, approved or broken");
}
inline const char* fiberStretchStateName(vo::FiberStretchState state)
{
    switch (state) {
    case vo::FiberStretchState::Undefined: return "undefined";
    case vo::FiberStretchState::Approved: return "approved";
    case vo::FiberStretchState::Broken: return "broken";
    }
    throw std::invalid_argument("fiber approval: unknown stretch state");
}
inline nb::list fiberStretchesToPy(const std::vector<vo::FiberStretch>& stretches)
{
    nb::list result;
    for (const auto& s : stretches) {
        nb::dict d;
        d["aspect"] = s.aspect == vo::FiberApprovalAspect::Normals ? "normals" : "geometry";
        d["state"] = fiberStretchStateName(s.state);
        d["from"] = s.from; d["to"] = s.to; d["fingerprint"] = s.fingerprint;
        result.append(d);
    }
    return result;
}
inline std::vector<vo::FiberStretch> fiberStretchesFromPy(const nb::handle& input)
{
    std::vector<vo::FiberStretch> result;
    for (const auto item : input) {
        const auto d = nb::cast<nb::dict>(item);
        result.push_back({fiberApprovalAspectFromPy(nb::cast<std::string>(d["aspect"])),
                          fiberStretchStateFromPy(nb::cast<std::string>(d["state"])), nb::cast<double>(d["from"]),
                          nb::cast<double>(d["to"]), nb::cast<uint64_t>(d["fingerprint"])});
    }
    return result;
}
inline nb::list fiberNormalPointsToPy(const std::vector<vo::FiberNormalPoint>& points)
{
    nb::list result;
    for (const auto& p : points) {
        nb::dict d; d["position"] = p.position; d["normal"] = p.normal; d["fingerprint"] = p.fingerprint;
        result.append(d);
    }
    return result;
}
inline std::vector<vo::FiberNormalPoint> fiberNormalPointsFromPy(const nb::handle& input)
{
    std::vector<vo::FiberNormalPoint> result;
    for (const auto item : input) {
        const auto d = nb::cast<nb::dict>(item);
        result.push_back({nb::cast<double>(d["position"]), nb::cast<std::array<double, 3>>(d["normal"]),
                          nb::cast<uint64_t>(d["fingerprint"])});
    }
    return result;
}
inline nb::dict fiberApprovalLayerToPy(const vo::FiberApprovalLayer& layer)
{
    nb::dict d;
    d["fiber_uuid"] = layer.fiberUuid; d["generation"] = layer.generation; d["fiber_version"] = layer.fiberVersion;
    d["author"] = layer.author; d["saved_at"] = layer.savedAt; d["host"] = layer.host;
    d["baseline_run"] = layer.baselineRun;
    d["stretches"] = fiberStretchesToPy(layer.stretches);
    d["normal_points"] = fiberNormalPointsToPy(layer.normalPoints);
    d["line_normals"] = layer.lineNormals;
    return d;
}
// Only "fiber_uuid" is required; everything else defaults as in FiberApprovalLayer.
inline vo::FiberApprovalLayer fiberApprovalLayerFromPy(const nb::dict& d)
{
    vo::FiberApprovalLayer layer;
    const auto textOf = [&](const char* key) {
        return d.contains(key) && !d[key].is_none() ? nb::cast<std::string>(d[key]) : std::string();
    };
    layer.fiberUuid = nb::cast<std::string>(d["fiber_uuid"]);
    if (d.contains("generation")) layer.generation = nb::cast<int64_t>(d["generation"]);
    if (d.contains("fiber_version")) layer.fiberVersion = nb::cast<int64_t>(d["fiber_version"]);
    layer.author = textOf("author"); layer.savedAt = textOf("saved_at"); layer.host = textOf("host");
    layer.baselineRun = textOf("baseline_run");
    if (d.contains("stretches")) layer.stretches = fiberStretchesFromPy(d["stretches"]);
    if (d.contains("normal_points")) layer.normalPoints = fiberNormalPointsFromPy(d["normal_points"]);
    if (d.contains("line_normals"))
        layer.lineNormals = nb::cast<std::vector<std::array<double, 3>>>(d["line_normals"]);
    return layer;
}

inline const char* fiberConstraintKindName(vo::FiberConstraintKind kind)
{
    switch (kind) {
    case vo::FiberConstraintKind::Perpendicular: return "perpendicular";
    case vo::FiberConstraintKind::Parallel: return "parallel";
    case vo::FiberConstraintKind::Enclosure: return "enclosure";
    }
    throw std::invalid_argument("fiber constraint approval: unknown kind");
}
inline vo::FiberConstraintKind fiberConstraintKindFromPy(const std::string& name)
{
    if (name == "perpendicular") return vo::FiberConstraintKind::Perpendicular;
    if (name == "parallel") return vo::FiberConstraintKind::Parallel;
    if (name == "enclosure") return vo::FiberConstraintKind::Enclosure;
    throw std::invalid_argument("fiber constraint approval: kind must be perpendicular, parallel or enclosure");
}
inline nb::object fiberSpanFingerprintToPy(const std::optional<vo::FiberSpanFingerprint>& span)
{
    if (!span) return nb::none();
    return nb::make_tuple(span->idLow, span->idHigh, span->hash);
}
inline std::optional<vo::FiberSpanFingerprint> fiberSpanFingerprintFromPy(const nb::handle& value)
{
    if (value.is_none()) return std::nullopt;
    const auto entry = nb::cast<std::tuple<int64_t, int64_t, uint64_t>>(value);
    return vo::FiberSpanFingerprint{std::get<0>(entry), std::get<1>(entry), std::get<2>(entry)};
}
inline nb::dict fiberConstraintApprovalToPy(const vo::FiberConstraintApproval& row)
{
    nb::dict d;
    d["fiber_a"] = row.identity.fiberA; d["fiber_b"] = row.identity.fiberB;
    d["kind"] = fiberConstraintKindName(row.identity.kind);
    d["arc_a"] = row.identity.arcA; d["arc_b"] = row.identity.arcB;
    d["row_uuid"] = row.rowUuid; d["approval_uuid"] = row.approvalUuid;
    d["state"] = row.state == vo::FiberConstraintApprovalState::Revoked ? "revoked" : "approved";
    d["distance"] = row.distance == vo::FiberConstraintDistance::Distant ? "distant" : "direct";
    d["supersedes"] = row.supersedes ? nb::cast(*row.supersedes) : nb::none();
    d["fiber_version_a"] = row.fiberVersionA; d["fiber_version_b"] = row.fiberVersionB;
    d["span_a"] = fiberSpanFingerprintToPy(row.spanA); d["span_b"] = fiberSpanFingerprintToPy(row.spanB);
    d["author"] = row.author; d["approved_at"] = row.approvedAt; d["host"] = row.host;
    d["changed_by"] = row.changedBy ? nb::cast(*row.changedBy) : nb::none();
    d["changed_at"] = row.changedAt ? nb::cast(*row.changedAt) : nb::none();
    return d;
}
inline vo::FiberConstraintApproval fiberConstraintApprovalFromPy(const nb::dict& d)
{
    vo::FiberConstraintApproval row;
    const auto optionalText = [&](const char* key) -> std::optional<std::string> {
        if (!d.contains(key) || d[key].is_none()) return std::nullopt;
        return nb::cast<std::string>(d[key]);
    };
    row.identity.fiberA = nb::cast<std::string>(d["fiber_a"]);
    row.identity.fiberB = nb::cast<std::string>(d["fiber_b"]);
    row.identity.kind = fiberConstraintKindFromPy(nb::cast<std::string>(d["kind"]));
    row.identity.arcA = nb::cast<int64_t>(d["arc_a"]);
    row.identity.arcB = nb::cast<int64_t>(d["arc_b"]);
    row.rowUuid = nb::cast<std::string>(d["row_uuid"]);
    row.approvalUuid = nb::cast<std::string>(d["approval_uuid"]);
    const auto state = nb::cast<std::string>(d["state"]);
    if (state == "approved") row.state = vo::FiberConstraintApprovalState::Approved;
    else if (state == "revoked") row.state = vo::FiberConstraintApprovalState::Revoked;
    else throw std::invalid_argument("fiber constraint approval: state must be approved or revoked");
    const auto distance = nb::cast<std::string>(d["distance"]);
    if (distance == "direct") row.distance = vo::FiberConstraintDistance::Direct;
    else if (distance == "distant") row.distance = vo::FiberConstraintDistance::Distant;
    else throw std::invalid_argument("fiber constraint approval: distance must be direct or distant");
    row.supersedes = optionalText("supersedes");
    row.fiberVersionA = nb::cast<int64_t>(d["fiber_version_a"]);
    row.fiberVersionB = nb::cast<int64_t>(d["fiber_version_b"]);
    if (d.contains("span_a")) row.spanA = fiberSpanFingerprintFromPy(d["span_a"]);
    if (d.contains("span_b")) row.spanB = fiberSpanFingerprintFromPy(d["span_b"]);
    row.author = nb::cast<std::string>(d["author"]);
    row.approvedAt = nb::cast<std::string>(d["approved_at"]);
    row.host = d.contains("host") && !d["host"].is_none() ? nb::cast<std::string>(d["host"]) : std::string();
    row.changedBy = optionalText("changed_by");
    row.changedAt = optionalText("changed_at");
    return row;
}

inline void bindFiberConstraintApproval(nb::module_& module)
{
    module.attr("FIBER_CONSTRAINT_APPROVAL_TABLE") = vo::kFiberConstraintApprovalTable;
    module.attr("FIBER_CONSTRAINT_ARC_QUANTUM") = vo::kFiberConstraintArcQuantum;
    module.def("fiber_constraint_approval_table", [] { return extensionTableToPy(vo::fiberConstraintApprovalTable()); },
        "The ext_fiber_constraint_approval definition (docs/extensions.md).")
    .def("fiber_constraint_identity", [](const std::string& fiberA, double arcA, const std::string& fiberB,
                                         double arcB, const std::string& kind, double quantum) {
        const auto identity = vo::fiberConstraintIdentity(fiberA, arcA, fiberB, arcB,
                                                          fiberConstraintKindFromPy(kind), quantum);
        nb::dict d;
        d["fiber_a"] = identity.fiberA; d["fiber_b"] = identity.fiberB;
        d["kind"] = fiberConstraintKindName(identity.kind);
        d["arc_a"] = identity.arcA; d["arc_b"] = identity.arcB;
        return d;
    }, nb::arg("fiber_a"), nb::arg("arc_a"), nb::arg("fiber_b"), nb::arg("arc_b"), nb::arg("kind"),
       nb::arg("quantum") = vo::kFiberConstraintArcQuantum,
       "The identity of one measured crossing: both fibers in UUID order and both anchors in arc quanta.")
    .def("fiber_constraint_approval_from_row", [](const nb::sequence& values) {
        return fiberConstraintApprovalToPy(vo::decodeFiberConstraintApprovalRow(extensionRowFromPy(values)));
    }, nb::arg("values"), "Decode a typed row of ext_fiber_constraint_approval into a dict.")
    .def("fiber_constraint_approval_to_row", [](const nb::dict& row) {
        return extensionRowToPy(vo::fiberConstraintApprovalRow(fiberConstraintApprovalFromPy(row)));
    }, nb::arg("row"), "Encode an approval dict as the typed row to append.")
    .def("revoke_fiber_constraint_approval", [](const nb::dict& row, const std::string& changedBy,
                                                const std::string& now) {
        return fiberConstraintApprovalToPy(
            vo::revokeFiberConstraintApproval(fiberConstraintApprovalFromPy(row), changedBy, now));
    }, nb::arg("row"), nb::arg("changed_by"), nb::arg("now"),
       "The revocation row superseding a live constraint approval.");
}

inline void bindFiberApproval(nb::module_& module)
{
    using Line = std::vector<std::array<double, 3>>;
    module.attr("FIBER_APPROVAL_TABLE") = vo::kFiberApprovalTable;
    module.attr("FIBER_APPROVAL_STRETCH_TABLE") = vo::kFiberApprovalStretchTable;
    module.attr("FIBER_APPROVAL_NORMAL_POINT_TABLE") = vo::kFiberApprovalNormalPointTable;
    module.def("fiber_approval_tables", [] {
        nb::list result;
        for (const auto& table : vo::fiberApprovalTables()) result.append(extensionTableToPy(table));
        return result;
    }, "The three ext_fiber_approval* definitions: header, stretches, normal points (docs/fiber-approval.md).")
    .def("fiber_approval_rows", [](const nb::dict& layer) {
        const auto rows = vo::fiberApprovalRows(fiberApprovalLayerFromPy(layer));
        nb::list stretches, normalPoints;
        for (const auto& row : rows.stretches) stretches.append(extensionRowToPy(row));
        for (const auto& row : rows.normalPoints) normalPoints.append(extensionRowToPy(row));
        nb::dict d;
        d["header"] = extensionRowToPy(rows.header); d["stretches"] = stretches; d["normal_points"] = normalPoints;
        return d;
    }, nb::arg("layer"), "The typed rows of one generation of a layer dict: one header row, its stretch and normal point rows.")
    .def("fiber_line_fingerprint", [](const Line& line, double from, double to) {
        return vo::fiberLineFingerprint(line, from, to);
    }, nb::arg("line"), nb::arg("from_position"), nb::arg("to_position"),
       "FNV-1a 64 over the line points floor(from) .. ceil(to), clamped to the line.")
    .def("set_fiber_stretch_state", [](const nb::sequence& stretches, const std::string& aspect, double from, double to,
                                       const std::string& state, const Line& line) {
        auto values = fiberStretchesFromPy(stretches);
        vo::setFiberStretchState(values, fiberApprovalAspectFromPy(aspect), from, to, fiberStretchStateFromPy(state), line);
        return fiberStretchesToPy(values);
    }, nb::arg("stretches"), nb::arg("aspect"), nb::arg("from_position"), nb::arg("to_position"), nb::arg("state"),
       nb::arg("line"),
       "The stretches with `state` painted over [from, to] of one aspect; undefined only clears.")
    .def("fiber_stretch_state_at", [](const nb::sequence& stretches, const std::string& aspect, double position) {
        return fiberStretchStateName(vo::fiberStretchStateAt(fiberStretchesFromPy(stretches),
                                                             fiberApprovalAspectFromPy(aspect), position));
    }, nb::arg("stretches"), nb::arg("aspect"), nb::arg("position"))
    .def("set_fiber_normal_point", [](const nb::sequence& points, double position, const std::array<double, 3>& normal,
                                      const Line& line, double tolerance) {
        auto values = fiberNormalPointsFromPy(points);
        const auto index = vo::setFiberNormalPoint(values, position, normal, line, tolerance);
        return nb::make_tuple(fiberNormalPointsToPy(values), index);
    }, nb::arg("normal_points"), nb::arg("position"), nb::arg("normal"), nb::arg("line"), nb::arg("tolerance") = 0.5,
       "(normal points, index): the points with the one at `position` added or replaced, and where it is.")
    .def("fiber_approval_status", [](const nb::dict& layer, const Line& line) {
        const auto status = vo::fiberApprovalStatus(fiberApprovalLayerFromPy(layer), line);
        nb::dict d;
        d["stretch_current"] = std::vector<bool>(status.stretchCurrent.begin(), status.stretchCurrent.end());
        d["normal_point_current"] = std::vector<bool>(status.normalPointCurrent.begin(), status.normalPointCurrent.end());
        d["line_normals_usable"] = status.lineNormalsUsable;
        d["stale_stretches"] = status.staleStretches; d["stale_normal_points"] = status.staleNormalPoints;
        return d;
    }, nb::arg("layer"), nb::arg("line"), "Which items of a layer dict still describe `line`, the fiber's stored line points.")
    .def("approved_fiber_line_normals", [](const nb::dict& layer, const Line& line) {
        const auto value = fiberApprovalLayerFromPy(layer);
        return vo::approvedFiberLineNormals(value, vo::fiberApprovalStatus(value, line), line.size());
    }, nb::arg("layer"), nb::arg("line"),
       "Per line point: the saved normal on the current stretches approved for normals and at current normal points, "
       "zero elsewhere. Empty when there is none.");
}

inline void bindExtensionHelpers(nb::module_& module)
{
    bindFiberApproval(module);
    bindFiberConstraintApproval(module);
    module.def("encode_fiber_arc",[](double value){const auto b=vo::encodeFiberArc(value);return nb::bytes(reinterpret_cast<const char*>(b.data()),b.size());});
    module.def("encode_fiber_direction",[](const std::array<double,3>& value){const auto b=vo::encodeFiberDirection(value);return nb::bytes(reinterpret_cast<const char*>(b.data()),b.size());});
    module.def("encode_fiber_winding",[](double value)->nb::object{const auto b=vo::encodeFiberWindingDistance(value);if(!b)return nb::none();const char c=char(*b);return nb::bytes(&c,1);});
    module.def("decode_fiber_arc",[](nb::bytes value){return vo::decodeFiberArc({reinterpret_cast<const uint8_t*>(value.c_str()),value.size()});});
    module.def("decode_fiber_direction",[](nb::bytes value){return vo::decodeFiberDirection({reinterpret_cast<const uint8_t*>(value.c_str()),value.size()});});
    module.def("decode_fiber_winding",[](nb::bytes value){return vo::decodeFiberWindingDistance({reinterpret_cast<const uint8_t*>(value.c_str()),value.size()});});
    module.def("extension_uuid", [](const nb::handle& input) -> nb::object {
        if (nb::isinstance<nb::str>(input)) return extensionValueToPy(vo::extensionUuid(nb::cast<std::string>(input)));
        return nb::cast(vo::extensionUuid(extensionValueFromPy(input)));
    }, nb::arg("uuid"), "Convert canonical UUID text to 16 bytes, or 16 bytes back to UUID text.")
    .def("fiber_source_extension", [](const std::string& uuid, const std::optional<std::string>& sha256,
                                     std::optional<int64_t> fibers, std::optional<int64_t> points) {
        return extensionTableToPy(vo::fiberSourceExtension({uuid, sha256, fibers, points}));
    }, nb::arg("source_uuid"), nb::arg("sha256") = nb::none(),
       nb::arg("fiber_count") = nb::none(), nb::arg("point_count") = nb::none())
    .def("fiber_constraint_extension", [](const std::string& uuid, const std::optional<std::string>& sha256,
                                         std::optional<int64_t> count, const nb::list& basedOn,
                                         const std::optional<std::string>& coordinateSpace) {
        return extensionTableToPy(vo::fiberConstraintExtension({uuid, sha256, count, coordinateSpace},
            vo::extensionSourcesFromJson(jsonDumps(basedOn))));
    }, nb::arg("fiber_source_uuid"), nb::arg("sha256") = nb::none(),
       nb::arg("constraint_count") = nb::none(), nb::arg("based_on") = nb::list(),
       nb::arg("coordinate_space") = nb::none())
    .def("fiber_graph_solve_table", [](const std::string& role) {
        return extensionTableToPy(vo::fiberGraphSolveTable(fiberGraphTableFromPy(role)));
    }, nb::arg("role"))
    .def("fiber_graph_solve_row", [](const std::string& role, const nb::dict& fields) {
        std::map<std::string, vo::ExtensionValue> values;
        for (const auto [name, value] : fields)
            values.emplace(nb::cast<std::string>(name), extensionValueFromPy(value));
        const auto row = vo::fiberGraphSolveRow(fiberGraphTableFromPy(role), values);
        nb::dict result; result["table"] = row.table; result["values"] = extensionRowToPy(row.values); return result;
    }, nb::arg("role"), nb::arg("fields"))
    .def("fiber_graph_run_row", [](const std::string& uuid, const std::string& stage,
                                  const std::string& method, const nb::dict& input) {
        const auto settings=fiberGraphSettingsFromPy(input);
        const auto row = vo::fiberGraphRunRow(uuid, stage, method, settings);
        nb::dict result; result["table"] = row.table; result["values"] = extensionRowToPy(row.values); return result;
    }, nb::arg("run_uuid"), nb::arg("stage"), nb::arg("method"), nb::arg("settings") = nb::dict())
    .def("fiber_graph_input_row", [](const std::string& uuid, const std::string& role,
                                    const nb::dict& source, const std::optional<std::string>& inputRun) {
        nb::list input; input.append(source);
        const auto refs = vo::extensionSourcesFromJson(jsonDumps(input));
        const auto row = vo::fiberGraphInputRow(uuid, role, refs.at(0), inputRun);
        nb::dict result; result["table"] = row.table; result["values"] = extensionRowToPy(row.values); return result;
    }, nb::arg("run_uuid"), nb::arg("role"), nb::arg("source"), nb::arg("input_run_uuid") = nb::none())
    .def("fiber_source_row", [](const std::string& uuid, int64_t sourceId, const nb::dict& annotation) {
        const auto row = vo::fiberSourceRow(uuid, sourceId, jsonDumps(annotation));
        nb::dict result; result["table"] = row.table; result["values"] = extensionRowToPy(row.values); return result;
    }, nb::arg("fiber_uuid"), nb::arg("source_id"), nb::arg("annotation"));
}
} // namespace
