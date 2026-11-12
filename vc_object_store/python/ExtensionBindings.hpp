#pragma once
#include "vc_object_store/objects/FiberGraphSolveExtension.hpp"

// Included after StorageBindings.hpp: use the same transaction wrappers and
// JSON metadata codec, but never round-trip typed row data through JSON.
namespace {
vo::FiberGraphTable fiberGraphTableFromPy(const std::string& name)
{
    using Table = vo::FiberGraphTable;
    const std::map<std::string, Table> names = {{"runs", Table::Runs}, {"inputs", Table::Inputs},
        {"fibers", Table::FiberResults}, {"samples", Table::Samples}, {"hypotheses", Table::Hypotheses},
        {"widths", Table::Widths}, {"positions", Table::Positions}, {"relations", Table::ConstraintResults},
        {"metrics", Table::Metrics}};
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
    .def("extension_rows", [](T& self, const std::string& table, const nb::sequence& prefix) {
        const auto key = extensionRowFromPy(prefix);
        nb::list result;
        self.withRead([&](auto& db, auto& txn) {
            db.scanExtensionRows(txn, table, [&](const auto& row) { result.append(extensionRowToPy(row)); return true; }, key);
        });
        return result;
    }, nb::arg("table"), nb::arg("key_prefix") = nb::tuple())
    .def("scan_extension_rows", [](T& self, const std::string& table, const nb::callable& visitor, const nb::sequence& prefix) {
        const auto key = extensionRowFromPy(prefix);
        size_t visited = 0;
        self.withRead([&](auto& db, auto& txn) {
            db.scanExtensionRows(txn, table, [&](const auto& row) {
                ++visited;
                const auto keepGoing = visitor(extensionRowToPy(row));
                return keepGoing.is_none() || nb::cast<bool>(keepGoing);
            }, key);
        });
        return visited;
    }, nb::arg("table"), nb::arg("visitor"), nb::arg("key_prefix") = nb::tuple())
    .def("extension_rows_at_version", [](T& self, const std::string& table, const nb::dict& ref) {
        const bool object = has(ref, "object"), relationship = has(ref, "relationship");
        if (object == relationship) throw std::invalid_argument("extension reference needs exactly one object or relationship UUID");
        nb::list result;
        const auto visitor = [&](const vo::ExtensionRow& row) { result.append(extensionRowToPy(row)); return true; };
        const auto store = getStr(ref, "store"), generation = getStr(ref, "generation");
        const auto version = getInt(ref, "version");
        self.withRead([&](auto& db, auto& txn) {
            if (object) db.scanExtensionRowsAtVersion(txn, table,
                vo::ObjectVersion{store, generation, getStr(ref, "object"), version}, visitor);
            else db.scanExtensionRowsAtVersion(txn, table,
                vo::RelationshipVersion{store, generation, getStr(ref, "relationship"), version}, visitor);
        });
        return result;
    }, nb::arg("table"), nb::arg("reference"));
}

inline void bindExtensionHelpers(nb::module_& module)
{
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
        std::map<std::string, vo::FiberGraphSetting> settings;
        for (const auto [name, value] : input) {
            const auto key = nb::cast<std::string>(name);
            if (nb::isinstance<nb::bool_>(value)) settings.emplace(key, nb::cast<bool>(value));
            else if (nb::isinstance<nb::int_>(value)) settings.emplace(key, nb::cast<int64_t>(value));
            else if (nb::isinstance<nb::float_>(value)) settings.emplace(key, nb::cast<double>(value));
            else throw std::invalid_argument("fiber graph settings must be explicit boolean/integer/real scalars");
        }
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
