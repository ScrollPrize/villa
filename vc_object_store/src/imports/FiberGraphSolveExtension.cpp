#include "vc_object_store/objects/FiberGraphSolveExtension.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/FiberConstraintGraph.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace vc::objects {
namespace {
using Type = ExtensionType;
using Json = nlohmann::json;
ExtensionColumn required(const char* name, Type type) { return {name, type, false}; }
ExtensionColumn optional(const char* name, Type type) { return {name, type, true}; }

void finiteValue(const ExtensionValue& value)
{
    if (const auto* number = std::get_if<double>(&value); number && !std::isfinite(*number))
        throw std::invalid_argument("fiber graph: nonfinite number");
}
} // namespace

ExtensionTable fiberGraphSolveTable(FiberGraphTable kind)
{
    ExtensionTable table;
    table.extension = kFiberGraphSolveExtension;
    table.provisioning = R"({"length_units":"base_voxels","winding_units":"turns"})";
    const auto fiberTable = [&](const char* suffix) {
        table.name = std::string(kFiberGraphSolveExtension) + suffix;
        table.keySpace = "objects";
        table.columns = {required("fiber_uuid", Type::Blob), required("run_uuid", Type::Blob)};
        table.primaryKey = {"fiber_uuid", "run_uuid"};
        table.indexes = {{{"run_uuid", "fiber_uuid"}, false}};
    };
    switch (kind) {
    case FiberGraphTable::Runs:
        table.name = std::string(kFiberGraphSolveExtension) + "_runs";
        table.columns = {required("run_uuid", Type::Blob), required("stage", Type::Text),
                         required("method", Type::Text), required("settings", Type::Text)};
        table.primaryKey = {"run_uuid"};
        table.indexes = {{{"stage", "run_uuid"}, false}};
        break;
    case FiberGraphTable::Inputs:
        table.name = std::string(kFiberGraphSolveExtension) + "_inputs";
        table.columns = {required("run_uuid", Type::Blob), required("role", Type::Text),
            required("store_uuid", Type::Blob), required("generation_uuid", Type::Blob),
            required("commit_sequence", Type::Integer), optional("input_run_uuid", Type::Blob)};
        table.primaryKey = {"run_uuid", "role"};
        table.indexes = {{{"input_run_uuid", "run_uuid"}, false}};
        break;
    case FiberGraphTable::FiberResults:
        fiberTable("_fibers");
        for (auto name : {"family", "enabled", "normal_sign", "selected_winding", "component"})
            table.columns.push_back(optional(name, Type::Integer));
        for (auto name : {"p_h", "p_v", "p_broken", "sign_probability", "winding_mean",
                          "winding_sigma", "selected_probability", "joint_probability",
                          "anchor_arc", "anchor_u_turns", "anchor_v_vx"})
            table.columns.push_back(optional(name, Type::Real));
        table.indexes.push_back({{"run_uuid", "enabled", "family"}, false});
        break;
    case FiberGraphTable::Samples:
        fiberTable("_samples");
        table.columns.push_back(required("sample_index", Type::Integer));
        table.columns.push_back(required("arc", Type::Real));
        for (auto name : {"direction_x", "direction_y", "direction_z", "normal_x", "normal_y", "normal_z"})
            table.columns.push_back(optional(name, Type::Real));
        table.columns.push_back(required("normal_valid", Type::Integer));
        table.primaryKey.push_back("sample_index");
        break;
    case FiberGraphTable::Hypotheses:
        fiberTable("_hypotheses");
        table.columns.push_back(required("winding", Type::Integer));
        table.columns.push_back(required("probability", Type::Real));
        table.primaryKey.push_back("winding");
        break;
    case FiberGraphTable::Widths:
        table.name = std::string(kFiberGraphSolveExtension) + "_widths";
        table.columns = {required("run_uuid", Type::Blob), required("winding", Type::Integer),
                         required("width_vx", Type::Real)};
        table.primaryKey = {"run_uuid", "winding"};
        break;
    case FiberGraphTable::Positions:
        fiberTable("_positions");
        table.columns.push_back(required("layer", Type::Integer));
        table.columns.push_back(required("hypothesis", Type::Integer));
        table.columns.push_back(required("node_index", Type::Integer));
        table.columns.push_back(required("arc", Type::Real));
        table.columns.push_back(optional("edge_uuid", Type::Blob));
        table.columns.push_back(optional("endpoint", Type::Integer));
        for (auto name : {"u_turns", "u_vx", "v_vx"}) table.columns.push_back(optional(name, Type::Real));
        table.primaryKey.insert(table.primaryKey.end(), {"layer", "hypothesis", "node_index"});
        table.indexes.push_back({{"run_uuid", "edge_uuid"}, false});
        break;
    case FiberGraphTable::ConstraintResults:
        table.name = std::string(kFiberGraphSolveExtension) + "_relations";
        table.keySpace = "edges";
        table.columns = {required("edge_uuid", Type::Blob), required("run_uuid", Type::Blob),
            required("layer", Type::Integer), required("evaluation_index", Type::Integer),
            required("included", Type::Integer)};
        for (auto name : {"hypothesis_a", "hypothesis_b", "fulfilled", "sign_violation"})
            table.columns.push_back(optional(name, Type::Integer));
        for (auto name : {"probability", "same_sheet_probability", "meeting_weight", "required_step",
                          "realized_step", "winding_error", "u_error_vx", "v_error_vx", "loss"})
            table.columns.push_back(optional(name, Type::Real));
        table.primaryKey = {"edge_uuid", "run_uuid", "layer", "evaluation_index"};
        table.indexes = {{{"run_uuid", "edge_uuid"}, false},
                         {{"run_uuid", "included", "sign_violation"}, false}};
        break;
    case FiberGraphTable::Metrics:
        table.name = std::string(kFiberGraphSolveExtension) + "_metrics";
        table.columns = {required("run_uuid", Type::Blob), required("iteration", Type::Integer),
            required("scope", Type::Text), required("name", Type::Text),
            optional("value", Type::Real), optional("count", Type::Integer), required("units", Type::Text)};
        table.primaryKey = {"run_uuid", "iteration", "scope", "name"};
        break;
    default:
        throw std::invalid_argument("fiber graph: unknown table");
    }
    return table;
}

FiberGraphResultColumns readFiberGraphResultColumns(const ObjectCollection& store,const CollectionTxn& txn,
                                                   const std::string& runUuid)
{
    const auto expected=fiberGraphSolveTable(FiberGraphTable::FiberResults);
    bool found=false;
    for(const auto& table:store.extensionTables(txn))if(table.name==expected.name){
        if(table.extension!=expected.extension||table.formatVersion!=expected.formatVersion||
           table.columns!=expected.columns||table.primaryKey!=expected.primaryKey||table.keySpace!=expected.keySpace||table.versioned)
            throw std::invalid_argument("fiber graph: unsupported fiber result table layout");
        found=true;
    }
    if(!found)throw std::invalid_argument("fiber graph: no fiber result table");
    struct Row {std::array<unsigned char,16> uuid;std::array<int64_t,5> values{};std::array<uint8_t,5> valid{};};
    std::vector<Row> rows;
    store.scanExtensionRows(txn,expected.name,[&](const ExtensionRow& source){
        Row row;const auto& key=std::get<std::vector<uint8_t>>(source[0]);
        if(key.size()!=16)throw std::invalid_argument("fiber graph: invalid fiber UUID");
        std::copy(key.begin(),key.end(),row.uuid.begin());
        for(size_t i=0;i<5;++i)if(!std::holds_alternative<std::monostate>(source[i+2])){
            row.values[i]=std::get<int64_t>(source[i+2]);row.valid[i]=1;
        }
        rows.push_back(row);return true;
    },{},{{"run_uuid",extensionUuid(runUuid)}});
    std::sort(rows.begin(),rows.end(),[](const auto& a,const auto& b){return a.uuid<b.uuid;});
    FiberGraphResultColumns result;
    result.fiberUuids.reserve(rows.size());result.integers.reserve(rows.size());result.valid.reserve(rows.size());
    for(auto& row:rows){result.fiberUuids.push_back(row.uuid);result.integers.push_back(row.values);result.valid.push_back(row.valid);}
    return result;
}

void validateFiberGraphField(const ExtensionColumn& column,const ExtensionValue& value)
{
    if(std::holds_alternative<std::monostate>(value)){
        if(!column.nullable)throw std::invalid_argument("fiber graph: NULL "+column.name);
        return;
    }
    const bool valid=column.type==Type::Integer?std::holds_alternative<int64_t>(value):
        column.type==Type::Real?std::holds_alternative<double>(value):
        column.type==Type::Text?std::holds_alternative<std::string>(value):
        std::holds_alternative<std::vector<uint8_t>>(value);
    if(!valid)throw std::invalid_argument("fiber graph: wrong type for "+column.name);
    finiteValue(value);
    if(column.type==Type::Blob)(void)extensionUuid(value);
    if(column.name.starts_with("p_")||column.name.find("probability")!=std::string::npos){
        const auto p=std::get<double>(value);
        if(p<0||p>1)throw std::invalid_argument("fiber graph: probability outside [0,1]");
    }
    if(column.name=="normal_sign"&&std::get<int64_t>(value)!=-1&&std::get<int64_t>(value)!=1)
        throw std::invalid_argument("fiber graph: normal sign must be -1 or +1");
    if(column.name=="layer"&&(std::get<int64_t>(value)<0||std::get<int64_t>(value)>3))
        throw std::invalid_argument("fiber graph: invalid coordinate layer");
    if(column.name=="family"&&(std::get<int64_t>(value)<0||std::get<int64_t>(value)>2))
        throw std::invalid_argument("fiber graph: family must be H=0, V=1 or broken=2");
    if(column.name=="enabled"||column.name=="normal_valid"||column.name=="included"||
       column.name=="fulfilled"||column.name=="sign_violation"||column.name=="endpoint")
        if(std::get<int64_t>(value)!=0&&std::get<int64_t>(value)!=1)
            throw std::invalid_argument("fiber graph: binary flag must be 0 or 1");
    if(column.name=="iteration"||column.name=="commit_sequence"||column.name=="count"||
       column.name=="node_index"||column.name=="sample_index"||column.name=="evaluation_index")
        if(std::get<int64_t>(value)<0)throw std::invalid_argument("fiber graph: negative count/index");
    if(column.name=="arc"||column.name=="anchor_arc"||column.name=="winding_sigma")
        if(std::get<double>(value)<0)throw std::invalid_argument("fiber graph: negative arc/uncertainty");
    if(column.name=="width_vx"&&std::get<double>(value)<=0)
        throw std::invalid_argument("fiber graph: width must be positive");
}

InitialExtensionRow fiberGraphSolveRow(FiberGraphTable kind,
                                      const std::map<std::string, ExtensionValue>& fields)
{
    const auto table = fiberGraphSolveTable(kind);
    InitialExtensionRow row{table.name, {}};
    row.values.reserve(table.columns.size());
    size_t supplied = 0;
    for (const auto& column : table.columns) {
        const auto found = fields.find(column.name);
        if (found == fields.end()) {
            if (!column.nullable) throw std::invalid_argument("fiber graph: missing " + column.name);
            row.values.emplace_back(std::monostate{});
            continue;
        }
        ++supplied;
        const auto& value = found->second;
        validateFiberGraphField(column,value);
        row.values.push_back(value);
    }
    if (supplied != fields.size()) throw std::invalid_argument("fiber graph: unknown field");
    const auto present = [&](const char* name) {
        const auto it = fields.find(name);
        return it != fields.end() && !std::holds_alternative<std::monostate>(it->second);
    };
    if (kind == FiberGraphTable::Metrics && present("value") == present("count"))
        throw std::invalid_argument("fiber graph: metric requires exactly one real value or integer count");
    if (kind == FiberGraphTable::Samples) {
        for (const auto* prefix : {"direction_", "normal_"}) {
            const auto x = std::string(prefix) + "x", y = std::string(prefix) + "y", z = std::string(prefix) + "z";
            const auto n = int(present(x.c_str())) + int(present(y.c_str())) + int(present(z.c_str()));
            if (n != 0 && n != 3) throw std::invalid_argument("fiber graph: partial vector");
        }
        if (std::get<int64_t>(fields.at("normal_valid")) && !present("normal_x"))
            throw std::invalid_argument("fiber graph: valid normal requires coordinates");
    }
    if (kind == FiberGraphTable::Positions) {
        if (std::get<int64_t>(fields.at("layer")) == int64_t(FiberGraphLayer::Shared) &&
            std::get<int64_t>(fields.at("hypothesis")) != 0)
            throw std::invalid_argument("fiber graph: shared coordinates use hypothesis=0 sentinel");
        if (present("edge_uuid") != present("endpoint"))
            throw std::invalid_argument("fiber graph: constraint node requires both edge UUID and endpoint");
    }
    return row;
}

InitialExtensionRow fiberGraphRunRow(const std::string& uuid, const std::string& stage,
    const std::string& method, const std::map<std::string, FiberGraphSetting>& settings)
{
    if (stage.empty() || method.empty()) throw std::invalid_argument("fiber graph: stage/method required");
    auto json = Json::object();
    for (const auto& [name, value] : settings) {
        if (name.empty()) throw std::invalid_argument("fiber graph: setting name required");
        std::visit([&](auto number) {
            if constexpr (std::is_same_v<decltype(number), double>)
                if (!std::isfinite(number)) throw std::invalid_argument("fiber graph: nonfinite setting");
            json[name] = number;
        }, value);
    }
    return fiberGraphSolveRow(FiberGraphTable::Runs, {{"run_uuid", extensionUuid(uuid)},
        {"stage", stage}, {"method", method}, {"settings", json.dump()}});
}

InitialExtensionRow fiberGraphInputRow(const std::string& uuid, const std::string& role,
    const ExtensionSource& input, const std::optional<std::string>& inputRun)
{
    if (role.empty() || input.commitSequence < 0) throw std::invalid_argument("fiber graph: invalid input reference");
    return fiberGraphSolveRow(FiberGraphTable::Inputs, {{"run_uuid", extensionUuid(uuid)}, {"role", role},
        {"store_uuid", extensionUuid(input.storeUuid)}, {"generation_uuid", extensionUuid(input.generation)},
        {"commit_sequence", input.commitSequence},
        {"input_run_uuid", inputRun ? extensionUuid(*inputRun) : ExtensionValue(std::monostate{})}});
}
} // namespace vc::objects
