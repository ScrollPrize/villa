#include "TestSupport.hpp"
#include "vc_object_store/objects/FiberGraphSolveExtension.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/objects/ObjectCopy.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "internal/Database.hpp"
#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>

using namespace vc::objects;
namespace {
using Json = nlohmann::json;
using Table = FiberGraphTable;
const std::string sourceUuid = "10000000-0000-4000-8000-000000000001";
const std::string lineageUuid = "20000000-0000-4000-8000-000000000001";
const std::array<std::string,2> runUuids{
    "30000000-0000-4000-8000-000000000001", "30000000-0000-4000-8000-000000000002"};
const std::array<Table,9> tables{Table::Runs, Table::Inputs, Table::FiberResults, Table::Samples,
    Table::Hypotheses, Table::Widths, Table::Positions, Table::ConstraintResults, Table::Metrics};

size_t column(Table kind, const std::string& name)
{
    const auto spec = fiberGraphSolveTable(kind);
    const auto found = std::find_if(spec.columns.begin(), spec.columns.end(), [&](const auto& c) { return c.name == name; });
    require(found != spec.columns.end(), "test column exists in the declared workflow schema");
    return size_t(found - spec.columns.begin());
}

void append(ObjectStore& store, WriteTxn& txn, const InitialExtensionRow& row)
{
    store.appendExtensionRows(txn, row.table, std::span<const ExtensionRow>(&row.values, 1));
}

std::vector<InitialExtensionRow> fiberRows(const std::string& fiber, const std::string& edge, size_t run)
{
    const auto id = extensionUuid(fiber), runId = extensionUuid(runUuids[run]);
    const int64_t winding = int64_t(5 + run);
    std::vector<InitialExtensionRow> rows;
    rows.push_back(fiberGraphSolveRow(Table::FiberResults, {
        {"fiber_uuid", id}, {"run_uuid", runId}, {"family", int64_t(run)}, {"enabled", int64_t(1)},
        {"normal_sign", run ? int64_t(-1) : int64_t(1)}, {"selected_winding", winding},
        {"p_h", run ? 0.1 : 0.9}, {"p_v", run ? 0.8 : 0.05}, {"p_broken", run ? 0.1 : 0.05},
        {"sign_probability", 0.95}, {"winding_mean", run ? 6.1 : 5.5}, {"winding_sigma", 0.6},
        {"selected_probability", run ? 0.9 : 0.5}, {"joint_probability", run ? 0.81 : 0.475},
        {"anchor_arc", 0.0}, {"anchor_u_turns", 2.3}, {"anchor_v_vx", 100.0}}));
    if (!run) {
        rows.push_back(fiberGraphSolveRow(Table::Samples, {{"fiber_uuid", id}, {"run_uuid", runId},
            {"sample_index", int64_t(0)}, {"arc", 0.0}, {"direction_x", 1.0}, {"direction_y", 0.0},
            {"direction_z", 0.0}, {"normal_x", 0.0}, {"normal_y", 1.0}, {"normal_z", 0.0},
            {"normal_valid", int64_t(1)}}));
    } else {
        rows.push_back(fiberGraphSolveRow(Table::Samples, {{"fiber_uuid", id}, {"run_uuid", runId},
            {"sample_index", int64_t(0)}, {"arc", 0.0}, {"normal_valid", int64_t(0)}}));
    }
    for (int64_t alternative = 0; alternative < 2; ++alternative)
        rows.push_back(fiberGraphSolveRow(Table::Hypotheses, {{"fiber_uuid", id}, {"run_uuid", runId},
            {"winding", winding + alternative}, {"probability", run ? (alternative ? 0.1 : 0.9) : 0.5}}));
    for (const auto layer : {FiberGraphLayer::Shared, FiberGraphLayer::Intermediate,
                             FiberGraphLayer::Final, FiberGraphLayer::Refined}) {
        std::map<std::string, ExtensionValue> fields{{"fiber_uuid", id}, {"run_uuid", runId},
            {"layer", int64_t(layer)}, {"hypothesis", layer == FiberGraphLayer::Shared ? int64_t(0) : winding},
            {"node_index", int64_t(0)}, {"arc", 250.0}, {"edge_uuid", extensionUuid(edge)},
            {"endpoint", int64_t(0)}, {"u_turns", 2.3 + 0.01 * int64_t(layer)},
            {"v_vx", 100.0 + int64_t(layer)}};
        if (layer == FiberGraphLayer::Final || layer == FiberGraphLayer::Refined)
            fields["u_vx"] = 70000.0 + 10.0 * int64_t(layer);
        rows.push_back(fiberGraphSolveRow(Table::Positions, fields));
    }
    return rows;
}

InitialExtensionRow relationRow(const std::string& edge, size_t run)
{
    return fiberGraphSolveRow(Table::ConstraintResults, {{"edge_uuid", extensionUuid(edge)},
        {"run_uuid", extensionUuid(runUuids[run])}, {"layer", int64_t(FiberGraphLayer::Final)},
        {"evaluation_index", int64_t(0)}, {"included", int64_t(1)}, {"hypothesis_a", int64_t(5 + run)},
        {"hypothesis_b", int64_t(5 + run)}, {"fulfilled", int64_t(run == 0)}, {"sign_violation", int64_t(0)},
        {"probability", 0.8}, {"same_sheet_probability", 0.7}, {"meeting_weight", 0.4},
        {"required_step", 0.0}, {"realized_step", 0.02}, {"winding_error", 0.02},
        {"u_error_vx", 2.5}, {"v_error_vx", 1.5}, {"loss", 8.5}});
}

void definitionsAndValidation()
{
    const auto source = fiberSourceExtension(FiberSourceInfo{sourceUuid, std::string(64, 'a'), 2, 100});
    const auto constraints = fiberConstraintExtension(FiberConstraintSourceInfo{sourceUuid, {}, 1});
    require(source.name == "ext_fiber_graph_solve_sources" && constraints.name == "ext_fiber_graph_solve_constraints",
            "import measurements and workflow results share the named extension family");
    require(Json::parse(source.provisioning) == Json({{"source_uuid", sourceUuid}, {"source_sha256", std::string(64, 'a')},
                {"fiber_count", 2}, {"point_count", 100}}), "source provisioning is explicitly constructed compact identity/count data");
    require(Json::parse(constraints.provisioning) == Json({{"fiber_source_uuid", sourceUuid}, {"constraint_count", 1},
                {"length_units", "base_voxels"}, {"winding_units", "turns"}, {"endpoint_order", "canonical"},
                {"arc_encoding","float32-le"},{"direction_encoding","lasagna-xy-u8-hemisphere-i8"},
                {"connector_direction","a_to_b"},{"winding_scale",16},{"winding_code_offset",1},{"winding_invalid_code",0}}),
            "constraint provenance identifies its fiber source, not an invented constraint file UUID");
    for (const auto kind : tables) {
        const auto spec = fiberGraphSolveTable(kind);
        require(spec.extension == kFiberGraphSolveExtension && spec.formatVersion == 1 && !spec.versioned,
                "workflow definitions are optional immutable extension format one");
        require(Json::parse(spec.provisioning) == Json({{"length_units", "base_voxels"}, {"winding_units", "turns"}}),
                "workflow provisioning contains units, not copied configuration trees or source reports");
    }
    const auto run = fiberGraphRunRow(runUuids[0], "winding", "irls", {{"iterations", int64_t(100)}, {"tolerance", 0.01}, {"fixed_width", true}});
    require(Json::parse(std::get<std::string>(run.values[column(Table::Runs, "settings")])) ==
                Json({{"iterations", 100}, {"tolerance", 0.01}, {"fixed_width", true}}),
            "run settings retain explicitly supplied scalar values only");
    mustFail([&] { fiberGraphRunRow(runUuids[0], "", "irls"); });
    mustFail([&] { fiberGraphRunRow(runUuids[0], "winding", "irls", {{"bad", std::numeric_limits<double>::infinity()}}); });
    mustFail([&] { fiberGraphInputRow(runUuids[0], "source", {sourceUuid, lineageUuid, -1}); });
    mustFail([&] { fiberGraphSolveRow(Table::FiberResults, {{"fiber_uuid", extensionUuid(sourceUuid)}}); });
    mustFail([&] { fiberGraphSolveRow(Table::FiberResults, {{"fiber_uuid", extensionUuid(sourceUuid)},
        {"run_uuid", extensionUuid(runUuids[0])}, {"unknown", 1.0}}); });
    mustFail([&] { fiberGraphSolveRow(Table::FiberResults, {{"fiber_uuid", extensionUuid(sourceUuid)},
        {"run_uuid", extensionUuid(runUuids[0])}, {"p_h", 1.01}}); });
    mustFail([&] { fiberGraphSolveRow(Table::FiberResults, {{"fiber_uuid", extensionUuid(sourceUuid)},
        {"run_uuid", extensionUuid(runUuids[0])}, {"normal_sign", int64_t(0)}}); });
    mustFail([&] { fiberGraphSolveRow(Table::Widths, {{"run_uuid", std::string(runUuids[0])},
        {"winding", int64_t(5)}, {"width_vx", 9000.0}}); });
}

void multipleRuns()
{
    TemporaryDirectory dir;
    auto store = ObjectStore::create(dir.path / "results.vcobj", OpenOptions{.author = "fiber-graph-test"});
    const auto firstFiber = util::newUuid(), secondFiber = util::newUuid(), edgeUuid = util::newUuid();
    {
        auto txn = store.write();
        store.createExtensionTable(txn, fiberSourceExtension(FiberSourceInfo{sourceUuid, {}, 2, 100}));
        store.createExtensionTable(txn, fiberConstraintExtension(FiberConstraintSourceInfo{sourceUuid, {}, 1}));
        for (const auto& uuid : {firstFiber, secondFiber}) {
            ObjectRecord fiber; fiber.uuid = uuid; fiber.kind = "fiber"; store.upsertObject(txn, fiber);
        }
        EdgeRecord edge; edge.uuid = edgeUuid; edge.kind = "perpendicular"; edge.aUuid = firstFiber; edge.bUuid = secondFiber;
        edge.aAnchor.wholeGeometry = edge.bAnchor.wholeGeometry = true; store.addEdge(txn, edge);
        txn.commit();
    }
    ExtensionSource source;
    {
        auto txn = store.read(); const auto status = store.status(txn);
        source = {status.storeUuid, status.generation, status.commitSequence};
        require(store.extensionTables(txn).size() == 2 && store.provisioning(txn).schemaVersion == 22,
                "import creates only its two optional tables without changing the core schema");
    }
    ExtensionSource priorRun;
    for (size_t run = 0; run < runUuids.size(); ++run) {
        auto txn = store.write();
        for (const auto kind : tables) store.createExtensionTable(txn, fiberGraphSolveTable(kind));
        append(store, txn, fiberGraphRunRow(runUuids[run], "winding", run ? "lemon" : "irls", {{"iterations", int64_t(100)}}));
        append(store, txn, fiberGraphInputRow(runUuids[run], "source", source));
        if (run) append(store, txn, fiberGraphInputRow(runUuids[run], "previous_result", priorRun, runUuids[0]));
        for (const auto& row : fiberRows(firstFiber, edgeUuid, run)) append(store, txn, row);
        append(store, txn, relationRow(edgeUuid, run));
        append(store, txn, fiberGraphSolveRow(Table::Widths, {{"run_uuid", extensionUuid(runUuids[run])},
            {"winding", int64_t(5)}, {"width_vx", run ? 9500.0 : 9000.0}}));
        append(store, txn, fiberGraphSolveRow(Table::Metrics, {{"run_uuid", extensionUuid(runUuids[run])},
            {"iteration", int64_t(100)}, {"scope", std::string("reference")}, {"name", std::string("perpendicular_l1")},
            {"value", run ? 0.2 : 0.25}, {"units", std::string("turns")}}));
        append(store, txn, fiberGraphSolveRow(Table::Metrics, {{"run_uuid", extensionUuid(runUuids[run])},
            {"iteration", int64_t(100)}, {"scope", std::string("reference")}, {"name", std::string("perpendicular_count")},
            {"count", int64_t(12)}, {"units", std::string("constraints")}}));
        txn.commit();
        auto read = store.read(); const auto status = store.status(read);
        priorRun = {status.storeUuid, status.generation, status.commitSequence};
    }
    auto txn = store.read();
    require(store.extensionTables(txn).size() == 11 && store.objectCount(txn, "fiber", true) == 2,
            "multiple runs share core fibers instead of creating result proxy objects");
    for (size_t run = 0; run < runUuids.size(); ++run) {
        for (const auto& expected : fiberRows(firstFiber, edgeUuid, run)) {
            const std::vector<ExtensionValue> prefix{extensionUuid(firstFiber), extensionUuid(runUuids[run])};
            bool found = false;
            store.scanExtensionRows(txn, expected.table, [&](const auto& actual) { if (actual == expected.values) found = true; return true; }, prefix);
            require(found, "classification, samples, hypotheses and every coordinate layer retain independent run values");
        }
        const auto expected = relationRow(edgeUuid, run);
        const std::vector<ExtensionValue> prefix{extensionUuid(edgeUuid), extensionUuid(runUuids[run])};
        std::vector<ExtensionRow> found;
        store.scanExtensionRows(txn, expected.table, [&](const auto& actual) { found.push_back(actual); return true; }, prefix);
        require(found == std::vector<ExtensionRow>{expected.values}, "constraint fulfillment is stored per relation and run");
    }
    auto* db = ObjectStoreImplAccess::conn(txn);
    detail::Statement binary(db, "SELECT typeof(fiber_uuid),length(fiber_uuid),typeof(run_uuid),length(run_uuid),component IS NULL "
                                "FROM ext_fiber_graph_solve_fibers");
    require(binary.row() && binary.text(0) == "blob" && binary.i64(1) == 16 && binary.text(2) == "blob" &&
                binary.i64(3) == 16 && binary.i64(4) == 1, "run/entity IDs use binary UUIDs and absent result values remain NULL");
    detail::Statement gap(db, "SELECT normal_valid,normal_x IS NULL,normal_y IS NULL,normal_z IS NULL "
                             "FROM ext_fiber_graph_solve_samples WHERE run_uuid=?", detail::parseUuid(runUuids[1]));
    require(gap.row() && gap.i64(0) == 0 && gap.i64(1) && gap.i64(2) && gap.i64(3),
            "invalid normal samples are explicit gaps, not fabricated zero normals");
    detail::Statement input(db, "SELECT commit_sequence,input_run_uuid FROM ext_fiber_graph_solve_inputs "
                               "WHERE run_uuid=? AND role='previous_result'", detail::parseUuid(runUuids[1]));
    require(input.row() && input.i64(0) == source.commitSequence + 1 && detail::formatUuid(input.blob16(1)) == runUuids[0],
            "a dependent run pins the earlier output commit rather than its own write commit");
    detail::Statement phase(db, "SELECT winding_mean,anchor_u_turns FROM ext_fiber_graph_solve_fibers WHERE run_uuid=?",
                            detail::parseUuid(runUuids[0]));
    require(phase.row() && phase.f64(0) == 5.5 && phase.f64(1) == 2.3,
            "integer-distribution center N and continuous phase U remain separate stored values");
    detail::Statement widths(db, "SELECT width_vx FROM ext_fiber_graph_solve_widths ORDER BY run_uuid");
    require(widths.row() && widths.f64(0) == 9000.0 && widths.row() && widths.f64(0) == 9500.0 && !widths.row(),
            "independent runs keep distinct winding width models");
    detail::Statement metrics(db, "SELECT value,count IS NULL FROM ext_fiber_graph_solve_metrics "
                                 "WHERE name='perpendicular_l1' ORDER BY run_uuid");
    require(metrics.row() && metrics.f64(0) == 0.25 && metrics.i64(1) == 1 &&
                metrics.row() && metrics.f64(0) == 0.2 && metrics.i64(1) == 1,
            "floating metrics retain each run's value without overloading integer counts");
    detail::Statement counts(db, "SELECT count,value IS NULL FROM ext_fiber_graph_solve_metrics WHERE name='perpendicular_count'");
    require(counts.row() && counts.i64(0) == 12 && counts.i64(1) == 1,
            "sample counts use separate integer metric rows");
}

void shardedResults(bool packed)
{
    TemporaryDirectory dir;
    constexpr size_t count = 2;
    InitialImportOptions options; options.store.author = "fiber-graph-test"; options.shards.clear();
    for (size_t i = 0; i < count; ++i) options.shards.push_back({{int64_t(i * 100 + 1), int64_t((i + 1) * 100)}});
    for (const auto kind : tables) options.extensions.push_back(fiberGraphSolveTable(kind));
    if (packed) { options.packed.emplace(); options.packed->memoryFirst = true; options.packed->disposableBulk = true;
        options.packed->backgroundFlush = false; options.packed->residentBytes = 64 * 1024 * 1024; }
    std::array<std::string,count> fibers, edges;
    for (size_t i = 0; i < count; ++i) {
        do { fibers[i] = util::newUuid(); } while (uuidShardIndex(fibers[i], count) != i);
        do { edges[i] = util::newUuid(); } while (uuidShardIndex(edges[i], count) != i);
    }
    const auto result = importInitialCollection(dir.path / "results", options,
        [&](size_t i, InitialShardImport& shard) {
            InitialObjectRecord row; row.record.uuid = fibers[i]; row.record.kind = "fiber";
            for (size_t run = 0; run < runUuids.size(); ++run) {
                for (auto& value : fiberRows(fibers[i], edges[i], run)) row.extensions.push_back(std::move(value));
                if (i == 0) row.extensions.push_back(fiberGraphRunRow(runUuids[run], "winding", "irls"));
            }
            shard.appendObjects(std::span<const InitialObjectRecord>(&row, 1));
        }, [&](size_t i, InitialShardImport& shard) {
            bool sent = false;
            shard.importEdges([&](std::vector<InitialCollectionEdge>& batch) {
                if (sent) return false; sent = true;
                InitialCollectionEdge row; row.a = {shard.storeUuid(), fibers[i]};
                row.b = {shard.storeUuid(), fibers[(i + 1) % count]};
                row.record.uuid = edges[i]; row.record.kind = "perpendicular";
                row.record.aAnchor.wholeGeometry = row.record.bAnchor.wholeGeometry = true;
                for (size_t run = 0; run < runUuids.size(); ++run) row.extensions.push_back(relationRow(edges[i], run));
                batch.push_back(std::move(row)); return true;
            });
        });
    auto storeReference = ObjectCollection::create(dir.path / "store-reference.vcobj", {result.path},
        OpenOptions{.history = HistoryMode::Off, .author = "fiber-graph-test"});
    {
        auto overlay=ObjectStore::open(dir.path / "store-reference.vcobj",OpenOptions{.readOnly=true});
        auto read=overlay.read();
        const LocalStoreReader reader(overlay,read);
        require(reader.status().extensionsPresent,"referenced extension presence is visible through an ordinary reader");
        mustFail([&]{(void)readObjectCopy(reader,fibers[0]);});
    }
    auto txn = storeReference.read();
    for (size_t i = 0; i < count; ++i) for (size_t run = 0; run < runUuids.size(); ++run) {
        for (const auto& expected : fiberRows(fibers[i], edges[i], run)) {
            bool found = false;
            const std::vector<ExtensionValue> prefix{extensionUuid(fibers[i]), extensionUuid(runUuids[run])};
            storeReference.scanExtensionRows(txn, expected.table, [&](const auto& actual) { if (actual == expected.values) found = true; return true; }, prefix);
            require(found, "fiber result rows route with their owner through store_references to plain/packed shards");
        }
        const auto expected = relationRow(edges[i], run);
        const std::vector<ExtensionValue> prefix{extensionUuid(edges[i]), extensionUuid(runUuids[run])};
        std::vector<ExtensionRow> found;
        storeReference.scanExtensionRows(txn, expected.table, [&](const auto& actual) { found.push_back(actual); return true; }, prefix);
        require(found == std::vector<ExtensionRow>{expected.values}, "constraint result rows route with their cross-shard edge");
    }
    size_t runs = 0;
    storeReference.scanExtensionRows(txn, fiberGraphSolveTable(Table::Runs).name, [&](const auto&) { ++runs; return true; });
    require(runs == 2, "run descriptors are stored once and discovered without duplicating them per fiber");
}
} // namespace

int main()
{
    definitionsAndValidation();
    multipleRuns();
    shardedResults(false);
    shardedResults(true);
}
