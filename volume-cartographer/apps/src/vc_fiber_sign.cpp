// Native CPU normal-sign solve over an extracted fiber-constraint store.
//
// Reads one store's compact constraint graph and its radial priors, assembles
// the sign problem the same way the reference stage does, runs one of the CPU
// backends and writes a JSON summary. Optionally writes the per-fiber signs
// and the prepared events, so another solver's result can be compared against
// the same inputs. With `support_output` it instead reports the sign support
// of given signs (FiberSignSupport.hpp) and solves nothing.
#include "vc/fiber_tracer/FiberNormalSign.hpp"
#include "vc/fiber_tracer/FiberSignSupport.hpp"

#include <vc_object_store/objects/FiberConstraintGraph.hpp>
#include <vc_object_store/objects/FiberGraphSolveExtension.hpp>
#include <vc_object_store/objects/ObjectCollection.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string_view>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace vc::fiber_tracer;

Json readJson(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot read " + path);
    Json value;
    stream >> value;
    return value;
}

void writeJson(const std::string& path, const Json& value)
{
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write " + path);
    stream << value.dump(1) << '\n';
}

double seconds(const Clock::time_point since)
{
    return std::chrono::duration<double>(Clock::now() - since).count();
}

FiberSignOptions optionsFrom(const Json& cfg)
{
    FiberSignOptions options;
    const auto backend = cfg.value("backend", std::string{"irls"});
    if (backend == "irls") options.backend = FiberSignBackend::Irls;
    else if (backend == "ceres") options.backend = FiberSignBackend::Ceres;
    else throw std::invalid_argument("backend must be irls or ceres");
    const auto solver = cfg.value("ceres_linear_solver", std::string{"sparse_normal_cholesky"});
    if (solver == "sparse_normal_cholesky") options.linearSolver = FiberSignLinearSolver::SparseNormalCholesky;
    else if (solver == "sparse_schur") options.linearSolver = FiberSignLinearSolver::SparseSchur;
    else if (solver == "iterative_schur") options.linearSolver = FiberSignLinearSolver::IterativeSchur;
    else if (solver == "cgnr") options.linearSolver = FiberSignLinearSolver::Cgnr;
    else if (solver == "dense_normal_cholesky") options.linearSolver = FiberSignLinearSolver::DenseNormalCholesky;
    else if (solver == "dense_qr") options.linearSolver = FiberSignLinearSolver::DenseQr;
    else if (solver == "dense_schur") options.linearSolver = FiberSignLinearSolver::DenseSchur;
    else throw std::invalid_argument("ceres_linear_solver must be sparse_normal_cholesky, sparse_schur, "
                                     "iterative_schur, cgnr, dense_normal_cholesky, dense_qr or dense_schur");
    const auto distance = cfg.value("winding_distance_weight", std::string{"inverse_square"});
    if (distance == "inverse_square") options.distanceWeight = FiberSignDistanceWeight::InverseSquare;
    else if (distance == "none") options.distanceWeight = FiberSignDistanceWeight::None;
    else throw std::invalid_argument("winding_distance_weight must be inverse_square or none");
    options.dotWeightPower = cfg.value("normal_dot_weight_power", options.dotWeightPower);
    options.priorWeight = cfg.value("prior_weight", options.priorWeight);
    options.fixedSignWeight = cfg.value("fixed_sign_weight", options.fixedSignWeight);
    options.brokenPenalty = cfg.value("broken_penalty", options.brokenPenalty);
    options.epsilon = cfg.value("epsilon", options.epsilon);
    options.iterations = cfg.value("iterations", options.iterations);
    options.cgIterations = cfg.value("cg_iterations", options.cgIterations);
    options.cgTolerance = cfg.value("cg_tolerance", options.cgTolerance);
    options.keepRounds = cfg.value("keep_rounds", options.keepRounds);
    options.tolerance = cfg.value("tolerance", options.tolerance);
    options.fixedKeep = cfg.value("fixed_keep", options.fixedKeep);
    options.threads = cfg.value("threads", options.threads);
    options.history = cfg.value("history", true);
    return options;
}

Json optionsJson(const Json& cfg, const FiberSignOptions& options)
{
    return Json{{"backend", cfg.value("backend", std::string{"irls"})},
                {"ceres_linear_solver", cfg.value("ceres_linear_solver", std::string{"sparse_normal_cholesky"})},
                {"winding_distance_weight", cfg.value("winding_distance_weight", std::string{"inverse_square"})},
                {"normal_dot_weight_power", options.dotWeightPower},
                {"prior_weight", options.priorWeight},
                {"fixed_sign_weight", options.fixedSignWeight},
                {"broken_penalty", options.brokenPenalty},
                {"epsilon", options.epsilon},
                {"iterations", options.iterations},
                {"cg_iterations", options.cgIterations},
                {"cg_tolerance", options.cgTolerance},
                {"keep_rounds", options.keepRounds},
                {"tolerance", options.tolerance},
                {"fixed_keep", options.fixedKeep},
                {"threads", options.threads}};
}

// The reference stage's metric names, so a comparison is a literal diff.
Json metricsJson(const FiberSignMetrics& metrics)
{
    return Json{{"broken_fibers_percent", metrics.brokenFibersPercent},
                {"expected_broken_percent", metrics.expectedBrokenPercent},
                {"retained_events_percent", metrics.retainedEventsPercent},
                {"weighted_disagreement_all_percent", metrics.weightedDisagreementPercent},
                {"weighted_disagreement_retained_percent", metrics.weightedRetainedDisagreementPercent},
                {"retained_disagreement_original_weight_percent", metrics.retainedDisagreementOriginalWeightPercent},
                {"outward_agreement", metrics.outwardAgreement},
                {"uncertain_signs_percent", metrics.uncertainSignsPercent},
                {"strict_events", metrics.strictEvents},
                {"strict_disagreement_percent", metrics.strictDisagreementPercent},
                {"strict_retained_disagreement_percent", metrics.strictRetainedDisagreementPercent}};
}

// The crossings the sign solve uses: perpendicular, with a valid normal on both
// sides (the reference stage's selection). Counts the perpendicular ones.
std::vector<std::size_t> signEventRows(const vc::objects::FiberGraphColumns& columns, std::size_t& perpendicular)
{
    std::vector<std::size_t> rows;
    perpendicular = 0;
    for (std::size_t e = 0; e < columns.events; ++e) {
        if (columns.parallel[e] != 0) continue;
        ++perpendicular;
        if (columns.validA[e] != 0 && columns.validB[e] != 0) rows.push_back(e);
    }
    return rows;
}

double eventDot(const vc::objects::FiberGraphColumns& columns, const std::size_t e)
{
    const auto& a = columns.normalA[e];
    const auto& b = columns.normalB[e];
    return std::clamp(a[0] * b[0] + a[1] * b[1] + a[2] * b[2], -1.0, 1.0);
}

// Sign support of the store's normal-sign run (`sign_run` selects one when
// there are several) over the sign solve's crossings that agree with its
// signs, weighted as the solve weights them (same config keys). Writes `support_output` (VCSUPP1:
// magic, u64 fibers, then per fiber a 16-byte uuid, u64 n and n samples of
// arc, missed length, missed crossing cost and local cut (fiberSignCut) as
// doubles) and its JSON summary next to it.
int runSupport(const Json& cfg)
{
    const auto started = Clock::now();
    vc::objects::OpenOptions storeOptions;
    storeOptions.readOnly = true;
    auto store = vc::objects::ObjectCollection::open(cfg.at("database").get<std::string>(), storeOptions);
    const auto txn = store.read();
    const auto columns = vc::objects::readFiberGraphColumns(store, txn, cfg.value("workers", std::size_t{32}));

    // Signs of the store's sign-stage run (or `sign_run`), read natively from its fiber result table.
    const std::string signRun = vc::objects::selectFiberGraphRun(store, txn, "normal_sign",
        cfg.contains("sign_run") ? std::optional<std::string>(cfg.at("sign_run").get<std::string>()) : std::nullopt);
    const auto results = vc::objects::readFiberGraphResultColumns(store, txn, signRun);
    constexpr std::size_t signColumn = 2;
    static_assert(std::string_view(vc::objects::FiberGraphResultColumns::names[signColumn]) == "normal_sign");
    std::map<vc::objects::FiberGraphUuid, int> lookup;
    for (std::size_t r = 0; r < results.fiberUuids.size(); ++r)
        if (results.valid[r][signColumn] != 0) lookup[results.fiberUuids[r]] = int(results.integers[r][signColumn]);
    std::vector<int> sign(columns.nodes, 0);
    std::vector<double> lengths(columns.lengths.get(), columns.lengths.get() + columns.nodes);
    for (std::size_t v = 0; v < columns.nodes; ++v) {
        const auto found = lookup.find(columns.fiberUuids[v]);
        if (found != lookup.end()) sign[v] = found->second;
    }

    // The sign solve's crossings and weights; those disagreeing with the signs are detected flips.
    const FiberSignOptions options = optionsFrom(cfg);
    std::size_t perpendicular = 0;
    std::vector<FiberSupportCrossing> crossings, disagreeingCrossings;
    std::int64_t used = 0, disagreeing = 0, weak = 0;
    const double minWeight = cfg.value("support_min_weight", 0.1);
    if (!(minWeight > 0.0)) throw std::invalid_argument("support_min_weight must be positive");
    for (const std::size_t e : signEventRows(columns, perpendicular)) {
        const auto a = columns.endpoints[e][0], b = columns.endpoints[e][1];
        if (a == b || sign[a] == 0 || sign[b] == 0) continue;
        const double dot = eventDot(columns, e);
        const double weight = fiberSignEventWeight(dot, columns.distance[e], options);
        // Weak crossings (perpendicular or far-apart axes) support nothing: 1 / weight would explode.
        if (!(weight >= minWeight)) {
            ++weak;
            continue;
        }
        ++used;
        const FiberSupportCrossing crossing{a, b, columns.arcs[e][0], columns.arcs[e][1], weight};
        if (dot * sign[a] * sign[b] < 0.0) {
            ++disagreeing;
            disagreeingCrossings.push_back(crossing);
            continue;
        }
        crossings.push_back(crossing);
    }
    const double spacing = cfg.value("support_spacing", 64.0);
    const FiberSupportResult result = fiberSignSupport(lengths, crossings, spacing);
    FiberCutOptions cutOptions;
    cutOptions.cap = cfg.value("support_cut_cap", cutOptions.cap);
    cutOptions.maxHops = cfg.value("support_cut_hops", cutOptions.maxHops);
    cutOptions.maxFibers = cfg.value("support_cut_fibers", cutOptions.maxFibers);
    cutOptions.chainWeightLength = cfg.value("support_cut_chain_length", cutOptions.chainWeightLength);
    const auto cutStarted = Clock::now();
    const FiberCutResult cut = fiberSignCut(lengths, crossings, spacing, cutOptions);
    const double cutSeconds = seconds(cutStarted);

    const std::string output = cfg.at("support_output").get<std::string>();
    std::ofstream file(output, std::ios::binary);
    if (!file) throw std::runtime_error("cannot write " + output);
    const char magic[8] = {'V', 'C', 'S', 'U', 'P', 'P', '1', 0};
    file.write(magic, sizeof magic);
    const std::uint64_t fibers = columns.nodes;
    file.write(reinterpret_cast<const char*>(&fibers), sizeof fibers);
    std::vector<double> buffer;
    double linked = 0.0, total = 0.0;
    for (std::size_t v = 0; v < columns.nodes; ++v) {
        const auto& points = result.points[v];
        file.write(reinterpret_cast<const char*>(columns.fiberUuids[v].data()), 16);
        const std::uint64_t n = points.size();
        file.write(reinterpret_cast<const char*>(&n), sizeof n);
        buffer.clear();
        for (std::size_t k = 0; k < points.size(); ++k) {
            const auto& p = points[k];
            buffer.insert(buffer.end(), {p.arc, p.missedLength, p.missedCrossingCost, cut.points[v][k].cut});
            total += 1.0;
            if (std::isfinite(p.missedLength)) linked += 1.0;
        }
        file.write(reinterpret_cast<const char*>(buffer.data()), std::streamsize(buffer.size() * sizeof(double)));
    }
    if (!file) throw std::runtime_error("write failed " + output);

    Json summary{{"store", store.uuid(txn)},
                       {"sign_run", signRun},
                       {"fibers", columns.nodes},
                       {"signed_fibers", std::count_if(sign.begin(), sign.end(), [](int s) { return s != 0; })},
                       {"crossings", used},
                       {"min_weight", minWeight},
                       {"weak_dropped", weak},
                       {"disagreeing_dropped", disagreeing},
                       {"crossing_ends", 2 * crossings.size()},
                       {"chain_edges", result.chainEdges},
                       {"bridge_chain_edges", result.bridgeChainEdges},
                       {"bridge_crossings", result.bridgeCrossings},
                       {"two_edge_connected_components", result.components},
                       {"core_ends", result.coreEnds},
                       {"unlinked_ends", result.unlinkedEnds},
                       {"spacing", spacing},
                       {"cut", Json{{"cap", cutOptions.cap}, {"hops", cutOptions.maxHops},
                                    {"fibers", cutOptions.maxFibers}, {"chain_length", cutOptions.chainWeightLength},
                                    {"segments", cut.segments}, {"seconds", cutSeconds}}},
                       {"samples", total},
                       {"linked_sample_percent", 100.0 * linked / std::max(total, 1.0)},
                       {"seconds", seconds(started)}};
    // lambda and q from the core's detectable flips (FiberSignSupport.hpp).
    const auto calibration = fiberSignCalibration(lengths, crossings, disagreeingCrossings,
                                                  cfg.value("calibration_min_bin_flips", std::int64_t{30}));
    const auto binsJson = [](const std::vector<FiberCalibrationBin>& bins, const char* exposure) {
        Json out = Json::array();
        for (const auto& b : bins)
            out.push_back(Json{{"lo", b.lo}, {"hi", std::isfinite(b.hi) ? Json(b.hi) : Json(nullptr)},
                               {"count", b.count}, {"flips", b.events}, {exposure, b.exposure},
                               {"rate", b.exposure > 0.0 ? Json(double(b.events) / b.exposure) : Json(nullptr)}});
        return out;
    };
    summary["calibration"] = Json{{"lambda_per_voxel", calibration.lambda},
                                  {"q_per_unit_weight", calibration.q},
                                  {"lambda_bound_per_voxel", calibration.lambdaBound},
                                  {"q_bound_per_unit_weight", calibration.qBound},
                                  {"core_crossings", calibration.coreCrossings},
                                  {"core_disagreeing", calibration.coreDisagreeing},
                                  {"chain_flips", calibration.chainSwitches},
                                  {"wrong_crossings", calibration.wrongCrossings},
                                  {"core_chain_length_voxels", calibration.coreChainLength},
                                  {"core_inverse_weight", calibration.coreInverseWeight},
                                  {"by_chain_gap_voxels", binsJson(calibration.gapBins, "chain_voxels")},
                                  {"by_crossing_weight", binsJson(calibration.weightBins, "inverse_weight")}};
    summary["seconds"] = seconds(started);
    writeJson(output + ".json", summary);
    std::cout << summary.dump() << '\n';
    return 0;
}

int run(const std::string& configPath)
{
    const Json cfg = readJson(configPath);
    if (cfg.contains("support_output")) return runSupport(cfg);
    const FiberSignOptions options = optionsFrom(cfg);
    const auto readStarted = Clock::now();

    vc::objects::OpenOptions storeOptions;
    storeOptions.readOnly = true;
    auto store = vc::objects::ObjectCollection::open(cfg.at("database").get<std::string>(), storeOptions);
    const auto txn = store.read();
    const auto columns =
        vc::objects::readFiberGraphColumns(store, txn, cfg.value("workers", std::size_t{32}));

    std::vector<std::string> ids(columns.nodes);
    for (std::size_t v = 0; v < columns.nodes; ++v) ids[v] = vc::objects::fiberGraphUuid(columns.fiberUuids[v]);

    const Json priors = readJson(cfg.at("priors").get<std::string>());
    if (priors.value("format", std::string{}) != "fiber-outward-priors")
        throw std::invalid_argument("priors are not fiber-outward-priors");
    const auto priorIds = priors.at("source_ids").get<std::vector<std::string>>();
    const auto agreement = priors.at("agreement").get<std::vector<double>>();
    if (priorIds.size() != columns.nodes || agreement.size() != columns.nodes)
        throw std::invalid_argument("priors cover a different fiber set");
    for (std::size_t v = 0; v < columns.nodes; ++v)
        if (priorIds[v] != ids[v]) throw std::invalid_argument("prior and graph fiber order differ");

    // Perpendicular crossings with a valid normal on both sides, exactly the
    // reference stage's selection.
    std::size_t perpendicular = 0;
    const auto rows = signEventRows(columns, perpendicular);
    std::vector<FiberSignEvent> events;
    events.reserve(rows.size());
    for (const std::size_t e : rows) {
        FiberSignEvent event;
        event.a = columns.endpoints[e][0];
        event.b = columns.endpoints[e][1];
        event.dot = eventDot(columns, e);
        event.windingDistance = columns.distance[e];
        events.push_back(event);
    }

    std::vector<int> fixedSigns;
    std::size_t unknownFixed = 0;
    if (cfg.contains("fixed_signs")) {
        const Json fixed = readJson(cfg.at("fixed_signs").get<std::string>());
        std::map<std::string, int> lookup;
        for (const auto& [key, value] : fixed.items()) lookup[key] = value.get<int>();
        fixedSigns.assign(columns.nodes, 0);
        std::size_t used = 0;
        for (std::size_t v = 0; v < columns.nodes; ++v) {
            const auto found = lookup.find(ids[v]);
            if (found == lookup.end()) continue;
            fixedSigns[v] = found->second;
            ++used;
        }
        unknownFixed = lookup.size() - used;
    }
    const double readSeconds = seconds(readStarted);

    const FiberSignResult result = solveFiberNormalSigns(agreement, events, options, fixedSigns);

    Json history = Json::array();
    for (const auto& row : result.history)
        history.push_back(Json{{"step", row.step},
                               {"loss", row.objective},
                               {"pair_loss", row.pairLoss},
                               {"prior_loss", row.priorLoss},
                               {"broken_loss", row.brokenLoss},
                               {"exact_wrapped_l1_objective", row.exactObjective},
                               {"linear_iterations", row.linearIterations},
                               {"backtracks", row.backtracks},
                               {"seconds", row.seconds}});

    Json summary{
        {"config", cfg},
        {"input",
         Json{{"store", store.uuid(txn)},
              {"fibers", result.nodes},
              {"perpendicular_events", perpendicular},
              {"used_events", result.events},
              {"invalid_events_skipped", std::int64_t(perpendicular) - result.events},
              {"self_events", result.selfEvents},
              {"events_with_winding_distance", result.eventsWithWindingDistance},
              {"isolated_fibers", result.isolatedNodes},
              {"node_chunks", result.nodeChunks},
              {"event_chunks", result.eventChunks},
              {"fixed_signs", result.fixedNodes},
              {"fixed_signs_for_unknown_fibers", unknownFixed},
              {"read_seconds", readSeconds}}},
        {"options", optionsJson(cfg, options)},
        {"result",
         Json{{"objective", result.objective},
              {"pair_loss", result.pairLoss},
              {"prior_loss", result.priorLoss},
              {"broken_loss", result.brokenLoss},
              {"exact_wrapped_l1_objective", result.exactObjective},
              {"iterations", result.iterations},
              {"linear_iterations", result.linearIterations},
              {"converged", result.converged},
              {"linear_solver", result.linearSolver},
              {"sparse_library", result.sparseLibrary},
              {"threads_used", result.threadsUsed},
              {"solve_seconds", result.seconds}}},
        {"metrics", metricsJson(result.metrics)},
        {"history", history}};

    if (cfg.contains("signs_output")) {
        Json signs{{"ids", ids},
                   {"normal_convention", "inward"},
                   {"normal_sign", Json::array()},
                   {"sign_probability", result.signProbability},
                   {"broken", Json::array()},
                   {"wrapped_phase", Json::array()}};
        for (std::size_t v = 0; v < columns.nodes; ++v) {
            signs["normal_sign"].push_back(int(result.sign[v]));
            signs["broken"].push_back(result.broken[v] != 0);
            double wrapped = std::fmod(result.phase[v] + 1.0, 2.0);
            if (wrapped < 0.0) wrapped += 2.0;
            signs["wrapped_phase"].push_back(wrapped);
        }
        writeJson(cfg.at("signs_output").get<std::string>(), signs);
    }
    if (cfg.contains("events_output")) {
        Json dump{{"i", Json::array()}, {"j", Json::array()}, {"dot", Json::array()}, {"distance", Json::array()}};
        for (const auto& event : events) {
            dump["i"].push_back(event.a);
            dump["j"].push_back(event.b);
            dump["dot"].push_back(event.dot);
            dump["distance"].push_back(event.windingDistance);
        }
        writeJson(cfg.at("events_output").get<std::string>(), dump);
    }
    if (cfg.contains("output")) writeJson(cfg.at("output").get<std::string>(), summary);

    Json line = summary;
    line.erase("history");
    line.erase("config");
    std::cout << line.dump() << '\n';
    return 0;
}

}  // namespace

int main(const int argc, const char* const* argv)
{
    if (argc != 2) {
        std::cerr << "usage: vc_fiber_sign <config.json>\n";
        return 2;
    }
    try {
        return run(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "vc_fiber_sign: " << error.what() << '\n';
        return 1;
    }
}
