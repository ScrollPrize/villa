// Native CPU normal-sign solve over an extracted fiber-constraint store.
//
// Reads one store's compact constraint graph and its radial priors, assembles
// the sign problem the same way the reference stage does, runs one of the CPU
// backends and writes a JSON summary. Optionally writes the per-fiber signs
// and the prepared events, so another solver's result can be compared against
// the same inputs.
#include "vc/fiber_tracer/FiberNormalSign.hpp"

#include <vc_object_store/objects/FiberConstraintGraph.hpp>
#include <vc_object_store/objects/ObjectCollection.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
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

int run(const std::string& configPath)
{
    const Json cfg = readJson(configPath);
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
    std::vector<FiberSignEvent> events;
    events.reserve(columns.events);
    std::size_t perpendicular = 0;
    for (std::size_t e = 0; e < columns.events; ++e) {
        if (columns.parallel[e] != 0) continue;
        ++perpendicular;
        if (columns.validA[e] == 0 || columns.validB[e] == 0) continue;
        const auto& a = columns.normalA[e];
        const auto& b = columns.normalB[e];
        FiberSignEvent event;
        event.a = columns.endpoints[e][0];
        event.b = columns.endpoints[e][1];
        event.dot = std::clamp(a[0] * b[0] + a[1] * b[1] + a[2] * b[2], -1.0, 1.0);
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
