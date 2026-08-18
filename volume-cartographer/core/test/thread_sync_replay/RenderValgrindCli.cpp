#include "thread_sync_replay/RenderValgrindCli.hpp"

#include "thread_sync_replay/RenderValgrind.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace vc::thread_sync_replay
{
namespace
{

using json = nlohmann::json;

std::map<std::string, std::string> parseArguments(int argc, char** argv)
{
    std::map<std::string, std::string> result;
    for (int index = 2; index < argc; index += 2) {
        if (index + 1 >= argc || !std::string(argv[index]).starts_with("--")) {
            throw std::runtime_error("evaluate-render arguments must be --name value pairs");
        }
        const std::string name = std::string(argv[index]).substr(2);
        if (!result.emplace(name, argv[index + 1]).second) {
            throw std::runtime_error("duplicate evaluate-render argument --" + name);
        }
    }
    return result;
}

const std::string& required(const std::map<std::string, std::string>& arguments, const std::string& name)
{
    const auto found = arguments.find(name);
    if (found == arguments.end() || found->second.empty()) {
        throw std::runtime_error("missing evaluate-render argument --" + name);
    }
    return found->second;
}

json readJson(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open JSON file " + path.string());
    }
    return json::parse(stream);
}

void writeJsonAtomic(const std::filesystem::path& path, const json& value)
{
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary);
        if (!stream) {
            throw std::runtime_error("cannot write JSON file " + temporary.string());
        }
        stream << value.dump(2) << '\n';
    }
    std::filesystem::rename(temporary, path);
}

EventCostModel parseEventModel(const json& model)
{
    const auto& value = model.at("event_cost_model");
    return {
        .feature_names = value.at("feature_names").get<std::vector<std::string>>(),
        .coefficients_ns = value.at("coefficients_ns").get<std::vector<double>>(),
        .stall_overlap_fraction = value.value("stall_overlap_fraction", 0.0),
    };
}

ReplayOptions parseReplayOptions(const json& model)
{
    const auto& value = model.at("replay");
    return {
        .cores = value.at("cores").get<std::size_t>(),
        .tie_policy = value.at("tie_policy").get<std::string>(),
        .wake_latency = value.value("wake_latency_ns", 0.0),
        .cross_thread_latency = model.value("cross_thread_release_ns", 0.0),
        .replay_idle_scale = value.value("replay_idle_scale", 1.0),
        .dependency_excess_scale = value.value("dependency_excess_scale", 1.0),
    };
}

json replayJson(const ReplayResult& result)
{
    return {
        {"modeled_work", result.modeled_work},
        {"modeled_makespan", result.modeled_makespan},
        {"simulated_core_idle", result.simulated_core_idle},
        {"logical_sync_delay", result.logical_sync_delay},
        {"utilization", result.utilization},
        {"raw_replay_makespan", result.raw_replay_makespan},
        {"dependency_critical_path", result.dependency_critical_path},
        {"hard_dependency_critical_path", result.hard_dependency_critical_path},
        {"work_per_core_lower_bound", result.work_per_core_lower_bound},
        {"hard_schedule_lower_bound", result.hard_schedule_lower_bound},
        {"schedule_lower_bound", result.schedule_lower_bound},
        {"dependency_excess", result.dependency_excess},
        {"raw_replay_excess", result.raw_replay_excess},
        {"raw_simulated_core_idle", result.raw_simulated_core_idle},
    };
}

void validateMetadata(const json& metadata, const std::string& fixture, const std::string& scenario)
{
    if (metadata.at("fixture") != fixture || metadata.at("scenario") != scenario) {
        throw std::runtime_error("benchmark metadata does not match requested case");
    }
    const auto repetitions = metadata.at("repetitions").get<std::int64_t>();
    const auto width = metadata.at("width").get<std::int64_t>();
    const auto height = metadata.at("height").get<std::int64_t>();
    if (repetitions <= 0 || width <= 0 || height <= 0 || metadata.at("measured_pixels").get<std::int64_t>() != width * height * repetitions) {
        throw std::runtime_error("benchmark metadata has inconsistent dimensions");
    }
}

void validatePair(const json& callgrind, const json& drd)
{
    for (const std::string name :
         {"fixture", "scenario", "width", "height", "tile_size", "repetitions", "measured_pixels", "worker_override", "checksum"}) {
        if (callgrind.at(name) != drd.at(name)) {
            throw std::runtime_error("Callgrind and DRD metadata differ for " + name);
        }
    }
}

double referenceTolerance(const json& reference, const std::map<std::string, std::string>& arguments)
{
    const auto explicit_tolerance = arguments.find("tolerance");
    const double value = explicit_tolerance == arguments.end() ? reference.at("tolerance").get<double>() : std::stod(explicit_tolerance->second);
    if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
        throw std::runtime_error("reference tolerance must be in [0, 1)");
    }
    return value;
}

}  // namespace

int renderValgrindCli(int argc, char** argv)
{
    const auto arguments = parseArguments(argc, argv);
    const auto fixture = required(arguments, "fixture");
    const auto scenario = required(arguments, "scenario");
    if (fixture != "serial" && fixture != "parallel") {
        throw std::runtime_error("fixture must be serial or parallel");
    }
    const auto output = std::filesystem::path(required(arguments, "output"));
    const auto metadata = readJson(required(arguments, "callgrind-metadata"));
    validateMetadata(metadata, fixture, scenario);
    const auto model = readJson(required(arguments, "model"));
    if (model.value("renderer_inputs_used", true) || model.value("timing_claims_enabled", true)) {
        throw std::runtime_error("render model must be synthetic-only and relative");
    }
    const auto event_model = parseEventModel(model);
    const auto callgrind = parsePeriodicCallgrind(required(arguments, "callgrind-prefix"));
    const auto repetitions = metadata.at("repetitions").get<double>();

    double score = 0.0;
    json paired = nullptr;
    if (fixture == "serial") {
        if (arguments.contains("callgrind-scheduler") || arguments.contains("drd-trace") || arguments.contains("drd-metadata")) {
            throw std::runtime_error("serial evaluation must not use scheduler traces or DRD");
        }
        for (const auto& [thread, profile] : callgrind.totals) {
            (void)thread;
            score += modeledProfileCostNs(profile, event_model);
        }
        score /= repetitions;
    } else {
        const auto drd_metadata = readJson(required(arguments, "drd-metadata"));
        validateMetadata(drd_metadata, fixture, scenario);
        validatePair(metadata, drd_metadata);
        const auto callgrind_scheduler = parseMeasuredScheduler(required(arguments, "callgrind-scheduler"));
        const auto drd = parseMeasuredDrd(required(arguments, "drd-trace"));
        const auto& replay_config = model.at("replay");
        const auto result = replayPaired(
            callgrind,
            callgrind_scheduler,
            drd,
            event_model,
            {
                .residual_fraction = replay_config.at("residual_fraction").get<double>(),
                .split_policy = replay_config.at("split_policy").get<std::string>(),
                .scheduler_bins = 16,
                .scheduler_quantum_slack = 1.0,
                .maximum_makespan_spread = 0.02,
                .maximum_mappings = 100000,
                .replay = parseReplayOptions(model),
            });
        score = result.conservative.modeled_makespan / repetitions;
        json mapping = json::object();
        for (const auto& [trace_thread, source_thread] : result.selected_source_by_trace_thread) {
            mapping[std::to_string(trace_thread)] = source_thread;
        }
        paired = {
            {"result", replayJson(result.conservative)},
            {"minimum_makespan", result.minimum_makespan},
            {"maximum_makespan", result.conservative.modeled_makespan},
            {"mapping_count", result.mapping_count},
            {"evaluated_mapping_count", result.evaluated_mapping_count},
            {"best_assignment_score", result.best_assignment_score},
            {"assignment_score_tolerance", result.assignment_score_tolerance},
            {"maximum_assignment_score", result.maximum_assignment_score},
            {"makespan_relative_spread", result.makespan_relative_spread},
            {"selected_source_by_trace_thread", std::move(mapping)},
            {"callgrind_scheduler_quanta", callgrind_scheduler.quantum_threads.size()},
            {"callgrind_scheduler_begin_line", callgrind_scheduler.begin_line},
            {"callgrind_scheduler_end_line", callgrind_scheduler.end_line},
            {"drd_event_count", drd.events.size()},
            {"drd_happens_before_edges", drd.happens_before_edges},
            {"drd_dropped_pre_window_edges", drd.dropped_pre_window_edges},
            {"drd_parsed_segment_count", drd.parsed_segment_count},
            {"drd_retained_segment_count", drd.retained_segment_count},
            {"drd_begin_line", drd.begin_line},
            {"drd_end_line", drd.end_line},
        };
    }
    if (!std::isfinite(score) || score <= 0.0) {
        throw std::runtime_error("modeled runtime score is not finite and positive");
    }

    json result = {
        {"schema_version", 2},
        {"kind", "evaluation"},
        {"case", fixture + "/" + scenario},
        {"model_id", model.at("model_id")},
        {"score_semantics", "relative_modeled_runtime_not_absolute_wall_time"},
        {"modeled_runtime_score_ns", score},
        {"modeled_nanoseconds_per_pixel", score / (metadata.at("width").get<double>() * metadata.at("height").get<double>())},
        {"checksum", metadata.at("checksum")},
        {"callgrind_periodic_dump_count", callgrind.periodic_dump_count},
        {"paired_replay", std::move(paired)},
        {"status", "passed"},
    };

    bool failed = false;
    const auto reference_path = arguments.find("reference");
    if (reference_path != arguments.end()) {
        const auto reference = readJson(reference_path->second);
        const auto& reference_case = reference.at("cases").at(fixture + "/" + scenario);
        const double expected = reference_case.at("modeled_runtime_score_ns").get<double>();
        const double ratio = score / expected;
        result["reference_modeled_runtime_score_ns"] = expected;
        result["reference_ratio"] = ratio;
        result["relative_error"] = ratio - 1.0;
        if (ratio > 1.0 + referenceTolerance(reference, arguments)) {
            result["status"] = "failed";
            failed = true;
        }
    }
    auto failed_output = output;
    failed_output.replace_filename(output.stem().string() + ".failed" + output.extension().string());
    if (failed) {
        std::filesystem::remove(output);
        writeJsonAtomic(failed_output, result);
    } else {
        std::filesystem::remove(failed_output);
        writeJsonAtomic(output, result);
    }
    std::cout << fixture << '/' << scenario << ": " << score << " modeled ns/call";
    if (result.contains("reference_ratio")) {
        std::cout << ", " << result.at("reference_ratio").get<double>() << "x reference";
    }
    std::cout << '\n';
    if (failed) {
        throw std::runtime_error("modeled runtime exceeds reference tolerance");
    }
    return 0;
}

}  // namespace vc::thread_sync_replay
