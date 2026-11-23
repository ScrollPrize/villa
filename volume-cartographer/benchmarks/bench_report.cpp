#include "bench_fixture.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace vcbench {

double msSince(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// ---------------------------------------------------------------------------
// Args

Args::Args(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0)
            throw std::invalid_argument("unexpected argument: " + key);
        key = key.substr(2);
        const auto eq = key.find('=');
        if (eq != std::string::npos) {
            values_[key.substr(0, eq)] = key.substr(eq + 1);
        } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            values_[key] = argv[++i];
        } else {
            values_[key] = "1";
        }
    }
}

std::optional<std::string> Args::get(const std::string& key) const
{
    used_.insert(key);
    const auto it = values_.find(key);
    if (it == values_.end())
        return std::nullopt;
    return it->second;
}

std::string Args::str(const std::string& key, const std::string& fallback) const
{
    return get(key).value_or(fallback);
}

bool Args::flag(const std::string& key) const
{
    const auto value = str(key);
    return !value.empty() && value != "0" && value != "false";
}

void Args::rejectUnused() const
{
    for (const auto& [key, value] : values_)
        if (!used_.count(key))
            throw std::invalid_argument("unknown option --" + key);
}

// ---------------------------------------------------------------------------
// Statistics and reporting

Summary summarize(std::vector<double> values)
{
    Summary s;
    s.n = values.size();
    if (values.empty())
        return s;
    std::sort(values.begin(), values.end());
    const auto at = [&](double q) {
        const double pos = q * double(values.size() - 1);
        const auto lo = std::size_t(std::floor(pos));
        const auto hi = std::min(values.size() - 1, lo + 1);
        return values[lo] + (values[hi] - values[lo]) * (pos - double(lo));
    };
    s.total = std::accumulate(values.begin(), values.end(), 0.0);
    s.mean = s.total / double(values.size());
    s.min = values.front();
    s.max = values.back();
    s.p50 = at(0.5);
    s.p95 = at(0.95);
    return s;
}

Report::Report(std::string benchmark, std::string pipeline, std::string caseName, std::string cacheState,
               std::map<std::string, std::string> config, RunInfo run)
    : benchmark_(std::move(benchmark)), pipeline_(std::move(pipeline)), case_(std::move(caseName)),
      cacheState_(std::move(cacheState)), config_(std::move(config)), run_(std::move(run))
{
}

void Report::add(const std::string& group, const std::string& metric, double value)
{
    const auto key = group + "|" + metric;
    if (!samples_.count(key))
        order_.push_back(key);
    samples_[key].push_back(value);
}

namespace {

// RFC 4180: fields with a comma, quote or line break are quoted, quotes doubled.
std::string csvField(const std::string& text)
{
    if (text.find_first_of(",\"\r\n") == std::string::npos)
        return text;
    std::string out = "\"";
    for (const char c : text)
        out += c == '"' ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

// The shortest text that reads back as the same double.
std::string number(double value)
{
    return nlohmann::json(value).dump();
}

constexpr const char* kCsvColumns[] = {"timestamp", "run_id", "commit", "benchmark", "case", "pipeline",
                                       "cache_state", "step", "metric", "n", "mean", "p50", "p95", "min",
                                       "max", "total", "config"};

} // namespace

void Report::finish() const
{
    std::printf("\n== %s / %s [%s, %s]\n", benchmark_.c_str(), case_.c_str(), pipeline_.c_str(), cacheState_.c_str());
    for (const auto& [key, value] : config_)
        std::printf("   %s: %s\n", key.c_str(), value.c_str());
    std::printf("%-34s %-14s %6s %10s %10s %10s %10s %10s\n", "step", "metric", "n", "mean", "p50", "p95", "min",
                "max");
    for (const auto& key : order_) {
        const auto bar = key.find('|');
        const auto s = summarize(samples_.at(key));
        std::printf("%-34s %-14s %6zu %10.2f %10.2f %10.2f %10.2f %10.2f\n", key.substr(0, bar).c_str(),
                    key.substr(bar + 1).c_str(), s.n, s.mean, s.p50, s.p95, s.min, s.max);
    }
    std::fflush(stdout);

    const nlohmann::json config(config_);
    if (!run_.jsonPath.empty()) {
        std::ofstream out(run_.jsonPath, std::ios::app);
        if (!out)
            throw std::runtime_error("cannot append to " + run_.jsonPath);
        for (const auto& key : order_) {
            const auto bar = key.find('|');
            const auto s = summarize(samples_.at(key));
            const nlohmann::json line{{"timestamp", run_.timestamp}, {"run_id", run_.runId}, {"commit", run_.commit},
                                      {"benchmark", benchmark_},     {"case", case_},       {"pipeline", pipeline_},
                                      {"cache_state", cacheState_},  {"step", key.substr(0, bar)},
                                      {"metric", key.substr(bar + 1)}, {"n", s.n},          {"mean", s.mean},
                                      {"p50", s.p50},                {"p95", s.p95},        {"min", s.min},
                                      {"max", s.max},                {"total", s.total},    {"config", config}};
            out << line.dump() << "\n";
        }
    }
    if (!run_.csvPath.empty()) {
        std::error_code ec;
        const bool header = !std::filesystem::exists(run_.csvPath, ec) || std::filesystem::file_size(run_.csvPath, ec) == 0;
        std::ofstream out(run_.csvPath, std::ios::app);
        if (!out)
            throw std::runtime_error("cannot append to " + run_.csvPath);
        if (header) {
            std::string line;
            for (const char* column : kCsvColumns)
                line += (line.empty() ? "" : ",") + std::string(column);
            out << line << "\n";
        }
        const auto configText = config.dump();
        for (const auto& key : order_) {
            const auto bar = key.find('|');
            const auto s = summarize(samples_.at(key));
            const std::vector<std::string> fields{
                run_.timestamp,  run_.runId,         run_.commit,       benchmark_,         case_,
                pipeline_,       cacheState_,        key.substr(0, bar), key.substr(bar + 1), std::to_string(s.n),
                number(s.mean),  number(s.p50),      number(s.p95),     number(s.min),      number(s.max),
                number(s.total), configText};
            std::string line;
            for (const auto& field : fields)
                line += (line.empty() ? "" : ",") + csvField(field);
            out << line << "\n";
        }
    }
}

} // namespace vcbench
