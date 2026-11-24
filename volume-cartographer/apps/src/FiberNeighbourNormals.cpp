#include "FiberNeighbourNormals.hpp"
#include "vc/fiber_tracer/FiberRibbonCrossings.hpp"
#include "vc/fiber_tracer/FiberGapFalloff.hpp"
#include <vc_object_store/objects/FiberConstraintGraph.hpp>
#include <vc_object_store/objects/FiberGraphSolveExtension.hpp>
#include <vc_object_store/objects/ObjectCollection.hpp>
#include <opencv2/core/utility.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace {
using Json = nlohmann::json;
Json readJson(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("neighbour normals: cannot read " + path);
    Json j;
    in >> j;
    return j;
}

using vc::fiber_tracer::FiberNeighbourFrame;
using vc::fiber_tracer::FiberNeighbourVariant;
}  // namespace

void writeFiberStripSamples(const std::filesystem::path& output, const std::vector<std::string>& ids,
    const std::vector<std::vector<FiberStripSample>>& samples)
{
    std::vector<vc::fiber_tracer::FiberStripUuid> uuids(ids.size());
    for (size_t f = 0; f < ids.size(); ++f) uuids[f] = vc::objects::fiberGraphUuid(ids[f]);
    vc::fiber_tracer::writeFiberStripSampleFile(output, uuids, samples);
}

std::vector<std::vector<FiberStripSample>> readFiberStripSamples(const std::filesystem::path& path,
    const std::vector<std::string>& ids, size_t& unknownFibers)
{
    auto file = vc::fiber_tracer::readFiberStripSampleFile(path);
    std::map<vc::objects::FiberGraphUuid, size_t> index;
    for (size_t i = 0; i < ids.size(); ++i) index.emplace(vc::objects::fiberGraphUuid(ids[i]), i);
    std::vector<std::vector<FiberStripSample>> samples(ids.size());
    unknownFibers = 0;
    for (size_t f = 0; f < file.ids.size(); ++f) {
        const auto found = index.find(file.ids[f]);
        if (found == index.end()) { ++unknownFibers; continue; }
        samples[found->second] = std::move(file.samples[f]);
    }
    return samples;
}

Json inferFiberNeighbourNormals(const Json& cfg)
{
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    const size_t workers = cfg.value("workers", size_t{32});
    std::vector<FiberNeighbourVariant> variants;
    double cell = 0.;
    for (const auto& v : cfg.at("neighbour_variants")) {
        const auto frameName = v.value("frame", std::string("isotropic"));
        const FiberNeighbourFrame frame = frameName == "isotropic" ? FiberNeighbourFrame::Isotropic : frameName == "own" ? FiberNeighbourFrame::Own
            : frameName == "prediction" ? FiberNeighbourFrame::Prediction : frameName == "two_pass" ? FiberNeighbourFrame::TwoPass
            : frameName == "fiber" ? FiberNeighbourFrame::Fiber
            : throw std::invalid_argument("neighbour_variants: frame must be isotropic, own, prediction, two_pass or fiber");
        std::array<double, 3> scales;
        if (v.contains("scales")) {
            const auto s = v.at("scales").get<std::vector<double>>();
            if (s.size() != 3) throw std::invalid_argument("neighbour_variants: scales are (tangent, side, normal)");
            scales = {s[0], s[1], s[2]};
        } else {
            const double sigma = v.value("sigma", 64.);
            scales = {sigma, sigma, sigma};
        }
        std::array<double, 3> parallelScales = scales;
        if (v.contains("parallel_scales")) {
            const auto s = v.at("parallel_scales").get<std::vector<double>>();
            if (s.size() != 3) throw std::invalid_argument("neighbour_variants: parallel_scales are (tangent, side, normal)");
            parallelScales = {s[0], s[1], s[2]};
        }
        FiberNeighbourVariant variant{parseGapFalloff(v.at("falloff").get<std::string>()), frame, scales, parallelScales, v.at("radius").get<double>(),
                                      v.value("first_sigma", 64.), v.value("fiber_cap", false), false};
        if (!(variant.radius > 0.) || !(scales[0] > 0. && scales[1] > 0. && scales[2] > 0.) || !(variant.firstSigma > 0.)
            || !(parallelScales[0] > 0. && parallelScales[1] > 0. && parallelScales[2] > 0.))
            throw std::invalid_argument("neighbour_variants: positive radius, scales and first_sigma required");
        cell = std::max(cell, variant.radius);
        variants.push_back(variant);
    }
    const size_t V = variants.size();
    if (!V) throw std::invalid_argument("neighbour_variants is empty");

    // Strip samples of every fiber, signed by the sign stage (unsigned fibers are no source).
    std::map<vc::objects::FiberGraphUuid, int> signs;
    const Json signsJson = readJson(cfg.at("fiber_signs").get<std::string>());
    for (const auto& [key, value] : signsJson.items()) signs[vc::objects::fiberGraphUuid(key)] = value.get<int>();
    auto strip = vc::fiber_tracer::readFiberStripSampleFile(cfg.at("strip_samples").get<std::string>());
    const size_t listed = strip.ids.size();
    std::vector<int> fiberSign(listed, 0);
    for (size_t f = 0; f < listed; ++f) {
        const auto s = signs.find(strip.ids[f]);
        fiberSign[f] = s == signs.end() ? 0 : s->second;
    }
    const vc::fiber_tracer::FiberNeighbourField field(std::move(strip.samples), fiberSign, cell);

    // Queries: fibers with the uuids whose samples they must not use.
    std::map<vc::objects::FiberGraphUuid, size_t> index;
    for (size_t f = 0; f < listed; ++f) index.emplace(strip.ids[f], f);
    struct Query { size_t fiber; std::set<uint32_t> excluded; std::string id; };
    std::vector<Query> queries;
    size_t unknownQueries = 0;
    const Json queriesJson = readJson(cfg.at("neighbour_queries").get<std::string>());
    for (const auto& [id, excluded] : queriesJson.items()) {
        const auto found = index.find(vc::objects::fiberGraphUuid(id));
        if (found == index.end()) { ++unknownQueries; continue; }
        Query q{found->second, {}, id};
        for (const auto& e : excluded) {
            const auto x = index.find(vc::objects::fiberGraphUuid(e.get<std::string>()));
            if (x != index.end()) q.excluded.insert(uint32_t(x->second));
        }
        queries.push_back(std::move(q));
    }

    // Record per (query sample, variant): query, sample, variant, arc, inferred normal xyz,
    // own signed normal xyz, weight sum, coherence, sign margin, neighbour samples,
    // effective samples (sum w)^2 / sum w^2, effective fibers (same over per-fiber sums).
    constexpr size_t kOut = 16;
    std::vector<std::vector<double>> results(queries.size());
    std::atomic<size_t> visited{0}, frameFallbacks{0}, crossingPairs{0}, parallelPairs{0};
    cv::parallel_for_(cv::Range(0, int(queries.size())), [&](const cv::Range& r) {
        for (int qi = r.start; qi < r.end; ++qi) {
            const auto& q = queries[size_t(qi)];
            const auto& own = field.fiber(q.fiber);
            vc::fiber_tracer::FiberNeighbourCounts counts;
            const auto estimates = field.infer(own, variants, std::int64_t(q.fiber), &q.excluded, &counts);
            auto& out = results[size_t(qi)];
            for (size_t s = 0; s < own.size(); ++s) {
                const cv::Vec3d signedOwn = own[s].normal * double(fiberSign[q.fiber]);
                for (size_t v = 0; v < V; ++v) {
                    const auto& est = estimates[s * V + v];
                    out.insert(out.end(), {double(qi), double(s), double(v), own[s].arc, est.axis[0], est.axis[1], est.axis[2],
                                           signedOwn[0], signedOwn[1], signedOwn[2], est.weight, est.coherence, est.margin, est.count,
                                           est.effectiveSamples, est.effectiveFibers});
                }
            }
            visited += own.size();
            frameFallbacks += counts.frameFallbacks;
            crossingPairs += counts.crossingPairs;
            parallelPairs += counts.parallelPairs;
        }
    }, int(workers));

    const std::filesystem::path output = cfg.at("neighbour_normals_output").get<std::string>();
    std::ofstream bin(output, std::ios::binary);
    if (!bin) throw std::runtime_error("neighbour normals: cannot write " + output.string());
    for (const auto& r : results) bin.write(reinterpret_cast<const char*>(r.data()), std::streamsize(r.size() * sizeof(double)));
    if (!bin) throw std::runtime_error("neighbour normals: write failed " + output.string());
    Json ids = Json::array();
    for (const auto& q : queries) ids.push_back(q.id);
    if (cfg.contains("neighbour_samples_output")) {
        const size_t pick = cfg.value("neighbour_samples_variant", size_t{0});
        if (pick >= V) throw std::invalid_argument("neighbour_samples_variant out of range");
        std::vector<std::string> names;
        std::vector<std::vector<FiberStripSample>> inferred(queries.size());
        for (size_t qi = 0; qi < queries.size(); ++qi) {
            names.push_back(queries[qi].id);
            const auto& r = results[qi];
            for (size_t k = pick * kOut; k < r.size(); k += V * kOut) {
                if (!(r[k + 13] > 0.)) continue;
                const auto& own = field.fiber(queries[qi].fiber)[size_t(r[k + 1])];
                inferred[qi].push_back({r[k + 3], own.point, {r[k + 4], r[k + 5], r[k + 6]}, own.prediction});
            }
        }
        writeFiberStripSamples(cfg.at("neighbour_samples_output").get<std::string>(), names, inferred);
    }
    Json summary{{"record_doubles", kOut}, {"columns", {"query", "sample", "variant", "arc", "nx", "ny", "nz", "own_x", "own_y", "own_z",
                                                        "weight", "coherence", "sign_margin", "neighbours", "effective_samples", "effective_fibers"}},
                 {"queries", ids}, {"unknown_queries", unknownQueries}, {"variants", cfg.at("neighbour_variants")},
                 {"fibers", listed}, {"samples", field.sampleCount()}, {"signed_samples", field.signedSampleCount()}, {"query_samples", visited.load()},
                 {"frame_fallbacks", frameFallbacks.load()}, {"fiber_frame_crossing_pairs", crossingPairs.load()},
                 {"fiber_frame_parallel_pairs", parallelPairs.load()},
                 {"seconds", std::chrono::duration<double>(Clock::now() - started).count()}};
    std::ofstream(output.string() + ".json") << summary.dump(1) << "\n";
    return summary;
}

Json writeSignedFiberStripSamples(const Json& cfg)
{
    const std::string signStore = cfg.at("sign_store").get<std::string>();
    vc::objects::OpenOptions storeOptions;
    storeOptions.readOnly = true;
    auto store = vc::objects::ObjectCollection::open(signStore, storeOptions);
    const auto txn = store.read();
    // The whole-fiber signs of the store's normal-sign run, from its fiber result table.
    const std::string signRun = vc::objects::selectFiberGraphRun(store, txn, "normal_sign",
        cfg.contains("sign_run") ? std::optional<std::string>(cfg.at("sign_run").get<std::string>()) : std::nullopt);
    const auto results = vc::objects::readFiberGraphResultColumns(store, txn, signRun);
    constexpr size_t signColumn = 2;
    static_assert(std::string_view(vc::objects::FiberGraphResultColumns::names[signColumn]) == "normal_sign");
    std::map<vc::objects::FiberGraphUuid, int> signs;
    for (size_t r = 0; r < results.fiberUuids.size(); ++r)
        if (results.valid[r][signColumn] != 0 && results.integers[r][signColumn] != 0)
            signs[results.fiberUuids[r]] = results.integers[r][signColumn] > 0 ? 1 : -1;
    auto strip = vc::fiber_tracer::readFiberStripSampleFile(cfg.at("strip_samples").get<std::string>());
    size_t fibers = 0, samples = 0, unsignedFibers = 0;
    for (size_t f = 0; f < strip.ids.size(); ++f) {
        const auto s = signs.find(strip.ids[f]);
        if (s == signs.end()) {
            unsignedFibers += !strip.samples[f].empty();
            strip.samples[f].clear();
            continue;
        }
        for (auto& sample : strip.samples[f]) sample.normal *= double(s->second);
        fibers += !strip.samples[f].empty();
        samples += strip.samples[f].size();
    }
    const std::filesystem::path output = cfg.at("signed_strip_samples_output").get<std::string>();
    vc::fiber_tracer::writeFiberStripSampleFile(output, strip.ids, strip.samples);
    Json statement{{"format", "signed_strip_samples"},
                   {"coordinate_space", cfg.at("coordinate_space")},
                   {"volume_uuid", cfg.at("volume_uuid")},
                   {"sign_store", store.uuid(txn)},
                   {"sign_run", signRun},
                   {"fibers", fibers},
                   {"samples", samples},
                   {"unsigned_fibers_left_out", unsignedFibers}};
    std::ofstream(output.string() + ".json") << statement.dump(1) << "\n";
    return statement;
}

nlohmann::json writeFiberRibbonCrossings(const nlohmann::json& cfg)
{
    using Json = nlohmann::json;
    const auto strip = vc::fiber_tracer::readFiberStripSampleFile(cfg.at("strip_samples").get<std::string>());
    vc::fiber_tracer::FiberRibbonOptions options;
    options.halfHeight = cfg.value("ribbon_half_height", options.halfHeight);
    options.maxTangentCosine = cfg.value("ribbon_max_tangent_cosine", options.maxTangentCosine);
    options.threads = cfg.value("workers", 0u);
    auto crossings = vc::fiber_tracer::fiberRibbonCrossings(strip.samples, options);
    const double arcGap = cfg.value("ribbon_encounter_arc_gap", 3000.0);
    const auto encounters = vc::fiber_tracer::fiberRibbonEncounters(crossings, arcGap);
    std::vector<double> rows;
    rows.reserve(crossings.size() * 11);
    for (const auto& c : crossings)
        rows.insert(rows.end(), {double(c.ribbon), double(c.fiber), c.ribbonArc, c.fiberArc, c.height, double(c.direction),
                                 c.tangentCosine, c.normalDot, c.point[0], c.point[1], c.point[2]});
    const std::filesystem::path output = cfg.at("ribbon_crossings_output").get<std::string>();
    std::ofstream out(output, std::ios::binary);
    out.write(reinterpret_cast<const char*>(rows.data()), std::streamsize(rows.size() * sizeof(double)));
    if (!out) throw std::runtime_error("ribbon crossings: cannot write " + output.string());
    {
        std::vector<double> table;
        table.reserve(encounters.size() * 7);
        for (const auto& e : encounters)
            table.insert(table.end(), {double(e.ribbon), double(e.fiber), double(e.first), double(e.count), double(e.closest),
                                       double(e.left), double(e.remaining)});
        std::ofstream file(output.string() + ".encounters", std::ios::binary);
        file.write(reinterpret_cast<const char*>(table.data()), std::streamsize(table.size() * sizeof(double)));
        if (!file) throw std::runtime_error("ribbon crossings: cannot write the encounters of " + output.string());
    }
    Json statement{{"format", "ribbon_crossings"}, {"columns", 11}, {"fibers", strip.samples.size()},
                   {"crossings", crossings.size()}, {"encounters", encounters.size()}, {"encounter_columns", 7}, {"encounter_arc_gap", arcGap}, {"half_height", options.halfHeight},
                   {"max_tangent_cosine", options.maxTangentCosine}};
    std::ofstream(output.string() + ".json") << statement.dump(1) << "\n";
    return statement;
}

nlohmann::json writeFiberCrossingFilter(const nlohmann::json& cfg)
{
    using Json = nlohmann::json;
    using namespace vc::fiber_tracer;
    const auto strip = readFiberStripSampleFile(cfg.at("strip_samples").get<std::string>());
    FiberRibbonOptions surface;
    surface.halfHeight = cfg.value("ribbon_half_height", 512.0);
    surface.maxTangentCosine = 1.0;  // the event is the crossing; its further passages have no angle limit
    surface.threads = cfg.value("workers", 0u);
    auto crossings = fiberRibbonCrossings(strip.samples, surface);
    const double arcGap = cfg.value("ribbon_encounter_arc_gap", 3000.0);
    const auto encounters = fiberRibbonEncounters(crossings, arcGap);

    vc::objects::OpenOptions storeOptions;
    storeOptions.readOnly = true;
    auto store = vc::objects::ObjectCollection::open(cfg.at("database").get<std::string>(), storeOptions);
    const auto txn = store.read();
    const auto columns = vc::objects::readFiberGraphColumns(store, txn, cfg.value("workers", size_t{32}));
    std::map<vc::objects::FiberGraphUuid, std::int64_t> stripIndex;
    for (size_t f = 0; f < strip.ids.size(); ++f) stripIndex[strip.ids[f]] = std::int64_t(f);
    std::vector<std::int64_t> node(columns.nodes, -1);
    for (size_t n = 0; n < columns.nodes; ++n)
        if (const auto found = stripIndex.find(columns.fiberUuids[n]); found != stripIndex.end()) node[n] = found->second;

    std::vector<FiberCrossingEventView> views;
    std::vector<size_t> rows;
    size_t withoutStrip = 0;
    for (size_t e = 0; e < columns.events; ++e) {
        if (columns.parallel[e] != 0 || columns.validA[e] == 0 || columns.validB[e] == 0) continue;
        const auto a = node[size_t(columns.endpoints[e][0])], b = node[size_t(columns.endpoints[e][1])];
        if (a < 0 || b < 0) { ++withoutStrip; continue; }
        rows.push_back(e);
        views.push_back({a, b, columns.arcs[e][0], columns.arcs[e][1]});
        views.push_back({b, a, columns.arcs[e][1], columns.arcs[e][0]});
    }
    if (withoutStrip != 0)
        throw std::runtime_error("crossing filter: " + std::to_string(withoutStrip) +
                                 " events lie on fibers the strip samples do not hold; they belong to another extraction");
    FiberCrossingFilterOptions filter;
    filter.matchTolerance = cfg.value("crossing_filter_match_tolerance", filter.matchTolerance);
    filter.minimumDot = cfg.value("crossing_filter_minimum_dot", filter.minimumDot);
    filter.nearHeight = cfg.value("crossing_filter_near_height", filter.nearHeight);
    const auto verdicts = fiberCrossingVerdicts(crossings, encounters, views, filter);
    // One record per sign event: id, finding bits (1 matched, 2 multiple, 4 nothing left, 8 mixed,
    // 16 mixed near, 32 several left) and the |normal dot| of the nearest alternative crossing (0: none).
    struct Record { std::int64_t event, findings; double alternative; };
    static_assert(sizeof(Record) == 24);
    std::vector<Record> table;
    table.reserve(rows.size());
    size_t nothingLeft = 0, mixed = 0, mixedNear = 0, severalLeft = 0, multiple = 0, unmatched = 0, alternatives = 0;
    for (size_t r = 0; r < rows.size(); ++r) {
        const auto& first = verdicts[2 * r];
        const auto& second = verdicts[2 * r + 1];
        const bool nothing = first.nothingLeft || second.nothingLeft, both = first.mixed || second.mixed,
                   bothNear = first.mixedNear || second.mixedNear, several = first.severalLeft || second.severalLeft;
        nothingLeft += nothing; mixed += both; mixedNear += bothNear; severalLeft += several;
        multiple += first.multiple || second.multiple;
        unmatched += !first.matched && !second.matched;
        // Of the two surfaces' alternatives the one nearer its fiber.
        const auto& nearer = second.alternativeDot != 0.0 &&
            (first.alternativeDot == 0.0 || second.alternativeHeight < first.alternativeHeight) ? second : first;
        alternatives += nearer.alternativeDot != 0.0;
        table.push_back({columns.eventIds[rows[r]],
                         (first.matched || second.matched ? 1 : 0) | (first.multiple || second.multiple ? 2 : 0) |
                             (nothing ? 4 : 0) | (both ? 8 : 0) | (bothNear ? 16 : 0) | (several ? 32 : 0),
                         nearer.alternativeDot});
    }
    std::sort(table.begin(), table.end(), [](const Record& x, const Record& y) { return x.event < y.event; });
    const std::filesystem::path output = cfg.at("crossing_filter_output").get<std::string>();
    {
        std::ofstream out(output, std::ios::binary);
        out.write(reinterpret_cast<const char*>(table.data()), std::streamsize(table.size() * sizeof(Record)));
        if (!out) throw std::runtime_error("crossing filter: cannot write " + output.string());
    }
    Json statement{{"format", "fiber_crossing_filter"}, {"version", 2}, {"store", store.uuid(txn)},
                   {"events", columns.events}, {"sign_events", rows.size()}, {"record_bytes", sizeof(Record)},
                   {"multiple", multiple}, {"nothing_left", nothingLeft}, {"mixed", mixed}, {"mixed_near", mixedNear},
                   {"several_left", severalLeft}, {"alternatives", alternatives},
                   {"without_surface_crossing", unmatched}, {"surface_crossings", crossings.size()},
                   {"half_height", surface.halfHeight}, {"encounter_arc_gap", arcGap},
                   {"match_tolerance", filter.matchTolerance}, {"minimum_dot", filter.minimumDot}};
    std::ofstream(output.string() + ".json") << statement.dump(1) << "\n";
    return statement;
}
