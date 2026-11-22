#include "FiberNeighbourNormals.hpp"
#include <vc_object_store/objects/FiberConstraintGraph.hpp>
#include <opencv2/core/utility.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace {
using Json = nlohmann::json;
constexpr char kMagic[8] = {'V', 'C', 'S', 'T', 'R', 'I', 'P', '1'};
constexpr size_t kRecord = 7;  // arc, point xyz, normal xyz

Json readJson(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("neighbour normals: cannot read " + path);
    Json j;
    in >> j;
    return j;
}

enum class Falloff { None, Gaussian, Inverse, InverseSquare };
struct Variant { Falloff falloff; double sigma, radius; };
double weight(const Variant& v, double d)
{
    if (d > v.radius) return 0.;
    switch (v.falloff) {
    case Falloff::None: return 1.;
    case Falloff::Gaussian: return std::exp(-.5 * d * d / (v.sigma * v.sigma));
    case Falloff::Inverse: return 1. / (1. + d / v.sigma);
    case Falloff::InverseSquare: return 1. / (1. + d * d / (v.sigma * v.sigma));
    }
    return 0.;
}
}  // namespace

void writeFiberStripSamples(const std::filesystem::path& output, const std::vector<std::string>& ids,
    const std::vector<std::vector<FiberStripSample>>& samples)
{
    if (ids.size() != samples.size()) throw std::invalid_argument("strip samples: ids and samples differ in size");
    std::ofstream file(output, std::ios::binary);
    if (!file) throw std::runtime_error("strip samples: cannot write " + output.string());
    const auto put = [&](const void* data, size_t bytes) { file.write(static_cast<const char*>(data), std::streamsize(bytes)); };
    put(kMagic, sizeof kMagic);
    uint64_t listed = 0;
    for (const auto& s : samples) listed += !s.empty();
    put(&listed, sizeof listed);
    std::vector<double> buffer;
    for (size_t f = 0; f < ids.size(); ++f) {
        if (samples[f].empty()) continue;
        const auto uuid = vc::objects::fiberGraphUuid(ids[f]);
        put(uuid.data(), uuid.size());
        const uint64_t n = samples[f].size();
        put(&n, sizeof n);
        buffer.clear();
        for (const auto& s : samples[f])
            buffer.insert(buffer.end(), {s.arc, s.point[0], s.point[1], s.point[2], s.normal[0], s.normal[1], s.normal[2]});
        put(buffer.data(), buffer.size() * sizeof(double));
    }
    if (!file) throw std::runtime_error("strip samples: write failed " + output.string());
}

std::vector<std::vector<FiberStripSample>> readFiberStripSamples(const std::filesystem::path& path,
    const std::vector<std::string>& ids, size_t& unknownFibers)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("strip samples: cannot read " + path.string());
    const auto get = [&](void* data, size_t bytes) {
        if (!file.read(static_cast<char*>(data), std::streamsize(bytes))) throw std::runtime_error("strip samples: truncated " + path.string());
    };
    char magic[8];
    get(magic, sizeof magic);
    if (std::memcmp(magic, kMagic, sizeof magic) != 0) throw std::runtime_error("strip samples: not a strip samples file " + path.string());
    std::map<vc::objects::FiberGraphUuid, size_t> index;
    for (size_t i = 0; i < ids.size(); ++i) index.emplace(vc::objects::fiberGraphUuid(ids[i]), i);
    std::vector<std::vector<FiberStripSample>> samples(ids.size());
    uint64_t listed = 0;
    get(&listed, sizeof listed);
    unknownFibers = 0;
    std::vector<double> buffer;
    for (uint64_t k = 0; k < listed; ++k) {
        vc::objects::FiberGraphUuid uuid;
        get(uuid.data(), uuid.size());
        uint64_t n = 0;
        get(&n, sizeof n);
        buffer.resize(n * kRecord);
        get(buffer.data(), buffer.size() * sizeof(double));
        const auto found = index.find(uuid);
        if (found == index.end()) { ++unknownFibers; continue; }
        auto& out = samples[found->second];
        out.resize(n);
        for (size_t j = 0; j < n; ++j) {
            const double* v = &buffer[j * kRecord];
            out[j] = {v[0], {v[1], v[2], v[3]}, {v[4], v[5], v[6]}};
        }
    }
    return samples;
}

Json inferFiberNeighbourNormals(const Json& cfg)
{
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    const size_t workers = cfg.value("workers", size_t{32});
    std::vector<Variant> variants;
    double cell = 0.;
    for (const auto& v : cfg.at("neighbour_variants")) {
        const auto name = v.at("falloff").get<std::string>();
        const Falloff f = name == "none" ? Falloff::None : name == "gaussian" ? Falloff::Gaussian
            : name == "inverse" ? Falloff::Inverse : name == "inverse_square" ? Falloff::InverseSquare
            : throw std::invalid_argument("neighbour_variants: unknown falloff " + name);
        variants.push_back({f, v.value("sigma", 64.), v.at("radius").get<double>()});
        if (!(variants.back().radius > 0.) || !(variants.back().sigma > 0.)) throw std::invalid_argument("neighbour_variants: positive radius and sigma required");
        cell = std::max(cell, variants.back().radius);
    }
    const size_t V = variants.size();
    if (!V) throw std::invalid_argument("neighbour_variants is empty");

    // Strip samples of every fiber, signed by the sign stage (unsigned fibers are no source).
    std::map<vc::objects::FiberGraphUuid, int> signs;
    const Json signsJson = readJson(cfg.at("fiber_signs").get<std::string>());
    for (const auto& [key, value] : signsJson.items())
        signs[vc::objects::fiberGraphUuid(key)] = value.get<int>();
    const std::string path = cfg.at("strip_samples").get<std::string>();
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("strip samples: cannot read " + path);
    const auto get = [&](void* data, size_t bytes) {
        if (!file.read(static_cast<char*>(data), std::streamsize(bytes))) throw std::runtime_error("strip samples: truncated " + path);
    };
    char magic[8];
    get(magic, sizeof magic);
    if (std::memcmp(magic, kMagic, sizeof magic) != 0) throw std::runtime_error("strip samples: not a strip samples file " + path);
    uint64_t listed = 0;
    get(&listed, sizeof listed);
    std::vector<vc::objects::FiberGraphUuid> fiberIds(listed);
    std::vector<int> fiberSign(listed, 0);
    std::vector<size_t> begin(listed + 1, 0);
    std::vector<double> data;
    for (uint64_t f = 0; f < listed; ++f) {
        get(fiberIds[f].data(), fiberIds[f].size());
        uint64_t n = 0;
        get(&n, sizeof n);
        const size_t old = data.size();
        data.resize(old + n * kRecord);
        get(data.data() + old, n * kRecord * sizeof(double));
        begin[f + 1] = begin[f] + n;
        const auto s = signs.find(fiberIds[f]);
        fiberSign[f] = s == signs.end() ? 0 : s->second;
    }
    const size_t total = begin.back();
    std::vector<uint32_t> owner(total);
    for (uint64_t f = 0; f < listed; ++f) std::fill(owner.begin() + long(begin[f]), owner.begin() + long(begin[f + 1]), uint32_t(f));
    const auto point = [&](size_t i) { const double* p = &data[i * kRecord + 1]; return cv::Vec3d(p[0], p[1], p[2]); };
    const auto normal = [&](size_t i) { const double* p = &data[i * kRecord + 4]; return cv::Vec3d(p[0], p[1], p[2]); };

    // Uniform grid (cell = largest radius) over the signed samples.
    const auto key = [&](const cv::Vec3d& p, int dx = 0, int dy = 0, int dz = 0) {
        const auto c = [&](double x, int d) { return uint64_t(int64_t(std::floor(x / cell)) + d + (1 << 20)) & 0x1fffff; };
        return (c(p[0], dx) << 42) | (c(p[1], dy) << 21) | c(p[2], dz);
    };
    std::vector<std::pair<uint64_t, uint32_t>> keyed;
    keyed.reserve(total);
    for (size_t i = 0; i < total; ++i)
        if (fiberSign[owner[i]] != 0) keyed.emplace_back(key(point(i)), uint32_t(i));
    std::sort(keyed.begin(), keyed.end());
    std::unordered_map<uint64_t, std::pair<size_t, size_t>> cells;
    for (size_t a = 0; a < keyed.size();) {
        size_t b = a;
        while (b < keyed.size() && keyed[b].first == keyed[a].first) ++b;
        cells.emplace(keyed[a].first, std::make_pair(a, b));
        a = b;
    }

    // Queries: fibers with the uuids whose samples they must not use.
    std::map<vc::objects::FiberGraphUuid, size_t> index;
    for (size_t f = 0; f < listed; ++f) index.emplace(fiberIds[f], f);
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
    // own signed normal xyz, weight sum, coherence, sign margin, neighbour samples.
    constexpr size_t kOut = 14;
    std::vector<std::vector<double>> results(queries.size());
    std::atomic<size_t> visited{0};
    cv::parallel_for_(cv::Range(0, int(queries.size())), [&](const cv::Range& r) {
        std::vector<cv::Matx33d> tensor(V);
        std::vector<cv::Vec3d> sum(V);
        std::vector<double> wsum(V);
        std::vector<size_t> count(V);
        for (int qi = r.start; qi < r.end; ++qi) {
            const auto& q = queries[size_t(qi)];
            const size_t b = begin[q.fiber], e = begin[q.fiber + 1];
            auto& out = results[size_t(qi)];
            for (size_t s = b; s < e; ++s) {
                cv::Vec3d t = point(std::min(s + 1, e - 1)) - point(s > b ? s - 1 : s);
                const double tl = cv::norm(t);
                t = tl > 0. ? t / tl : t;
                const cv::Vec3d p = point(s);
                for (size_t v = 0; v < V; ++v) { tensor[v] = cv::Matx33d::zeros(); sum[v] = {0, 0, 0}; wsum[v] = 0; count[v] = 0; }
                for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
                    const auto c = cells.find(key(p, dx, dy, dz));
                    if (c == cells.end()) continue;
                    for (size_t k = c->second.first; k < c->second.second; ++k) {
                        const uint32_t j = keyed[k].second, o = owner[j];
                        if (o == q.fiber || q.excluded.count(o)) continue;
                        const double d = cv::norm(point(j) - p);
                        if (d > cell) continue;
                        const cv::Vec3d n = normal(j) * double(fiberSign[o]);
                        const cv::Matx33d nn = n * n.t();
                        for (size_t v = 0; v < V; ++v) {
                            const double w = weight(variants[v], d);
                            if (!(w > 0.)) continue;
                            tensor[v] += w * nn; sum[v] += w * n; wsum[v] += w; ++count[v];
                        }
                    }
                }
                const cv::Matx33d P = cv::Matx33d::eye() - t * t.t();
                const cv::Vec3d own = normal(s) * double(fiberSign[q.fiber]);
                for (size_t v = 0; v < V; ++v) {
                    cv::Vec3d axis{0, 0, 0};
                    double coherence = 0., margin = 0.;
                    if (wsum[v] > 0.) {
                        const cv::Matx33d M = P * tensor[v] * P;
                        cv::Mat values, vectors;
                        cv::eigen(cv::Mat(M), values, vectors);
                        axis = {vectors.at<double>(0, 0), vectors.at<double>(0, 1), vectors.at<double>(0, 2)};
                        const double trace = M(0, 0) + M(1, 1) + M(2, 2);
                        coherence = trace > 0. ? (values.at<double>(0) - values.at<double>(1)) / trace : 0.;
                        const double m = sum[v].dot(axis);
                        if (m < 0.) axis = -axis;
                        margin = std::abs(m) / wsum[v];
                    }
                    out.insert(out.end(), {double(qi), double(s - b), double(v), data[s * kRecord], axis[0], axis[1], axis[2],
                                           own[0], own[1], own[2], wsum[v], coherence, margin, double(count[v])});
                }
            }
            visited += e - b;
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
                const size_t s = begin[queries[qi].fiber] + size_t(r[k + 1]);
                inferred[qi].push_back({r[k + 3], point(s), {r[k + 4], r[k + 5], r[k + 6]}});
            }
        }
        writeFiberStripSamples(cfg.at("neighbour_samples_output").get<std::string>(), names, inferred);
    }
    Json summary{{"record_doubles", kOut}, {"columns", {"query", "sample", "variant", "arc", "nx", "ny", "nz", "own_x", "own_y", "own_z",
                                                        "weight", "coherence", "sign_margin", "neighbours"}},
                 {"queries", ids}, {"unknown_queries", unknownQueries}, {"variants", cfg.at("neighbour_variants")},
                 {"fibers", listed}, {"samples", total}, {"signed_samples", keyed.size()}, {"query_samples", visited.load()},
                 {"seconds", std::chrono::duration<double>(Clock::now() - started).count()}};
    std::ofstream(output.string() + ".json") << summary.dump(1) << "\n";
    return summary;
}
