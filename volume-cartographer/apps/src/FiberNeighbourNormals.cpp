#include "FiberNeighbourNormals.hpp"
#include "FiberGapFalloff.hpp"
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
constexpr char kMagic[8] = {'V', 'C', 'S', 'T', 'R', 'I', 'P', '2'};
constexpr size_t kRecord = 10;  // arc, point xyz, normal xyz, prediction axis xyz

Json readJson(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("neighbour normals: cannot read " + path);
    Json j;
    in >> j;
    return j;
}

struct StripFile {
    std::vector<vc::objects::FiberGraphUuid> ids;
    std::vector<std::vector<FiberStripSample>> samples;
};

StripFile readStripFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("strip samples: cannot read " + path.string());
    const auto get = [&](void* data, size_t bytes) {
        if (!file.read(static_cast<char*>(data), std::streamsize(bytes))) throw std::runtime_error("strip samples: truncated " + path.string());
    };
    char magic[8];
    get(magic, sizeof magic);
    if (std::memcmp(magic, kMagic, sizeof magic) != 0) throw std::runtime_error("strip samples: not a strip samples file " + path.string());
    uint64_t listed = 0;
    get(&listed, sizeof listed);
    StripFile out;
    out.ids.resize(listed);
    out.samples.resize(listed);
    std::vector<double> buffer;
    for (uint64_t f = 0; f < listed; ++f) {
        get(out.ids[f].data(), out.ids[f].size());
        uint64_t n = 0;
        get(&n, sizeof n);
        buffer.resize(n * kRecord);
        get(buffer.data(), buffer.size() * sizeof(double));
        auto& list = out.samples[f];
        list.resize(n);
        for (size_t j = 0; j < n; ++j) {
            const double* v = &buffer[j * kRecord];
            list[j] = {v[0], {v[1], v[2], v[3]}, {v[4], v[5], v[6]}, {v[7], v[8], v[9]}};
        }
    }
    return out;
}

cv::Vec3d unitOrZero(const cv::Vec3d& v)
{
    const double n = cv::norm(v);
    return n > 0. ? v / n : cv::Vec3d(0, 0, 0);
}

// Weighting of one variant: the falloff of the neighbour offset, measured per
// axis in a frame (own tangent, side, normal) whose normal comes from `frame`:
// isotropic (no frame, scales[0] for all axes), own (the fiber's own solved
// normal), prediction (the own sample's prediction axis) or two_pass (an
// isotropic Gaussian estimate of first_sigma from the same neighbours).
// fiber: distances along the fibers. A neighbour fiber's samples belong to its
// nearest contact (local minimum of its gap to the own fiber); at a crossing
// contact (more than 30 degrees) the scales are (own arc from the query sample
// to the contact, gap at the contact, partner arc from the contact to the
// sample); along a parallel contact the offset is measured like the prediction
// frame with parallel_scales (tangent, side, normal).
// fiber_cap scales each neighbour fiber's weights so they sum to its largest one.
enum class Frame { Isotropic, Own, Prediction, TwoPass, Fiber };
struct Variant {
    GapFalloff falloff;
    Frame frame;
    std::array<double, 3> scales, parallelScales;
    double radius, firstSigma;
    bool fiberCap;
};

struct Neighbour { uint32_t owner, index; cv::Vec3d offset, normal; double distance; };
struct Contact { double ownArc, gap, partnerArc; bool crossing; };
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
            buffer.insert(buffer.end(), {s.arc, s.point[0], s.point[1], s.point[2], s.normal[0], s.normal[1], s.normal[2],
                                         s.prediction[0], s.prediction[1], s.prediction[2]});
        put(buffer.data(), buffer.size() * sizeof(double));
    }
    if (!file) throw std::runtime_error("strip samples: write failed " + output.string());
}

std::vector<std::vector<FiberStripSample>> readFiberStripSamples(const std::filesystem::path& path,
    const std::vector<std::string>& ids, size_t& unknownFibers)
{
    auto file = readStripFile(path);
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
    std::vector<Variant> variants;
    double cell = 0., firstSigma = 0.;
    for (const auto& v : cfg.at("neighbour_variants")) {
        const auto frameName = v.value("frame", std::string("isotropic"));
        const Frame frame = frameName == "isotropic" ? Frame::Isotropic : frameName == "own" ? Frame::Own
            : frameName == "prediction" ? Frame::Prediction : frameName == "two_pass" ? Frame::TwoPass
            : frameName == "fiber" ? Frame::Fiber
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
        Variant variant{parseGapFalloff(v.at("falloff").get<std::string>()), frame, scales, parallelScales, v.at("radius").get<double>(),
                        v.value("first_sigma", 64.), v.value("fiber_cap", false)};
        if (!(variant.radius > 0.) || !(scales[0] > 0. && scales[1] > 0. && scales[2] > 0.) || !(variant.firstSigma > 0.)
            || !(parallelScales[0] > 0. && parallelScales[1] > 0. && parallelScales[2] > 0.))
            throw std::invalid_argument("neighbour_variants: positive radius, scales and first_sigma required");
        cell = std::max(cell, variant.radius);
        if (frame == Frame::TwoPass) {
            if (firstSigma > 0. && firstSigma != variant.firstSigma) throw std::invalid_argument("neighbour_variants: one first_sigma for all two_pass variants");
            firstSigma = variant.firstSigma;
        }
        variants.push_back(variant);
    }
    const size_t V = variants.size();
    if (!V) throw std::invalid_argument("neighbour_variants is empty");
    const bool anyFiber = std::any_of(variants.begin(), variants.end(), [](const auto& v) { return v.frame == Frame::Fiber; });

    // Strip samples of every fiber, signed by the sign stage (unsigned fibers are no source).
    std::map<vc::objects::FiberGraphUuid, int> signs;
    const Json signsJson = readJson(cfg.at("fiber_signs").get<std::string>());
    for (const auto& [key, value] : signsJson.items()) signs[vc::objects::fiberGraphUuid(key)] = value.get<int>();
    const auto strip = readStripFile(cfg.at("strip_samples").get<std::string>());
    const size_t listed = strip.ids.size();
    std::vector<int> fiberSign(listed, 0);
    std::vector<size_t> begin(listed + 1, 0);
    for (size_t f = 0; f < listed; ++f) {
        begin[f + 1] = begin[f] + strip.samples[f].size();
        const auto s = signs.find(strip.ids[f]);
        fiberSign[f] = s == signs.end() ? 0 : s->second;
    }
    const size_t total = begin.back();
    std::vector<const FiberStripSample*> sample(total);
    std::vector<uint32_t> owner(total);
    for (size_t f = 0; f < listed; ++f)
        for (size_t j = 0; j < strip.samples[f].size(); ++j) { sample[begin[f] + j] = &strip.samples[f][j]; owner[begin[f] + j] = uint32_t(f); }

    // Uniform grid (cell = largest radius) over the signed samples.
    const auto key = [&](const cv::Vec3d& p, int dx = 0, int dy = 0, int dz = 0) {
        const auto c = [&](double x, int d) { return uint64_t(int64_t(std::floor(x / cell)) + d + (1 << 20)) & 0x1fffff; };
        return (c(p[0], dx) << 42) | (c(p[1], dy) << 21) | c(p[2], dz);
    };
    std::vector<std::pair<uint64_t, uint32_t>> keyed;
    keyed.reserve(total);
    for (size_t i = 0; i < total; ++i)
        if (fiberSign[owner[i]] != 0) keyed.emplace_back(key(sample[i]->point), uint32_t(i));
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
        std::vector<double> w;
        for (int qi = r.start; qi < r.end; ++qi) {
            const auto& q = queries[size_t(qi)];
            const size_t b = begin[q.fiber], e = begin[q.fiber + 1];
            auto& out = results[size_t(qi)];
            size_t fallbacks = 0, crossingCount = 0, parallelCount = 0;
            const auto tangentAt = [&](size_t i, size_t lo, size_t hi) {
                return unitOrZero(sample[std::min(i + 1, hi - 1)]->point - sample[i > lo ? i - 1 : i]->point);
            };
            std::vector<std::vector<Neighbour>> gathered(e - b);
            for (size_t s = b; s < e; ++s) {
                const cv::Vec3d p = sample[s]->point;
                auto& near = gathered[s - b];
                for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
                    const auto c = cells.find(key(p, dx, dy, dz));
                    if (c == cells.end()) continue;
                    for (size_t k = c->second.first; k < c->second.second; ++k) {
                        const uint32_t j = keyed[k].second, o = owner[j];
                        if (o == q.fiber || q.excluded.count(o)) continue;
                        const cv::Vec3d d = sample[j]->point - p;
                        const double distance = cv::norm(d);
                        if (distance > cell) continue;
                        near.push_back({o, j, d, sample[j]->normal * double(fiberSign[o]), distance});
                    }
                }
                std::stable_sort(near.begin(), near.end(), [](const auto& x, const auto& y) { return x.owner < y.owner; });
            }
            // Contacts (fiber frame): every gathered neighbour sample's nearest own
            // sample and gap; along each run of consecutive samples of a neighbour
            // fiber the local gap minima are its contacts.
            std::unordered_map<uint32_t, Contact> contacts;
            if (anyFiber) {
                std::unordered_map<uint32_t, std::pair<double, size_t>> nearest;
                for (size_t s = b; s < e; ++s)
                    for (const auto& nb : gathered[s - b]) {
                        auto [it, inserted] = nearest.try_emplace(nb.index, nb.distance, s);
                        if (!inserted && nb.distance < it->second.first) it->second = {nb.distance, s};
                    }
                std::vector<uint32_t> js;
                js.reserve(nearest.size());
                for (const auto& kv : nearest) js.push_back(kv.first);
                std::sort(js.begin(), js.end());
                for (size_t r0 = 0; r0 < js.size();) {
                    size_t r1 = r0 + 1;
                    while (r1 < js.size() && js[r1] == js[r1 - 1] + 1 && owner[js[r1]] == owner[js[r0]]) ++r1;
                    const auto gap = [&](size_t k) { return nearest.at(js[k]).first; };
                    std::vector<size_t> minima;
                    for (size_t k = r0; k < r1; ++k)
                        if ((k == r0 || gap(k) < gap(k - 1)) && (k + 1 == r1 || gap(k) <= gap(k + 1))) minima.push_back(k);
                    const uint32_t o = owner[js[r0]];
                    std::vector<Contact> made;
                    for (size_t m : minima) {
                        const auto [g, own] = nearest.at(js[m]);
                        const cv::Vec3d x = tangentAt(own, b, e).cross(tangentAt(js[m], begin[o], begin[o + 1]));
                        made.push_back({sample[own]->arc, g, sample[js[m]]->arc, cv::norm(x) > .5});
                    }
                    size_t c = 0;
                    for (size_t k = r0; k < r1; ++k) {
                        const auto away = [&](size_t m) { return m > k ? m - k : k - m; };
                        while (c + 1 < minima.size() && away(minima[c + 1]) <= away(minima[c])) ++c;
                        contacts[js[k]] = made[c];
                    }
                    r0 = r1;
                }
            }
            for (size_t s = b; s < e; ++s) {
                const cv::Vec3d t = tangentAt(s, b, e);
                const auto& near = gathered[s - b];
                const cv::Matx33d P = cv::Matx33d::eye() - t * t.t();
                // Principal axis perpendicular to t, signed by the weighted majority.
                struct Estimate { cv::Vec3d axis{0, 0, 0}; double wsum = 0, coherence = 0, margin = 0, count = 0, effSamples = 0, effFibers = 0; };
                const auto estimate = [&](const std::vector<double>& weight) {
                    Estimate est;
                    cv::Matx33d tensor = cv::Matx33d::zeros();
                    cv::Vec3d sum{0, 0, 0};
                    double w2 = 0., fiberSum = 0., fiber2 = 0.;
                    for (size_t k = 0; k < near.size(); ++k) {
                        if (k > 0 && near[k].owner != near[k - 1].owner) { fiber2 += fiberSum * fiberSum; fiberSum = 0.; }
                        const double x = weight[k];
                        if (!(x > 0.)) continue;
                        tensor += x * (near[k].normal * near[k].normal.t());
                        sum += x * near[k].normal;
                        est.wsum += x; w2 += x * x; fiberSum += x; ++est.count;
                    }
                    fiber2 += fiberSum * fiberSum;
                    if (!(est.wsum > 0.)) return est;
                    const cv::Matx33d M = P * tensor * P;
                    cv::Vec3d values;
                    cv::Matx33d vectors;
                    cv::eigen(M, values, vectors);
                    est.axis = {vectors(0, 0), vectors(0, 1), vectors(0, 2)};
                    const double trace = values[0] + values[1] + values[2];
                    est.coherence = trace > 0. ? (values[0] - values[1]) / trace : 0.;
                    const double m = sum.dot(est.axis);
                    if (m < 0.) est.axis = -est.axis;
                    est.margin = std::abs(m) / est.wsum;
                    est.effSamples = est.wsum * est.wsum / w2;
                    est.effFibers = fiber2 > 0. ? est.wsum * est.wsum / fiber2 : 0.;
                    return est;
                };
                cv::Vec3d firstAxis{0, 0, 0};
                if (firstSigma > 0.) {
                    w.assign(near.size(), 0.);
                    for (size_t k = 0; k < near.size(); ++k) w[k] = std::exp(-0.5 * near[k].distance * near[k].distance / (firstSigma * firstSigma));
                    firstAxis = estimate(w).axis;
                }
                const cv::Vec3d own = sample[s]->normal * double(fiberSign[q.fiber]);
                for (size_t v = 0; v < V; ++v) {
                    const auto& variant = variants[v];
                    cv::Vec3d n{0, 0, 0};
                    if (variant.frame == Frame::Own) n = sample[s]->normal;
                    if (variant.frame == Frame::Prediction) n = sample[s]->prediction;
                    if (variant.frame == Frame::TwoPass) n = firstAxis;
                    n = unitOrZero(P * n);  // perpendicular to the own tangent
                    const bool framed = variant.frame != Frame::Isotropic && cv::norm(n) > 0. && cv::norm(t) > 0.;
                    fallbacks += variant.frame != Frame::Isotropic && variant.frame != Frame::Fiber && !framed;
                    w.assign(near.size(), 0.);
                    if (variant.frame == Frame::Fiber) {
                        const cv::Vec3d pn = unitOrZero(P * sample[s]->prediction);
                        const bool parallelFramed = cv::norm(pn) > 0. && cv::norm(t) > 0.;
                        const bool count = &variant == &*std::find_if(variants.begin(), variants.end(), [](const auto& x) { return x.frame == Frame::Fiber; });
                        for (size_t k = 0; k < near.size(); ++k) {
                            if (near[k].distance > variant.radius) continue;
                            const auto& c = contacts.at(near[k].index);
                            double g;
                            if (c.crossing) {
                                const double a = (c.ownArc - sample[s]->arc) / variant.scales[0], h = c.gap / variant.scales[1],
                                             pa = (sample[near[k].index]->arc - c.partnerArc) / variant.scales[2];
                                g = std::sqrt(a * a + h * h + pa * pa);
                            } else {
                                g = parallelFramed ? scaledFrameGap(near[k].offset, t, pn, variant.parallelScales) : near[k].distance / variant.parallelScales[0];
                            }
                            if (count) (c.crossing ? crossingCount : parallelCount) += 1;
                            w[k] = gapFalloffFactor(variant.falloff, g);
                        }
                    } else
                    for (size_t k = 0; k < near.size(); ++k) {
                        if (near[k].distance > variant.radius) continue;
                        const double g = framed ? scaledFrameGap(near[k].offset, t, n, variant.scales) : near[k].distance / variant.scales[0];
                        w[k] = gapFalloffFactor(variant.falloff, g);
                    }
                    if (variant.fiberCap)
                        for (size_t a = 0; a < near.size();) {
                            size_t c = a;
                            double sum = 0., top = 0.;
                            while (c < near.size() && near[c].owner == near[a].owner) { sum += w[c]; top = std::max(top, w[c]); ++c; }
                            if (sum > 0.) for (size_t k = a; k < c; ++k) w[k] *= top / sum;
                            a = c;
                        }
                    const auto est = estimate(w);
                    out.insert(out.end(), {double(qi), double(s - b), double(v), sample[s]->arc, est.axis[0], est.axis[1], est.axis[2],
                                           own[0], own[1], own[2], est.wsum, est.coherence, est.margin, est.count, est.effSamples, est.effFibers});
                }
            }
            visited += e - b;
            frameFallbacks += fallbacks;
            crossingPairs += crossingCount;
            parallelPairs += parallelCount;
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
                const auto& own = *sample[begin[queries[qi].fiber] + size_t(r[k + 1])];
                inferred[qi].push_back({r[k + 3], own.point, {r[k + 4], r[k + 5], r[k + 6]}, own.prediction});
            }
        }
        writeFiberStripSamples(cfg.at("neighbour_samples_output").get<std::string>(), names, inferred);
    }
    Json summary{{"record_doubles", kOut}, {"columns", {"query", "sample", "variant", "arc", "nx", "ny", "nz", "own_x", "own_y", "own_z",
                                                        "weight", "coherence", "sign_margin", "neighbours", "effective_samples", "effective_fibers"}},
                 {"queries", ids}, {"unknown_queries", unknownQueries}, {"variants", cfg.at("neighbour_variants")},
                 {"fibers", listed}, {"samples", total}, {"signed_samples", keyed.size()}, {"query_samples", visited.load()},
                 {"frame_fallbacks", frameFallbacks.load()}, {"fiber_frame_crossing_pairs", crossingPairs.load()},
                 {"fiber_frame_parallel_pairs", parallelPairs.load()},
                 {"seconds", std::chrono::duration<double>(Clock::now() - started).count()}};
    std::ofstream(output.string() + ".json") << summary.dump(1) << "\n";
    return summary;
}
