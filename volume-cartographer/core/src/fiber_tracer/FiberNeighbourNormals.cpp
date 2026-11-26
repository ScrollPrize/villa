#include "vc/fiber_tracer/FiberNeighbourNormals.hpp"

#include "vc/core/util/SpatialBatches.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace vc::fiber_tracer
{
namespace
{
constexpr char kMagic[8] = {'V', 'C', 'S', 'T', 'R', 'I', 'P', '2'};
constexpr std::size_t kRecord = 10;  // arc, point xyz, normal xyz, prediction axis xyz

cv::Vec3d unitOrZero(const cv::Vec3d& v)
{
    const double n = cv::norm(v);
    return n > 0. ? v / n : cv::Vec3d(0, 0, 0);
}

struct Contact { double ownArc, gap, partnerArc; bool crossing; };
}  // namespace

FiberStripSampleFile readFiberStripSampleFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("strip samples: cannot read " + path.string());
    const auto truncated = [&] { return std::runtime_error("strip samples: truncated " + path.string()); };
    const auto get = [&](void* data, std::size_t bytes) {
        if (!file.read(static_cast<char*>(data), std::streamsize(bytes))) throw truncated();
    };
    char magic[8];
    get(magic, sizeof magic);
    if (std::memcmp(magic, kMagic, sizeof magic) != 0) throw std::runtime_error("strip samples: not a strip samples file " + path.string());
    std::uint64_t listed = 0;
    get(&listed, sizeof listed);
    std::error_code sizeError;
    const auto fileBytes = std::filesystem::file_size(path, sizeError);
    if (sizeError) throw std::runtime_error("strip samples: cannot read " + path.string());
    FiberStripSampleFile out;
    out.ids.resize(listed);
    out.samples.resize(listed);
    // The headers in file order - each fiber's id, its sample count and where
    // its samples start - then the samples themselves on several threads.
    std::vector<std::uint64_t> counts(listed), offsets(listed);
    std::uint64_t at = sizeof magic + sizeof listed;
    for (std::uint64_t f = 0; f < listed; ++f) {
        file.seekg(std::streamoff(at));
        get(out.ids[f].data(), out.ids[f].size());
        get(&counts[f], sizeof counts[f]);
        offsets[f] = at + out.ids[f].size() + sizeof counts[f];
        if (counts[f] > (fileBytes - std::min<std::uint64_t>(fileBytes, offsets[f])) / (kRecord * sizeof(double)))
            throw truncated();
        at = offsets[f] + counts[f] * kRecord * sizeof(double);
    }
    const std::size_t threads = std::clamp<std::size_t>(std::thread::hardware_concurrency(), 1, 16);
    std::atomic<std::uint64_t> next{0};
    std::atomic<bool> failed{false};
    const auto work = [&] {
        std::ifstream own(path, std::ios::binary);
        std::vector<double> buffer;
        constexpr std::uint64_t block = 256;
        for (std::uint64_t first; !failed && (first = next.fetch_add(block)) < listed;)
            for (std::uint64_t f = first; f < std::min(first + block, listed); ++f) {
                const auto n = counts[f];
                buffer.resize(n * kRecord);
                own.seekg(std::streamoff(offsets[f]));
                if (!own.read(reinterpret_cast<char*>(buffer.data()), std::streamsize(buffer.size() * sizeof(double)))) {
                    failed = true;
                    return;
                }
                auto& list = out.samples[f];
                list.resize(n);
                for (std::size_t j = 0; j < n; ++j) {
                    const double* v = &buffer[j * kRecord];
                    list[j] = {v[0], {v[1], v[2], v[3]}, {v[4], v[5], v[6]}, {v[7], v[8], v[9]}};
                }
            }
    };
    std::vector<std::thread> pool;
    for (std::size_t t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& thread : pool) thread.join();
    if (failed) throw truncated();
    return out;
}

void writeFiberStripSampleFile(const std::filesystem::path& output, std::span<const FiberStripUuid> ids,
                               std::span<const std::vector<FiberStripSample>> samples)
{
    if (ids.size() != samples.size()) throw std::invalid_argument("strip samples: ids and samples differ in size");
    std::ofstream file(output, std::ios::binary);
    if (!file) throw std::runtime_error("strip samples: cannot write " + output.string());
    const auto put = [&](const void* data, std::size_t bytes) { file.write(static_cast<const char*>(data), std::streamsize(bytes)); };
    put(kMagic, sizeof kMagic);
    std::uint64_t listed = 0;
    for (const auto& s : samples) listed += !s.empty();
    put(&listed, sizeof listed);
    std::vector<double> buffer;
    for (std::size_t f = 0; f < ids.size(); ++f) {
        if (samples[f].empty()) continue;
        put(ids[f].data(), ids[f].size());
        const std::uint64_t n = samples[f].size();
        put(&n, sizeof n);
        buffer.clear();
        for (const auto& s : samples[f])
            buffer.insert(buffer.end(), {s.arc, s.point[0], s.point[1], s.point[2], s.normal[0], s.normal[1], s.normal[2],
                                         s.prediction[0], s.prediction[1], s.prediction[2]});
        put(buffer.data(), buffer.size() * sizeof(double));
    }
    if (!file) throw std::runtime_error("strip samples: write failed " + output.string());
}

FiberNeighbourField::FiberNeighbourField(std::vector<std::vector<FiberStripSample>> samples, std::vector<int> signs, const double radius)
    : samples_(std::move(samples)), signs_(std::move(signs)), cell_(radius)
{
    if (samples_.size() != signs_.size()) throw std::invalid_argument("neighbour field: one sign per fiber required");
    if (!(cell_ > 0.)) throw std::invalid_argument("neighbour field: radius must be positive");
    const std::size_t listed = samples_.size();
    begin_.assign(listed + 1, 0);
    for (std::size_t f = 0; f < listed; ++f) begin_[f + 1] = begin_[f] + samples_[f].size();
    const std::size_t total = begin_.back();
    sample_.resize(total);
    owner_.resize(total);
    // Uniform grid (cell = radius) over the signed samples, in the order of
    // their cell and then of the samples.
    std::vector<std::size_t> keyedBegin(listed + 1, 0);
    for (std::size_t f = 0; f < listed; ++f) keyedBegin[f + 1] = keyedBegin[f] + (signs_[f] != 0 ? samples_[f].size() : 0);
    keyed_.resize(keyedBegin.back());
    const int threads = int(std::clamp<unsigned>(std::thread::hardware_concurrency(), 1, 16));
    {
        std::atomic<std::size_t> next{0};
        constexpr std::size_t block = 1024;
        const auto work = [&] {
            for (std::size_t first; (first = next.fetch_add(block)) < listed;)
                for (std::size_t f = first; f < std::min(first + block, listed); ++f)
                    for (std::size_t j = 0; j < samples_[f].size(); ++j) {
                        const std::size_t i = begin_[f] + j;
                        sample_[i] = &samples_[f][j];
                        owner_[i] = std::uint32_t(f);
                        if (signs_[f] != 0) keyed_[keyedBegin[f] + j] = {key(samples_[f][j].point), std::uint32_t(i)};
                    }
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < threads; ++t) pool.emplace_back(work);
        work();
        for (auto& thread : pool) thread.join();
    }
    // The pairs are all different, so the order is the same on any number of threads.
    core::util::parallelSort(keyed_, std::less<>{}, threads);
    for (std::size_t a = 0; a < keyed_.size();) {
        std::size_t b = a;
        while (b < keyed_.size() && keyed_[b].first == keyed_[a].first) ++b;
        cells_.emplace(keyed_[a].first, std::make_pair(a, b));
        a = b;
    }
}

std::uint64_t FiberNeighbourField::key(const cv::Vec3d& p, const int dx, const int dy, const int dz) const
{
    const auto c = [&](double x, int d) { return std::uint64_t(std::int64_t(std::floor(x / cell_)) + d + (1 << 20)) & 0x1fffff; };
    return (c(p[0], dx) << 42) | (c(p[1], dy) << 21) | c(p[2], dz);
}

std::vector<std::vector<FiberNeighbourField::Near>> FiberNeighbourField::gather(const std::span<const FiberStripSample> query,
                                                                              const std::int64_t self,
                                                                              const std::set<std::uint32_t>* excluded) const
{
    std::vector<std::vector<Near>> gathered(query.size());
    for (std::size_t s = 0; s < query.size(); ++s) {
        const cv::Vec3d p = query[s].point;
        auto& near = gathered[s];
        for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
            const auto c = cells_.find(key(p, dx, dy, dz));
            if (c == cells_.end()) continue;
            for (std::size_t k = c->second.first; k < c->second.second; ++k) {
                const std::uint32_t j = keyed_[k].second, o = owner_[j];
                if (std::int64_t(o) == self || (excluded && excluded->count(o))) continue;
                const cv::Vec3d d = sample_[j]->point - p;
                const double distance = cv::norm(d);
                if (distance > cell_) continue;
                near.push_back({o, j, distance});
            }
        }
        std::stable_sort(near.begin(), near.end(), [](const auto& x, const auto& y) { return x.owner < y.owner; });
    }
    return gathered;
}

FiberNeighbourField::Contacts FiberNeighbourField::contacts(const std::vector<std::vector<Near>>& gathered) const
{
    // Every gathered neighbour sample's nearest own sample and gap.
    Contacts out;
    for (std::size_t s = 0; s < gathered.size(); ++s)
        for (const auto& nb : gathered[s]) {
            auto [it, inserted] = out.nearest.try_emplace(nb.index, nb.distance, s);
            if (!inserted && nb.distance < it->second.first) it->second = {nb.distance, s};
        }
    auto& js = out.samples;
    js.reserve(out.nearest.size());
    for (const auto& kv : out.nearest) js.push_back(kv.first);
    std::sort(js.begin(), js.end());
    for (std::size_t r0 = 0; r0 < js.size();) {
        std::size_t r1 = r0 + 1;
        while (r1 < js.size() && js[r1] == js[r1 - 1] + 1 && owner_[js[r1]] == owner_[js[r0]]) ++r1;
        out.runs.emplace_back(r0, r1);
        r0 = r1;
    }
    return out;
}

std::vector<std::size_t> FiberNeighbourField::Contacts::minima(const std::size_t first, const std::size_t last) const
{
    const auto gap = [&](std::size_t k) { return nearest.at(samples[k]).first; };
    std::vector<std::size_t> found;
    for (std::size_t k = first; k < last; ++k)
        if ((k == first || gap(k) < gap(k - 1)) && (k + 1 == last || gap(k) <= gap(k + 1))) found.push_back(k);
    return found;
}

std::vector<FiberCrossingAxis> FiberNeighbourField::crossings(const std::span<const FiberStripSample> query,
                                                              const std::int64_t self,
                                                              const std::set<std::uint32_t>* excluded) const
{
    std::vector<FiberCrossingAxis> out;
    const std::size_t Q = query.size();
    if (Q < 2) return out;
    const auto found = contacts(gather(query, self, excluded));
    const auto unit = [](const cv::Vec3d& v) { const double n = cv::norm(v); return n > 0. ? v / n : cv::Vec3d(0, 0, 0); };
    for (const auto [r0, r1] : found.runs) {
        const std::uint32_t o = owner_[found.samples[r0]];
        const std::size_t lo = begin_[o], hi = begin_[o + 1];
        for (const std::size_t m : found.minima(r0, r1)) {
            const std::uint32_t j = found.samples[m];
            const auto [gap, own] = found.nearest.at(j);
            // Tangents over the neighbouring samples, as the contacts of infer() take them.
            const cv::Vec3d ownTangent = unit(query[std::min(own + 1, Q - 1)].point - query[own > 0 ? own - 1 : own].point);
            const cv::Vec3d partnerTangent = unit(sample_[std::min<std::size_t>(j + 1, hi - 1)]->point - sample_[j > lo ? j - 1 : j]->point);
            const cv::Vec3d x = ownTangent.cross(partnerTangent);
            if (!(cv::norm(x) > .5)) continue;
            FiberCrossingAxis crossing;
            crossing.arc = query[own].arc;
            crossing.axis = x;
            crossing.offset = sample_[j]->point - query[own].point;
            crossing.gap = gap;
            crossing.ownTangent = ownTangent;
            out.push_back(crossing);
        }
    }
    std::sort(out.begin(), out.end(), [](const auto& x, const auto& y) { return x.arc < y.arc; });
    return out;
}

std::vector<FiberNeighbourEstimate> FiberNeighbourField::infer(const std::span<const FiberStripSample> query,
                                                               const std::span<const FiberNeighbourVariant> variants,
                                                               const std::int64_t self,
                                                               const std::set<std::uint32_t>* excluded,
                                                               FiberNeighbourCounts* counts) const
{
    const std::size_t V = variants.size(), Q = query.size();
    bool anyFiber = false;
    double firstSigma = 0.;
    for (const auto& v : variants) {
        if (!(v.radius > 0.) || v.radius > cell_) throw std::invalid_argument("neighbour field: a variant's radius must be positive and within the field's");
        if (!(v.scales[0] > 0. && v.scales[1] > 0. && v.scales[2] > 0.) || !(v.firstSigma > 0.)
            || !(v.parallelScales[0] > 0. && v.parallelScales[1] > 0. && v.parallelScales[2] > 0.))
            throw std::invalid_argument("neighbour field: positive scales and first_sigma required");
        anyFiber = anyFiber || v.frame == FiberNeighbourFrame::Fiber;
        if (v.frame == FiberNeighbourFrame::TwoPass || (v.frame == FiberNeighbourFrame::Fiber && v.estimatedParallelFrame)) {
            if (firstSigma > 0. && firstSigma != v.firstSigma) throw std::invalid_argument("neighbour field: one first_sigma for all variants that use a first estimate");
            firstSigma = v.firstSigma;
        }
    }
    std::vector<FiberNeighbourEstimate> out(Q * V);
    if (Q == 0 || V == 0) return out;
    std::vector<double> w;
    std::size_t fallbacks = 0, crossingCount = 0, parallelCount = 0;
    const auto queryTangent = [&](std::size_t i) {
        return unitOrZero(query[std::min(i + 1, Q - 1)].point - query[i > 0 ? i - 1 : i].point);
    };
    const auto fieldTangent = [&](std::size_t i, std::size_t lo, std::size_t hi) {
        return unitOrZero(sample_[std::min(i + 1, hi - 1)]->point - sample_[i > lo ? i - 1 : i]->point);
    };
    const auto gathered = gather(query, self, excluded);
    // Contacts (fiber frame): along each run of consecutive samples of a
    // neighbour fiber the local gap minima are its contacts.
    std::unordered_map<std::uint32_t, Contact> contacts;
    if (anyFiber) {
        const auto found = this->contacts(gathered);
        const auto& js = found.samples;
        for (const auto [r0, r1] : found.runs) {
            const auto minima = found.minima(r0, r1);
            const std::uint32_t o = owner_[js[r0]];
            std::vector<Contact> made;
            for (std::size_t m : minima) {
                const auto [g, own] = found.nearest.at(js[m]);
                const cv::Vec3d x = queryTangent(own).cross(fieldTangent(js[m], begin_[o], begin_[o + 1]));
                made.push_back({query[own].arc, g, sample_[js[m]]->arc, cv::norm(x) > .5});
            }
            std::size_t c = 0;
            for (std::size_t k = r0; k < r1; ++k) {
                const auto away = [&](std::size_t m) { return m > k ? m - k : k - m; };
                while (c + 1 < minima.size() && away(minima[c + 1]) <= away(minima[c])) ++c;
                contacts[js[k]] = made[c];
            }
        }
    }
    const auto firstFiber = std::find_if(variants.begin(), variants.end(), [](const auto& x) { return x.frame == FiberNeighbourFrame::Fiber; });
    for (std::size_t s = 0; s < Q; ++s) {
        const cv::Vec3d t = queryTangent(s);
        const auto& near = gathered[s];
        const auto offsetOf = [&](const Near& nb) { return sample_[nb.index]->point - query[s].point; };
        const auto normalOf = [&](const Near& nb) { return sample_[nb.index]->normal * double(signs_[nb.owner]); };
        const cv::Matx33d P = cv::Matx33d::eye() - t * t.t();
        // Principal axis perpendicular to t, signed by the weighted majority.
        const auto estimate = [&](const std::vector<double>& weight) {
            FiberNeighbourEstimate est;
            cv::Matx33d tensor = cv::Matx33d::zeros();
            cv::Vec3d sum{0, 0, 0};
            double w2 = 0., fiberSum = 0., fiber2 = 0.;
            for (std::size_t k = 0; k < near.size(); ++k) {
                if (k > 0 && near[k].owner != near[k - 1].owner) { fiber2 += fiberSum * fiberSum; fiberSum = 0.; }
                const double x = weight[k];
                if (!(x > 0.)) continue;
                const cv::Vec3d normal = normalOf(near[k]);
                tensor += x * (normal * normal.t());
                sum += x * normal;
                est.weight += x; w2 += x * x; fiberSum += x; ++est.count;
            }
            fiber2 += fiberSum * fiberSum;
            if (!(est.weight > 0.)) return est;
            const cv::Matx33d M = P * tensor * P;
            cv::Vec3d values;
            cv::Matx33d vectors;
            cv::eigen(M, values, vectors);
            est.axis = {vectors(0, 0), vectors(0, 1), vectors(0, 2)};
            const double trace = values[0] + values[1] + values[2];
            est.coherence = trace > 0. ? (values[0] - values[1]) / trace : 0.;
            const double m = sum.dot(est.axis);
            if (m < 0.) est.axis = -est.axis;
            est.margin = std::abs(m) / est.weight;
            est.effectiveSamples = est.weight * est.weight / w2;
            est.effectiveFibers = fiber2 > 0. ? est.weight * est.weight / fiber2 : 0.;
            return est;
        };
        cv::Vec3d firstAxis{0, 0, 0};
        if (firstSigma > 0.) {
            w.assign(near.size(), 0.);
            for (std::size_t k = 0; k < near.size(); ++k) w[k] = std::exp(-0.5 * near[k].distance * near[k].distance / (firstSigma * firstSigma));
            firstAxis = estimate(w).axis;
        }
        for (std::size_t v = 0; v < V; ++v) {
            const auto& variant = variants[v];
            cv::Vec3d n{0, 0, 0};
            if (variant.frame == FiberNeighbourFrame::Own) n = query[s].normal;
            if (variant.frame == FiberNeighbourFrame::Prediction) n = query[s].prediction;
            if (variant.frame == FiberNeighbourFrame::TwoPass) n = firstAxis;
            n = unitOrZero(P * n);  // perpendicular to the own tangent
            const bool framed = variant.frame != FiberNeighbourFrame::Isotropic && cv::norm(n) > 0. && cv::norm(t) > 0.;
            fallbacks += variant.frame != FiberNeighbourFrame::Isotropic && variant.frame != FiberNeighbourFrame::Fiber && !framed;
            w.assign(near.size(), 0.);
            if (variant.frame == FiberNeighbourFrame::Fiber) {
                cv::Vec3d pn = unitOrZero(P * query[s].prediction);
                if (variant.estimatedParallelFrame && !(cv::norm(pn) > 0.)) pn = unitOrZero(P * firstAxis);
                const bool parallelFramed = cv::norm(pn) > 0. && cv::norm(t) > 0.;
                const bool count = &variant == &*firstFiber;
                for (std::size_t k = 0; k < near.size(); ++k) {
                    if (near[k].distance > variant.radius) continue;
                    const auto& c = contacts.at(near[k].index);
                    double g;
                    if (c.crossing) {
                        const double a = (c.ownArc - query[s].arc) / variant.scales[0], h = c.gap / variant.scales[1],
                                     pa = (sample_[near[k].index]->arc - c.partnerArc) / variant.scales[2];
                        g = std::sqrt(a * a + h * h + pa * pa);
                    } else {
                        g = parallelFramed ? scaledFrameGap(offsetOf(near[k]), t, pn, variant.parallelScales) : near[k].distance / variant.parallelScales[0];
                    }
                    if (count) (c.crossing ? crossingCount : parallelCount) += 1;
                    w[k] = gapFalloffFactor(variant.falloff, g);
                }
            } else
            for (std::size_t k = 0; k < near.size(); ++k) {
                if (near[k].distance > variant.radius) continue;
                const double g = framed ? scaledFrameGap(offsetOf(near[k]), t, n, variant.scales) : near[k].distance / variant.scales[0];
                w[k] = gapFalloffFactor(variant.falloff, g);
            }
            if (variant.fiberCap)
                for (std::size_t a = 0; a < near.size();) {
                    std::size_t c = a;
                    double sum = 0., top = 0.;
                    while (c < near.size() && near[c].owner == near[a].owner) { sum += w[c]; top = std::max(top, w[c]); ++c; }
                    if (sum > 0.) for (std::size_t k = a; k < c; ++k) w[k] *= top / sum;
                    a = c;
                }
            out[s * V + v] = estimate(w);
        }
    }
    if (counts) {
        counts->frameFallbacks += fallbacks;
        counts->crossingPairs += crossingCount;
        counts->parallelPairs += parallelCount;
    }
    return out;
}

std::size_t setFiberNeighbourTargets(const FiberStripChain& chain, const std::span<const FiberStripSample> list,
                                     const double weight, const double sigma, std::vector<FiberStripAxisTarget>& targets)
{
    if (!(sigma > 0.) || weight < 0.) throw std::invalid_argument("neighbour_sigma must be positive and neighbour_weight >= 0");
    if (chain.points.empty() || list.empty()) return 0;
    if (targets.size() != chain.points.size()) throw std::invalid_argument("neighbour targets: one target per chain point required");
    std::size_t nodes = 0;
    for (std::size_t k = 0; k < chain.points.size(); ++k) {
        cv::Vec3d mean{0, 0, 0};
        double sum = 0.;
        const auto lo = std::lower_bound(list.begin(), list.end(), chain.arc[k] - 3 * sigma, [](const auto& a, double x) { return a.arc < x; });
        for (auto it = lo; it != list.end() && it->arc <= chain.arc[k] + 3 * sigma; ++it) {
            const double d = it->arc - chain.arc[k], f = std::exp(-0.5 * d * d / (sigma * sigma));
            mean += f * it->normal;
            sum += f;
        }
        const double length = cv::norm(mean);
        if (!(sum > 0.) || !(length > 0.)) continue;
        auto& target = targets[k];
        target.signedAxis = mean / length;
        target.signedWeight = weight * std::min(1., sum) * length / sum;
        ++nodes;
    }
    return nodes;
}

}  // namespace vc::fiber_tracer
