// helirec/surface_composite.hpp: BandComposite, surface layers reconstructed where their points lie.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

#include "helirec/surface_composite.hpp"
#include "io/json.hpp"
#include "vc_reader/composite.hpp"
#include "vc_reader/resident_sampling.hpp"

namespace helirec::surface {

BandComposite::BandComposite(Reconstruction& recon, const float* pts, int L, int H, int W, const std::uint8_t* valid, std::optional<std::array<int, 2>> h_range,
                             int n_sets)
    : r_(recon), L_(L), H_(H), W_(W), n_sets_(n_sets) {
    const int N = recon.reconstructor()->grid_size();
    const auto [lo, hi] = recon.doable();
    const std::size_t px = static_cast<std::size_t>(H) * W, n = static_cast<std::size_t>(L) * px;
    // a point is usable where its pixel is valid, it is finite and inside the slice
    std::vector<std::uint8_t> ok(n);
    double hmin = std::numeric_limits<double>::infinity(), hmax = -hmin;
    for (std::size_t i = 0; i < n; ++i) {
        const float* p = pts + 3 * i;
        ok[i] = valid[i % px] && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) && p[0] >= 0 && p[1] >= 0 && p[0] <= N - 1 && p[1] <= N - 1;
        if (ok[i]) hmin = std::min(hmin, p[2] + double(lo)), hmax = std::max(hmax, p[2] + double(lo));
    }
    int h0 = h_range ? (*h_range)[0] : static_cast<int>(std::floor(hmin)), h1 = h_range ? (*h_range)[1] : static_cast<int>(std::floor(hmax)) + 2;
    h0 = std::max(h0, lo), h1 = std::min(h1, hi);
    h_range_ = {h0, h1};
    valid_.assign(px, 1);
    for (std::size_t i = 0; i < n; ++i) {
        const double h = pts[3 * i + 2] + double(lo);
        if (!(ok[i] && h >= h0 && h <= h1 - 1)) valid_[i % px] = 0;  // every layer samplable
    }
    for (std::size_t p = 0; p < px; ++p)
        if (valid_[p]) sel_.push_back(static_cast<std::int64_t>(p));
    // the points, layer by layer over the valid pixels, ordered by their lower slice
    const std::size_t m = static_cast<std::size_t>(L) * sel_.size();
    std::vector<std::int32_t> zlo(m);
    std::vector<double> hh(m);
    for (int l = 0; l < L; ++l)
        for (std::size_t k = 0; k < sel_.size(); ++k) {
            const std::size_t i = static_cast<std::size_t>(l) * sel_.size() + k;
            hh[i] = pts[3 * (static_cast<std::size_t>(l) * px + static_cast<std::size_t>(sel_[k])) + 2] + double(lo);
            zlo[i] = static_cast<std::int32_t>(std::min(std::floor(hh[i]), double(h1 - 2)));
        }
    order_.resize(m);
    std::iota(order_.begin(), order_.end(), 0);
    std::stable_sort(order_.begin(), order_.end(), [&](std::int64_t a, std::int64_t b) { return zlo[static_cast<std::size_t>(a)] < zlo[static_cast<std::size_t>(b)]; });
    zlo_.resize(m), fz_.resize(m), y_.resize(m), x_.resize(m);
    for (std::size_t j = 0; j < m; ++j) {
        const std::size_t i = static_cast<std::size_t>(order_[j]);
        const std::size_t l = i / sel_.size(), k = i % sel_.size();
        const float* p = pts + 3 * (l * px + static_cast<std::size_t>(sel_[k]));
        zlo_[j] = zlo[i];
        fz_[j] = static_cast<float>(hh[i] - zlo[i]);
        y_[j] = p[1], x_[j] = p[0];
    }
    const Json bp = Json::parse(backprojector_config(Json::parse(recon.parameters()).find("backprojector")->string));
    if (bp.find("backprojector")->string == "fbp") {  // the exact backprojector at the band pixels, sampled on the reconstructor's device
        sampler_ = std::make_unique<BandSampler>(recon.device(), zlo_.data(), fz_.data(), y_.data(), x_.data(), m, N, n_sets);
        const BandBox& b = sampler_->box();
        box_ = {b.x0, b.x0 + b.w, b.y0, b.y0 + b.h};
    } else {  // whole-slice methods: the points' box slice by slice
        int x0 = N, x1 = 0, y0 = N, y1 = 0;
        for (std::size_t j = 0; j < m; ++j) {
            const int iy = std::min(static_cast<int>(std::floor(y_[j])), N - 2), ix = std::min(static_cast<int>(std::floor(x_[j])), N - 2);
            x0 = std::min(x0, ix), x1 = std::max(x1, ix + 2), y0 = std::min(y0, iy), y1 = std::max(y1, iy + 2);
        }
        box_ = {x0, x1, y0, y1};
        vals_.assign(static_cast<std::size_t>(n_sets) * m, std::numeric_limits<float>::quiet_NaN());
    }
}

BandComposite::~BandComposite() = default;

std::size_t BandComposite::n_points() const { return zlo_.size(); }
std::array<int, 4> BandComposite::box() const { return box_; }

void BandComposite::run_slices(int chunk, const std::vector<std::string>& sweep, const std::function<void(int, int, double)>& progress) {
    const auto [h0, h1] = h_range_;
    const auto [x0, x1, y0, y1] = box_;
    const int bw = x1 - x0, bh = y1 - y0;
    const std::string base = Json::parse(r_.parameters()).find("filters")->text();
    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t m = zlo_.size();
    std::vector<float> points, out;
    for (int k = 0; k < std::max<int>(1, static_cast<int>(sweep.size())); ++k) {
        if (!sweep.empty()) r_.set_parameters("{\"filters\": " + sweep[static_cast<std::size_t>(k)] + "}");
        std::vector<float> prev;
        for (int c0 = h0; c0 < h1; c0 += chunk) {
            const int c1 = std::min(h1, c0 + chunk);
            const std::size_t slice = static_cast<std::size_t>(bw) * bh;
            std::vector<float> vol(prev);  // the chunk's slices, the one before them first
            vol.resize(prev.size() + static_cast<std::size_t>(c1 - c0) * slice);
            SliceRoi roi{x0, x1, y0, y1};
            r_.reconstructor()->reconstruct(c0, c1, vol.data() + prev.size(), nullptr, &roi);
            const int lo = c0 - static_cast<int>(prev.size() / slice), nz = static_cast<int>(vol.size() / slice);
            // the points whose two slices are here
            const auto a = std::lower_bound(zlo_.begin(), zlo_.end(), lo) - zlo_.begin();
            const auto b = std::lower_bound(zlo_.begin(), zlo_.end(), lo + nz - 1) - zlo_.begin();
            if (b > a) {
                points.resize(3 * static_cast<std::size_t>(b - a));
                for (auto j = a; j < b; ++j) {  // ZYX indices into the block
                    float* q = points.data() + 3 * static_cast<std::size_t>(j - a);
                    q[0] = static_cast<float>(zlo_[static_cast<std::size_t>(j)] - lo) + fz_[static_cast<std::size_t>(j)];
                    q[1] = y_[static_cast<std::size_t>(j)] - y0, q[2] = x_[static_cast<std::size_t>(j)] - x0;
                }
                vcr::sampling::Dense d;  // (vc_reader's defaults: linear)
                d.values = vol.data();
                d.points = points.data();
                d.batches = 1, d.channels = 1, d.queries = static_cast<std::size_t>(b - a);
                d.shape[0] = nz, d.shape[1] = bh, d.shape[2] = bw;
                out.resize(d.queries);
                vcr::sampling::sampleDense(d, out.data());
                std::copy(out.begin(), out.end(), vals_.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(k) * m) + a);
            }
            prev.assign(vol.end() - static_cast<std::ptrdiff_t>(slice), vol.end());
            if (progress) progress(c1 - h0, h1 - h0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    if (!sweep.empty()) r_.set_parameters("{\"filters\": " + base + "}");
}

std::string BandComposite::run(int chunk, bool timing, const std::vector<std::string>& sweep, bool prefetch, const std::function<void(int, int, double)>& progress) {
    const auto [h0, h1] = h_range_;
    if (!sweep.empty() && static_cast<int>(sweep.size()) != n_sets_)
        throw Error(std::to_string(sweep.size()) + " parameter sets for a sampler with " + std::to_string(n_sets_) + " value sets");
    if (sweep.empty() && n_sets_ != 1) throw Error("this sampler has several value sets: give one filter chain per set");
    const auto t0 = std::chrono::steady_clock::now();
    const auto wall = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    if (!sampler_) {
        run_slices(chunk, sweep, progress);
        return Json::dict().set("wall", Json::of(wall())).text();
    }
    std::vector<FilterChain> chains;
    for (const std::string& c : sweep) chains.push_back(filter_chain(c));
    sampler_->reset();
    std::map<std::string, double> sec;
    for (int c0 = h0; c0 < h1; c0 += chunk) {
        const int c1 = std::min(h1, c0 + chunk);
        const bool next = prefetch && c1 < h1;
        ChunkDebug dbg;
        dbg.timing = timing;
        r_.reconstructor()->reconstruct_band(c0, c1, *sampler_, &dbg, chains.empty() ? nullptr : &chains, next ? c1 : 0, next ? std::min(h1, c1 + chunk) : 0);
        for (const auto& [k, v] : dbg.seconds) sec[k] += v;
        for (const auto& [k, v] : dbg.stats) sec["stat_" + k] += v;
        if (progress) progress(c1 - h0, h1 - h0, wall());
    }
    Json out = Json::dict();
    for (const auto& [k, v] : sec) out.set(k, Json::of(v));
    return out.set("wall", Json::of(wall())).text();
}

std::vector<float> BandComposite::layers(int set) const {
    const std::size_t m = zlo_.size(), px = static_cast<std::size_t>(H_) * W_;
    std::vector<float> got(m);
    if (sampler_) sampler_->values(got.data(), set);
    else std::copy(vals_.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(set) * m), vals_.begin() + static_cast<std::ptrdiff_t>((set + 1) * m), got.begin());
    std::vector<float> out(static_cast<std::size_t>(L_) * px, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t j = 0; j < m; ++j) {
        const std::size_t i = static_cast<std::size_t>(order_[j]), l = i / sel_.size(), k = i % sel_.size();
        out[l * px + static_cast<std::size_t>(sel_[k])] = got[j];
    }
    return out;
}

std::array<std::vector<float>, 2> BandComposite::composites(int set) const {
    const std::vector<float> v = layers(set);
    const std::size_t px = static_cast<std::size_t>(H_) * W_;
    std::array<std::vector<float>, 2> out;
    const vcr::composite::Method methods[2] = {vcr::composite::Method::Max, vcr::composite::Method::Mean};
    for (int k = 0; k < 2; ++k) {
        vcr::composite::Options o;
        o.method = methods[k];
        o.isoCutoff = -std::numeric_limits<float>::infinity();
        out[static_cast<std::size_t>(k)].resize(px);
        vcr::composite::reduce(v.data(), static_cast<std::size_t>(L_), px, o, out[static_cast<std::size_t>(k)].data());
    }
    return out;
}

}  // namespace helirec::surface
