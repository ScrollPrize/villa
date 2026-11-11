// The record against every frame on disk: the inventory pass, the dark and flat series, and the beam's level.
#include <omp.h>

#include <cmath>
#include <fstream>

#include "calibration/util.hpp"
#include "helirec/calibration/verify.hpp"

namespace helirec::calibration {

namespace {

std::string join(const std::vector<std::string>& v, const char* sep = "; ") {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + v[i];
    return s;
}

int n_strips(const Inventory& inv, int cols) { return (cols + inv.strip_cols - 1) / inv.strip_cols; }

}  // namespace

Inventory check_inventory(const Record& rec, Frames& frames, Report& rep) {
    const std::string scan = rec.label;
    const int rows = frames.rows(), cols = frames.cols();
    Inventory inv;
    inv.stretches.resize(rec.stretches.size());
    const int ns = n_strips(inv, cols);
    const int max_threads = std::max(64, omp_get_max_threads());

    std::int64_t frames_read = 0, compared = 0, mismatched = 0;
    std::vector<std::string> first_bad;
    for (std::size_t si = 0; si < rec.stretches.size(); ++si) {
        const Stretch& s = rec.stretches[si];
        Inventory::PerStretch& out = inv.stretches[si];
        const std::size_t n = static_cast<std::size_t>(s.n_frames);
        // the rectangle of the recorded statistics; without one, the whole frame
        const bool have = s.stat_sum.size() == n && s.stat_w > 0 && s.stat_h > 0;
        const int x0 = have ? s.stat_x : 0, y0 = have ? s.stat_y : 0, w = have ? s.stat_w : cols, h = have ? s.stat_h : rows;
        if (x0 < 0 || y0 < 0 || x0 + w > cols || y0 + h > rows) throw Error("stretch " + s.entry + ": statistics rectangle outside the frame");
        out.sum.assign(n, 0);
        out.min.assign(n, 0);
        out.max.assign(n, 0);
        out.frame_max.assign(n, 0);
        out.strips.assign(n * ns, 0);
        out.histogram.assign(65536, 0);
        std::vector<std::vector<std::int64_t>> hist(max_threads);
        frames.visit(static_cast<int>(si), 0, s.n_frames, [&](std::int64_t f, const std::uint16_t* data) {
            std::vector<std::int64_t>& my = hist[omp_get_thread_num()];
            if (my.empty()) my.assign(65536, 0);
            std::uint16_t top = 0;
            for (std::size_t p = 0; p < static_cast<std::size_t>(rows) * cols; ++p) {
                ++my[data[p]];
                top = std::max(top, data[p]);
            }
            std::uint64_t sum = 0;
            std::uint16_t lo = 65535, hi = 0;
            std::vector<std::uint64_t> strip(ns, 0);
            for (int r = y0; r < y0 + h; ++r) {
                const std::uint16_t* row = data + static_cast<std::size_t>(r) * cols;
                for (int c = x0; c < x0 + w; ++c) {
                    lo = std::min(lo, row[c]);
                    hi = std::max(hi, row[c]);
                }
                for (int k = 0; k < ns; ++k) {
                    const int a = std::max(x0, k * inv.strip_cols), b = std::min(x0 + w, (k + 1) * inv.strip_cols);
                    std::uint64_t t = 0;
                    for (int c = a; c < b; ++c) t += row[c];
                    strip[k] += t;
                }
            }
            for (int k = 0; k < ns; ++k) {
                const int a = std::max(x0, k * inv.strip_cols), b = std::min(x0 + w, (k + 1) * inv.strip_cols);
                sum += strip[k];
                out.strips[static_cast<std::size_t>(f) * ns + k] = b > a ? static_cast<float>(static_cast<double>(strip[k]) / (static_cast<double>(b - a) * h)) : 0.0f;
            }
            out.sum[f] = static_cast<double>(sum);
            out.min[f] = lo;
            out.max[f] = hi;
            out.frame_max[f] = top;
        });
        for (const auto& my : hist)
            if (!my.empty())
                for (int v = 0; v < 65536; ++v) out.histogram[v] += my[v];
        frames_read += s.n_frames;
        if (!have) continue;
        for (std::size_t f = 0; f < n; ++f) {
            ++compared;
            const bool same = out.sum[f] == s.stat_sum[f] && (s.stat_min.size() != n || out.min[f] == s.stat_min[f]) &&
                              (s.stat_max.size() != n || out.max[f] == s.stat_max[f]) &&
                              (s.stat_avg.size() != n || std::fabs(out.sum[f] / (static_cast<double>(w) * h) - s.stat_avg[f]) <= 1e-9 * s.stat_avg[f]);
            if (!same) {
                ++mismatched;
                if (first_bad.size() < 8) first_bad.push_back(fmt("%s frame %zu (sum %.0f, recorded %.0f)", s.entry.c_str(), f, out.sum[f], s.stat_sum[f]));
            }
        }
    }
    {
        Finding& f = rep.add("4.1", scan, "every frame on disk against the statistics recorded for it at acquisition");
        f.stated = "sum, min, max and mean of each frame over the statistics rectangle";
        f.found = fmt("%lld frames read, %lld compared, %lld differ", static_cast<long long>(frames_read), static_cast<long long>(compared), static_cast<long long>(mismatched));
        if (!first_bad.empty()) f.found += ": " + join(first_bad);
        f.verdict = compared == 0 ? "open" : mismatched == 0 ? "holds" : "fails";
        f.note = "the sums are integers and are compared exactly: frame k of the record is the frame the detector software saw as k, unaltered. "
                 "This does not show that the camera delivered a frame for every trigger";
        f.numbers = {{"frames", static_cast<double>(frames_read)}, {"compared", static_cast<double>(compared)}, {"mismatched", static_cast<double>(mismatched)}};
    }
    // 8.1.4, 8.1.7: the raw values
    {
        Finding& f = rep.add("8.1.4 8.1.7", scan, "range of the raw values, whole frames");
        std::vector<std::string> parts;
        for (std::size_t si = 0; si < rec.stretches.size(); ++si) {
            const Stretch& s = rec.stretches[si];
            const auto& hgram = inv.stretches[si].histogram;
            std::int64_t total = 0;
            for (std::int64_t v : hgram) total += v;
            int lowest = 0, highest = 65535;
            while (lowest < 65535 && !hgram[lowest]) ++lowest;
            while (highest > 0 && !hgram[highest]) --highest;
            // the top of the continuous distribution: the highest value below which no run of 64 values is empty
            int bulk = lowest, empty = 0;
            for (int v = lowest; v <= highest; ++v) {
                if (hgram[v]) bulk = v, empty = 0;
                else if (++empty >= 64) break;
            }
            std::int64_t above = 0;
            for (int v = bulk + 1; v <= 65535; ++v) above += hgram[v];
            std::int64_t frames_top = 0, first = -1;
            for (std::size_t k = 0; k < inv.stretches[si].frame_max.size(); ++k)
                if (inv.stretches[si].frame_max[k] == 65535) {
                    ++frames_top;
                    if (first < 0) first = static_cast<std::int64_t>(k);
                }
            std::string part = fmt("%s %s: %d to %d; continuous up to %d, %lld values above it; %lld values at 65535", s.entry.c_str(), role_name(s.role), lowest, highest,
                                   bulk, static_cast<long long>(above), static_cast<long long>(hgram[65535]));
            if (frames_top) part += fmt(" in %lld frames (first: %lld)", static_cast<long long>(frames_top), static_cast<long long>(first));
            parts.push_back(part);
            f.numbers.push_back({"max_" + s.entry, static_cast<double>(highest)});
            f.numbers.push_back({"bulk_max_" + s.entry, static_cast<double>(bulk)});
            f.numbers.push_back({"above_bulk_" + s.entry, static_cast<double>(above)});
            (void)total;
        }
        f.stated = "16-bit values; the saturation flag of the detector software is off";
        f.found = join(parts);
        f.note = "values above the continuous distribution are isolated outliers (8.1.7). A value is the mean of binned sensor pixels: one saturated sensor pixel does not reach 65535";
    }
    return inv;
}

void save_inventory(const Inventory& inv, const Record& rec, const std::string& path) {
    std::ofstream bin(path + ".bin", std::ios::binary);
    std::ofstream idx(path + ".json");
    if (!bin || !idx) throw Error("cannot write " + path);
    idx << "{\"strip_cols\": " << inv.strip_cols << ", \"layout\": \"per stretch: sum f64[n], min f64[n], max f64[n], frame_max u16[n], strips f32[n, n_strips], histogram i64[65536]\", \"stretches\": [";
    for (std::size_t i = 0; i < inv.stretches.size(); ++i) {
        const auto& s = inv.stretches[i];
        const std::size_t n = s.sum.size();
        idx << (i ? ", " : "") << "{\"entry\": \"" << rec.stretches[i].entry << "\", \"n\": " << n << ", \"n_strips\": " << (n ? s.strips.size() / n : 0)
            << ", \"offset\": " << bin.tellp() << "}";
        bin.write(reinterpret_cast<const char*>(s.sum.data()), n * 8);
        bin.write(reinterpret_cast<const char*>(s.min.data()), n * 8);
        bin.write(reinterpret_cast<const char*>(s.max.data()), n * 8);
        bin.write(reinterpret_cast<const char*>(s.frame_max.data()), n * 2);
        bin.write(reinterpret_cast<const char*>(s.strips.data()), s.strips.size() * 4);
        bin.write(reinterpret_cast<const char*>(s.histogram.data()), s.histogram.size() * 8);
    }
    idx << "]}\n";
}

Inventory load_inventory(const Record& rec, int cols, const std::string& path) {
    std::ifstream bin(path + ".bin", std::ios::binary);
    if (!bin) throw Error("cannot read " + path + ".bin");
    Inventory inv;
    const int ns = n_strips(inv, cols);
    inv.stretches.resize(rec.stretches.size());
    for (std::size_t i = 0; i < rec.stretches.size(); ++i) {
        auto& s = inv.stretches[i];
        const std::size_t n = static_cast<std::size_t>(rec.stretches[i].n_frames);
        s.sum.resize(n), s.min.resize(n), s.max.resize(n), s.frame_max.resize(n), s.strips.resize(n * ns), s.histogram.resize(65536);
        bin.read(reinterpret_cast<char*>(s.sum.data()), n * 8);
        bin.read(reinterpret_cast<char*>(s.min.data()), n * 8);
        bin.read(reinterpret_cast<char*>(s.max.data()), n * 8);
        bin.read(reinterpret_cast<char*>(s.frame_max.data()), n * 2);
        bin.read(reinterpret_cast<char*>(s.strips.data()), s.strips.size() * 4);
        bin.read(reinterpret_cast<char*>(s.histogram.data()), s.histogram.size() * 8);
        if (!bin) throw Error(path + ".bin does not match the record");
    }
    return inv;
}

// ------------------------------------------------------------------------------------------------ series

namespace {

// mean of `v` (rows, cols) over blocks of `size` x `size` pixels
std::vector<double> block_means(const std::vector<double>& v, int rows, int cols, int size, int& br, int& bc) {
    br = rows / size, bc = cols / size;
    std::vector<double> out(static_cast<std::size_t>(br) * bc, 0.0);
    for (int i = 0; i < br; ++i)
        for (int j = 0; j < bc; ++j) {
            double s = 0;
            for (int r = i * size; r < (i + 1) * size; ++r)
                for (int c = j * size; c < (j + 1) * size; ++c) s += v[static_cast<std::size_t>(r) * cols + c];
            out[static_cast<std::size_t>(i) * bc + j] = s / (static_cast<double>(size) * size);
        }
    return out;
}

}  // namespace

void check_series(const Record& rec, Frames& frames, const FlatField& ff, Report& rep, const Record* other, Frames* other_frames, const FlatField* other_ff) {
    const std::string scan = rec.label;
    const int rows = ff.rows, cols = ff.cols;
    const std::size_t np = static_cast<std::size_t>(rows) * cols;
    const int size = 32;
    int br = 0, bc = 0;

    // ---- darks: 4.2, 8.1.1
    const SeriesStats* dark = nullptr;
    std::vector<const SeriesStats*> flats;
    std::vector<std::string> flat_entries;
    for (std::size_t k = 0; k < ff.series.size(); ++k) {
        const Stretch& s = rec.stretches[ff.series_stretch[k]];
        if (s.role == Role::Dark && !dark) dark = &ff.series[k];
        if (s.role == Role::Flat) flats.push_back(&ff.series[k]), flat_entries.push_back(s.entry);
    }
    if (dark) {
        const Spread level = spread(dark->mean);
        long double var = 0;
        for (double v : dark->variance) var += v;
        const double noise = std::sqrt(static_cast<double>(var / np));
        const std::size_t n = dark->frame_mean.size();
        const Spread fm = spread(dark->frame_mean), head = spread(dark->frame_mean, 0, std::min<std::size_t>(50, n)), tail = spread(dark->frame_mean, n - std::min<std::size_t>(50, n), n);
        std::int64_t odd = 0;
        for (double v : dark->mean)
            if (std::fabs(v - level.mean) > 10 * noise) ++odd;
        Finding& f = rep.add("4.2 8.1.1", scan, "dark frames");
        f.stated = "the dark stretch";
        f.found = fmt("%lld frames; level %.3f counts (pixel means %.1f to %.1f, spread %.2f rms); noise of one pixel from frame to frame %.2f counts rms; "
                      "frame means %.3f to %.3f, first 50 frames %.3f, last 50 %.3f; %lld pixels differ from the level by more than ten times the noise",
                      static_cast<long long>(dark->n), level.mean, level.min, level.max, level.sd, noise, fm.min, fm.max, head.mean, tail.mean, static_cast<long long>(odd));
        f.verdict = level.mean < 2000 && fm.max - fm.min < 5 ? "holds" : "fails";
        f.note = "verdict: the frames are darks (a constant low level, no beam)";
        f.numbers = {{"level", level.mean}, {"noise_rms", noise}, {"odd_pixels", static_cast<double>(odd)}};
    }
    // ---- flats: 4.2, 8.2.2, 8.2.5
    if (dark && !flats.empty()) {
        std::vector<std::string> parts;
        // a projection for comparison: transmission of the first projection frame against the mean flat
        std::vector<double> t(np, 1.0);
        const auto proj = rec.stretches_with(Role::Projection);
        std::string proj_text;
        if (!proj.empty()) {
            std::vector<std::uint16_t> raw(np);
            frames.read(proj[0], 0, 1, raw.data());
            for (std::size_t p = 0; p < np; ++p) t[p] = (raw[p] - ff.dark[p]) / ff.gain[p];
            const auto bm = block_means(t, rows, cols, size, br, bc);
            const Spread sp = spread(bm);
            std::int64_t low = 0;
            for (double v : bm)
                if (v < 0.9) ++low;
            proj_text = fmt("for comparison, projection 0 of %s over the mean flat: %.3f to %.3f, %.0f %% of the blocks below 0.9", rec.stretches[proj[0]].entry.c_str(),
                            sp.min, sp.max, 100.0 * low / bm.size());
        }
        bool ok = true;
        for (std::size_t k = 0; k < flats.size(); ++k) {
            std::vector<double> ratio(np);
            for (std::size_t p = 0; p < np; ++p) ratio[p] = (flats[k]->mean[p] - dark->mean[p]) / ff.gain[p];
            const auto bm = block_means(ratio, rows, cols, size, br, bc);
            const Spread sp = spread(bm);
            const std::size_t n = flats[k]->frame_mean.size();
            const Spread head = spread(flats[k]->frame_mean, 0, std::min<std::size_t>(20, n)), tail = spread(flats[k]->frame_mean, n - std::min<std::size_t>(20, n), n);
            long double level = 0;
            for (std::size_t p = 0; p < np; ++p) level += flats[k]->mean[p] - dark->mean[p];
            parts.push_back(fmt("%s: mean %.1f counts above dark; over the mean of all flats, in %dx%d blocks: %.4f to %.4f (%.4f rms about %.4f); frame mean changes by %+.2f %% through the series",
                                flat_entries[k].c_str(), static_cast<double>(level / np), size, size, sp.min, sp.max, sp.sd, sp.mean, (tail.mean - head.mean) / (head.mean - spread(dark->mean).mean) * 100));
            ok &= sp.min > 0.9 && sp.max < 1.1;
        }
        Finding& f = rep.add("4.2 8.2.2 8.2.5", scan, "flat frames");
        f.stated = "the flat stretches, taken with the sample moved out sideways";
        f.found = join(parts) + (proj_text.empty() ? "" : "; " + proj_text);
        f.verdict = ok ? "holds" : "fails";
        f.note = "verdict: the flat series agree with each other to within 10 % everywhere, so none of them holds something the others do not. "
                 "Something that stays in the beam in every flat cannot be seen this way";
    }
    // ---- 8.1.2: the dark of the other scan
    if (dark && other && other_ff) {
        const SeriesStats* dark2 = nullptr;
        for (std::size_t k = 0; k < other_ff->series.size(); ++k)
            if (other->stretches[other_ff->series_stretch[k]].role == Role::Dark && !dark2) dark2 = &other_ff->series[k];
        if (dark2) {
            std::vector<double> diff(np);
            long double expect = 0;
            for (std::size_t p = 0; p < np; ++p) {
                diff[p] = dark2->mean[p] - dark->mean[p];
                expect += dark->variance[p] / dark->n + dark2->variance[p] / dark2->n;
            }
            const Spread d = spread(diff);
            const double sigma = std::sqrt(static_cast<double>(expect / np));
            std::int64_t moved = 0;
            for (std::size_t p = 0; p < np; ++p)
                if (std::fabs(diff[p] - d.mean) > 6 * std::sqrt(dark->variance[p] / dark->n + dark2->variance[p] / dark2->n) + 1e-9) ++moved;
            Finding& f = rep.add("8.1.2", rec.label + "+" + other->label, "dark level per pixel in the two scans");
            f.stated = "a dark series at the start of each scan, none at the end";
            f.found = fmt("second minus first: %+.3f counts on average; %.3f counts rms from pixel to pixel where the noise of the two means gives %.3f; %lld pixels changed by more than six times their noise",
                          d.mean, d.sd, sigma, static_cast<long long>(moved));
            f.verdict = std::fabs(d.mean) < 1.0 && d.sd < 2 * sigma ? "holds" : "fails";
            f.note = "verdict: the offset is the same in both scans to within a count. Nothing shows the offset during a scan";
            f.numbers = {{"mean_difference", d.mean}, {"rms_difference", d.sd}, {"noise_expected", sigma}};
        }
    }
    (void)other_frames;
}

// ------------------------------------------------------------------------------------------------ beam

namespace {

// linear interpolation of the monitor's reading at time t; NaN readings are skipped
double monitor_at(const Monitor& m, double t) {
    const std::size_t n = std::min(m.epoch.size(), m.current.size());
    if (!n) return NAN;
    std::size_t hi = std::lower_bound(m.epoch.begin(), m.epoch.begin() + n, t) - m.epoch.begin();
    std::size_t lo = hi ? hi - 1 : 0;
    while (hi < n && !std::isfinite(m.current[hi])) ++hi;
    while (lo > 0 && !std::isfinite(m.current[lo])) --lo;
    if (hi >= n) return m.current[lo];
    if (!std::isfinite(m.current[lo]) || hi == lo || m.epoch[hi] == m.epoch[lo]) return m.current[hi];
    const double w = std::clamp((t - m.epoch[lo]) / (m.epoch[hi] - m.epoch[lo]), 0.0, 1.0);
    return m.current[lo] + w * (m.current[hi] - m.current[lo]);
}

}  // namespace

void check_beam(const Record& rec, const Inventory& inv, const FlatField& ff, Report& rep) {
    const std::string scan = rec.label;
    const int cols = ff.cols;
    const int ns = n_strips(inv, cols);
    // mean ring current over the flat frames: the reference the mean flat belongs to
    double flat_current = 0, flat_frames = 0;
    for (int i : rec.stretches_with(Role::Flat)) {
        const Stretch& s = rec.stretches[i];
        for (double c : s.monitor.current)
            if (std::isfinite(c)) flat_current += c, flat_frames += 1;
    }
    flat_current = flat_frames ? flat_current / flat_frames : NAN;

    for (int si : rec.stretches_with(Role::Projection)) {
        const Stretch& s = rec.stretches[si];
        const auto& data = inv.stretches.at(si);
        const std::size_t n = static_cast<std::size_t>(s.n_frames);
        if (data.strips.size() != n * ns || s.stat_h <= 0) continue;
        // dark and gain of each strip over the statistics rectangle
        std::vector<double> dark(ns, 0), gain(ns, 0);
        for (int k = 0; k < ns; ++k) {
            const int a = std::max(s.stat_x, k * inv.strip_cols), b = std::min(s.stat_x + s.stat_w, (k + 1) * inv.strip_cols);
            double d = 0, g = 0;
            for (int r = s.stat_y; r < s.stat_y + s.stat_h; ++r)
                for (int c = a; c < b; ++c) d += ff.dark[static_cast<std::size_t>(r) * cols + c], g += ff.gain[static_cast<std::size_t>(r) * cols + c];
            const double count = static_cast<double>(std::max(1, b - a)) * s.stat_h;
            dark[k] = d / count, gain[k] = g / count;
        }
        // strips that stay in air through the whole stretch: transmission never below 0.9
        std::vector<int> air;
        for (int k = 0; k < ns; ++k) {
            double lo = 1e30;
            for (std::size_t f = 0; f < n; ++f) lo = std::min(lo, (data.strips[f * ns + k] - dark[k]) / gain[k]);
            if (lo >= 0.9) air.push_back(k);
        }
        Finding& f = rep.add("4.7 8.2.1 8.2.3 8.2.4 8.2.5", scan, "level of the air beside the sample, stretch " + s.entry);
        f.stated = fmt("flats before and after the scan (mean ring current %.3f mA) and the ring current through the stretch", flat_current);
        if (air.empty()) {
            f.found = "no column strip stays above a transmission of 0.9 through the stretch";
            f.verdict = "open";
            continue;
        }
        std::vector<double> level(n), current(n), ratio(n), index(n);
        for (std::size_t fr = 0; fr < n; ++fr) {
            double t = 0;
            for (int k : air) t += (data.strips[fr * ns + k] - dark[k]) / gain[k];
            level[fr] = t / air.size();
            current[fr] = s.epoch_trig.size() == n ? monitor_at(s.monitor, s.epoch_trig[fr]) : NAN;
            ratio[fr] = level[fr] / (current[fr] / flat_current);
            index[fr] = static_cast<double>(fr);
        }
        const Spread lv = spread(level), rt = spread(ratio);
        // frame-to-frame scatter: rms of the first difference / sqrt(2)
        long double dd = 0;
        for (std::size_t fr = 1; fr < n; ++fr) dd += (level[fr] - level[fr - 1]) * (level[fr] - level[fr - 1]);
        const double jitter = std::sqrt(static_cast<double>(dd / (n - 1)) / 2);
        const std::size_t m = std::min<std::size_t>(200, n / 4);
        const double l0 = spread(level, 0, m).mean, l1 = spread(level, n - m, n).mean, c0 = spread(current, 0, m).mean, c1 = spread(current, n - m, n).mean;
        f.found = fmt("air = columns %d..%d of the statistics rows (%zu strips of %d columns that never fall below 0.9); transmission %.4f on average, %.4f to %.4f; "
                      "from frame to frame %.4f rms; first %zu frames %.4f, last %.4f (%+.2f %%) while the ring current goes %.3f -> %.3f mA (%+.2f %%); "
                      "air level over (current / current of the flats): %.4f on average, %.4f to %.4f",
                      air.front() * inv.strip_cols, std::min(cols, (air.back() + 1) * inv.strip_cols) - 1, air.size(), inv.strip_cols, lv.mean, lv.min, lv.max, jitter, m, l0, l1,
                      (l1 / l0 - 1) * 100, c0, c1, (c1 / c0 - 1) * 100, rt.mean, rt.min, rt.max);
        f.numbers = {{"air_mean", lv.mean}, {"air_min", lv.min}, {"air_max", lv.max}, {"air_over_current_mean", rt.mean}, {"air_over_current_min", rt.min},
                     {"air_over_current_max", rt.max}, {"air_strips", static_cast<double>(air.size())}};
        // a rise of the beam: where the air level steps up, and the monitor's delay that best explains it
        std::size_t rise0 = n, rise1 = 0;
        {
            // smoothed level over 15 frames; a rise is where it exceeds the running minimum of the past by 2 %
            const std::size_t win = 15;
            std::vector<double> smooth(n);
            for (std::size_t fr = 0; fr < n; ++fr) smooth[fr] = spread(level, fr >= win / 2 ? fr - win / 2 : 0, std::min(n, fr + win / 2 + 1)).mean;
            for (std::size_t fr = 400; fr < n; ++fr) {
                const double before = smooth[fr - 400];
                if (smooth[fr] > 1.02 * before) {
                    rise0 = std::min(rise0, fr - 400);
                    rise1 = std::max(rise1, fr);
                }
            }
        }
        if (rise0 < rise1 && s.epoch_trig.size() == n) {
            // narrow to where the level actually changes: first and last frame at 2 % and 98 % of the step
            const double base = spread(level, rise0 > 100 ? rise0 - 100 : 0, rise0 + 1).mean, top = spread(level, rise1, std::min(n, rise1 + 100)).mean;
            std::size_t a = rise0, b = rise1;
            while (a < rise1 && level[a] < base + 0.02 * (top - base)) ++a;
            while (b > a && level[b] > base + 0.98 * (top - base)) --b;
            // delay: the level is modelled as alpha * current(t - delay) + beta over the frames around the rise
            const std::size_t w0 = a > 300 ? a - 300 : 0, w1 = std::min(n, b + 300);
            double best_delay = NAN, best_rms = 1e30, best_alpha = NAN;
            for (double delay = -10.0; delay <= 10.0001; delay += 0.05) {
                std::vector<double> x(w1 - w0), y(w1 - w0);
                for (std::size_t fr = w0; fr < w1; ++fr) x[fr - w0] = monitor_at(s.monitor, s.epoch_trig[fr] + delay), y[fr - w0] = level[fr];
                const Line l = fit_line(x, y, 0, x.size());
                if (l.rms < best_rms) best_rms = l.rms, best_delay = delay, best_alpha = l.slope;
            }
            f.found += fmt("; the air level rises by %+.1f %% between projections %zu and %zu (2 %% and 98 %% of the step); modelled as a multiple of the ring current read %.2f s %s the frame "
                           "(residual %.4f rms, %.2f %% of the level per %% of current)",
                           (top / base - 1) * 100, a, b, std::fabs(best_delay), best_delay >= 0 ? "after" : "before", best_rms, best_alpha * spread(current, w0, w1).mean / spread(level, w0, w1).mean);
            f.numbers.push_back({"rise_first_frame", static_cast<double>(a)});
            f.numbers.push_back({"rise_last_frame", static_cast<double>(b)});
            f.numbers.push_back({"monitor_delay_s", best_delay});
        }
        f.note = "air is found from the frames themselves; nothing here assumes where the sample is. The flat used is the mean of all the scan's flats";
    }
}

}  // namespace helirec::calibration
